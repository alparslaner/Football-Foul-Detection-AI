#!/usr/bin/env python

from PyQt5 import QtCore
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import cv2
import pandas as pd
import json
import os
import numpy as np

# --- moviepy import'ları kod içinde kullanılmıyordu, tamamen kaldırıldı ---
# from moviepy.editor import *
# from moviepy.config import get_setting

from interface.model import MVNetwork
import torch
from torchvision.io.video import read_video
import torch.nn as nn
from interface.config.classes import (
    EVENT_DICTIONARY,
    INVERSE_EVENT_DICTIONARY_action_class,
    INVERSE_EVENT_DICTIONARY_offence_severity_class
)
from torchvision.models.video import MViT_V2_S_Weights

# Faul sınıflandırıcı için gerekli sabitler
FEATURE_WEIGHTS = np.array([
    1.0,   # Body Part
    1.0,   # Action Class
    0.8,   # Multiple Fouls
    0.7,   # Try to Play
    0.7,   # Touch Ball
    0.8,   # Close Goal Position
    1.35    # Severity
])

CATEGORY_MEANS_7D = {
    "NoFoul":            np.array([-0.0267, 0.4833, -0.2733, -0.2300,  0.00,   -0.2667, 0.3133]),
    "BorderlineNoFoul":  np.array([-0.1000, 0.5333, -0.3700, -0.2067,  0.0767,  0.1033, 0.2900]),
    "FoulNoCard":        np.array([-0.1233, 0.4800, -0.3167, -0.2000,  0.0700,  0.4933, 0.3367]),
    "FoulYellow":        np.array([-0.0933, 0.4867, -0.2767, -0.2400,  0.0400,  0.4900, 0.3800]),
    "FoulUnknown":       np.array([-0.1233, 0.4600, -0.2633, -0.1233,  0.0433,  0.5000, 0.3533]),
    "FoulRed":           np.array([-0.0767, 0.4800, -0.3033, -0.2300,  0.0367,  0.4667, 0.3400]),
}

def classify_foul_physical(features_7d):
    """
    7 feature'lı vektör için fiziksel ağırlık odaklı sınıflandırma
    """
    distances = {
        category: np.sqrt(np.sum(FEATURE_WEIGHTS * (features_7d - center) ** 2))
        for category, center in CATEGORY_MEANS_7D.items()
    }
    closest = min(distances, key=distances.get)

    # Karar metni
    if closest == "NoFoul":
        return "No Foul"
    elif closest in ["BorderlineNoFoul", "FoulNoCard"]:
        return "Foul - No Card"
    elif closest in ["FoulYellow", "FoulUnknown"]:
        return "Foul - Yellow Card"
    elif closest == "FoulRed":
        return "Foul - Red Card"
    else:
        return "Unknown"

class VideoWindow(QMainWindow):

    def __init__(self, parent=None, device=None):
        super(VideoWindow, self).__init__(parent)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # True if prediction is shown, False if not
        self.show_prediction = True
        rootdir = os.getcwd()

        # Load model
        self.model = MVNetwork(net_name="slowfast", agr_type="attention")
        path = os.path.join(rootdir, 'interface')
        path = os.path.join(path, 'best_model.pth')
        path = path.replace('\\', '/' )

        # Load weights
        load = torch.load(path, map_location=torch.device('cpu'))
        if isinstance(load, dict) and 'state_dict' in load:
            state_dict = load['state_dict']
            # Remove 'module.' prefix if it exists
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # Add 'mvnetwork.' prefix to match the model structure
            state_dict = {'mvnetwork.' + k: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            # Remove 'module.' prefix if it exists
            load = {k.replace('module.', ''): v for k, v in load.items()}
            # Add 'mvnetwork.' prefix to match the model structure
            load = {'mvnetwork.' + k: v for k, v in load.items()}
            self.model.load_state_dict(load, strict=False)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Set batch normalization layers to eval mode
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm3d):
                module.eval()
                module.track_running_stats = True
                
        self.softmax = nn.Softmax(dim=1)

        self.setWindowTitle("Video Assistent Referee System")

        path = os.path.join(rootdir, 'interface')
        path_image = os.path.join(path, 'vars_logo.png')
        path_image = path_image.replace('\\', '/' )

        self.setStyleSheet("background: #0F0F65;")

        #######################
        # CREATE VIDEO WIDGETS#
        #######################

        self.mediaPlayers = []
        self.videoWidgets = []
        self.frame_duration_ms = 40        

        self.files = []
        self.current_clips = None

        for i in range(4):
            self.mediaPlayers.append(QMediaPlayer(
                None, QMediaPlayer.VideoSurface))
            self.videoWidgets.append(QVideoWidget())
            self.mediaPlayers[i].setVideoOutput(self.videoWidgets[i])
        
        # create video layout
        upperLayout = QHBoxLayout()
        upperLayout.setContentsMargins(0, 0, 0, 0)
        upperLayout.addWidget(self.videoWidgets[0])
        upperLayout.addWidget(self.videoWidgets[1])

        bottomLayout = QHBoxLayout()
        bottomLayout.setContentsMargins(0, 0, 0, 0)
        bottomLayout.addWidget(self.videoWidgets[2])
        bottomLayout.addWidget(self.videoWidgets[3])

        finalLayout = QVBoxLayout()
        finalLayout.setContentsMargins(0, 0, 0, 0)
        finalLayout.addLayout(upperLayout)
        finalLayout.addLayout(bottomLayout)

        # sidebar
        sidebar = QVBoxLayout()
        sidebar.insertSpacing(0, 100)

        ################################################
        # CREATE LAYOUT WITH PLAY STOP AND OPEN BUTTONS#
        ################################################

        # Create play button and shortcuts
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        font2 = QFont('Arial', 10)
        #font2.setBold(True)
        self.playButton.setFont(font2)
        self.playButton.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.playButton.setText("Play")
        self.playButton.clicked.connect(self.play)

        playShortcut = QShortcut(QKeySequence("Space"), self)
        playShortcut.activated.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                                      QSizePolicy.Maximum)
        self.errorLabel.hide()

        # Create open button
        openButton = QPushButton("Open files", self)
        openButton.setFont(font2)
        openButton.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        openButton.clicked.connect(self.openFile)
        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addWidget(openButton)

        ##################################################
        # LAYOUT FOR SHOWING GROUND TRUTH AND PREDICTIONS#
        ##################################################

        # Create the text labels for the annotated properties of the action
        decisionLayout = QVBoxLayout()
        font = QFont('Arial', 20)
        font.setBold(True)
        self.decisionTitle = QLabel("Groundtruth")
        self.decisionTitle.setAlignment(Qt.AlignCenter)
        self.decisionTitle.setFont(font)
        self.decisionTitle.setStyleSheet("color: rgb(255,255,255)")

        # Create the label for the classification result
        self.decisionResult = QLabel("")
        self.decisionResult.setAlignment(Qt.AlignCenter)
        font_result = QFont('Arial', 14)
        self.decisionResult.setFont(font_result)
        self.decisionResult.setStyleSheet("color: rgb(255,255,255); background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-top: 10px;")
        self.decisionResult.hide() # Hide initially

        fontText = QFont('Arial', 14)

        self.spacetext = QLabel("")

        self.actionText = QLabel("")
        self.actionText.setStyleSheet("color: rgb(255,255,255)")
        self.actionText.setAlignment(Qt.AlignCenter)
        self.actionText.setFont(fontText)

        self.offenceText = QLabel("")
        self.offenceText.setStyleSheet("color: rgb(255,255,255)")
        self.offenceText.setAlignment(Qt.AlignCenter)
        self.offenceText.setFont(fontText)
        
        self.prediction1Text = QLabel("")
        self.prediction1Text.setStyleSheet("color: rgb(255,255,255)")
        self.prediction1Text.setAlignment(Qt.AlignCenter)
        self.prediction1Text.setFont(fontText)
        self.prediction1Text.hide()

        self.prediction2Text = QLabel("")
        self.prediction2Text.setStyleSheet("color: rgb(255,255,255)")
        self.prediction2Text.setAlignment(Qt.AlignCenter)
        self.prediction2Text.setFont(fontText)
        self.prediction2Text.hide()

        self.prediction3Text = QLabel("")
        self.prediction3Text.setStyleSheet("color: rgb(255,255,255)")
        self.prediction3Text.setAlignment(Qt.AlignCenter)
        self.prediction3Text.setFont(fontText)
        self.prediction3Text.hide()

        self.prediction4Text = QLabel("")
        self.prediction4Text.setStyleSheet("color: rgb(255,255,255)")
        self.prediction4Text.setAlignment(Qt.AlignCenter)
        self.prediction4Text.setFont(fontText)
        self.prediction4Text.hide()

        decisionLayout.addWidget(self.decisionTitle)
        decisionLayout.addWidget(self.decisionResult)
        decisionLayout.addWidget(self.spacetext)
        decisionLayout.addWidget(self.offenceText)
        decisionLayout.addWidget(self.actionText)

        decisionLayout.addWidget(self.spacetext)
        decisionLayout.addWidget(self.spacetext)
        decisionLayout.addWidget(self.spacetext)
        decisionLayout.addWidget(self.prediction1Text)
        decisionLayout.addWidget(self.prediction2Text)
        decisionLayout.addWidget(self.spacetext)
        decisionLayout.addWidget(self.prediction3Text)
        decisionLayout.addWidget(self.prediction4Text)

        # Add Feature display area
        self.featureTitle = QLabel("Extracted Features")
        self.featureTitle.setFont(font)
        self.featureTitle.setAlignment(Qt.AlignCenter)
        self.featureTitle.setStyleSheet("color: rgb(255,255,255)")

        # Create feature labels
        self.feature_labels = {}
        features = [
            "Body Part",
            "Action Class",
            "Multiple Fouls",
            "Try to Play",
            "Touch Ball",
            "Close Goal Position",
            "Severity"
        ]

        # Create a grid layout for features
        feature_layout = QGridLayout()
        feature_layout.setSpacing(10)

        for i, feature in enumerate(features):
            # Feature name label
            name_label = QLabel(feature)
            name_label.setFont(font)
            name_label.setStyleSheet("color: rgb(255,255,255); font-weight: bold;")
            feature_layout.addWidget(name_label, i, 0)

            # Feature value label
            value_label = QLabel("0.00")
            value_label.setFont(font)
            value_label.setStyleSheet("color: rgb(255,255,255); padding: 5px; background-color: rgba(0,0,0,0.3); border-radius: 3px;")
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            feature_layout.addWidget(value_label, i, 1)
            self.feature_labels[feature] = value_label

        # Add feature layout to main layout
        decisionLayout.addWidget(self.featureTitle)
        decisionLayout.addLayout(feature_layout)

        ##############################################
        # LAYOUT FOR BUTTONS TO SHOW A SPECIFIC VIDEO#
        ##############################################

        holder = QVBoxLayout()
        self.holdertext = QLabel("")
        holder.addWidget(self.holdertext)

        showVideoLayout = QVBoxLayout()

         # Create show video 1 button
        self.showVid1 = QPushButton("Show video 1", self)
        self.showVid1.setFont(font2)
        self.showVid1.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.showVid1.clicked.connect(self.enlargeV1)

        # Create show video 2 button
        self.showVid2 = QPushButton("Show video 2", self)
        self.showVid2.setFont(font2)
        self.showVid2.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.showVid2.clicked.connect(self.enlargeV2)

        # Create show video 3 button
        self.showVid3 = QPushButton("Show video 3", self)
        self.showVid3.setFont(font2)
        self.showVid3.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.showVid3.clicked.connect(self.enlargeV3)

        # Create show video 4 button
        self.showVid4 = QPushButton("Show video 4", self)
        self.showVid4.setFont(font2)
        self.showVid4.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.showVid4.clicked.connect(self.enlargeV4)

        # Create show all buttons again
        self.showAllVid = QPushButton("Show all videos", self)
        self.showAllVid.setFont(font2)
        self.showAllVid.setStyleSheet("background:#DBDBDB;"
            "color: rgb(0,0,0)")
        self.showAllVid.clicked.connect(self.allVideos)

        showVideoLayout.addWidget(self.showVid1)
        showVideoLayout.addWidget(self.showVid2)
        showVideoLayout.addWidget(self.showVid3)
        showVideoLayout.addWidget(self.showVid4)
        showVideoLayout.addWidget(self.showAllVid)

        self.decisionTitle.hide()
        self.showVid1.hide()
        self.showVid2.hide()
        self.showVid3.hide()
        self.showVid4.hide()
        self.showAllVid.hide()
        
        sidebar.addLayout(decisionLayout, 0)
        sidebar.addLayout(holder, 1)
        sidebar.addLayout(showVideoLayout, 2)

        #################################
        # ADD ALL LAYOUTS TO MAIN LAYOUT#
        #################################

        mainScreen = QGridLayout()
        mainScreen.addLayout(finalLayout, 0, 0)
        mainScreen.addLayout(controlLayout, 3, 0)
        mainScreen.addWidget(self.errorLabel, 4, 0)
        mainScreen.addLayout(sidebar, 0, 1)

        # Set widget to contain window contents
        wid.setLayout(mainScreen)

        # creating label
        self.label = QLabel(self)
        self.label.setGeometry(QtCore.QRect(500, 0, 1000, 900))
        # loading image
        self.pixmap = QPixmap(path_image)
        # adding image to label
        self.label.setPixmap(self.pixmap)


        for i in self.mediaPlayers:
            i.stateChanged.connect(self.mediaStateChanged)
            i.positionChanged.connect(self.positionChanged)
            i.durationChanged.connect(self.durationChanged)
            i.error.connect(self.handleError)

        for i in self.videoWidgets:
            i.hide()

        # __init__ içinde, uygun bir yere ekleyin:
        self.extractFeaturesButton = QPushButton("Extract Features")
        self.extractFeaturesButton.clicked.connect(self.process_video)
        decisionLayout.addWidget(self.extractFeaturesButton)
        self.extractFeaturesButton.setStyleSheet("background-color: rgb(255,255,255)")
    # Function to only show video 1
    def enlargeV1(self):
        for i in self.videoWidgets:
            i.hide()
        cou = 0
        index = 1
        for w in self.videoWidgets:
            cou += 1
            if cou > len(self.files):
                continue
            if index == cou:
                w.show()

        for m, f in zip(self.mediaPlayers, self.files):
            m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()

    # Function to only show video 2
    def enlargeV2(self):
        for i in self.videoWidgets:
            i.hide()
        cou = 0
        index = 2
        for w in self.videoWidgets:
            cou += 1
            if cou > len(self.files):
                continue
            if index == cou:
                w.show()
        for m, f in zip(self.mediaPlayers, self.files):
            m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()

    # Function to only show video 3
    def enlargeV3(self):
        for i in self.videoWidgets:
            i.hide()
        cou = 0
        index = 3
        for w in self.videoWidgets:
            cou += 1
            if cou > len(self.files):
                continue
            if index == cou:
                w.show()

        for m, f in zip(self.mediaPlayers, self.files):
            m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()

    # Function to only show video 4
    def enlargeV4(self):
        for i in self.videoWidgets:
            i.hide()
        cou = 0
        index = 4
        for w in self.videoWidgets:
            cou += 1
            if cou > len(self.files):
                continue
            if index == cou:
                w.show()

        for m, f in zip(self.mediaPlayers, self.files):
            m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()
        
    # Function to show all videos
    def allVideos(self):
        cou = 0
        for w in self.videoWidgets:
            cou += 1
            if cou > len(self.files):
                continue
            w.show()

        for m, f in zip(self.mediaPlayers, self.files):
            m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()

    # EVENTS FOR PRESSING KEYS
    def keyPressEvent(self, event):

        # Move one frame forward in time
        if event.text() == "a" and self.mediaPlayers[0].state() != QMediaPlayer.PlayingState:
            position = self.mediaPlayers[0].position()
            if position > self.frame_duration_ms:
                for i in self.mediaPlayers:
                    i.setPosition(position-self.frame_duration_ms)
                    self.setFocus()

        # Move one frame backwards in time
        if event.text() == "d" and self.mediaPlayers[0].state() != QMediaPlayer.PlayingState:
            position = self.mediaPlayers[0].position()
            duration = self.mediaPlayers[0].duration()
            if position < duration - self.frame_duration_ms:
                for i in self.mediaPlayers:
                    i.setPosition(position+self.frame_duration_ms)
                    self.setFocus()

        # Set the playback rate to normal
        if event.key() == Qt.Key_F1:
            position = self.mediaPlayers[0].position()
            for i in self.mediaPlayers:
                i.setPlaybackRate(1)
                i.setPosition(position)
                self.setFocus()

        # Set the playback rate to 0.5
        if event.key() == Qt.Key_F2:
            position = self.mediaPlayers[0].position()
            for i in self.mediaPlayers:
                i.setPlaybackRate(0.5)
                i.setPosition(position)
                self.setFocus()

        # Set the playback rate to 0.3
        if event.key() == Qt.Key_F3:
            position = self.mediaPlayers[0].position()
            for i in self.mediaPlayers:
                i.setPlaybackRate(0.3)
                i.setPosition(position)
                self.setFocus()

        # Set the playback rate to 0.25
        if event.key() == Qt.Key_F4:
            position = self.mediaPlayers[0].position()
            for i in self.mediaPlayers:
                i.setPlaybackRate(0.25)
                i.setPosition(position)
                self.setFocus()

        # Set the playback rate to 0.2
        if event.key() == Qt.Key_F5:
            position = self.mediaPlayers[0].position()
            for i in self.mediaPlayers:
                i.setPlaybackRate(0.2)
                i.setPosition(position)
                self.setFocus()

        # Start the clip from the beginning
        if event.text() == "s":
            for i in self.mediaPlayers:
                i.setPosition(2500)
                i.play()
                i.setMuted(True)

        # Set the clip at the moment where we have the annotation
        if event.text() == "k":
            for i in self.mediaPlayers:
                i.setPosition(3000)

        # Open a file
        if event.text() == "o":
            self.openFile()

    # Function to open files
    def openFile(self):
        for i in self.videoWidgets:
            i.hide()

        files, _ = QFileDialog.getOpenFileNames(
            self, "Select up to 4 files", QDir.homePath())

        if len(files) != 0:
            self.files = files
            self.decisionTitle.hide()
            self.showVid1.hide()
            self.showVid2.hide()
            self.showVid3.hide()
            self.showVid4.hide()
            self.showAllVid.hide()
            self.label.hide()

            if self.show_prediction:
                factor = (85 - 65) / (((85 - 65) / 25) * 21)
                clips = []
                for num_view in range(len(files)):
                    video, _, _ = read_video(files[num_view], output_format="THWC")
                    frames = video[65:85,:,:,:]
                    final_frames = None
                    transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()

                    for j in range(len(frames)):
                        if j%factor<1:
                            if final_frames == None:
                                final_frames = frames[j,:,:,:].unsqueeze(0)
                            else:
                                final_frames = torch.cat((final_frames, frames[j,:,:,:].unsqueeze(0)), 0)

                    final_frames = final_frames.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
                    final_frames = transforms_model(final_frames)

                    # Print shape after transform
                    print("Shape after transform:", final_frames.shape)

                    # Prepare input for SlowFast model
                    # Slow path: sample every 8th frame (alpha=8)
                    # transforms sonrası final_frames: [T, C, H, W]
                    alpha = 8
                    T_slow = 16  # Changed from 8 to 16
                    T_fast = alpha * T_slow  # 128 frames

                    # Frame sayısını garanti altına al
                    if final_frames.shape[0] < T_fast:
                        repeat_times = (T_fast + final_frames.shape[0] - 1) // final_frames.shape[0]
                        final_frames = final_frames.repeat(repeat_times, 1, 1, 1)[:T_fast]
                    else:
                        final_frames = final_frames[:T_fast]

                    # Sampling with adjusted stride
                    stride = T_fast // T_slow  # Calculate stride to get exactly T_slow frames
                    slow_frames = final_frames[::stride]       # [16, C, H, W]
                    fast_frames = final_frames                # [128, C, H, W]

                    # Ensure both paths have the same number of channels
                    num_channels = 3  # SlowFast model expects 3 channels
                    slow_frames = slow_frames[:, :num_channels, :, :]
                    fast_frames = fast_frames[:, :num_channels, :, :]

                    # Ensure temporal dimensions match
                    if slow_frames.shape[0] != T_slow:
                        slow_frames = slow_frames[:T_slow]
                    if fast_frames.shape[0] != T_fast:
                        fast_frames = fast_frames[:T_fast]

                    print("Slow frames shape:", slow_frames.shape)
                    print("Fast frames shape:", fast_frames.shape)

                    # Her view için slow ve fast frame'leri bir tuple olarak ekle
                    clips.append((slow_frames, fast_frames))

                # self.current_clips'i doldur
                self.current_clips = clips
                self.extractFeaturesButton.setEnabled(True)
                print("Video(lar) yüklendi ve buton aktif!")
            else:
                print("Video(lar) okunamadı!")

            self.label.hide()
            cou = 0
            for w in self.videoWidgets:
                cou += 1
                if cou > len(files):
                    continue
                w.show()

            self.decisionTitle.show()
            self.offenceText.show()

            if len(files) >= 2:
                self.showVid1.show()
                self.showVid2.show()
                self.showAllVid.show()

            if len(files) >= 3:
                self.showVid3.show()

            if len(files) >= 4:
                self.showVid4.show()

            for m, f in zip(self.mediaPlayers, files):
                m.setMedia(QMediaContent(QUrl.fromLocalFile(f)))
            self.playButton.setEnabled(True)
            self.setPosition(2500)
            self.play()
        else:
            self.allVideos()


    def play(self):
        for i in self.mediaPlayers:
            if i.state() == QMediaPlayer.PlayingState:
                i.pause()
            else:
                i.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayers[0].state() == QMediaPlayer.PlayingState:
            self.playButton.setText("Pause")
        else:
            self.playButton.setText("Play")

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        for i in self.mediaPlayers:
            i.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayers[0].errorString())

    def process_video(self):
        print("Extract Features butonuna basıldı!")
        if not hasattr(self, 'current_clips') or self.current_clips is None:
            print("Video yüklenmemiş!")
            return
        try:
            mvinputs = self.prepare_model_input(self.current_clips).to(self.device)
            # extract_features now returns [B,7] tensor directly
            with torch.no_grad():
                feats = self.model.extract_features(mvinputs)
            feats = feats.squeeze(0).cpu().numpy()  # [7]
            print(f"Features shape: {feats.shape}")
            print(f"Features values: {feats}")

            # Update feature labels
            features = [
                "Body Part",
                "Action Class",
                "Multiple Fouls",
                "Try to Play",
                "Touch Ball",
                "Close Goal Position",
                "Severity"
            ]
            for i, (feature, value) in enumerate(zip(features, feats)):
                self.feature_labels[feature].setText(f"{value:.2f}")
            
            # Sınıflandırma yap ve sonucu göster
            decision = classify_foul_physical(feats)
            
            # Update the decision result label
            self.decisionResult.setText(decision)
            self.decisionResult.show()
                
        except Exception as e:
            print(f"Error in process_video: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_current_frame(self):
        if hasattr(self, 'cap') and self.cap is not None:
            pos_msec = self.mediaPlayers[0].position()  # milisaniye cinsinden pozisyon
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                return None
            frame_number = int((pos_msec / 1000.0) * fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def prepare_model_input(self, clips):
        """
        Prepare input for R(2+1)D network.
        Args:
            clips: List of video clips, each of shape [C, T, H, W] or [T, C, H, W]
        Returns:
            Tensor of shape [1, V, C, T, H, W] where:
            - V: number of views
            - C: channels (3 for RGB)
            - T: temporal dimension (16 frames)
            - H, W: spatial dimensions (224x224)
        """
        if clips is None:
            return None

        # 1) Convert to list if not already
        if not isinstance(clips, (list, tuple)):
            clips = [clips]
        else:
            # Flatten nested lists/tuples
            flat = []
            for c in clips:
                if isinstance(c, (list, tuple)):
                    flat.extend(list(c))
                else:
                    flat.append(c)
            clips = flat

        tensors = []
        for clip in clips:
            # A) Convert numpy array to tensor
            if isinstance(clip, np.ndarray):
                if clip.ndim == 4:
                    t = torch.from_numpy(clip).float().permute(2, 3, 0, 1)
                else:
                    raise ValueError(f"Unexpected np.ndarray ndim={clip.ndim}")
            # B) Handle torch tensor
            elif torch.is_tensor(clip):
                if clip.dim() == 4 and clip.shape[0] in (1,3):
                    # [C, T, H, W] format
                    t = clip.float()
                elif clip.dim() == 4 and clip.shape[1] in (1,3):
                    # [T, C, H, W] format
                    t = clip.permute(1, 0, 2, 3).float()
                else:
                    raise ValueError(f"Unexpected tensor shape: {tuple(clip.shape)}")
            else:
                raise TypeError(f"Unsupported clip type: {type(clip)}")
            tensors.append(t)

        # Ensure all tensors have same temporal dimension
        T0 = min(t.shape[1] for t in tensors)
        tensors = [ (t[:, :T0] if t.shape[1]>=T0 else torch.cat([t, t[:, -1:].repeat(1, T0-t.shape[1], 1, 1)],1))
                    for t in tensors ]
        
        # Stack views and add batch dimension
        mv = torch.stack(tensors, dim=0)
        return mv.unsqueeze(0).to(self.device)

    def extract_features(self, mvinputs):
        with torch.no_grad():
            return self.forward(mvinputs)

