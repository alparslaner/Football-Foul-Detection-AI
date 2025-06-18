from PyQt5.QtWidgets import (QMainWindow, QLabel, QGridLayout, QWidget, 
                            QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout)
from PyQt5.QtCore import Qt, QUrl, QDir
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtMultimedia import QMediaContent
from SoccerNet.Evaluation.MV_FoulRecognition import MV_FoulRecognition
import torch
import numpy as np
import cv2
import os
import json

class VideoWindow(QMainWindow):
    def __init__(self, model, device):
        super(VideoWindow, self).__init__()
        
        # Model ve device'ı kaydet
        self.model = model
        self.device = device
        self.current_clips = None
        
        # Action ve offence sınıflarını tanımla
        self.action_classes = [
    "Tackling",
    "Standing tackling",
    "High leg",
    "Holding",
    "Pushing",
    "Elbowing",
    "Challenge",
    "Dive"
]
        
        self.offence_classes = [
    "No offence",
    "Offence + No card",
    "Offence + Yellow card",
    "Offence + Red card"
]
        
        # Ana widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Ana layout
        self.mainLayout = QHBoxLayout()
        self.central_widget.setLayout(self.mainLayout)
        
        # Sol panel için bir widget ve layout oluştur
        self.leftPanel = QWidget()
        self.leftLayout = QVBoxLayout()
        self.leftPanel.setLayout(self.leftLayout)
        
        # Load Video butonu ekle
        self.loadButton = QPushButton("Load Video")
        self.loadButton.clicked.connect(self.openFile)
        self.leftLayout.insertWidget(0, self.loadButton)
        
        # Extract Features butonu
        self.extractButton = QPushButton("Extract Featuresss")
        self.extractButton.clicked.connect(self.on_extract_clicked)
        self.extractButton.setEnabled(False)  # Video yüklenince True yapılmalı
        self.extractButton.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border: 1px solid #ccc;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        self.leftLayout.addWidget(self.extractButton)
        
        # Feature değerlerini gösterecek QLabels
        self.bodyPartValueLabel = QLabel("0.00")
        self.actionClassValueLabel = QLabel("0.00")
        self.multipleFoulsLabel = QLabel("0.00")
        self.tryToPlayLabel = QLabel("0.00")
        self.touchBallLabel = QLabel("0.00")
        self.closeGoalPosLabel = QLabel("0.00")
        self.severityLabel = QLabel("0.00")
        
        # Label'ları sağa hizala
        for label in [self.bodyPartValueLabel, self.actionClassValueLabel, self.multipleFoulsLabel,
                     self.tryToPlayLabel, self.touchBallLabel, self.closeGoalPosLabel, self.severityLabel]:
            label.setAlignment(Qt.AlignRight)
        
        # Feature grid'ini oluştur
        self.featureWidget = QWidget()
        grid = QGridLayout()
        
        feature_names = ["Body Part", "Action Class", "Multiple Fouls",
                        "Try to Play", "Touch Ball", "Close Goal Position", "Severity"]
        value_labels = [self.bodyPartValueLabel,
                       self.actionClassValueLabel,
                       self.multipleFoulsLabel,
                       self.tryToPlayLabel,
                       self.touchBallLabel,
                       self.closeGoalPosLabel,
                       self.severityLabel]
        
        for i, (name, val_lbl) in enumerate(zip(feature_names, value_labels)):
            grid.addWidget(QLabel(name), i, 0)
            grid.addWidget(val_lbl, i, 1)
        
        self.featureWidget.setLayout(grid)
        self.leftLayout.addWidget(self.featureWidget)
        
        # Sol paneli ana layout'a ekle
        self.mainLayout.addWidget(self.leftPanel)
        
        # Sağ panel (karar layout'u) için
        self.rightWidget = QWidget()
        self.decisionLayout = QVBoxLayout()
        self.rightWidget.setLayout(self.decisionLayout)
        self.rightWidget.setFixedWidth(400)
        self.rightWidget.setStyleSheet("background-color: #0F0F65;")
        
        # Karar başlığı
        self.decisionTitle = QLabel("Decision")
        self.decisionTitle.setFont(QFont('Arial', 16, QFont.Bold))
        self.decisionTitle.setAlignment(Qt.AlignCenter)
        self.decisionTitle.setStyleSheet("color: rgb(255,255,255)")
        
        # Prediction başlığı
        self.predictionTitle = QLabel("Prediction")
        self.predictionTitle.setFont(QFont('Arial', 14, QFont.Bold))
        self.predictionTitle.setAlignment(Qt.AlignCenter)
        self.predictionTitle.setStyleSheet("color: rgb(255,255,255)")
        
        # Prediction sonucu için label
        self.predictionText = QLabel("")
        self.predictionText.setStyleSheet("""
            color: rgb(255,255,255);
            padding: 15px;
            background-color: rgba(0,0,0,0.3);
            border-radius: 5px;
            margin: 10px;
        """)
        self.predictionText.setAlignment(Qt.AlignCenter)
        self.predictionText.setFont(QFont('Arial', 12, QFont.Bold))
        self.predictionText.setWordWrap(True)
        self.predictionText.setMinimumHeight(80)
        
        # Groundtruth başlığı
        self.groundtruthTitle = QLabel("Groundtruth")
        self.groundtruthTitle.setFont(QFont('Arial', 14, QFont.Bold))
        self.groundtruthTitle.setAlignment(Qt.AlignCenter)
        self.groundtruthTitle.setStyleSheet("color: rgb(255,255,255)")
        
        # Groundtruth sonucu için label
        self.groundtruthText = QLabel("")
        self.groundtruthText.setStyleSheet("""
            color: rgb(255,255,255);
            padding: 15px;
            background-color: rgba(0,0,0,0.3);
            border-radius: 5px;
            margin: 10px;
        """)
        self.groundtruthText.setAlignment(Qt.AlignCenter)
        self.groundtruthText.setFont(QFont('Arial', 12, QFont.Bold))
        self.groundtruthText.setWordWrap(True)
        self.groundtruthText.setMinimumHeight(80)
        
        # Offence sonucu için label
        self.offenceText = QLabel("")
        self.offenceText.setStyleSheet("color: rgb(255,255,255)")
        self.offenceText.setAlignment(Qt.AlignCenter)
        self.offenceText.setFont(QFont('Arial', 14))
        
        # Action sonucu için label
        self.actionText = QLabel("")
        self.actionText.setStyleSheet("color: rgb(255,255,255)")
        self.actionText.setAlignment(Qt.AlignCenter)
        self.actionText.setFont(QFont('Arial', 14))
        
        # Layout'a widget'ları ekle
        self.decisionLayout.addWidget(self.decisionTitle)
        self.decisionLayout.addWidget(self.predictionTitle)
        self.decisionLayout.addWidget(self.predictionText)
        self.decisionLayout.addWidget(self.groundtruthTitle)
        self.decisionLayout.addWidget(self.groundtruthText)
        self.decisionLayout.addWidget(self.offenceText)
        self.decisionLayout.addWidget(self.actionText)
        
        # Ana layout'a sağ paneli ekle
        self.mainLayout.addWidget(self.rightWidget)
        
        # Pencere özelliklerini ayarla
        self.setWindowTitle("Video Feature Extractor")
        self.resize(1200, 800)

    def update_model_outputs(self, pred_action, pred_offence_severity):
        # Update action class scores
        action_scores = pred_action[0].detach().cpu().numpy()
        for i, (label, score) in enumerate(zip(self.actionClassLabels, action_scores)):
            label.setText(f"{self.action_classes[i]}: {score:.3f}")

        # Update offence severity scores
        severity_scores = pred_offence_severity[0].detach().cpu().numpy()
        for i, (label, score) in enumerate(zip(self.offenceSeverityLabels, severity_scores)):
            label.setText(f"{self.offence_classes[i]}: {score:.3f}")

    def openFile(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select up to 4 files", QDir.homePath())
        if not files:
            return
        clips = []
        for path in files:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            cap.release()
            if frames:
                clips.append(np.stack(frames, axis=0))  # [T, H, W, C]
        if clips:
            self.current_clips = clips
            self.extractButton.setEnabled(True)
            print("Video(lar) yüklendi ve buton aktif!")
        else:
            print("Video(lar) okunamadı!")

    def prepare_model_input(self, clips):
        # clips: [view1_frames, view2_frames, ...], her biri [T, H, W, C]
        view_tensors = []
        for frames in clips:
            # [T, H, W, C] -> [C, T, H, W]
            frames = np.transpose(frames, (3, 0, 1, 2))
            view_tensors.append(torch.from_numpy(frames).float())
        # [V, C, T, H, W]
        views = torch.stack(view_tensors, dim=0)
        # [1, V, C, T, H, W]
        return views.unsqueeze(0)

    def on_extract_clicked(self):
        print("Extract Features butonuna basıldı!")
        if self.current_clips is None:
            print("Video yüklenmemiş!")
            return
        try:
            mvinputs = self.prepare_model_input(self.current_clips)
            print(f"Input tensor shape: {mvinputs.shape}")
            mvinputs = mvinputs.to(self.device)
            
            # Model çıktılarını al
            with torch.no_grad():
                pred_offence_severity, pred_action, _ = self.model(mvinputs)
                print("\n=== MODEL TAHMİN SONUÇLARI ===")
                print(f"Offence scores: {pred_offence_severity[0].detach().cpu().numpy()}")
                print(f"Action scores: {pred_action[0].detach().cpu().numpy()}")
            
            # Model tahminlerini göster
            offence_scores = pred_offence_severity[0].detach().cpu().numpy()
            max_offence_idx = np.argmax(offence_scores)
            offence_result = self.offence_classes[max_offence_idx]
            
            action_scores = pred_action[0].detach().cpu().numpy()
            max_action_idx = np.argmax(action_scores)
            action_result = self.action_classes[max_action_idx]
            
            # Prediction sonucunu göster
            prediction_result = f"İhlal: {offence_result}\n\nAksiyon: {action_result}"
            self.predictionText.setText(prediction_result)
            print(f"\nModel Kararı:")
            print(f"İhlal: {offence_result}")
            print(f"Aksiyon: {action_result}")
            print("==============================\n")
            
            # Özellik değerlerini göster
            feats = self.model.extract_features(mvinputs)
            print(f"Features shape: {feats.shape}")  # [1,7] olmalı
            feats = feats.squeeze(0).cpu().numpy()
            print(f"Features values: {feats}")
            labels = [
                self.bodyPartValueLabel,
                self.actionClassValueLabel,
                self.multipleFoulsLabel,
                self.tryToPlayLabel,
                self.touchBallLabel,
                self.closeGoalPosLabel,
                self.severityLabel
            ]
            for i, (lbl, val) in enumerate(zip(labels, feats)):
                print(f"Setting label {i} to value {val:.2f}")
                lbl.setText(f"{val:.2f}")
            
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc() 