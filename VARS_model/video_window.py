from PyQt5.QtWidgets import (QMainWindow, QLabel, QGridLayout, QWidget, 
                            QPushButton, QFileDialog, QVBoxLayout)
from PyQt5.QtCore import Qt
import torch
import numpy as np
import cv2

class VideoWindow(QMainWindow):
    def __init__(self, model, device):
        # QMainWindow'u başlat
        super(VideoWindow, self).__init__()
        
        # Model ve device'ı kaydet
        self.model = model
        self.device = device
        self.current_clips = None
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # Video yükleme butonu
        self.loadButton = QPushButton("Load Video")
        self.loadButton.clicked.connect(self.on_load_clicked)
        self.main_layout.addWidget(self.loadButton)
        
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
        
        # Feature widget ve layout'u
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
        
        # Feature grid'ini oluştur
        for i, (name, val_lbl) in enumerate(zip(feature_names, value_labels)):
            grid.addWidget(QLabel(name), i, 0)
            grid.addWidget(val_lbl, i, 1)
        
        self.featureWidget.setLayout(grid)
        self.main_layout.addWidget(self.featureWidget)
        
        # Extract Features butonu
        self.extractButton = QPushButton("Extract Features")
        self.extractButton.clicked.connect(self.on_extract_clicked)
        self.extractButton.setEnabled(False)  # Başlangıçta devre dışı
        self.main_layout.addWidget(self.extractButton)
        
        self.setWindowTitle("Video Feature Extractor")
        self.resize(400, 300)
    
    def on_load_clicked(self):
        """Video dosyası seç ve yükle"""
        print("Load button clicked!")  # Debug print
        
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        
        if file_name:
            print(f"Selected file: {file_name}")  # Debug print
            # Video'yu oku
            cap = cv2.VideoCapture(file_name)
            frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Frame'i RGB'ye çevir ve normalize et
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            cap.release()
            
            if frames:
                print(f"Loaded {len(frames)} frames")  # Debug print
                # Frame'leri [C, T, H, W] formatına dönüştür
                frames = np.stack(frames, axis=1)  # [H, W, C, T] -> [H, W, C, T]
                frames = np.transpose(frames, (2, 3, 0, 1))  # [C, T, H, W]
                print(f"Final tensor shape: {frames.shape}")  # Debug print
                self.set_current_clips(frames)
                self.extractButton.setEnabled(True)  # Video yüklendiğinde butonu aktif et
                print("Video loaded successfully!")  # Debug print
    
    def prepare_model_input(self, clips):
        """
        Video kliplerini model için uygun tensor formatına dönüştürür
        """
        if clips is None:
            return None
            
        # [B, C, T, H, W] formatına dönüştür
        return torch.from_numpy(clips).float().unsqueeze(0)  # Batch dimension ekle
    
    def on_extract_clicked(self):
        print("Extract button clicked!")  # Debug print
        
        if self.current_clips is None:
            print("No video loaded!")  # Debug print
            return
            
        try:
            # A) Video framelerini al ve tensor formatına dönüştür
            mvinputs = self.prepare_model_input(self.current_clips)
            print(f"Input tensor shape: {mvinputs.shape}")  # Debug print
            mvinputs = mvinputs.to(self.device)
            
            # B) Model'den 7-feature'ı al
            with torch.no_grad():
                # Model çıktısını al
                pooled_view, _ = self.model.mvnetwork.aggregation_model(mvinputs)
                print(f"Pooled view shape: {pooled_view.shape}")  # Debug print
                
                # Feature extractor'ı çalıştır
                feats = self.model.mvnetwork.feature_extractor(pooled_view)
                print(f"Features shape: {feats.shape}")  # Debug print
                
                # Tensor'ı numpy'a çevir
                feats = feats.cpu().numpy()
                print(f"Features values before squeeze: {feats}")  # Debug print
                
                # Eğer batch dimension varsa kaldır
                if len(feats.shape) > 1:
                    feats = feats.squeeze()
                print(f"Features values after squeeze: {feats}")  # Debug print
                
                # Eğer 7 özellik yoksa, sıfırlarla doldur
                if len(feats) != 7:
                    print(f"Warning: Expected 7 features but got {len(feats)}")
                    feats = np.zeros(7)
            
            # C) Her QLabel'e değerleri yazdır
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
                print(f"Setting label {i} to value {val:.2f}")  # Debug print
                lbl.setText(f"{val:.2f}")
                
        except Exception as e:
            print(f"Error in on_extract_clicked: {str(e)}")  # Debug print
            import traceback
            traceback.print_exc()
    
    def set_current_clips(self, clips):
        """
        Mevcut video kliplerini ayarla
        """
        self.current_clips = clips 