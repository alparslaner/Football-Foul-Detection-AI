# ⚽ Football Match Foul Detection with AI-Based Video Analysis

## 🚀 Project Overview

In football matches, referee decisions significantly impact the outcome of games. Yet, human referees are prone to subjective judgments and occasional errors. This project aims to mitigate these issues by implementing an automated foul detection system leveraging advanced video analysis, multi-camera integration, and state-of-the-art deep learning models.

Important Note: This Project has not been developed completely from scratch. Necessary interface support has been provided from https://github.com/SoccerNet/sn-mvfoul. We thank them for allowing us to use it and for their help.

![Ekran görüntüsü 2025-06-17 160833](https://github.com/user-attachments/assets/2c866b8c-4ada-4442-97f5-ef513e807732)


## 🎯 Main Objectives

* **Automatic Detection:** Real-time identification and classification of fouls from video footage.
* **Decision Support:** Provide referees with immediate, reliable support to enhance match officiating.
* **Accuracy Improvement:** Minimize subjectivity and human error in critical match decisions.

## 📹 Data Source

**SoccerNet Multi-View Foul Dataset (MVFoul)**

* Multi-camera synchronized video sequences.
* Detailed annotations covering action classes, severity levels, and offence types.

## ⚙️ Technical Approach

### 📌 Video Processing and Analysis

* Multi-view feature extraction for comprehensive spatial and temporal scene understanding.
* Frame preprocessing, alignment, and synchronization across multiple camera views.

### 🤖 Deep Learning Models Explored

* **R(2+1)D:** Effective spatiotemporal feature extraction.
* **SlowFast:** Dual-pathway network for temporal resolution and speed.
* **MViT:** Multiscale Vision Transformer optimized for complex action recognition.
* **YOLOv12:** High-speed, real-time object detection adapted for foul type classification.

Through extensive experimentation, we benchmarked these models to identify the best-performing architecture in terms of accuracy and inference speed.


![image](https://github.com/user-attachments/assets/4399131d-088f-4e10-8c68-d5656bad5c50)



### 🧠 Explainable AI Techniques

* **Grad-CAM++:** Utilized to enhance model interpretability and provide visual explanations of decision-making processes.

## 🔧 Technologies Used

### Programming Languages & Frameworks

* **Python**
* **PyTorch**
* **TensorFlow**

### Development Environment

* **Google Colab:** GPU-accelerated training environment.

### Libraries & Tools

* **OpenCV:** Video processing and image manipulation.
* **NumPy, Pandas:** Data handling and processing.
* **Scikit-learn:** Performance evaluation metrics.
* **Matplotlib, Seaborn:** Visualization of results and analysis.

## 📊 Project Structure

```
.
├── data_processing         # Scripts for data collection and preprocessing
├── models                  # Definitions and architectures of deep learning models
├── training_scripts        # Scripts for training and evaluation
├── utils                   # Utility functions and common tools
├── notebooks               # Colab notebooks for experimentation
└── results                 # Saved model checkpoints, results, and visualizations
```

## 📈 Results & Future Work

Significant milestones were achieved, yet the journey toward optimal performance continues. Future plans include:

* Enhancing model accuracy and improving inference speed.
* Expanding the dataset to incorporate diverse football actions and additional camera views.
* Developing real-time inference capabilities suited for live football match scenarios.


![Ekran görüntüsü 2025-06-17 143222](https://github.com/user-attachments/assets/7980ecf5-7473-449c-8037-990ec96699ef)


---

Important Note: The pth code in Football-Foul-Detection-AI\VARS interface\interface is missing due to file size.

