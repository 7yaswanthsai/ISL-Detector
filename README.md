# Indian Sign Language (ISL) Detector

This project is a **real-time Indian Sign Language (ISL) letter and digit detector**.  
It recognizes hand signs (A–Z and 0–9) and predicts them using a deep learning model trained on custom datasets collected from multiple sources.

---

## 🚀 Project Overview
- Built a baseline **CNN model** for initial experiments.
- Finalized the system using **MobileNetV2 transfer learning** for better accuracy and robustness.
- Implemented a **testing pipeline** to validate predictions on unseen images.
- Developed a **real-time detection system** using webcam/video input.

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ISL-Detector.git
   cd ISL-Detector
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt

## 📊 Dataset
- The dataset was collected from **multiple online sources** and preprocessed for training.  
- Contains images of **35 classes** (A–Z and 0–9).  
- Augmentation techniques were applied for better generalization.  

---

## 🧠 Models Used
### Baseline CNN Model
- Simple custom CNN architecture.  
- Served as a proof of concept.  

### MobileNetV2 (Transfer Learning) [Final Model]
- Pretrained MobileNetV2 used as backbone.  
- Fine-tuned for ISL hand sign classification.  
- Achieved robust results on unseen data.  

---

## 👥 Contributors

- **Vamsi**  
- **Siva**
- **Yaswanth Sai**
