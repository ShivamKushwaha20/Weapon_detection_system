# 🛡️ Weapon Detection Using YOLOv11

This repository contains the **inference code** and **model training results** for a weapon detection model capable of identifying **pistols and knives** in real-time environments using deep learning.  
The model leverages the **YOLOv11 object detection architecture** and is trained on a publicly available dataset to achieve robust detection performance under varying lighting and environmental conditions.

---

## 🔍 Project Overview
The model was trained on a labeled dataset from [**Kaggle**](https://www.kaggle.com/datasets/iqmansingh/guns-knives-object-detection), fine-tuned to detect pistols and knives, and reached an average accuracy of approximately **90% mean average precision (mAP)**.  

📓 **Model Training Code and Notebook:** Available on Kaggle — [View Notebook](https://www.kaggle.com/code/overwatch2003/weapon-detection-model-yolo11/notebook?scriptVersionId=256756529)

---

## 📦 Installation & Usage

### 1. Clone the Repository:
```git clone https://github.com/ShivamKushwaha20/Weapon_detection_system.git```<br/>
```cd Weapon_detection_system```


### 2. Create a Virtual Environment and Activate
```python -m venv venv```<br/>
```source venv/bin/activate # On Mac/Linux```<br/>
```venv\Scripts\activate # On Windows```


### 3. Install Dependencies
```pip install -r requirements.txt```


### 4. Run Inference
Use the provided inference script to run detection on images or videos:
```python scripts\detect.py```

---

## 📊 Model Training Results

- **Training Dataset:** Guns and Knives detection dataset from Kaggle  
- **GPU Used:** NVIDIA Tesla P100  
- **Final Accuracy:** ~90% mAP  
- **Training Logs and Weights:** Available in the `runs/train/` directory within the repository

---

## 📂 Code Repository
- **GitHub Repo:** [https://github.com/ShivamKushwaha20/Weapon_detection_system](https://github.com/ShivamKushwaha20/Weapon_detection_system)

---

## 👨‍💻 Contributors

- [**Vaibhav Sharma**](https://github.com/torq125) — Data collection, preprocessing, model training, performance evaluation  
- [**Shivam Kushwaha**](https://github.com/ShivamKushwaha20) — Deployment pipeline and integration  

---

## 🙌 Acknowledgements

- **YOLOv11 community** for the object detection framework  
- **Kaggle dataset providers** for the annotated guns and knives dataset  

---

💡 This repository focuses on the trained model and inference pipeline, facilitating real-time weapon detection for practical applications.


