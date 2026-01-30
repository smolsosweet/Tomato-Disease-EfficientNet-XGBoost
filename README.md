# 🍅 Tomato Disease Detection: Hybrid EfficientNet-XGBoost Architecture

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Model](https://img.shields.io/badge/Model-EfficientNet%20%2B%20XGBoost-orange.svg)]()

A robust web application for diagnosing tomato leaf diseases using a **Hybrid Ensemble Learning** approach. By combining **EfficientNet** for deep feature extraction and **XGBoost** for classification, this system achieves superior accuracy and inference speed compared to traditional CNNs.

## 🚀 Key Features

* **Hybrid Architecture:** Leverages the power of Deep Learning (EfficientNet) and Gradient Boosting (XGBoost).
* **Dual Input Modes:**
    * 📂 **File Upload:** Analyze existing images.
    * 📷 **Live Camera/Canvas:** Capture and analyze in real-time.
* **Vietnamese Localization:** Full support for Vietnamese disease names (e.g., *Early Blight* $\rightarrow$ *Bệnh mốc sớm*).
* **Confidence Scoring:** Displays Top-3 probable diseases with confidence percentages.

## 🧠 Model Pipeline (How it works)

The system moves beyond standard Softmax classification by using a two-stage pipeline:

```mermaid
graph LR
    A[Input Image] -->|Preprocessing| B[EfficientNet Backbone]
    B -->|Feature Extraction| C[High-Dim Feature Vector]
    C -->|Input| D{XGBoost Classifier}
    D -->|Prediction| E[Disease Class & Probability]
