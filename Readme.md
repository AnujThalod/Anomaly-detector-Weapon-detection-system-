# Real-Time Violence & Weapon Detection System

An intelligent, edge-optimized deep learning application designed to monitor video or image feeds, automatically identifying weapons (firearms, knives) and active violence. This project utilizes a state-of-the-art hybrid vision network deployed locally to minimize network latency and maintain absolute data privacy.

---

## 🚀 Key Features
* **State-of-the-Art Architecture:** Utilizes **YOLO26 (Nano)**, featuring a native end-to-end head that eliminates Non-Maximum Suppression (NMS) overhead for ultra-fast performance.
* **100% Offline Edge Inference:** Weights are cached locally on the machine, running completely free-forever on local hardware with zero external API call dependencies or subscription paywalls.
* **Secure Flask Backend Pipeline:** Features a structured Python Flask proxy server to securely ingest frame payloads from frontend web interfaces.
* **Cross-Origin Resource Sharing (CORS):** Fully configured middleware enabling secure connection to modern frontend frameworks (React/Webcam streams).

---

## 📊 Model Performance Metrics
The model was trained on a custom security dataset on cloud infrastructure, achieving exceptional convergence and accuracy scores:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **mAP@50** | **84.4%** | Highly accurate spatial bounding box placement and alignment. |
| **Precision** | **87.3%** | Exceptional false-alarm suppression (only 12.7% false alerts). |
| **Recall** | **81.6%** | High sensitivity, successfully flagging over 81% of threats present. |
| **F1 Score** | **84.3%** | Optimal harmonic balance between Precision and Recall performance. |

---

## 📂 Model Storage & Configuration Data

To ensure the system runs permanently without any active internet dependencies or subscription blockades, the backend is locked into a local caching configuration framework:

* **Workspace ID Location:** `anuj-thalod`
* **Project Identifier Endpoint:** `violence-weapon-detection-k02oa`
* **Active Trained Model Version:** Version 1 (`/1`)
* **Local Workspace Root Directory:** `C:\Users\anujt\OneDrive\Desktop\ML\anomaly-detector`
* **Local Persistent Model Cache Path:** `./model_cache`

### Target Detection Classes
The model actively categorizes architectural threats into two specialized classification tags:
1. `weapon` (Firearms, knives, handheld tactical gear)
2. `violence` (Active physical combat, fighting, physical assault postures)

---

## 🛠️ System Architecture & Tech Stack
* **Core Vision Model:** YOLO26 (Nano Object Detection Model)
* **Backend Runtime Environment:** Python 3.10+ / Flask / Virtual Environments (`venv`)
* **Compilation & Execution Engine:** ONNX Runtime / Roboflow Inference Engine
* **Dataset Management:** Roboflow Ecosystem

---

## 💻 Backend Installation & Setup Guide

### 1. Navigate to Project Directory
```bash
cd anomaly-detector