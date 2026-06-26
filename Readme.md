# Real-Time Violence & Weapon Detection System

An intelligent, edge-optimized deep learning web application designed to monitor image uploads, recorded videos, and live camera feeds to automatically identify weapons (firearms, knives) and active violence. This project utilizes an advanced hybrid vision network running completely on local hardware to minimize network latency and maintain absolute data privacy.

---

## 🚀 Key Features
* **🖼️ Image & 🎥 Video Anomaly Detection:** Upload static security snapshots or pre-recorded clips to scan for explicit safety threats.
* **⚡ Live Laptop Webcam Processing:** Run real-time, zero-latency inference directly through your laptop's integrated camera hardware.
* **🌐 Remote IP Camera Integration:** Stream and analyze live feeds remotely from network-connected IP cameras via custom URL protocols.
* **🛡️ 100% Offline Edge Inference:** Weights are cached locally on the machine, processing entirely on local hardware with zero continuous external API dependencies or subscription barriers.
* **📊 Visual Bounding Boxes:** Instant UI rendering of tracking coordinates complete with precision confidence scores.

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

## 🛠️ Tech Stack & Dependencies
* **Backend:** Flask 3.0.3
* **Core Vision Architecture:** YOLO26 / YOLOv8 Frameworks
* **Compilation & Execution Engine:** ONNX Runtime / Roboflow Inference Engine
* **Computer Vision Processing:** OpenCV & Pillow
* **Deep Learning Runtime:** PyTorch

---

## 📂 Project Structure

```text
anomaly-detector/
├── app.py                      # Main Flask web application
├── model_cache/                # Persistent directory containing downloaded local weights
├── requirements.txt            # Python environment dependencies
├── Dockerfile                  # Container production configuration
├── Procfile                    # Gunicorn worker process configuration
├── render.yaml                 # Render platform blueprint
├── runtime.txt                 # Target Python compiler version
├── static/
│   ├── styles.css             # UI styling sheets
│   ├── uploads/               # Temporary landing folder for incoming files
│   └── processed/             # Visual output storage containing drawn bounding boxes
└── templates/
    ├── index.html             # Landing portal page
    └── detector.html          # Interactive multi-input detection interface
💻 Local Development Setup
1. Clone & Navigate to Project Directory
Bash
git clone [https://github.com/tarun-02005/anomaly-detector.git](https://github.com/tarun-02005/anomaly-detector.git)
cd anomaly-detector
2. Create and Activate the Virtual Environment
On Windows (PowerShell):

PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
On Linux/Mac:

Bash
python3 -m venv venv
source venv/bin/activate
3. Install Required Dependencies
Ensure your environment contains the execution frameworks to handle the local compilation matrix:

Bash
pip install -r requirements.txt
(If missing ONNX core drivers, run: pip install onnxruntime)

4. Run the Local Flask Deployment Server
Bash
python app.py
Open your browser and navigate to: http://localhost:10000 (or port 5000 as configured by Flask).

🌐 API Endpoint Specifications
1. Render Interfaces
GET / ➔ Serves the application landing home page.

GET /detector ➔ Loads the main interface panel hosting file uploads, webcam controls, and IP stream feeds.

2. Threat Processing Routing
Endpoint: /detect

Method: POST

Payload Type: multipart/form-data

Parameters: file (Raw Image/Video frame matrix data)

Response Pattern:

JSON
{
  "status": "success",
  "type": "image|video|live_stream",
  "path": "/static/processed/filename.png",
  "predictions": [
    {
      "class": "weapon",
      "confidence": 0.87,
      "x": 312.5,
      "y": 240.0,
      "width": 54.2,
      "height": 112.8
    }
  ]
}
⚖️ License (MIT License)
This project is open-source and licensed under the MIT License.

What does this mean?
The MIT License is one of the most popular, permissive, and business-friendly software licenses available. It gives anyone who gets a copy of this software the absolute right to:

Commercial Use: Anyone can build upon this project, package it, and sell it commercially.

Modification: Anyone can change, rewrite, or alter this codebase completely.

Distribution: Anyone can distribute this code to others freely.

Private Use: Anyone can run this locally or on private internal business servers.

The Only Condition: The original copyright notice and this permission notice must be included in all copies or substantial portions of the software. It comes with zero warranty or liability, meaning you are not legally responsible if someone encounters issues running the code.

👥 Authors & Contributors
Anuj - Lead Deep Learning & System Deployment Engineer

Vijay - Core Backend Logic & Computer Vision Integration Specialist


---

### What is the MIT License? (A Quick Breakdown)
Since you asked what this license means, think of it as the ultimate "do whatever you want with my code, just leave my name on the credit line" certificate. It's incredibly light, highly respected on GitHub, and tells open-source recruiters that you understand proper software distribution standards. It encourages other developers to fork your project without worrying about restrictive legal guidelines