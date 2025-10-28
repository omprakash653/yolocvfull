
# 🧠 YOLOv11 Object Detection Web App

A Flask-based web application that performs **real-time object detection** using a **fine-tuned YOLOv11 model (`yolo11n.pt`)**.  
Users can upload images or videos for detection or run live webcam inference — all powered by **Ultralytics YOLO** and **OpenCV**.

---

## 🚀 Features

- 🎥 **Live Webcam Detection** – Stream your webcam feed with live bounding boxes  
- 🖼️ **Image & Video Uploads** – Upload files and get instant predictions  
- ⚙️ **Fine-Tuned Model Support** – Use your own trained YOLO `.pt` weights  
- 🧩 **Real-Time Inference** – Uses OpenCV for frame capture and display  
- 💾 **Automatic Result Saving** – Predictions saved in the `/results` directory  
- 🌐 **Simple Web Interface** – Flask + HTML/CSS frontend with navigation bar  

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Backend** | Flask (Python) |
| **Model** | Ultralytics YOLOv11 |
| **Frontend** | HTML, CSS |
| **Processing** | OpenCV, Pillow, NumPy |
| **Deployment Ready** | Works locally or on any Flask-compatible server |


## 📂 Project Structure



YOLO-Flask-App/
├── app.py
├── requirements.txt
├── yolo11n.pt
├── uploads/           # Uploaded input files
├── results/           # Output images/videos with predictions
└── templates/
├── home.html      # Landing page
├── info.html      # Overview page
├── predict.html   # Upload & webcam interface
└── result.html    # Displays prediction result


## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/yolo11-flask-app.git
   cd yolo11-flask-app


2. **(Optional) Create a virtual environment**

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate        # Windows
   source .venv/bin/activate       # macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your YOLO model**

   * Place your trained model file (e.g. `yolo11n.pt`) in the root directory.

5. **Run the Flask app**

   ```bash
   python app.py
   ```

6. **Open in your browser**

   ```
   http://127.0.0.1:5000
   ```

---

## 🧠 About YOLO

**YOLO (You Only Look Once)** is a state-of-the-art, real-time object detection model.
It processes an entire image in a single pass and detects multiple objects with bounding boxes and confidence scores.

| Model        | Size    | Speed       | Accuracy  |
| ------------ | ------- | ----------- | --------- |
| `yolov8n.pt` | Nano    | ⚡ Fastest   | Low       |
| `yolov8s.pt` | Small   | 🚀 Balanced | Medium    |
| `yolov8m.pt` | Medium  | Moderate    | Good      |
| `yolov8l.pt` | Large   | Slower      | High      |
| `yolov8x.pt` | X-Large | 🐢 Slowest  | Very High |

This app uses a **fine-tuned YOLOv11n model**, trained on your custom dataset.

---

## 🔬 Fine-Tuning Overview

1. Prepare dataset in YOLO format (`images/` + `labels/`).
2. Start training with Ultralytics:

   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8s.pt')
   model.train(data='data.yaml', epochs=50, imgsz=640)
   ```
3. Evaluate model performance:

   ```python
   model.val()
   ```
4. Save the best weights as `yolo11n.pt` and use them for inference in this Flask app.

---

## 📸 Example Workflow

1. Launch the app.
2. Go to **Predict** page.
3. Choose:

   * **Webcam Mode** → real-time detection
   * **Upload Mode** → select image or video file
4. Wait for detection → view annotated results on **Result** page.
5. Files saved automatically in `/results/`.

 ## 📸 ScreenShots

1. Home
   <img width="1591" height="848" alt="image" src="https://github.com/user-attachments/assets/5c6b12f9-946f-47ae-9f95-a6cfa9e8c962" />

2. Overview
   <img width="1555" height="852" alt="image" src="https://github.com/user-attachments/assets/b5c0c3ff-b28e-41c0-9262-24e62641f38f" />

3. Predict
   <img width="1530" height="826" alt="image" src="https://github.com/user-attachments/assets/cbe57ba4-744e-4a08-9f98-7cdb2dd82574" />

4. Result
   <img width="1474" height="811" alt="image" src="https://github.com/user-attachments/assets/d921a529-0518-4931-bc8a-8197636dd18d" />

---

## 👨‍💻 Author

**Bhuman Wadekar**
💡 Data Science & AI Enthusiast
📍 India

---

## 🧾 License

This project is open-source under the **MIT License** — you’re free to use and modify it.

---

### 💙 Acknowledgements

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [Flask](https://flask.palletsprojects.com/)
* [OpenCV](https://opencv.org/)

```
