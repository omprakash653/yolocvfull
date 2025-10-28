import os
from datetime import datetime
import cv2
from flask import (
    Flask, render_template, request, Response, redirect,
    url_for, send_from_directory
)
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ----------------- Paths -----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(APP_DIR, "uploads")
RESULTS = os.path.join(APP_DIR, "results")

os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# ----------------- Flask App -----------------
app = Flask(__name__)

# Load fine-tuned YOLO model
model = YOLO("yolo11n.pt")

ALLOWED_IMAGE_EXTS = {"png", "jpg", "jpeg", "bmp", "gif"}
ALLOWED_VIDEO_EXTS = {"mp4", "avi", "mov", "mkv"}

def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return (ext in ALLOWED_IMAGE_EXTS) or (ext in ALLOWED_VIDEO_EXTS)

def is_video(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_VIDEO_EXTS

# ----------------- Webcam Stream -----------------
def gen_frames(conf: float = 0.25, cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            results = model(source=frame, conf=conf, verbose=False)
            annotated = results[0].plot()  # BGR numpy array
            ret, buffer = cv2.imencode(".jpg", annotated)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

# ----------------- Routes -----------------

@app.route("/")
@app.route("/home")
def home():
    """Main landing page"""
    return render_template("home.html")

@app.route("/info")
def info():
    """Overview page"""
    return render_template("info.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction (upload + webcam)"""
    if request.method == "GET":
        return render_template("predict.html")

    # POST = file upload
    if "file" not in request.files:
        return render_template("predict.html", error="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("predict.html", error="No file selected")

    if not allowed_file(file.filename):
        return render_template("predict.html", error="Unsupported file format")

    conf = float(request.form.get("conf", 0.25))
    filename = secure_filename(file.filename)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    in_name = f"{ts}_{filename}"
    in_path = os.path.join(UPLOADS, in_name)
    file.save(in_path)

    out_basename = os.path.splitext(in_name)[0] + "_pred"

    # Image
    if not is_video(in_name):
        results = model.predict(source=in_path, save=False, conf=conf, verbose=False)
        img = results[0].plot()  # BGR
        out_path = os.path.join(RESULTS, out_basename + ".jpg")
        cv2.imwrite(out_path, img)
        return redirect(url_for("show_result", filename=os.path.basename(out_path)))

    # Video
    run_name = out_basename
    model.predict(
        source=in_path,
        conf=conf,
        save=True,
        project=RESULTS,
        name=run_name,
        exist_ok=True,
        verbose=False
    )

    # Locate output
    run_dir = os.path.join(RESULTS, run_name)
    candidate = None
    if os.path.isdir(run_dir):
        for f in os.listdir(run_dir):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                candidate = os.path.join(run_dir, f)
                break
        if candidate is None:
            files = [os.path.join(run_dir, f) for f in os.listdir(run_dir)]
            files.sort()
            candidate = files[0] if files else None

    if not candidate:
        return render_template("predict.html", error="Prediction completed, but no output file found.")

    # Copy result to main results folder
    out_path = os.path.join(RESULTS, os.path.basename(candidate))
    if not os.path.exists(out_path):
        import shutil
        shutil.copy(candidate, out_path)

    return redirect(url_for("show_result", filename=os.path.basename(out_path)))

@app.route("/video_feed")
def video_feed():
    """Webcam feed for live detection"""
    conf = float(request.args.get("conf", 0.25))
    return Response(
        gen_frames(conf=conf),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/results/<path:filename>")
def show_result(filename):
    """Display result page"""
    file_url = url_for("serve_result", filename=filename)
    return render_template("result.html", file_url=file_url, filename=filename)

@app.route("/_results/<path:filename>")
def serve_result(filename):
    """Serve result file"""
    return send_from_directory(RESULTS, filename, as_attachment=False)

# ----------------- Run -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
