from flask import Flask, render_template, Response
from src.video_processing.process_video import VideoProcessor, PoseEstimator
import cv2

app = Flask(__name__)

# Instantiate the VideoProcessor here
pose_estimator = PoseEstimator(model_complexity=1)
video_processor = VideoProcessor(pose_estimator, webcam=0)


@app.route("/")
def index():
    return render_template("index.html")


def generate_frames():
    global video_processor  # Use the VideoProcessor instance initialized earlier

    if video_processor is None:
        return

    for annotated_frame in video_processor.process_video(show=True, flask=True):
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            break
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
