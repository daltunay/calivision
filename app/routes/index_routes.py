from flask import Blueprint, Response, current_app, redirect, render_template, request, url_for

index_routes = Blueprint("index_routes", __name__)

import logging

logging = logging.getLogger()


@index_routes.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        logging.info("************************************* POST REQUEST")
        if current_app.app_instance.pose_estimation_active:
            current_app.app_instance.terminate_estimation()
        else:
            min_detection_confidence = float(request.form["min_detection_confidence"])
            min_tracking_confidence = float(request.form["min_tracking_confidence"])
            model_complexity = int(request.form["model_complexity"])

            source_type = str(request.form["source_type"])  # either 'webcam' or 'upload'
            if source_type == "webcam":
                webcam = int(request.form["webcam"])
                path = None
            elif source_type == "upload":
                logging.info(str(request.files))
                webcam = None
                upload = request.files["video_upload"]
                path = f"saved/{upload.filename}"
                upload.save(path)

            current_app.app_instance.start_estimation(
                min_detection_confidence, min_tracking_confidence, model_complexity, path, webcam
            )

    if current_app.app_instance.pose_estimation_active:
        action_button_text = "END POSE ESTIMATION"
        process_button_disabled = "disabled"
    else:
        action_button_text = "START POSE ESTIMATION"
        process_button_disabled = (
            "disabled" if current_app.app_instance.landmarks_series is None else ""
        )
    return render_template(
        "index.html",
        action_button_text=action_button_text,
        process_button_disabled=process_button_disabled,
    )


@index_routes.route("/terminate", methods=["POST"])
def terminate():
    """Route to terminate pose estimation and redirect back to the homepage."""

    current_app.app_instance.terminate_estimation()
    return redirect(url_for("index"))


@index_routes.route("/process_data", methods=["GET"])
def process_data():
    """Route to process data and render data processing template."""

    current_app.app_instance.process_data()
    return render_template("process_data.html")


@index_routes.route("/video_feed")
def video_feed():
    """Route to provide video feed with annotated frames."""

    return Response(
        current_app.app_instance.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
