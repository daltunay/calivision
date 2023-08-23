from flask import Blueprint, Response, current_app, redirect, render_template, request, url_for

index_routes = Blueprint("index_routes", __name__)


@index_routes.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if current_app.app_instance.pose_estimation_active:
            current_app.app_instance.terminate_estimation()
        else:
            min_detection_confidence = float(request.form["min_detection_confidence"])
            min_tracking_confidence = float(request.form["min_tracking_confidence"])
            model_complexity = int(request.form["model_complexity"])

            current_app.app_instance.start_estimation(
                min_detection_confidence, min_tracking_confidence, model_complexity
            )

    if current_app.app_instance.pose_estimation_active:
        action_button_text = "End" if current_app.app_instance.pose_estimation_active else "Start"
    else:
        action_button_text = "Start"
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
