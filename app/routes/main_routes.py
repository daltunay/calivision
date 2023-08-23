from flask import Blueprint, Response, redirect, render_template, request, url_for, current_app


app_routes = Blueprint("app_routes", __name__)


@app_routes.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if current_app.app_instance.start_estimation_flag:
            current_app.app_instance.terminate_estimation()
        else:
            min_detection_confidence = float(request.form["min_detection_confidence"])
            min_tracking_confidence = float(request.form["min_tracking_confidence"])
            model_complexity = int(request.form["model_complexity"])

            current_app.app_instance.start_estimation(
                min_detection_confidence, min_tracking_confidence, model_complexity
            )

    if current_app.app_instance.start_estimation_flag:
        action_button_text = "End" if current_app.app_instance.pose_estimation_active else "Start"
        process_button_disabled = (
            "disabled" if current_app.app_instance.pose_estimation_active else ""
        )
    else:
        action_button_text = "Start"
        process_button_disabled = (
            "" if current_app.app_instance.pose_estimation_active else "disabled"
        )

    return render_template(
        "index.html",
        action_button_text=action_button_text,
        process_button_disabled=process_button_disabled,
    )


@app_routes.route("/terminate", methods=["POST"])
def terminate():
    """Route to terminate pose estimation and redirect back to the homepage."""

    current_app.app_instance.terminate_estimation()
    return redirect(url_for("index"))


@app_routes.route("/process_data", methods=["GET"])
def process_data():
    """Route to process data and render data processing template."""

    current_app.app_instance.process_data()
    return render_template("process_data.html")


@app_routes.route("/visualize_joints")
def visualize_joints():
    """Route to visualize joint series data."""

    return current_app.app_instance.visualize_joints()


@app_routes.route("/visualize_angles")
def visualize_angles():
    """Route to visualize angle series data."""

    return current_app.app_instance.visualize_angles()


@app_routes.route("/visualize_fourier")
def visualize_fourier():
    """Route to visualize Fourier series data."""

    return current_app.app_instance.visualize_fourier()


@app_routes.route("/video_feed")
def video_feed():
    """Route to provide video feed with annotated frames."""

    return Response(
        current_app.app_instance.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
