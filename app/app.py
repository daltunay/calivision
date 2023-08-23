from flask import Flask

from .pose_estimation import PoseEstimationApp
from .routes.main_routes import app_routes


app = Flask(__name__)
app.register_blueprint(app_routes)
app_instance = PoseEstimationApp()
app.app_instance = app_instance


if __name__ == "__main__":
    app.run(debug=True)
