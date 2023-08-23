from flask import Flask

from .pose_estimation import PoseEstimationApp
from .routes import data_routes, index_routes

app = Flask(__name__)

app.register_blueprint(index_routes)
app.register_blueprint(data_routes)

app_instance = PoseEstimationApp()
app.app_instance = app_instance


if __name__ == "__main__":
    app.run(debug=True)
