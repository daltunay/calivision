from flask import Flask

from .pose_estimation import PoseEstimationApp
from .action_prediction import ActionRecognitionApp
from .routes import data_routes, index_routes

app = Flask(__name__)

app.register_blueprint(index_routes)
app.register_blueprint(data_routes)

pose_estimation_app_instance = PoseEstimationApp()
app.pose_estimation_app_instance = pose_estimation_app_instance

action_recognition_app_instance = ActionRecognitionApp(model_name="daniel_knn_angles_model.pth")
app.action_recognition_app_instance = action_recognition_app_instance

if __name__ == "__main__":
    app.run(debug=True)
