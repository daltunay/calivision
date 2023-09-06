# Calivision - Your Fitness Companion

Calivision is a web application designed to enhance your fitness experience. It offers advanced body pose estimation and action recognition, making it the perfect tool for tracking your workouts.

## Features

1. **Body Pose Estimation**: Analyze body poses in real-time using your webcam or uploaded videos.

2. **Comprehensive Analysis**: Compute essential features and visualize interactive plots for better workout insights.

3. **Export Insights**: Export computed features like joint coordinate time series, angle evolution time series or Fourier transforms for in-depth analysis and sharing.

4. **Movement Classification**: Categorize your movements into five labels:
  - `BodyWeightSquats`
  - `JumpingJack`
  - `Lunges`
  - `PullUps`
  - `PushUps`

## Configuration Options

### Pose Estimation Parameters

Calivision provides flexibility with various pose estimation parameters:

| Parameter                | Description                                                                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| min_detection_confidence | The minimum confidence score for the pose detection to be considered successful                                                  |
| min_tracking_confidence  | The minimum confidence score for the pose tracking to be considered successful                                                   |
| model_complexity         | Complexity of the pose landmark model. Landmark accuracy as well as inference latency generally go up with the model complexity. |
| skip_frame               | Number of frames to skip between two consecutive processed frames.                                                               |
|                          |

### Action Recognition Options

Choose from different methods for action recognition:

| Model          | Description                                      |
| -------------- | ------------------------------------------------ |
| LSTM (PyTorch) | Deep learning-based model for action recognition |
| k-NN (Custom)  | Custom k-nearest neighbors implementation        |

| Input Type | Description                                                      |
| ---------- | ---------------------------------------------------------------- |
| Joints     | Input based on joint position coordinates (x, y, z)              |
| Angles     | Input based on joint angles (see /src/features/joints_data.yaml) |
| Fourier    | Input based on Fourier transforms (magnitude and phase)          |

| Metric | Description                |
| ------ | -------------------------- |
| L1     | Manhattan distance         |
| L2     | Euclidean distance         |
| DTW    | Dynamic Time Warping       |
| LCSS   | Longest Common Subsequence |
| EMD    | Earth Mover's Distance     |

## Screenshots

### Front page

Here, you can select the pose estimation parameters as well as the input source. The two buttons enable you to start/stop the pose estimation, as well as process the data.
![front page](screenshots/front_page.png)

### Pose estimation

This is what shows when you start the pose estimation.
![pose estimation](screenshots/pose_estimation.png)

### Explore data

On this page, you have the choice between visualizing several plots, exporting the data, or perform action recognition.
![explore data](screenshots/explore_data.png)

### Interactive 3D joints visualization

This is the interactive 4D plot : `(x, y, z, t)`.
![joints visualization](screenshots/joints_visualization.png)

### Angle evolution

As for the angles, there are two different plots on the page.
#### Time series

The first plot is a multivariate time series visualization, showing angle values over time.
![time series](screenshots/angle_evolution_time_series.png)
#### Heatmap

This second plot is a heatmap, which shows the same data as above but in another manner.
![heatmap](screenshots/angle_evolution_heatmap.png)

### Fourier transform

Once we have performed a FFT on the angle evolution data, we have access to the Fourier frequency spectrums.
#### Magnitude frequency domain

This shows the magnitude for each frequency value, for each angle.
![magnitude](screenshots/fourier_magnitude.png)
#### Phase frequency domain

This shows the phase for each frequency value, for each angle.
![phase](screenshots/fourier_phase.png)

### Action recognition

This is the page you land on after clicking on the prediction button. The results depend on the model and parameters you chose.
![action recognition](screenshots/action_recognition.png)
## Requirements

- Python >= 3.9
- Poetry (https://python-poetry.org/)

## Setup

0. (Optional) Install poetry:

```bash
pip install poetry
```

1. Clone this repository:

```bash
git clone https://github.com/daltunay/calivision.git
```

2. Navigate to the project directory:

```bash
cd calivision
```

3. Activate poetry shell and install dependencies:

```bash
poetry shell
poetry install
```

## Usage

To run web app, do the following:

```bash
cd app
flask run
```

Then go to **localhost**: http://127.0.0.1:5000/, and enjoy!
