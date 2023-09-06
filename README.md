# calivision

Bodyweight exercise classification via body pose estimation and k-NN.

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

Then go to **localhost**: http://127.0.0.1:5000/

## Screenshots

### Front page
![front page](screenshots/front_page.png)
### Pose estimation
![pose estimation](screenshots/pose_estimation.png)

### Explore data
![explore data](screenshots/explore_data.png)

### Interactive 3D joints visualization
![joints visualization](screenshots/joints_visualization.png)

### Angle evolution
![time series](screenshots/angle_evolution_time_series.png)
![heatmap](screenshots/angle_evolution_heatmap.png)

### Fourier transform
![magnitude](screenshots/fourier_magnitude.png)
![phase](screenshots/fourier_phase.png)

### Action recognition
![action recognition](screenshots/action_recognition.png)

## Architecture
```
┣ 📂app
┃ ┣ 📂routes
┃ ┣ 📂saved
┃ ┣ 📂static
┃ ┃ ┣ 📂css
┃ ┃ ┣ 📂img
┃ ┃ ┣ 📂js
┃ ┣ 📂templates
┣ 📂dataset
┃ ┣ 📂info
┃ ┣ 📂processed
┃ ┣ 📂raw
┃ ┃ ┣ 📂UCF101
┃ ┃ ┃ ┣ 📂BodyWeightSquats
┃ ┃ ┃ ┣ 📂JumpingJack
┃ ┃ ┃ ┣ 📂Lunges
┃ ┃ ┃ ┣ 📂PullUps
┃ ┃ ┃ ┣ 📂PushUps
┣ 📂models
┣ 📂notebooks
┣ 📂src
┃ ┣ 📂data
┃ ┣ 📂distance_metrics
┃ ┣ 📂features
┃ ┣ 📂models
┃ ┣ 📂utils
┃ ┣ 📂video_processing
┃ ┣ 📂visualization
┣ 📜poetry.lock
┣ 📜pyproject.toml
┗ 📜README.md
```