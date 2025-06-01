# traffic-contol
# Traffic Congestion Prediction

This project uses machine learning to predict traffic congestion levels (`Low`, `Medium`, `High`) based on factors like hour of day, day of week, holiday status, and weather conditions. It helps urban planners and traffic management authorities anticipate congestion and optimize traffic flow.

## Features

- Synthetic dataset simulating vehicle counts influenced by time and weather.
- Classification model using XGBoost to predict congestion level.
- Evaluation with classification report and confusion matrix.
- Visualization of feature importance to understand key traffic factors.

## Project Structure

- `traffic_prediction.py` — Main script to generate data, train model, and evaluate.
- `requirements.txt` — Python dependencies.
- `README.md` — Project overview and instructions.

## Getting Started

### Prerequisites

- Python 3.x
- Packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`

Install dependencies:

```bash
pip install -r requirements.txt
# traffic-contol
