# Student Performance Prediction System

A machine learning project that predicts student final marks using synthetic academic, lifestyle, and demographic data with Linear Regression and Streamlit.

> Educational use only: predictions are estimates and must not be used for high-stakes educational decisions.

## Features

- Synthetic dataset generation and preprocessing
- Missing-value handling and duplicate removal
- Categorical encoding and feature scaling
- Linear Regression training and evaluation
- Interactive Streamlit prediction interface
- Model evaluation and visualizations

## Technology Stack

Python, Streamlit, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib, and Pillow.

## Installation

Python 3.10 or later is recommended.

`powershell
git clone https://github.com/Vishal45-hub/Student-Performance-Prediction-System.git
cd Student-Performance-Prediction-System
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
`

## Retrain the Model

Run python student_performance_prediction.py from the project root. This may regenerate datasets, model files, and plots.

## Evaluation

The training pipeline reports MAE, MSE, RMSE, and R-squared. Refer to the actual training output for measured results.

## Limitations

- Data is synthetic and may not represent real-world student outcomes.
- Predictions may not generalize to actual students.
- Correlation does not establish causation.
- Do not use predictions for high-stakes educational decisions.

## License

No license has been selected. Reuse and redistribution are not automatically permitted without a license.
