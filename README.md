📊 Customer Churn Prediction

A Python project that predicts customer churn using Logistic Regression and generates an interactive HTML report with visualizations, metrics, and dataset insights.

🚀 Features

Loads CSV automatically from the data/ folder (or specify a file)

Preprocessing:

Handles missing values

Encodes categorical variables

Converts Churn (Yes/No → 1/0)

Model:

Logistic Regression (scikit-learn)

Train/test split (80/20)

Accuracy, precision, recall, F1-score, confusion matrix

Analysis:

Top 10 most important features (by model coefficients)

Correlation matrix of top features

Visualizations:

Churn distribution

Confusion matrix heatmap

Feature importance (bar chart)

Feature correlations (heatmap)

Model performance overview (accuracy, precision, recall, F1, dataset quality)

Reporting:

Auto-generates churn_prediction_results.html

Styled HTML with stats, tables, and charts

Opens automatically in the browser

⚙️ Setup

Clone the repository

Install dependencies from requirements.txt

pandas

numpy

scikit-learn

Place a dataset (CSV with a Churn column) in the data/ folder

▶️ Usage

Run:
python churn_predictor.py

The script will:

Load and preprocess data

Train the Logistic Regression model

Generate visualizations and metrics

Save results as churn_prediction_results.html

Open the report in your browser