Customer Churn Prediction

This is a Python project that predicts customer churn using Logistic Regression. It generates an interactive HTML report that includes visualizations, performance metrics, feature importance, correlations, and dataset insights.

The script automatically loads a CSV file from the data folder or you can provide a file path directly. It preprocesses the data by handling missing values, encoding categorical variables, and converting the Churn column from Yes or No to 1 or 0. The model is built with Logistic Regression using scikit-learn, with an 80/20 train test split. It evaluates accuracy, precision, recall, F1-score, and confusion matrix results.

The analysis includes the top 10 most important features ranked by model coefficients and a correlation matrix of the most relevant features. The visualizations include churn distribution, a confusion matrix heatmap, feature importance chart, feature correlation heatmap, and a performance overview with accuracy, precision, recall, F1, data quality, and dataset summary.

The script creates a styled HTML report named churn_prediction_results.html. This report includes tables, metrics, and all visualizations, and it opens automatically in your browser when the script finishes.

To set up, clone the repository, install the requirements listed in requirements.txt (pandas, numpy, scikit-learn), and place a dataset that contains a Churn column inside the data folder.

To run the project, execute python churn_predictor.py. The program will load and preprocess the dataset, train the model, generate the visualizations, create the HTML report, and open it in your browser.

