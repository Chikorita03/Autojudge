AutoJudge – Predicting Programming Problem Difficulty

Overview

AutoJudge is a machine learning system that predicts the difficulty of programming problems using only their textual descriptions.
It automates difficulty estimation that is typically done through human judgment on coding platforms.

Predictions
	•	Difficulty Class: Easy / Medium / Hard (classification)
	•	Difficulty Score: Numerical value (regression)

A simple web interface allows users to paste a problem statement and get instant predictions.

⸻

Dataset

Each problem includes:
	•	title
	•	description
	•	input_description
	•	output_description
	•	problem_class (Easy / Medium / Hard)
	•	problem_score (numeric)

All predictions are made solely using textual information.

⸻

Approach & Models

Data Preprocessing
	•	Combined all text fields into a single input
	•	Handled missing values
	•	Cleaned and normalized text

Feature Engineering
	•	TF-IDF vectorization
	•	Improved TF-IDF by:
	•	Tuning vocabulary size
	•	Adding bigrams
	•	Removing very rare and very common terms

⸻

Models Tried

Classification
	•	Logistic Regression ✅
	•	Random Forest

Final Model: Logistic Regression
Chosen for strong generalization and better performance on sparse TF-IDF features.

⸻

Regression
	•	Linear Regression
	•	Ridge Regression
	•	Gradient Boosting ✅

Final Model: Gradient Boosting Regressor
Chosen for capturing non-linear patterns with lower prediction error.

⸻

Evaluation Metrics
	•	Classification: Accuracy, Confusion Matrix
	•	Regression: MAE, RMSE

⸻

Web Interface

The project includes a Streamlit-based web application that allows users to:
	1.	Paste:
	•	Problem title
	•	Problem description
	•	Input description
	•	Output description
	2.	Click Predict
	3.	View:
	•	Predicted difficulty class (Easy / Medium / Hard)
	•	Predicted numerical difficulty score

⸻
