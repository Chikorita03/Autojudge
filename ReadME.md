AutoJudge – Predicting Programming Problem Difficulty

Overview

AutoJudge is a machine-learning system that predicts the difficulty of programming problems using only their textual descriptions.
It automates difficulty estimation typically done through human judgment on coding platforms.

Predictions:
	•	Difficulty Class → Easy / Medium / Hard (classification)
	•	Difficulty Score → Numerical value (regression)

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

Preprocessing
	•	Combined all text fields into one
	•	Handled missing values
	•	Cleaned and normalized text

Feature Engineering
	•	TF-IDF vectorization
	•	Improved TF-IDF with:
	   •	Tuned vocabulary size
	   •	Bigrams
	   •	Removal of very rare/common terms

⸻

Models Tried

Classification:
	•	Logistic Regression ✅ 
	•	Random Forest
Final: Logistic Regression
Best performance on sparse TF-IDF features and strong generalization.

Regression:
	•	Linear Regression 
	•	Ridge Regression
	•	Gradient Boosting ✅ 
Final: Gradient Boosting Regressor
Captured non-linear patterns with lower error.

⸻

Evaluation Metrics
	•	Classification: Accuracy, Confusion Matrix
	•	Regression: MAE, RMSE

⸻

Web UI Interface
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
