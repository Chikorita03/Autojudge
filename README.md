# AutoJudge – Predicting Programming Problem Difficulty

## Overview

AutoJudge is a machine learning system that predicts the difficulty of programming problems using only their textual descriptions.  
It automates difficulty estimation typically done through human judgment on coding platforms.

### Predictions
- Difficulty Class: Easy / Medium / Hard (classification)
- Difficulty Score: Numerical value (regression)

A simple web interface allows users to paste a problem statement and get instant predictions.

---

## Dataset

Each problem includes:
- title
- description
- input_description
- output_description
- problem_class (Easy / Medium / Hard)
- problem_score (numeric)

All predictions are made solely using textual information.

---

## Approach and Models

### Data Preprocessing
- Combined all text fields into a single input
- Handled missing values
- Cleaned and normalized text

### Feature Engineering
- TF-IDF vectorization
- Improved TF-IDF using:
  - Tuned vocabulary size
  - Bigrams
  - Removal of very rare and very common terms

---

## Models Tried

### Classification
- Logistic Regression (Final)
- Random Forest

**Final Choice:** Logistic Regression  
Chosen for strong generalization and effectiveness with sparse TF-IDF features.

---

### Regression
- Linear Regression
- Ridge Regression
- Gradient Boosting (Final)

**Final Choice:** Gradient Boosting Regressor  
Chosen for capturing non-linear patterns and achieving lower error.

---

## Evaluation Metrics
- Classification: Accuracy, Confusion Matrix
- Regression: MAE, RMSE

---

## Steps to Run the Project Locally

- Clone the project repository from GitHub to the local system.
- Navigate to the project directory created after cloning.
- Install all required Python dependencies specified in the `requirements.txt` file.
- Launch the Streamlit-based web application.
- Access the application through the local URL displayed in the terminal and use the interface to generate predictions.

```bash
git clone https://github.com/Chikorita03/Autojudge.git
cd Autojudge
pip install -r requirements.txt
streamlit run app.py 
```

---

## Web Interface

The project includes a Streamlit-based web application that allows users to:
1. Paste the problem title, description, input description, and output description
2. Click Predict
3. View the predicted difficulty class and numerical difficulty score
