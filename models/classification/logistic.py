from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_logistic_classifier(X_train, y_train):
    clf = LogisticRegression(
        max_iter=3000
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_logistic_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["easy", "medium", "hard"])
    report = classification_report(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report
    }