from sklearn.metrics import accuracy_score, classification_report

def evaluate_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return {"accuracy": acc, "report": report}