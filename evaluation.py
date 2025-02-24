from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, X_test, y_test, model_type='logistic_regression'):
    y_pred = model.predict(X_test) if model_type == 'logistic_regression' else model.predict(X_test).logits.argmax(axis=1)
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1-Score: {f1_score(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
