def create_model(data):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    if 'diagnosis' not in data.columns:
        raise ValueError("Input data must contain a 'diagnosis' column")

    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': acc,
        'classification_report': report,
        'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm
    }

    return model, metrics
