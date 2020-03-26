import sklearn


def model_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    log_loss = sklearn.metrics.log_loss(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    return round(log_loss, 3)
