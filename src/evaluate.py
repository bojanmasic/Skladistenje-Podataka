from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def plot_results(y_test, y_pred, model_name):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn,mx],[mn,mx],"r--")
    plt.xlabel("Stvarni AC_POWER")
    plt.ylabel("Predikcija AC_POWER")
    plt.title(f"{model_name}: Stvarno vs Predikcija")
    plt.show()

def plot_importance(model, features):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)
        plt.figure(figsize=(8,5))
        plt.barh(np.array(features)[order], importances[order])
        plt.xlabel("Važnost")
        plt.title("Feature Importance")
        plt.show()
