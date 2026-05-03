import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    preds = np.asarray(preds).ravel()
    labels = np.asarray(labels).ravel()

    if len(preds) == 0:
        return {"rmse": float("nan"), "r2": float("nan"), "mae": float("nan")}

    mse = mean_squared_error(labels, preds)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(labels, preds))
    mae = float(mean_absolute_error(labels, preds))

    return {"rmse": rmse, "r2": r2, "mae": mae}


def format_metrics(metrics: dict) -> str:
    parts: list[str] = []
    if "rmse" in metrics:
        parts.append(f"RMSE: {metrics['rmse']:.4f}")
    if "r2" in metrics:
        parts.append(f"R\u00b2: {metrics['r2']:.4f}")
    if "mae" in metrics:
        parts.append(f"MAE: {metrics['mae']:.4f}")
    return " | ".join(parts) if parts else str(metrics)
