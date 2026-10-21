
import numpy as np
from sklearn.metrics import confusion_matrix

FALSE_NEGATIVE_COST = 200
RETENTION_OFFER_COST = 25
SUCCESSFUL_RETENTION_BENEFIT = 100
RETENTION_SUCCESS_RATE = 0.50


def calculate_business_cost(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    offers_sent = tp + fp
    expected_retained_customers = tp * RETENTION_SUCCESS_RATE
    expected_benefit = (
        expected_retained_customers * SUCCESSFUL_RETENTION_BENEFIT
    )

    total_cost = (
        fn * FALSE_NEGATIVE_COST
        + offers_sent * RETENTION_OFFER_COST
        - expected_benefit
    )

    return {
        "threshold": float(threshold),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "offers_sent": int(offers_sent),
        "expected_retained_customers": float(expected_retained_customers),
        "expected_benefit": float(expected_benefit),
        "offer_cost": float(offers_sent * RETENTION_OFFER_COST),
        "total_cost": float(total_cost),
        "cost_per_customer": float(total_cost / len(y_true)),
    }
