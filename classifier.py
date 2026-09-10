import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def train_xgboost(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names
):

    print("\n" + "=" * 60)
    print("XGBOOST CLASSIFICATION")
    print("=" * 60)

    # Number of classes
    num_classes = len(np.unique(y_train))

    print("\nNumber of Classes :", num_classes)

    print(
        "Number of Selected Features :",
        len(feature_names)
    )

    # -------------------------------------------------
    # XGBoost Model
    # -------------------------------------------------

    model = XGBClassifier(

        n_estimators=100,

        max_depth=6,

        learning_rate=0.1,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="multi:softprob",

        num_class=num_classes,

        eval_metric="mlogloss",

        random_state=42,

        n_jobs=-1
    )

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    print("XGBoost Training Completed.")

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    print("\nPredicting Test Data...")

    predictions = model.predict(
        X_test
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # -------------------------------------------------
    # Display Results
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("XGBOOST RESULTS")
    print("=" * 60)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # -------------------------------------------------
    # Classification Report
    # -------------------------------------------------

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)

    print(cm)

    # -------------------------------------------------
    # Save Results
    # -------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    results_df = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Score": [
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    results_df.to_csv(
        "results/xgboost_results.csv",
        index=False
    )

    # Save confusion matrix

    cm_df = pd.DataFrame(cm)

    cm_df.to_csv(
        "results/confusion_matrix.csv",
        index=False
    )

    # Save predictions

    prediction_df = pd.DataFrame({

        "Actual": y_test.reset_index(drop=True),

        "Predicted": predictions

    })

    prediction_df.to_csv(
        "results/test_predictions.csv",
        index=False
    )

    print("\nResults saved successfully.")

    print("results/xgboost_results.csv")
    print("results/confusion_matrix.csv")
    print("results/test_predictions.csv")

    return model, predictions