import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def run_shap(
    model,
    X_test,
    feature_names
):

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY")
    print("=" * 60)

    os.makedirs(
        "results/shap",
        exist_ok=True
    )

    # -------------------------------------------------
    # Convert test data to DataFrame
    # -------------------------------------------------

    X_test_df = pd.DataFrame(
        X_test,
        columns=feature_names
    )

    # -------------------------------------------------
    # Use limited samples for SHAP
    # -------------------------------------------------

    max_samples = min(
        500,
        len(X_test_df)
    )

    X_shap = X_test_df.sample(
        n=max_samples,
        random_state=42
    )

    print(
        "\nSHAP Samples Used :",
        len(X_shap)
    )

    # -------------------------------------------------
    # Create SHAP Explainer
    # -------------------------------------------------

    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(
        model
    )

    print("Calculating SHAP values...")

    shap_values = explainer.shap_values(
        X_shap
    )

    print("SHAP calculation completed.")

    # -------------------------------------------------
    # GLOBAL IMPORTANCE
    # -------------------------------------------------

    print("\nCalculating Global Feature Importance...")

    # Different SHAP versions return different shapes.
    # Handle both formats.

    if isinstance(shap_values, list):

        # Older SHAP:
        # list of arrays, one for each class

        importance = np.mean(
            [
                np.abs(values)
                for values in shap_values
            ],
            axis=(0, 1)
        )

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:

            # samples × features × classes

            importance = np.mean(
                np.abs(shap_array),
                axis=(0, 2)
            )

        else:

            importance = np.mean(
                np.abs(shap_array),
                axis=0
            )

    # -------------------------------------------------
    # Create Importance Table
    # -------------------------------------------------

    importance_df = pd.DataFrame({

        "Feature": list(feature_names),

        "Mean_Absolute_SHAP": importance

    })

    importance_df.sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False,
        inplace=True
    )

    importance_df.reset_index(
        drop=True,
        inplace=True
    )

    # -------------------------------------------------
    # Display
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("GLOBAL SHAP FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        importance_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # Save CSV
    # -------------------------------------------------

    importance_df.to_csv(
        "results/shap/shap_feature_importance.csv",
        index=False
    )

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        importance_df["Feature"],
        importance_df["Mean_Absolute_SHAP"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Global SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "results/shap/global_shap_importance.png",
        dpi=300
    )

    plt.close()

    print(
        "\nSHAP plot saved:"
    )

    print(
        "results/shap/global_shap_importance.png"
    )

    return importance_df