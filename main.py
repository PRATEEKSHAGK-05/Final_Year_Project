from preprocessing import preprocess_data
from jaya import JayaFeatureSelection

from classifier import DDoSClassifier
from explainability import SHAPExplainer


def main():

    # =========================================================
    # STEP 1
    # PREPROCESSING
    # =========================================================

    print("\n")
    print("=" * 70)
    print("STEP 1 : DATA PREPROCESSING")
    print("=" * 70)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names
    ) = preprocess_data()

    # =========================================================
    # STEP 2
    # JAYA FEATURE SELECTION
    # =========================================================

    print("\n")
    print("=" * 70)
    print("STEP 2 : JAYA FEATURE SELECTION")
    print("=" * 70)

    jaya = JayaFeatureSelection(

        population_size=10,

        iterations=20,

        random_state=42

    )

    (
        selected_indices,
        selected_feature_names,
        best_score
    ) = jaya.fit(

        X_train,

        y_train,

        feature_names

    )

    print("\nJaya Feature Selection Completed.")

    print(
        "Best Jaya Fitness :",
        round(best_score, 4)
    )

    # =========================================================
    # STEP 3
    # APPLY SELECTED FEATURES
    # =========================================================

    print("\n")
    print("=" * 70)
    print("STEP 3 : APPLY SELECTED FEATURES")
    print("=" * 70)

    X_train_selected = X_train[
        :,
        selected_indices
    ]

    X_test_selected = X_test[
        :,
        selected_indices
    ]

    print(
        "\nOriginal Number of Features :",
        X_train.shape[1]
    )

    print(
        "Selected Number of Features :",
        X_train_selected.shape[1]
    )

    print(
        "Features Removed :",
        X_train.shape[1] -
        X_train_selected.shape[1]
    )

    print("\nSelected Features:")

    for feature in selected_feature_names:
        print(" -", feature)

    # =========================================================
    # STEP 4
    # XGBOOST CLASSIFICATION
    # =========================================================

    print("\n")
    print("=" * 70)
    print("STEP 4 : XGBOOST CLASSIFICATION")
    print("=" * 70)

    classifier = DDoSClassifier(
        random_state=42
    )

    results = classifier.train(

        X_train_selected,

        y_train,

        X_test_selected,

        y_test,

        selected_feature_names

    )

    # =========================================================
    # STEP 5
    # SHAP GLOBAL EXPLAINABILITY
    # =========================================================

    print("\n")
    print("=" * 70)
    print("STEP 5 : SHAP GLOBAL EXPLAINABILITY")
    print("=" * 70)

    shap_explainer = SHAPExplainer(
        classifier.model
    )

    (
        importance_df,
        explainer,
        shap_values,
        X_shap
    ) = shap_explainer.global_explanation(

        X_test_selected,

        selected_feature_names

    )

    # =========================================================
    # STEP 6
    # SHAP LOCAL EXPLANATION
    # =========================================================

    print("\n")
    print("=" * 70)
    print("STEP 6 : SHAP LOCAL EXPLANATION")
    print("=" * 70)

    local_df = shap_explainer.local_explanation(

        X_test_selected,

        selected_feature_names,

        sample_index=0

    )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print("\n")
    print("=" * 70)
    print("COMPLETE DDoS DETECTION PIPELINE COMPLETED")
    print("=" * 70)

    print(
        "\nJaya Best Fitness :",
        round(best_score, 4)
    )

    print(
        "Selected Features :",
        len(selected_feature_names)
    )

    print(
        "XGBoost Accuracy :",
        round(results["accuracy"], 4)
    )

    print(
        "XGBoost Precision :",
        round(results["precision"], 4)
    )

    print(
        "XGBoost Recall :",
        round(results["recall"], 4)
    )

    print(
        "XGBoost F1 Score :",
        round(results["f1"], 4)
    )

    print("\nGenerated Files:")

    print(
        " - results/selected_features.csv"
    )

    print(
        " - results/xgboost_results.csv"
    )

    print(
        " - results/confusion_matrix.csv"
    )

    print(
        " - results/test_predictions.csv"
    )

    print(
        " - results/shap/shap_feature_importance.csv"
    )

    print(
        " - results/shap/global_shap_importance.png"
    )

    print(
        " - results/shap/local_explanation.csv"
    )

    print(
        " - results/shap/local_shap_explanation.png"
    )

    print("\nPipeline finished successfully!")


if __name__ == "__main__":

    main()