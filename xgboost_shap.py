import pandas as pd

from preprocessing import preprocess_data
from classifier import train_xgboost
from explainability import run_shap


def main():

    # =====================================================
    # STEP 1
    # LOAD PREPROCESSED DATA
    # =====================================================

    print("\n" + "=" * 60)
    print("LOADING PREPROCESSED DATA")
    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names
    ) = preprocess_data()

    # =====================================================
    # STEP 2
    # LOAD JAYA SELECTED FEATURES
    # =====================================================

    print("\n" + "=" * 60)
    print("LOADING JAYA SELECTED FEATURES")
    print("=" * 60)

    selected_df = pd.read_csv(
        "results/selected_features.csv"
    )

    selected_features = (
        selected_df["Feature"]
        .values
    )

    print(
        "\nNumber of Jaya Selected Features :",
        len(selected_features)
    )

    print("\nSelected Features:")

    for feature in selected_features:
        print(" -", feature)

    # =====================================================
    # STEP 3
    # FIND FEATURE INDICES
    # =====================================================

    feature_names_list = list(
        feature_names
    )

    selected_indices = [
        feature_names_list.index(feature)
        for feature in selected_features
    ]

    # =====================================================
    # STEP 4
    # SELECT JAYA FEATURES
    # =====================================================

    X_train_selected = X_train[
        :,
        selected_indices
    ]

    X_test_selected = X_test[
        :,
        selected_indices
    ]

    print("\nOriginal Features :", X_train.shape[1])

    print(
        "Selected Features :",
        X_train_selected.shape[1]
    )

    print(
        "Training Shape :",
        X_train_selected.shape
    )

    print(
        "Testing Shape :",
        X_test_selected.shape
    )

    # =====================================================
    # STEP 5
    # XGBOOST
    # =====================================================

    model, predictions = train_xgboost(

        X_train_selected,

        X_test_selected,

        y_train,

        y_test,

        selected_features

    )

    # =====================================================
    # STEP 6
    # SHAP
    # =====================================================

    importance_df = run_shap(

        model,

        X_test_selected,

        selected_features

    )

    # =====================================================
    # COMPLETE
    # =====================================================

    print("\n" + "=" * 60)
    print("XGBOOST + SHAP COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    main()