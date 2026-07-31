from preprocessing import preprocess_data
from jaya import JayaFeatureSelection


def main():

    # Preprocessing
    X_train, X_test, y_train, y_test, feature_names = preprocess_data()

    # Initialize Jaya
    jaya = JayaFeatureSelection(
        population_size=10,
        iterations=20,
        random_state=42
    )

    # Run Jaya Feature Selection
    selected_indices, selected_feature_names, best_score = jaya.fit(
        X_train,
        y_train,
        feature_names
    )

    print("\nJaya Feature Selection Completed Successfully!")
    print("Best Fitness:", best_score)


if __name__ == "__main__":
    main()