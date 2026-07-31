import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


def preprocess_data():

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    # ---------------------------------
    # Load Dataset
    # ---------------------------------

    df = pd.read_csv("dataset/CICDDoS2019.csv")

    print("Dataset Loaded Successfully!")
    print("Dataset Shape :", df.shape)

    print("\nFirst 5 Rows")
    print(df.head())

    # ---------------------------------
    # Remove Identifier Columns
    # ---------------------------------

    print("\nRemoving Identifier Columns...")

    columns_to_drop = [
        "Unnamed: 0",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "SimillarHTTP"
    ]

    existing_columns = [c for c in columns_to_drop if c in df.columns]

    df.drop(columns=existing_columns, inplace=True)

    print("Removed Columns :", existing_columns)

    # ---------------------------------
    # Missing Values
    # ---------------------------------

    print("\nChecking Missing Values...")

    missing = df.isnull().sum().sum()

    print("Missing Values :", missing)

    df.dropna(inplace=True)

    # ---------------------------------
    # Duplicate Rows
    # ---------------------------------

    print("\nChecking Duplicate Rows...")

    duplicates = df.duplicated().sum()

    print("Duplicate Rows :", duplicates)

    df.drop_duplicates(inplace=True)

    # ---------------------------------
    # Infinity Values
    # ---------------------------------

    print("\nRemoving Infinity Values...")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)

    print("Current Dataset Shape :", df.shape)

    # ---------------------------------
    # Label Encoding
    # ---------------------------------

    target_column = "Label"

    label_encoder = LabelEncoder()

    df[target_column] = label_encoder.fit_transform(df[target_column])

    print("\nLabel Encoding Completed.")

    print("\nUnique Labels After Encoding:")
    print(df[target_column].unique())

    print("\nNumber of Classes:")
    print(df[target_column].nunique())

    print("\nLabel Mapping:")
    for original, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print(f"{original} --> {encoded}")

    # ---------------------------------
    # Separate Features and Label
    # ---------------------------------

    X = df.drop(columns=[target_column])

    y = df[target_column]

    # ---------------------------------
    # Remove Constant Features
    # ---------------------------------

    print("\nRemoving Constant Features...")

    selector = VarianceThreshold(threshold=0)

    selector.fit(X)

    constant_features = X.columns[~selector.get_support()]

    if len(constant_features) > 0:

        print("\nConstant Features Removed:")

        for feature in constant_features:
            print(feature)

    else:

        print("No Constant Features Found.")

    X = pd.DataFrame(
        selector.transform(X),
        columns=X.columns[selector.get_support()]
    )

    print("\nRemaining Features :", X.shape[1])

    # ---------------------------------
    # Train-Test Split
    # ---------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain-Test Split Completed.")

    print("Training Samples :", X_train.shape)

    print("Testing Samples :", X_test.shape)

    # ---------------------------------
    # Feature Scaling
    # ---------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    print("\nFeature Scaling Completed.")

    # ---------------------------------
    # Save Clean Dataset
    # ---------------------------------

    cleaned_df = pd.concat(
        [X.reset_index(drop=True),
         y.reset_index(drop=True)],
        axis=1
    )

    cleaned_df.to_csv(
        "results/cleaned_dataset.csv",
        index=False
    )

    print("cleaned_dataset.csv Saved.")

    # ---------------------------------
    # Save Scaled Dataset
    # ---------------------------------

    scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    )

    scaled_df[target_column] = y_train.reset_index(drop=True)

    scaled_df.to_csv(
        "results/scaled_dataset.csv",
        index=False
    )

    print("scaled_dataset.csv Saved.")

    print("\n" + "=" * 60)
    print("Preprocessing Completed Successfully!")
    print("=" * 60)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X.columns
    )


if __name__ == "__main__":
    preprocess_data()