import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# === Base directories (always relative to this project folder) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# === Classification Models ===
CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=500),
    "decision_tree_clf": DecisionTreeClassifier(),
    "random_forest_clf": RandomForestClassifier(n_estimators=100),
    "gradient_boosting_clf": GradientBoostingClassifier(),
    "svm_clf": SVC(probability=True),
    "knn_clf": KNeighborsClassifier()
}

# === Regression Models ===
REGRESSION_MODELS = {
    "linear_regression": LinearRegression(),
    "decision_tree_reg": DecisionTreeRegressor(),
    "random_forest_reg": RandomForestRegressor(n_estimators=100),
    "gradient_boosting_reg": GradientBoostingRegressor(),
    "svr": SVR(),
    "knn_reg": KNeighborsRegressor()
}


# === Dataset Helpers ===
def get_datasets():
    """Return list of available CSVs in datasets folder."""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    return [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]

def load_dataset(filename):
    """Load a dataset from datasets/"""
    path = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


# === Training ===
def train_models(dataset_file, features, target, mode="classification", selected_models=None):
    """
    Train models on a dataset with selected features and target.
    - dataset_file: CSV filename in datasets/
    - features: list of columns to use as X
    - target: column to predict (for regression or classification)
    - mode: "classification" or "regression"
    - selected_models: list of model keys to train (if None, train all in that category)
    """
    try:
        if not features or not target:
            print("No features or target provided. Aborting training.")
            return False

        df = load_dataset(dataset_file)

        if target not in df.columns:
            print(f"Target column {target} not found in dataset.")
            return False

        X = df[features]
        y = df[target]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        # Choose models based on mode
        models = CLASSIFICATION_MODELS if mode == "classification" else REGRESSION_MODELS

        if selected_models:
            models = {k: v for k, v in models.items() if k in selected_models}

        saved_models = []
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
                    pickle.dump({
                        "model": model,
                        "features": features,
                        "target": target,
                        "dataset": dataset_file,
                        "mode": mode,
                        "name": name
                    }, f)
                saved_models.append(name)
            except Exception as e:
                print(f"Failed to train {name}: {e}")

        if saved_models:
            print(f"Successfully trained and saved: {saved_models}")
            return True
        else:
            print("No models were trained successfully.")
            return False

    except Exception as e:
        print(f"Training failed with error: {e}")
        return False


# === Standalone usage (manual test) ===
if __name__ == "__main__":
    datasets = get_datasets()
    if datasets:
        df = load_dataset(datasets[0])
        features = df.columns[:-1].tolist()
        target = df.columns[-1]
        train_models(datasets[0], features, target, mode="classification")
    else:
        print("No datasets available in datasets/")
