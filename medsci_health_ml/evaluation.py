import os, pickle, io, base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from trainer import DATASET_DIR, MODEL_DIR

def evaluate_models():
    results = {}
    metrics = {"names": [], "accuracy": [], "roc_auc": []}

    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        print("⚠ No models found in models/ folder.")
        return {"error": "No models available. Train some first!"}

    for file in os.listdir(MODEL_DIR):
        if not file.endswith(".pkl"):
            continue

        model_name = file.replace(".pkl", "")
        try:
            with open(os.path.join(MODEL_DIR, file), "rb") as f:
                model_data = pickle.load(f)
        except Exception as e:
            print(f"❌ Could not load {file}: {e}")
            continue

        model = model_data.get("model")
        features = model_data.get("features")
        target = model_data.get("target")
        dataset_file = model_data.get("dataset")

        if not model or not features or not target or not dataset_file:
            print(f"⚠ Missing metadata in {file}, skipping.")
            continue

        dataset_path = os.path.join(DATASET_DIR, dataset_file)
        if not os.path.exists(dataset_path):
            print(f"⚠ Dataset {dataset_file} not found, skipping {model_name}.")
            continue

        try:
            df = pd.read_csv(dataset_path)
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred)

            results[model_name] = {
                "accuracy": round(acc, 3),
                "roc_auc": round(auc, 3),
                "confusion_matrix": cm.tolist(),
                "dataset": dataset_file,
                "features": features,
                "target": target
            }

            metrics["names"].append(model_name)
            metrics["accuracy"].append(acc)
            metrics["roc_auc"].append(auc)

        except Exception as e:
            print(f"❌ Failed evaluating {model_name}: {e}")
            continue

    # If no results, return error
    if not results:
        return {"error": "No models could be evaluated."}

    # --- Accuracy Plot ---
    if metrics["names"]:
        fig, ax = plt.subplots()
        ax.bar(metrics["names"], metrics["accuracy"], color="#3498db")
        ax.set_title("Model Accuracy")
        ax.set_ylabel("Accuracy")
        plt.xticks(rotation=30)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        results["accuracy_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        # --- ROC-AUC Plot ---
        fig, ax = plt.subplots()
        ax.bar(metrics["names"], metrics["roc_auc"], color="#e67e22")
        ax.set_title("Model ROC-AUC")
        ax.set_ylabel("ROC-AUC")
        plt.xticks(rotation=30)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        results["roc_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

        # --- Confusion Matrices ---
        results["conf_matrices"] = {}
        for model_name, info in results.items():
            if model_name in ["accuracy_plot", "roc_plot", "conf_matrices"]:
                continue
            cm = info["confusion_matrix"]
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix - {model_name}")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            results["conf_matrices"][model_name] = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.close()

    return results


# === Standalone test ===
if __name__ == "__main__":
    res = evaluate_models()
    if "error" in res:
        print(res["error"])
    else:
        print(f"Evaluated {len(res) - 3} models")
