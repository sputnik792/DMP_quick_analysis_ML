from flask import Flask, request, render_template, redirect, url_for
from trainer import get_datasets, load_dataset, train_models, MODEL_DIR
from evaluation import evaluate_models
import pickle, os, numpy as np

app = Flask(__name__)

@app.route("/setup", methods=["GET", "POST"])
def setup():
    datasets = get_datasets()
    dataset_file = request.args.get("dataset")
    columns = []

    if request.method == "POST":
        dataset_file = request.form.get("dataset")
        features = request.form.getlist("features")
        target = request.form.get("target")

        success = train_models(dataset_file, features, target)
        if success:
            return redirect(url_for("index"))
        else:
            # stay on setup with error
            return render_template("setup.html",
                                   datasets=datasets,
                                   columns=columns,
                                   selected=dataset_file,
                                   error="⚠ Training failed. Please check dataset, features, and target.")

    if dataset_file:
        df = load_dataset(dataset_file)
        columns = df.columns.tolist()

    return render_template("setup.html", datasets=datasets, columns=columns, selected=dataset_file)


@app.route("/", methods=["GET", "POST"])
def index():
    example_model = None
    features, target = [], None

    # ✅ Only try loading models if folder exists and has files
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            with open(os.path.join(MODEL_DIR, file), "rb") as f:
                example_model = pickle.load(f)
            break

    if example_model:
        features = example_model["features"]
        target = example_model["target"]
    else:
        # Show helpful message instead of infinite redirect
        return render_template("index.html",
                               features=[],
                               target=None,
                               error="⚠ No models trained yet. Please go to Setup first.")

    if request.method == "POST":
        inputs = []
        try:
            for feat in features:
                inputs.append(float(request.form.get(feat)))
        except Exception:
            return render_template("index.html",
                                   features=features,
                                   target=target,
                                   error="⚠ Invalid input values.")

        model_name = request.form.get("model")
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            return render_template("index.html",
                                   features=features,
                                   target=target,
                                   error=f"⚠ Model {model_name} not found.")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        input_data = np.array(inputs).reshape(1, -1)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return render_template("index.html",
                               prediction=int(prediction),
                               probability=round(float(probability), 3),
                               selected_model=model_name,
                               features=features,
                               target=target)

    return render_template("index.html", features=features, target=target)


@app.route("/dashboard")
def dashboard():
    results = evaluate_models()
    return render_template("dashboard.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
