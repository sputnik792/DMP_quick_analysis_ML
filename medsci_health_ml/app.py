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
        dataset_file = request.form.get("dataset") or dataset_file
        features = request.form.getlist("features")
        target = request.form.get("target")
        mode = request.form.get("mode")
        selected_models = request.form.getlist("models")

        success = train_models(dataset_file, features, target, mode=mode, selected_models=selected_models)
        if success:
            return redirect(url_for("index"))

        return render_template("setup.html", datasets=datasets, columns=columns,
                               selected=dataset_file, error="Training failed. Please check inputs.")

    if dataset_file:
        df = load_dataset(dataset_file)
        columns = df.columns.tolist()

    return render_template("setup.html", datasets=datasets, columns=columns, selected=dataset_file)



@app.route("/", methods=["GET", "POST"])
def index():
    example_model = None
    features, target, mode = [], None, "classification"
    model_list = []

    # Load any trained model metadata
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            if not file.endswith(".pkl"):
                continue
            with open(os.path.join(MODEL_DIR, file), "rb") as f:
                example_model = pickle.load(f)
            break

    if example_model:
        features = example_model.get("features", [])
        target = example_model.get("target")
        mode = example_model.get("mode", "classification")
        model_list = [f.replace(".pkl", "") for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    else:
        return redirect(url_for("setup"))

    if request.method == "POST":
        inputs = []
        for feat in features:
            try:
                inputs.append(float(request.form.get(feat)))
            except Exception:
                return render_template("index.html",
                                       features=features,
                                       target=target,
                                       models=model_list,
                                       error=f"Invalid input for {feat}")

        model_name = request.form.get("model")
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            return render_template("index.html",
                                   features=features,
                                   target=target,
                                   models=model_list,
                                   error=f"Model {model_name} not found.")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        input_data = np.array(inputs).reshape(1, -1)

        prediction = model.predict(input_data)[0]
        probability = None
        if mode == "classification" and hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]

        return render_template("index.html",
                               prediction=prediction,
                               probability=(round(float(probability), 3) if probability is not None else None),
                               selected_model=model_name,
                               features=features,
                               target=target,
                               models=model_list,
                               mode=mode)

    return render_template("index.html",
                           features=features,
                           target=target,
                           models=model_list,
                           mode=mode)




@app.route("/dashboard")
def dashboard():
    results = evaluate_models()
    return render_template("dashboard.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/")
def home():
    return redirect(url_for("setup"))

