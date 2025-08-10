
import os
import io
import traceback
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import requests
import re

# Optional pyspark
try:
    from pyspark.sql import SparkSession
    PysparkAvailable = True
except Exception:
    PysparkAvailable = False

app = Flask(__name__)
CORS(app)

# Folders
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
STATIC_FOLDER = "static"
for d in (UPLOAD_FOLDER, MODEL_FOLDER, STATIC_FOLDER):
    os.makedirs(d, exist_ok=True)

# In-memory registry of uploaded datasets
uploaded_datasets = {}  # filename -> {path, columns}

# Store last training results for plotting
last_train_result = {
    # example keys: 'y_test', 'y_pred', 'metric_name':value, ...
}
# GOOGLE_API_KEY = "AIzaSyDe54SYl1jRu7xGLtvNUTuff8jI2qZtRmc"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_dataset():
    try:
        if "dataset" not in request.files:
            return jsonify({"ok": False, "message": "No file part (use 'dataset' field)."}), 400
        file = request.files["dataset"]
        if file.filename == "":
            return jsonify({"ok": False, "message": "No file selected."}), 400

        filename = file.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        # Read CSV with pandas to detect columns (small safety limit)
        df = pd.read_csv(path)
        # Record numeric columns only for auto feature detection
        uploaded_datasets[filename] = {
            "path": path,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "shape": df.shape
        }
        return jsonify({
            "ok": True,
            "message": f"File '{filename}' uploaded successfully.",
            "filename": filename,
            "columns": uploaded_datasets[filename]["columns"],
            "numeric_columns": uploaded_datasets[filename]["numeric_columns"],
            "shape": uploaded_datasets[filename]["shape"]
        })
    except Exception:
        traceback.print_exc()
        return jsonify({"ok": False, "message": "Failed to upload/parse dataset."}), 500


@app.route("/datasets", methods=["GET"])
def list_datasets():
    # Return list of uploaded files and their columns
    return jsonify({
        "datasets": [
            {
                "filename": fn,
                "columns": uploaded_datasets[fn]["columns"],
                "numeric_columns": uploaded_datasets[fn]["numeric_columns"],
                "shape": uploaded_datasets[fn]["shape"]
            }
            for fn in uploaded_datasets
        ]
    })


@app.route("/init-spark", methods=["GET"])
def init_spark():
    if not PysparkAvailable:
        return jsonify({"ok": False, "message": "pyspark not installed. Install pyspark to use Spark features."}), 400
    try:
        spark = SparkSession.builder \
            .appName("ITS-Dashboard") \
            .master("local[*]") \
            .config("spark.sql.shuffle.partitions", "2") \
            .getOrCreate()
        # store small handle? We'll just return success message
        return jsonify({"ok": True, "message": "Spark session initialized (local[*])."})
    except Exception:
        traceback.print_exc()
        return jsonify({"ok": False, "message": "Failed to initialize Spark."}), 500


def _prepare_data_for_training(path, target_column):
    """
    Loads CSV, selects numeric features, fills/drops na, returns X, y.
    """
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError("Target column not found in dataset.")
    numeric = df.select_dtypes(include=[np.number])
    if target_column not in numeric.columns:
        # try to coerce target to numeric
        numeric[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    # drop rows with NA in numeric or target
    numeric = numeric.dropna()
    y = numeric[target_column]
    X = numeric.drop(columns=[target_column])
    if X.shape[1] == 0:
        raise ValueError("No numeric features found besides the target.")
    return X, y


def _train_and_evaluate(X, y, model_type="linear"):
    """
    Trains and returns model, metrics, and test predictions.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if model_type == "linear":
        model = LinearRegression()
    else:
        model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return model, {"mse": mse, "r2": r2}, (y_test.values.tolist(), y_pred.tolist())


@app.route("/linear-regression", methods=["POST"])
def run_linear_regression():
    """
    Expects JSON: { "filename": "<uploaded filename>", "target": "<target column>" }
    """
    try:
        data = request.get_json()
        filename = data.get("filename")
        target = data.get("target")
        if not filename or filename not in uploaded_datasets:
            return jsonify({"ok": False, "message": "Please provide a valid uploaded filename."}), 400
        if not target:
            return jsonify({"ok": False, "message": "Please provide a target column."}), 400

        path = uploaded_datasets[filename]["path"]
        X, y = _prepare_data_for_training(path, target)
        model, metrics, (y_test, y_pred) = _train_and_evaluate(X, y, model_type="linear")

        # save model
        model_path = os.path.join(MODEL_FOLDER, f"linear_{filename}.joblib")
        joblib.dump(model, model_path)

        # store last result for plotting
        last_train_result.clear()
        last_train_result.update({
            "y_test": y_test,
            "y_pred": y_pred,
            "metrics": metrics,
            "model_path": model_path,
            "filename": filename,
            "target": target,
            "model_type": "linear"
        })

        return jsonify({"ok": True, "message": "Linear Regression trained.", "metrics": metrics, "model_path": model_path})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "message": f"Training failed: {str(e)}"}), 500


@app.route("/decision-tree", methods=["POST"])
def run_decision_tree():
    """
    Expects JSON: { "filename": "<uploaded filename>", "target": "<target column>" }
    """
    try:
        data = request.get_json()
        filename = data.get("filename")
        target = data.get("target")
        if not filename or filename not in uploaded_datasets:
            return jsonify({"ok": False, "message": "Please provide a valid uploaded filename."}), 400
        if not target:
            return jsonify({"ok": False, "message": "Please provide a target column."}), 400

        path = uploaded_datasets[filename]["path"]
        X, y = _prepare_data_for_training(path, target)
        model, metrics, (y_test, y_pred) = _train_and_evaluate(X, y, model_type="tree")

        # save model
        model_path = os.path.join(MODEL_FOLDER, f"tree_{filename}.joblib")
        joblib.dump(model, model_path)

        # store last result for plotting
        last_train_result.clear()
        last_train_result.update({
            "y_test": y_test,
            "y_pred": y_pred,
            "metrics": metrics,
            "model_path": model_path,
            "filename": filename,
            "target": target,
            "model_type": "tree"
        })

        return jsonify({"ok": True, "message": "Decision Tree trained.", "metrics": metrics, "model_path": model_path})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "message": f"Training failed: {str(e)}"}), 500


@app.route("/accuracy-graph", methods=["GET"])
def accuracy_graph():
    """
    Returns the latest accuracy graph image (png) if available. It plots y_test vs y_pred (scatter).
    """
    try:
        if not last_train_result:
            return jsonify({"ok": False, "message": "No training run available to plot."}), 400

        y_test = np.array(last_train_result["y_test"])
        y_pred = np.array(last_train_result["y_pred"])
        metrics = last_train_result["metrics"]

        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, y_pred, alpha=0.6)
        mins = min(y_test.min(), y_pred.min())
        maxs = max(y_test.max(), y_pred.max())
        plt.plot([mins, maxs], [mins, maxs], linestyle="--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        title = f"{last_train_result.get('model_type','model').title()} — R²={metrics['r2']:.3f}, MSE={metrics['mse']:.3f}"
        plt.title(title)
        plt.tight_layout()

        out_path = os.path.join(STATIC_FOLDER, "accuracy.png")
        plt.savefig(out_path)
        plt.close()

        return send_file(out_path, mimetype="image/png")
    except Exception:
        traceback.print_exc()
        return jsonify({"ok": False, "message": "Failed to generate plot."}), 500


@app.route("/find-route", methods=["POST"])
def find_route():
    """
    Calls Google Directions API and returns JSON with:
    origin, destination, total distance_km, estimated_time_min, and steps (text + distance)
    """
    try:
        data = request.get_json()
        origin = data.get("origin")
        destination = data.get("destination")
        if not origin or not destination:
            return jsonify({"ok": False, "message": "origin and destination required."}), 400

        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            # "key": GOOGLE_API_KEY,
            "mode": "driving"
        }
        resp = requests.get(url, params=params)
        directions = resp.json()

        if directions["status"] != "OK":
            return jsonify({"ok": False, "message": f"Google API error: {directions['status']}"}), 400

        route = directions["routes"][0]
        leg = route["legs"][0]

        total_meters = leg["distance"]["value"]  # meters
        total_seconds = leg["duration"]["value"]  # seconds

        steps = []
        for step in leg["steps"]:
            # Remove HTML tags from instructions
            instruction = re.sub(r"<.*?>", "", step["html_instructions"])
            distance_text = step["distance"]["text"]
            steps.append(f"{instruction} — {distance_text}")

        route_json = {
            "origin": origin,
            "destination": destination,
            "distance_km": round(total_meters / 1000, 2),
            "estimated_time_min": round(total_seconds / 60),
            "steps": steps
        }
        return jsonify({"ok": True, "message": "Route calculated.", "route": route_json})

    except Exception as e:
        return jsonify({"ok": False, "message": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
