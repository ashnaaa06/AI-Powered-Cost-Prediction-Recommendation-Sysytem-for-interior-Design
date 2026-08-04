"""
STEP 3 — FLASK API BACKEND
===========================
Run:  python step3_app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

from chatbot import VastuCostChatbot

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── INIT ───────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── SAFE PATH CONFIG ───────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_FILE = MODEL_DIR / "cost_model.pkl"
INFO_FILE = MODEL_DIR / "model_info.json"

# ── LOAD MODEL SAFELY ──────────────────────────────────────────────────────

print("Loading model...")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"\n❌ Model file not found:\n{MODEL_FILE}")

if not os.path.exists(INFO_FILE):
    raise FileNotFoundError(f"\n❌ Model info file not found:\n{INFO_FILE}")

model = joblib.load(MODEL_FILE)

with open(INFO_FILE) as f:
    model_info = json.load(f)

print(f"✅ Model loaded: {model_info['best_model']}")
print(f"   R² = {model_info['r2_score']}  |  MAE = ₹{model_info['mae']:,.0f}")

# ── ROUTES ─────────────────────────────────────────────────────────────────

chatbot = VastuCostChatbot(PROJECT_ROOT)

GRADE_MAP = {"economy": 1, "standard": 2, "premium": 3, "luxury": 4}


def parse_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "included"}
    return bool(value)


def parse_grade(value):
    if isinstance(value, str):
        key = value.strip().lower()
        if key in GRADE_MAP:
            return GRADE_MAP[key]
    return int(float(value))


def build_model_input(data):
    """Recreate the numeric feature vector produced by step1 + step2."""
    feature_cols = model_info["feature_cols"]

    area = float(data["property_area_sqft"])
    rooms = max(float(data["rooms"]), 1.0)
    grade = parse_grade(data["materials_grade"])
    furniture = int(parse_bool(data["furniture_included"]))
    labour = max(float(data["estimated_labour_days"]), 1.0)
    duration = float(data.get("project_duration_days", labour))
    quarter = int(data.get("project_quarter", pd.Timestamp.today().quarter))
    n_materials = max(float(data["n_materials"]), 1.0)

    project_type = str(data["project_type"]).strip().lower()
    scope = str(data["scope"]).strip().lower()
    contractor = str(data["contractor_type"]).strip().lower()
    design_style = str(data["design_style"]).strip().lower()
    city_tier = str(data["city_tier"]).strip().lower()

    features = {
        "property_area_sqft": area,
        "rooms": rooms,
        "materials_grade": grade,
        "furniture_included": furniture,
        "estimated_labour_days": labour,
        "quoted_from": float(data.get("quoted_from", 0)),
        "project_duration_days": duration,
        "area_per_room_sqft": area / rooms,
        "project_quarter": quarter,
        "sqft_per_labour_day": area / labour,
        "n_materials": n_materials,
        "project_type_office": int(project_type == "office"),
        "project_type_residential": int(project_type == "residential"),
        "project_type_retail": int(project_type == "retail"),
        "scope_full": int(scope == "full"),
        "scope_partial": int(scope == "partial"),
        "contractor_type_contractor": int(contractor == "contractor"),
        "contractor_type_designer-led": int(contractor in {"designer-led", "designer led", "designer"}),
        "contractor_type_owner": int(contractor == "owner"),
        "design_style_colonial": int(design_style == "colonial"),
        "design_style_contemporary": int(design_style == "contemporary"),
        "design_style_industrial": int(design_style == "industrial"),
        "design_style_minimalist": int(design_style == "minimalist"),
        "design_style_modern": int(design_style == "modern"),
        "design_style_scandinavian": int(design_style == "scandinavian"),
        "design_style_traditional": int(design_style == "traditional"),
        "city_tier_tier1": int(city_tier == "tier1"),
        "city_tier_tier2": int(city_tier == "tier2"),
    }

    features.update({
        "area_x_grade": area * grade,
        "rooms_x_grade": rooms * grade,
        "avg_room_size_x_grade": (area / rooms) * grade,
        "log_area": np.log1p(area),
        "sqrt_area": np.sqrt(area),
        "area_sq": area ** 2 / 1_000_000,
        "log_labour": np.log1p(labour),
        "days_per_room": duration / rooms,
        "labour_per_100sqft": labour / max(area, 1.0) * 100,
        "materials_per_room": n_materials / rooms,
        "materials_x_grade": n_materials * grade,
        "furniture_x_grade": furniture * grade,
        "furniture_x_area": furniture * area,
    })

    features["is_full_scope"] = int(
        features["scope_full"] + features["scope_partial"] == 0
    )
    features["full_scope_x_area"] = features["is_full_scope"] * area
    features["is_metro"] = int(
        features["city_tier_tier1"] + features["city_tier_tier2"] == 0
    )
    features["metro_x_grade"] = features["is_metro"] * grade
    features["quarter_x_area"] = quarter * area

    return np.array([[features.get(col, 0.0) for col in feature_cols]], dtype=np.float64)


def estimate_cost(data):
    model_input = build_model_input(data)
    prediction_log = model.predict(model_input)[0]
    prediction = np.maximum(np.expm1(prediction_log), 0)
    prediction = max(0, round(float(prediction)))

    area = float(data["property_area_sqft"])

    return {
        "total": prediction,
        "low": round(prediction * 0.88),
        "high": round(prediction * 1.12),
        "cost_per_sqft": round(prediction / area) if area > 0 else 0,
        "model": model_info["best_model"],
        "r2": model_info["r2_score"],
        "mae": model_info["mae"],
    }


@app.route("/")
def index():
    return render_template("index.html", model_info=model_info)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        required = [
            "project_type", "property_area_sqft", "rooms", "scope",
            "materials_grade", "furniture_included",
            "estimated_labour_days", "contractor_type",
            "design_style", "city_tier", "n_materials"
        ]

        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        return jsonify(estimate_cost(data))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info")
def get_model_info():
    return jsonify(model_info)


@app.route("/chatbot/message", methods=["POST"])
def chatbot_message():
    try:
        data = request.get_json() or {}
        result = chatbot.handle(
            data.get("session_id"),
            data.get("message", ""),
            estimate_cost,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chatbot/reset", methods=["POST"])
def chatbot_reset():
    session_id = chatbot.reset((request.get_json() or {}).get("session_id"))
    return jsonify({"session_id": session_id, "status": "reset"})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": model_info["best_model"],
        "r2": model_info["r2_score"]
    })


# ── RUN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    print("\n" + "=" * 50)
    print("  VastuCost — Interior Cost Estimator")
    print("=" * 50)
    print(f"  Open browser: http://localhost:{port}")
    print(f"  API endpoint: http://localhost:{port}/predict")
    print("=" * 50 + "\n")

    app.run(debug=debug, port=port)
