# 🏗️ AI-Based Cost Prediction & Recommendation System

An end-to-end machine learning system that predicts interior design/renovation project costs from structured project inputs — benchmarking 18 regression models, serving predictions via a Flask REST API, and offering conversational cost estimation through a RAG-powered chatbot.

---

## 📌 Problem Statement

Interior design and renovation costs are notoriously unpredictable upfront. Clients receive wildly different quotes from different contractors with no fast, standardized way to sanity-check a fair price before committing to a project. This system solves that by turning measurable project inputs — area, materials grade, scope, contractor type — into an instant, data-driven cost estimate, removing the "black box" nature of quote-based pricing and giving both clients and contractors a transparent reference point.

---

## 🧠 How It Works

### Step 1 — Data Cleaning (`step1_clean_data.py`)
Raw project data (`interior_india_10000_extended_enriched.csv`) is cleaned through a defensive, production-style pipeline:
- Drops fully-empty columns and known **leakage columns** (`cost_per_sqft_inr`, `cost_per_room_inr`, `cost_category`) that are mathematically derived from the target and would let the model "cheat."
- Recovers missing `end_date` values using `start_date + estimated_labour_days`, then derives `project_duration_days`.
- Engineers `city_tier` (metro / tier1 / tier2) from raw city names based on real-estate pricing tiers.
- Engineers `n_materials` from a semicolon-separated materials list.
- Ordinal-encodes `materials_grade` (economy → luxury) and `project_quarter`, preserving their natural ordering rather than one-hot encoding them.
- Runs dtype validation, null-imputation via median, and a **target skewness check** — flagging when the target needs a log-transform before modeling.
- Saves a clean, model-ready `cleaned_data.csv`.

### Step 2 — Model Benchmarking (`step2_train_model.py`)
Rather than committing to one algorithm upfront, **18 regression models** are trained and compared on identical train/test splits:

| Category | Models |
|---|---|
| Linear | Ridge, Lasso, ElasticNet, Huber, Bayesian Ridge |
| Tree-based | Decision Tree, Random Forest, Extra Trees |
| Boosting | Gradient Boosting, HistGradientBoosting, AdaBoost, XGBoost, LightGBM, **CatBoost** |
| Other | Tweedie, KNN, SVR, MLP Neural Net |

- Target variable is **log-transformed** (`log1p`) before training to correct right-skew from high-cost outlier projects, with predictions inverse-transformed (`expm1`) back to real currency values.
- Each model is scored on **R², MAE, RMSE, MAPE, and Explained Variance**, with results ranked and saved to `model_info.json`.
- **CatBoost** was selected as the production model — R² 0.80, MAPE 24.4% — winning primarily due to its native handling of categorical features (project type, materials grade, contractor type) without manual one-hot encoding.
- Feature importances are extracted directly from the winning model, identifying `property_area_sqft`, `materials_grade`, and `scope` as the strongest cost predictors.

### Step 3 — Serving Layer (`step3_app.py`)
A Flask REST API wraps the trained model for real-time inference:
- **`/predict`** — accepts project parameters as JSON, reconstructs the exact feature vector the model expects (via `build_model_input`), and returns a predicted cost with a ±12% confidence band (`low`/`high`) alongside cost-per-sqft.
- **`/model-info`** — exposes model metadata (algorithm used, R², MAE) to the frontend.
- **`/health`** — basic liveness check.
- Robust input parsing helpers (`parse_bool`, `parse_grade`) normalize inconsistent user input (e.g., `"yes"`, `"1"`, `"true"` all map to boolean `True`).

### Step 4 — RAG-Powered Chatbot (`chatbot.py`)
A conversational layer sits on top of the prediction engine:
- Uses **TF-IDF vectorization + cosine similarity** to retrieve relevant context from prior conversation/domain knowledge before generating a response.
- Integrates with an external LLM for natural-language responses, with a **graceful fallback** if the LLM API is unavailable or unconfigured — the chatbot degrades to rule-based responses rather than crashing.
- Lets users ask conversational what-if questions (e.g., *"What if I upgrade to premium materials?"*) and get a re-estimated cost without refilling a form.

### Step 5 — Business Intelligence Layer (Power BI)
Model outputs are exported and visualized in a Power BI dashboard for non-technical stakeholders:
- **KPI cards** — R², MAE, MAPE, cross-validation R², number of models benchmarked.
- **Actual vs. Predicted scatter plot** — visual proof of model accuracy across the test set.
- **Feature importance chart** — communicates *what drives cost* in business terms.
- **Model comparison chart** — all 18 models ranked, showing the benchmarking rigor visually.

---

## 🏗️ Architecture Diagram

```
┌────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  Raw Project Data    │────▶│  Step 1: Cleaning     │────▶│  cleaned_data.csv      │
│  (CSV)                │     │  (leakage removal,    │     │                        │
│                        │     │  feature engineering) │     │                        │
└────────────────────┘     └─────────────────────┘     └──────────┬───────────┘
                                                                     │
                                                          ┌──────────▼───────────┐
                                                          │  Step 2: Benchmark     │
                                                          │  18 regression models  │
                                                          │  → model_info.json     │
                                                          │  → cost_model.pkl      │
                                                          └──────────┬───────────┘
                                                                     │
                       ┌─────────────────────────────────────────────┼───────────────────────────┐
                       ▼                                              ▼                            ▼
           ┌─────────────────────┐                     ┌──────────────────────┐     ┌─────────────────────┐
           │  Step 3: Flask API    │                     │  Step 4: RAG Chatbot   │     │  Power BI Dashboard   │
           │  /predict              │◀────────────────────│  (TF-IDF retrieval +   │     │  (KPIs, accuracy,     │
           │  /model-info            │   calls estimate_   │  LLM integration)      │     │  feature importance)  │
           │  /health                │   cost()             │                        │     │                        │
           └─────────────────────┘                     └──────────────────────┘     └─────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Data Cleaning & Feature Engineering | Python, Pandas, NumPy |
| Model Benchmarking | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Serving | Flask, joblib |
| Conversational Layer | TF-IDF (scikit-learn), LLM integration |
| BI / Reporting | Power BI, DAX |
| Evaluation | R², MAE, RMSE, MAPE, 5-fold Cross-Validation |

---

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.9+
- Power BI Desktop (optional, for viewing/editing the dashboard)

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ai-cost-prediction-system

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the cleaning + training pipeline (only needed if retraining)
python step1_clean_data.py
python step2_train_model.py

# 5. Start the Flask API
python step3_app.py
```

The app will be available at `http://localhost:5000`.

### Example API Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "project_type": "residential",
    "property_area_sqft": 1200,
    "rooms": 3,
    "scope": "full",
    "materials_grade": "premium",
    "furniture_included": true,
    "estimated_labour_days": 45,
    "contractor_type": "contractor",
    "design_style": "modern",
    "city_tier": "metro",
    "n_materials": 5
  }'
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Best Model | CatBoost |
| R² Score | 0.80 |
| MAE | ₹4,62,562 |
| RMSE | ₹8,15,104 |
| MAPE | 24.4% |
| Cross-Validation R² | 0.90 |
| Models Compared | 18 |

**Top cost-driving features:** property area (sq ft), materials grade, and project scope.

---

## ✅ Key Design Decisions

- **Benchmark broadly before optimizing narrowly** — comparing 18 models on identical splits revealed that simple linear models (Ridge, Lasso) already reached ~0.79 R², meaning the true ceiling on this dataset is close to what a much simpler model achieves. This is valuable information in its own right, not just a search for the "best" model.
- **Log-transform the target** — cost data is right-skewed by a small number of high-value luxury projects; training on `log1p(cost)` prevents outliers from dominating the loss function.
- **Drop leakage columns explicitly** — `cost_per_sqft_inr` and similar fields are mathematically derived from the target itself and would produce an artificially perfect but useless model if left in.
- **CatBoost's native categorical handling** — avoids the information loss and dimensionality explosion of one-hot encoding high-cardinality categorical fields.
- **RAG + graceful degradation in the chatbot** — the conversational layer never hard-fails if the LLM API is unavailable; it falls back to rule-based responses, prioritizing reliability over feature completeness.
- **Translate metrics into a BI layer** — raw R²/MAE numbers mean little to a non-technical stakeholder; the Power BI dashboard exists specifically to make model performance and cost drivers interpretable to a business audience.

---

## 🚧 Known Limitations & Future Improvements

- **24.4% MAPE is too high for fully automated pricing** — in its current state, this should be treated as a starting estimate for human review, not a final quote.
- **Limited geographic granularity** — `city_tier` (metro/tier1/tier2) is a coarse proxy; incorporating exact location or hyperlocal pricing data would likely tighten prediction error.
- **No real vendor/contractor quote data** — predictions are based on historical project records, not live market pricing, so estimates may drift from current material/labor costs over time.
- **Chatbot retrieval is TF-IDF-based**, not embedding-based — upgrading to semantic (vector) retrieval could improve context relevance for more nuanced conversational queries.
- **No automated retraining pipeline** — the model is static once trained; a production version would need scheduled retraining as new project data becomes available.

---

## 📄 License

This project is for educational/portfolio purposes.
