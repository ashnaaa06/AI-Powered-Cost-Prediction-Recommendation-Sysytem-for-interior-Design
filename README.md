# VastuCost — Interior Cost Estimator
### ML-Powered · Indian Interior Design Projects · 10,000 Real Samples

---

## 📁 Project Structure

```
vastucost/
│
├── data/
│   └── interior_india_10000_extended.csv   ← PUT YOUR DATASET HERE
│
├── model/                                  ← auto-created after training
│   ├── cost_model.pkl
│   └── model_info.json
│
├── templates/
│   └── index.html                          ← frontend UI
│
├── step1_clean_data.py                     ← data cleaning
├── step2_train_model.py                    ← ML model training
├── step3_app.py                            ← Flask web server
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run (Step by Step)

### Step 0 — Setup Python environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 1 — Place your dataset

Copy your CSV file into the `data/` folder and rename it:
```
data/interior_india_10000_extended.csv
```

---

### Step 2 — Clean the data

```bash
python step1_clean_data.py
```

**What happens:**
- Loads 10,000 rows × 43 columns
- Drops 21 fully-empty columns
- Drops leakage column (cost_per_sqft_inr)
- Drops irrelevant admin columns
- Engineers `city_tier` and `n_materials` features
- Saves → `data/cleaned_data.csv`

---

### Step 3 — Train the ML model

```bash
python step2_train_model.py
```

**What happens:**
- Loads cleaned data
- Builds preprocessing pipeline (StandardScaler + OneHotEncoder)
- Trains 3 models and compares them:

  | Model              | R²     | MAE        |
  |--------------------|--------|------------|
  | Ridge Regression   | ~0.66  | ~₹6.6L     |
  | Random Forest      | ~0.79  | ~₹4.8L     |
  | **Gradient Boosting** | **~0.80** | **~₹4.7L** |

- Saves best model → `model/cost_model.pkl`
- Saves metadata → `model/model_info.json`

---

### Step 4 — Run the web app

```bash
python step3_app.py
```

**Then open your browser:**
```
http://localhost:5000
```

---

## 🌐 API Reference

The app exposes a REST API so you can also call it programmatically.

### POST /predict

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "project_type": "residential",
    "property_area_sqft": 800,
    "rooms": 3,
    "scope": "full",
    "materials_grade": "premium",
    "furniture_included": 1,
    "estimated_labour_days": 30,
    "contractor_type": "designer-led",
    "design_style": "modern",
    "city_tier": "metro",
    "n_materials": 2
  }'
```

**Response:**
```json
{
  "total": 2856000,
  "low": 2513280,
  "high": 3198720,
  "cost_per_sqft": 3570,
  "model": "Gradient Boosting",
  "r2": 0.802,
  "mae": 476521.0
}
```

### GET /model-info
Returns model metadata, feature importances, cross-validation scores.

### GET /health
Quick health check.

---

## 🧠 ML Model Details

| Component         | Detail                                      |
|-------------------|---------------------------------------------|
| Algorithm         | Gradient Boosting Regressor                 |
| n_estimators      | 200                                         |
| max_depth         | 5                                           |
| learning_rate     | 0.10                                        |
| Preprocessing     | StandardScaler + OneHotEncoder              |
| Train/Test split  | 80% / 20%                                   |
| Cross-validation  | 5-fold                                      |
| R² Score          | ~0.80                                       |
| MAE               | ~₹4.76 Lakh                                 |

### Top Cost Drivers (Feature Importance)
1. Property area (30.7%)
2. Luxury grade materials (15.4%)
3. Full scope work (15.0%)
4. Metro city location (12.9%)
5. Premium materials (9.5%)

---

## 🛠 Troubleshooting

**"ModuleNotFoundError: No module named 'flask'"**
→ Make sure your virtual environment is activated and you ran `pip install -r requirements.txt`

**"FileNotFoundError: data/interior_india_10000_extended.csv"**
→ Copy your dataset CSV into the `data/` folder

**"FileNotFoundError: model/cost_model.pkl"**
→ Run `step2_train_model.py` before `step3_app.py`

**Browser shows "Cannot reach the server"**
→ Make sure `step3_app.py` is running in a terminal before opening the browser
