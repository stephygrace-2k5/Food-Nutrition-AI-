"""
app.py
------
Food Nutrition AI — Full-stack Flask web application.
Features:
  • NLP-powered semantic food search (TF-IDF + cosine similarity)
  • Interactive nutritional visualizations (Chart.js)
  • ML-powered caloric prediction (input custom nutrients)
  • EDA dashboard with 9 precomputed charts
  • Regression / Classification / Clustering model results
  • REST API endpoints for all features

Run:
    python app.py
Then open: http://localhost:5000
"""

import os
import sys
import json
import time
import threading

from flask import Flask, render_template, request, jsonify, send_from_directory

# Ensure src is importable
SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, SRC_DIR)

from data_engine import FoodDataEngine
from ml_models import MLEngine
from eda_plots import run_eda

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
PLOTS_DIR   = os.path.join(BASE_DIR, 'outputs', 'plots')
REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
engine     = None
ml_engine  = None
_init_done = False
_init_lock = threading.Lock()
_init_log  = []


def log(msg):
    _init_log.append(msg)
    print(msg)


def initialize():
    global engine, ml_engine, _init_done
    with _init_lock:
        if _init_done:
            return
        t0 = time.time()
        log("🔄 Loading data & building NLP index...")
        engine = FoodDataEngine()
        log(f"✅ Data loaded: {len(engine.df)} foods across {engine.df['Group'].nunique()} groups")

        log("📊 Running EDA plots...")
        run_eda(engine.df, PLOTS_DIR)

        log("🤖 Training ML models (this takes ~30-60s)...")
        ml_engine = MLEngine(PLOTS_DIR, REPORTS_DIR)
        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = engine.get_splits()
        class_names = list(engine.label_encoder.classes_)
        ml_engine.train_all(
            X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te,
            engine.feature_names, class_names,
            engine.X_scaled, engine.df_prep
        )

        elapsed = time.time() - t0
        log(f"🎉 All ready in {elapsed:.1f}s!")
        _init_done = True


# Start initialization in background thread
threading.Thread(target=initialize, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    return jsonify({
        'ready': _init_done,
        'log': _init_log[-5:],
        'stats': engine.summary_stats() if engine else {}
    })


@app.route('/api/search')
def search():
    if not engine:
        return jsonify({'error': 'Loading...'}), 503
    query = request.args.get('q', '').strip()
    top_k = int(request.args.get('top_k', 10))
    if not query:
        return jsonify({'results': []})
    df_res = engine.nlp_search(query, top_k=top_k)
    return jsonify({'results': df_res.fillna(0).to_dict(orient='records'), 'query': query})


@app.route('/api/food/<food_name>')
def food_detail(food_name):
    if not engine:
        return jsonify({'error': 'Loading...'}), 503
    detail = engine.get_food_detail(food_name)
    if not detail:
        return jsonify({'error': 'Food not found'}), 404
    # Convert numpy types to native Python
    clean = {k: (float(v) if hasattr(v, 'item') else v) for k, v in detail.items()}
    return jsonify(clean)


@app.route('/api/predict', methods=['POST'])
def predict():
    if not ml_engine or not ml_engine.is_trained:
        return jsonify({'error': 'Models not ready yet'}), 503
    data = request.get_json()
    result = ml_engine.predict_single(data, engine.scaler, engine.feature_names, engine.label_encoder)
    return jsonify(result)


@app.route('/api/eda/stats')
def eda_stats():
    if not engine:
        return jsonify({'error': 'Loading...'}), 503
    df = engine.df
    stats = {
        'caloric_distribution': df['Caloric Value'].describe().round(2).to_dict(),
        'protein_distribution': df['Protein'].describe().round(2).to_dict(),
        'category_counts': df['Caloric Category'].value_counts().to_dict(),
        'group_avg_calories': df.groupby('Group')['Caloric Value'].mean().round(1).to_dict(),
        'group_avg_protein':  df.groupby('Group')['Protein'].mean().round(2).to_dict(),
        'group_avg_fat':      df.groupby('Group')['Fat'].mean().round(2).to_dict(),
        'top_caloric': df.nlargest(10, 'Caloric Value')[['food', 'Caloric Value']].to_dict(orient='records'),
        'top_protein': df.nlargest(10, 'Protein')[['food', 'Protein']].to_dict(orient='records'),
        'top_nutrition_density': df.nlargest(10, 'Nutrition Density')[['food', 'Nutrition Density']].to_dict(orient='records'),
        'top_health_score': df.nlargest(10, 'Health_Score')[['food', 'Health_Score']].to_dict(orient='records'),
        'macro_means': {col: round(float(df[col].mean()), 2)
                        for col in ['Caloric Value','Fat','Protein','Carbohydrates','Dietary Fiber','Sugars']
                        if col in df.columns},
        'vitamin_means': {col: round(float(df[col].mean()), 4)
                          for col in ['Vitamin A','Vitamin B1','Vitamin C','Vitamin D','Vitamin E','Vitamin K']
                          if col in df.columns},
        'mineral_means': {col: round(float(df[col].mean()), 4)
                          for col in ['Calcium','Iron','Magnesium','Potassium','Zinc']
                          if col in df.columns},
    }
    # Convert category index (CategoricalIndex) to strings
    stats['category_counts'] = {str(k): int(v) for k, v in stats['category_counts'].items()}
    return jsonify(stats)


@app.route('/api/ml/results')
def ml_results():
    if not ml_engine or not ml_engine.is_trained:
        return jsonify({'ready': False})
    reg = ml_engine.reg_results.to_dict(orient='records')
    clf = ml_engine.clf_results.to_dict(orient='records')
    cluster = ml_engine.cluster_profile.reset_index().to_dict(orient='records')
    return jsonify({
        'ready': True,
        'regression': reg,
        'classification': clf,
        'clustering': cluster,
        'best_reg': ml_engine.best_reg[0],
        'best_clf': ml_engine.best_clf[0],
    })


@app.route('/api/plots')
def list_plots():
    plots = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')])
    return jsonify({'plots': plots})


@app.route('/outputs/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)


@app.route('/api/compare', methods=['POST'])
def compare_foods():
    """Compare two foods side by side."""
    if not engine:
        return jsonify({'error': 'Loading...'}), 503
    data = request.get_json()
    food1 = engine.get_food_detail(data.get('food1', ''))
    food2 = engine.get_food_detail(data.get('food2', ''))
    if not food1 or not food2:
        return jsonify({'error': 'One or both foods not found'}), 404

    def clean(d):
        return {k: (float(v) if hasattr(v, 'item') else v) for k, v in d.items()}

    return jsonify({'food1': clean(food1), 'food2': clean(food2)})


@app.route('/api/foods/autocomplete')
def autocomplete():
    if not engine:
        return jsonify({'suggestions': []})
    q = request.args.get('q', '').lower().strip()
    if len(q) < 2:
        return jsonify({'suggestions': []})
    mask = engine.df['food'].str.contains(q, na=False, regex=False)
    suggestions = engine.df[mask]['food'].head(8).tolist()
    return jsonify({'suggestions': suggestions})


if __name__ == '__main__':
    print("=" * 60)
    print("  🍎 Food Nutrition AI  —  Starting up")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("  (Models train in background — UI shows loading status)")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
