"""
data_engine.py
--------------
Loads, cleans, and indexes the food nutrition dataset.
Provides NLP-powered semantic search using TF-IDF + cosine similarity.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

NUMERIC_COLS = [
    'Caloric Value', 'Fat', 'Saturated Fats', 'Monounsaturated Fats',
    'Polyunsaturated Fats', 'Carbohydrates', 'Sugars', 'Protein',
    'Dietary Fiber', 'Cholesterol', 'Sodium', 'Water',
    'Vitamin A', 'Vitamin B1', 'Vitamin B11', 'Vitamin B12',
    'Vitamin B2', 'Vitamin B3', 'Vitamin B5', 'Vitamin B6',
    'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K',
    'Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese',
    'Phosphorus', 'Potassium', 'Selenium', 'Zinc', 'Nutrition Density'
]

MACRO_COLS = ['Caloric Value', 'Fat', 'Carbohydrates', 'Protein', 'Dietary Fiber', 'Sugars']
VITAMIN_COLS = ['Vitamin A', 'Vitamin B1', 'Vitamin B2', 'Vitamin B3',
                'Vitamin B5', 'Vitamin B6', 'Vitamin B11', 'Vitamin B12',
                'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K']
MINERAL_COLS = ['Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese',
                'Phosphorus', 'Potassium', 'Selenium', 'Zinc']

FEATURE_COLS = [
    'Fat', 'Saturated Fats', 'Monounsaturated Fats', 'Polyunsaturated Fats',
    'Carbohydrates', 'Sugars', 'Protein', 'Dietary Fiber',
    'Cholesterol', 'Sodium', 'Water',
    'Vitamin A', 'Vitamin B1', 'Vitamin B11', 'Vitamin B12',
    'Vitamin B2', 'Vitamin B3', 'Vitamin B5', 'Vitamin B6',
    'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K',
    'Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese',
    'Phosphorus', 'Potassium', 'Selenium', 'Zinc',
    'Nutrition Density', 'Sat_to_Unsat_Ratio'
]

# ─── NLP keyword enrichment dictionary ───────────────────────────────────────
NUTRITION_KEYWORDS = {
    'high protein': ['Protein'],
    'low calorie': ['Caloric Value'],
    'high fiber': ['Dietary Fiber'],
    'low fat': ['Fat'],
    'high calcium': ['Calcium'],
    'vitamin c': ['Vitamin C'],
    'iron rich': ['Iron'],
    'omega': ['Monounsaturated Fats', 'Polyunsaturated Fats'],
    'sugar free': ['Sugars'],
    'low sodium': ['Sodium'],
}


class FoodDataEngine:
    """Central data + NLP engine for the Food Nutrition AI app."""

    def __init__(self):
        self.df = None
        self.df_prep = None
        self.X_scaled = None
        self.feature_names = []
        self.scaler = None
        self.label_encoder = None
        self.tfidf = None
        self.tfidf_matrix = None
        self._load_and_prepare()

    # ── LOADING ───────────────────────────────────────────────────────────────

    def _load_and_prepare(self):
        frames = []
        for i in range(1, 6):
            path = os.path.join(DATA_DIR, f'FOOD-DATA-GROUP{i}.csv')
            df = pd.read_csv(path)
            drop_cols = [c for c in df.columns if c.startswith('Unnamed')]
            df.drop(columns=drop_cols, inplace=True)
            df['Group'] = f'Group {i}'
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined['food'] = combined['food'].str.strip().str.lower()
        combined.drop_duplicates(subset=['food'], keep='first', inplace=True)

        for col in NUMERIC_COLS:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')
                combined[col].fillna(combined[col].median(), inplace=True)

        combined.reset_index(drop=True, inplace=True)

        # Add engineered columns
        combined = self._add_caloric_category(combined)
        combined = self._add_fat_ratio(combined)
        combined = self._add_health_score(combined)

        self.df = combined
        self._build_ml_features()
        self._build_nlp_index()

    def _add_caloric_category(self, df):
        bins = [0, 100, 250, 450, float('inf')]
        labels = ['Low (<100 kcal)', 'Medium (100–250 kcal)',
                  'High (250–450 kcal)', 'Very High (>450 kcal)']
        df['Caloric Category'] = pd.cut(df['Caloric Value'], bins=bins, labels=labels)
        return df

    def _add_fat_ratio(self, df):
        unsat = df['Monounsaturated Fats'] + df['Polyunsaturated Fats']
        df['Sat_to_Unsat_Ratio'] = df['Saturated Fats'] / unsat.replace(0, np.nan)
        df['Sat_to_Unsat_Ratio'] = df['Sat_to_Unsat_Ratio'].fillna(0)
        return df

    def _add_health_score(self, df):
        """Composite health score (higher = more nutritious per calorie)."""
        cal = df['Caloric Value'].replace(0, 1)
        df['Health_Score'] = (
            df['Protein'] * 4 +
            df['Dietary Fiber'] * 3 +
            df['Vitamin C'] * 0.5 +
            df['Iron'] * 5 +
            df['Calcium'] * 0.02 -
            df['Saturated Fats'] * 2 -
            df['Sugars'] * 1.5
        ) / cal * 100
        df['Health_Score'] = df['Health_Score'].clip(-100, 100).round(2)
        return df

    # ── ML FEATURES ───────────────────────────────────────────────────────────

    def _build_ml_features(self):
        df = self.df.copy()
        le = LabelEncoder()
        df['Caloric_Label'] = le.fit_transform(df['Caloric Category'].astype(str))

        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].fillna(df[available].median())

        skewed = X.columns[X.skew().abs() > 2]
        X[skewed] = np.log1p(X[skewed].clip(lower=0))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.X_scaled = X_scaled
        self.feature_names = X.columns.tolist()
        self.scaler = scaler
        self.label_encoder = le
        self.df_prep = df

    # ── NLP INDEX ────────────────────────────────────────────────────────────

    def _build_nlp_index(self):
        """
        Build a TF-IDF index over enriched food descriptions.
        Each food item gets a text document that includes its name,
        nutritional properties, and keyword tags — enabling natural
        language queries like 'high protein low fat food'.
        """
        docs = []
        for _, row in self.df.iterrows():
            name = row['food']
            tags = [name]

            # Nutrition-level tags
            cal = row.get('Caloric Value', 0)
            protein = row.get('Protein', 0)
            fat = row.get('Fat', 0)
            fiber = row.get('Dietary Fiber', 0)
            sugar = row.get('Sugars', 0)
            carbs = row.get('Carbohydrates', 0)

            if cal < 50:   tags.append('very low calorie')
            elif cal < 100: tags.append('low calorie')
            elif cal > 400: tags.append('very high calorie high energy')
            else:           tags.append('moderate calorie')

            if protein > 15:  tags.append('high protein protein rich')
            if fat < 2:       tags.append('low fat fat free')
            if fat > 20:      tags.append('high fat')
            if fiber > 5:     tags.append('high fiber fiber rich')
            if sugar < 1:     tags.append('sugar free low sugar')
            if sugar > 20:    tags.append('high sugar sweet')
            if carbs < 5:     tags.append('low carb keto friendly')
            if carbs > 50:    tags.append('high carb')

            # Micronutrient tags
            for vit in VITAMIN_COLS:
                v = row.get(vit, 0)
                if v > self.df[vit].quantile(0.75):
                    tag = vit.lower().replace(' ', '_')
                    tags.append(f'rich in {tag} {tag}')

            for min_col in MINERAL_COLS:
                v = row.get(min_col, 0)
                if v > self.df[min_col].quantile(0.75):
                    tag = min_col.lower()
                    tags.append(f'rich in {tag} {tag} source')

            # Group tag
            grp = row.get('Group', '')
            tags.append(grp.lower().replace(' ', '_'))

            docs.append(' '.join(tags))

        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.tfidf.fit_transform(docs)

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def nlp_search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Semantic NLP food search.
        Returns top_k foods most relevant to the natural language query.
        """
        if not query.strip():
            return pd.DataFrame()

        # Expand query with nutrition synonyms
        expanded = query.lower()
        for phrase, cols in NUTRITION_KEYWORDS.items():
            if any(w in expanded for w in phrase.split()):
                expanded += ' ' + ' '.join(cols)

        query_vec = self.tfidf.transform([expanded])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        result = self.df.iloc[top_idx].copy()
        result['NLP_Score'] = (scores[top_idx] * 100).round(1)
        cols = ['food', 'NLP_Score', 'Caloric Value', 'Protein', 'Fat',
                'Carbohydrates', 'Dietary Fiber', 'Sugars',
                'Nutrition Density', 'Health_Score', 'Caloric Category', 'Group']
        return result[[c for c in cols if c in result.columns]]

    def get_food_detail(self, food_name: str) -> dict:
        """Full nutritional profile for a single food."""
        mask = self.df['food'].str.contains(food_name.lower(), na=False)
        if mask.sum() == 0:
            return {}
        row = self.df[mask].iloc[0]
        return row.to_dict()

    def get_splits(self):
        y_reg = self.df_prep['Caloric Value'].values
        y_clf = self.df_prep['Caloric_Label'].values
        return train_test_split(
            self.X_scaled, y_reg, y_clf,
            test_size=0.2, random_state=42, stratify=y_clf
        )

    def summary_stats(self) -> dict:
        df = self.df
        return {
            'total_foods': len(df),
            'num_groups': df['Group'].nunique(),
            'avg_calories': round(df['Caloric Value'].mean(), 1),
            'avg_protein': round(df['Protein'].mean(), 1),
            'avg_fat': round(df['Fat'].mean(), 1),
            'avg_fiber': round(df['Dietary Fiber'].mean(), 1),
            'num_features': len(self.feature_names),
        }
