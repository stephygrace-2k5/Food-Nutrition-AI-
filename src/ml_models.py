"""
ml_models.py
------------
Trains regression, classification, and clustering models.
Returns results and serialized models for the web app.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score
)
from sklearn.model_selection import cross_val_score

PALETTE = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
           '#00BCD4', '#FF5722', '#8BC34A', '#673AB7', '#FFC107']


class MLEngine:
    """Trains and holds all ML models for the food nutrition project."""

    def __init__(self, plots_dir, reports_dir):
        self.plots_dir = plots_dir
        self.reports_dir = reports_dir
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        self.reg_results = None
        self.clf_results = None
        self.cluster_profile = None
        self.best_reg = None
        self.best_clf = None
        self.cluster_labels = None
        self.pca_2d = None
        self.is_trained = False

    def train_all(self, X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te,
                  feature_names, class_names, X_full, df_orig):
        print("[ML] Training regression models...")
        self.reg_results, self.best_reg = self._train_regression(
            X_tr, X_te, yr_tr, yr_te, feature_names)

        print("[ML] Training classification models...")
        self.clf_results, self.best_clf = self._train_classification(
            X_tr, X_te, yc_tr, yc_te, feature_names, class_names)

        print("[ML] Running clustering...")
        self.cluster_labels, self.cluster_profile, self.pca_2d = self._train_clustering(
            X_full, df_orig)

        self.is_trained = True
        print("[ML] All models trained!")

    # ── REGRESSION ────────────────────────────────────────────────────────────

    def _train_regression(self, X_tr, X_te, y_tr, y_te, feature_names):
        models = {
            'Linear Regression':  LinearRegression(),
            'Ridge Regression':   Ridge(alpha=1.0),
            'Random Forest':      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR (RBF)':          SVR(kernel='rbf', C=10, epsilon=0.1),
        }

        results, best_model, best_r2 = [], None, -np.inf
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
            mae  = float(mean_absolute_error(y_te, preds))
            r2   = float(r2_score(y_te, preds))
            cv   = float(cross_val_score(model, X_tr, y_tr, cv=3, scoring='r2').mean())
            results.append({'Model': name, 'RMSE': round(rmse, 3),
                            'MAE': round(mae, 3), 'R²': round(r2, 4), 'CV R²': round(cv, 4)})
            if r2 > best_r2:
                best_r2 = r2
                best_preds = preds
                best_model = (name, model)

        df_res = pd.DataFrame(results).sort_values('R²', ascending=False)
        df_res.to_csv(os.path.join(self.reports_dir, 'regression_results.csv'), index=False)
        self._save_regression_plots(y_te, best_preds, best_model[0], df_res, feature_names, best_model[1])
        return df_res, best_model

    def _save_regression_plots(self, y_te, preds, name, df_res, feature_names, model):
        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(y_te, preds, alpha=0.4, s=18, color='#2196F3', label='Predictions')
        lims = [min(y_te.min(), preds.min()), max(y_te.max(), preds.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Fit')
        r2 = r2_score(y_te, preds)
        ax.text(0.05, 0.93, f'R² = {r2:.4f}', transform=ax.transAxes, fontsize=11,
                color='darkred', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax.set_xlabel('Actual Caloric Value'); ax.set_ylabel('Predicted')
        ax.set_title(f'Actual vs Predicted – {name}', fontweight='bold')
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '11_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Residuals
        residuals = y_te - preds
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.scatter(preds, residuals, alpha=0.4, s=15, color='#FF9800')
        ax1.axhline(0, color='red', linestyle='--')
        ax1.set_xlabel('Predicted'); ax1.set_ylabel('Residual')
        ax1.set_title(f'Residuals vs Fitted', fontweight='bold')
        ax2.hist(residuals, bins=50, color='#4CAF50', edgecolor='white')
        ax2.axvline(0, color='red', linestyle='--')
        ax2.set_title('Residual Distribution', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '12_residuals.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            idx = np.argsort(imp)[-20:]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh([feature_names[i] for i in idx], imp[idx], color='#2196F3', edgecolor='white')
            ax.set_title(f'Top 20 Feature Importances – Regression', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, '13_feature_importance_reg.png'), dpi=150, bbox_inches='tight')
            plt.close()

        # Model comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(df_res))]
        bars = ax.barh(df_res['Model'], df_res['R²'], color=colors, edgecolor='white')
        for bar, val in zip(bars, df_res['R²']):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)
        ax.set_xlabel('R²'); ax.set_title('Regression Model Comparison (R²)', fontweight='bold')
        ax.invert_yaxis(); plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '10_regression_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # ── CLASSIFICATION ────────────────────────────────────────────────────────

    def _train_classification(self, X_tr, X_te, y_tr, y_te, feature_names, class_names):
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree':       DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
            'K-Nearest Neighbours': KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        }

        results, best_model, best_acc = [], None, -np.inf
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            acc = float(accuracy_score(y_te, preds))
            cv  = float(cross_val_score(model, X_tr, y_tr, cv=3, scoring='accuracy').mean())
            results.append({'Model': name, 'Accuracy': round(acc, 4), 'CV Accuracy': round(cv, 4)})
            if acc > best_acc:
                best_acc = acc
                best_model = (name, model, preds)

        df_res = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
        df_res.to_csv(os.path.join(self.reports_dir, 'classification_results.csv'), index=False)

        report = classification_report(y_te, best_model[2], target_names=class_names, output_dict=True)
        pd.DataFrame(report).T.to_csv(os.path.join(self.reports_dir, 'classification_report.csv'))

        self._save_classification_plots(y_te, best_model[2], class_names,
                                         best_model[0], df_res, feature_names, best_model[1])
        return df_res, best_model

    def _save_classification_plots(self, y_te, preds, class_names, name, df_res, feature_names, model):
        # Confusion matrix
        cm = confusion_matrix(y_te, preds)
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    linewidths=0.5, linecolor='white')
        ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix – {name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '15_confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            idx = np.argsort(imp)[-20:]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh([feature_names[i] for i in idx], imp[idx], color='#4CAF50', edgecolor='white')
            ax.set_title(f'Top 20 Features – {name} (Classification)', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, '16_feature_importance_clf.png'), dpi=150, bbox_inches='tight')
            plt.close()

        # Model comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(df_res))]
        bars = ax.barh(df_res['Model'], df_res['Accuracy'], color=colors, edgecolor='white')
        for bar, val in zip(bars, df_res['Accuracy']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)
        ax.set_xlabel('Accuracy'); ax.set_title('Classification Model Comparison', fontweight='bold')
        ax.invert_yaxis(); plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '14_classification_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # ── CLUSTERING ────────────────────────────────────────────────────────────

    def _train_clustering(self, X, df_orig, n_clusters=5):
        # Elbow + Silhouette
        inertias, silhouettes = [], []
        k_range = range(2, 11)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X, km.labels_))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=7)
        ax1.set_title('Elbow Method', fontweight='bold')
        ax1.set_xlabel('k'); ax1.set_ylabel('Inertia')
        ax2.plot(list(k_range), silhouettes, 'gs-', linewidth=2, markersize=7)
        ax2.set_title('Silhouette Score vs k', fontweight='bold')
        ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '17_elbow_silhouette.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Final clusters
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        # PCA 2D
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.get_cmap('tab10', n_clusters)
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, alpha=0.5, s=20)
        plt.colorbar(sc, ax=ax, label='Cluster')
        ax.set_title(f'K-Means Clusters in PCA Space (k={n_clusters})', fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '18_pca_clusters.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Cluster profiles
        df_cl = df_orig.copy()
        df_cl['Cluster'] = labels
        profile = df_cl.groupby('Cluster')[
            ['Caloric Value', 'Fat', 'Protein', 'Carbohydrates', 'Nutrition Density']
        ].mean().round(2)
        profile['Count'] = df_cl.groupby('Cluster').size()
        profile.to_csv(os.path.join(self.reports_dir, 'cluster_profiles.csv'))

        return labels, profile, X_pca

    def predict_single(self, features: dict, scaler, feature_names, label_encoder=None) -> dict:
        """Predict caloric value and category for a single food input."""
        if not self.is_trained:
            return {'error': 'Models not trained yet'}

        try:
            import numpy as np
            row = np.array([[features.get(f, 0) for f in feature_names]])
            row_scaled = scaler.transform(row)

            cal_pred = float(self.best_reg[1].predict(row_scaled)[0])
            cat_pred = int(self.best_clf[1].predict(row_scaled)[0])
            if label_encoder is not None:
                cat_label = label_encoder.inverse_transform([cat_pred])[0]
            else:
                cat_label = str(cat_pred)

            if hasattr(self.best_clf[1], 'predict_proba'):
                proba = self.best_clf[1].predict_proba(row_scaled)[0]
                confidence = float(proba.max() * 100)
            else:
                confidence = 0.0

            return {
                'caloric_value': round(cal_pred, 1),
                'category': cat_label,
                'confidence': round(confidence, 1),
                'best_reg_model': self.best_reg[0],
                'best_clf_model': self.best_clf[0],
            }
        except Exception as e:
            return {'error': str(e)}
