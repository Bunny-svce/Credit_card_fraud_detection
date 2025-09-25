import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


# -------------------------
# Data Preprocessor
# -------------------------
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.isolation_forest = IsolationForest(
            contamination=0.1, random_state=42)

    def load_data(self, filepath=None):
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Dataset loaded: {df.shape}")
            return df
        except:
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        np.random.seed(42)
        n = 10000
        X_normal = np.random.normal(0, 1, (int(n * 0.99), 28))
        X_fraud = np.random.normal(2, 1.5, (int(n * 0.01), 28))
        X = np.vstack([X_normal, X_fraud])
        y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_fraud))])
        time = np.random.uniform(0, 172800, len(X))
        amount = np.random.lognormal(3, 1.5, len(X))
        df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
        df["Time"] = time
        df["Amount"] = amount
        df["Class"] = y
        print(f"✓ Synthetic data generated: {df.shape}")
        return df

    def handle_missing(self, df):
        if df.isnull().sum().sum() > 0:
            df[df.select_dtypes(include=[np.number]).columns] = self.imputer.fit_transform(
                df.select_dtypes(include=[np.number])
            )
        return df

    def remove_outliers(self, df):
        df_maj = df[df["Class"] == 0]
        df_min = df[df["Class"] == 1]

        features_maj = df_maj.drop("Class", axis=1)
        mask = self.isolation_forest.fit_predict(features_maj) == 1
        clean_maj = df_maj[mask].copy()

        clean = pd.concat([clean_maj, df_min])
        print(f"✓ Removed outliers: {df.shape[0] - clean.shape[0]} (maj only)")
        return clean

    def scale(self, X_train, X_test):
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        return X_train_s, X_test_s


# -------------------------
# Novel Sampling
# -------------------------
class NovelSamplingTechnique:
    def __init__(self, k_neighbors=5, n_clusters=5, random_seed=42):
        self.k = k_neighbors
        self.nc = n_clusters
        np.random.seed(random_seed)

    def identify_boundary_mass(self, X_maj, X_min):
        knn = NearestNeighbors(n_neighbors=self.k +
                               1).fit(np.vstack([X_maj, X_min]))
        boundary, mass = [], []
        for i, x in enumerate(X_maj):
            _, idx = knn.kneighbors([x])
            neigh = idx[0][1:]
            if np.any(neigh >= len(X_maj)):
                boundary.append(i)
            else:
                mass.append(i)
        return np.array(boundary, dtype=int), np.array(mass, dtype=int)

    def generate_sets(self, X_maj, X_min, n_sets=3):
        b_idx, m_idx = self.identify_boundary_mass(X_maj, X_min)
        X_b, X_m = X_maj[b_idx], X_maj[m_idx]

        print(
            f"Boundary samples: {len(X_b)}, Mass samples: {len(X_m)}, Minority samples: {len(X_min)}")

        sets = []
        for _ in range(n_sets):
            if len(X_b) == 0 and len(X_m) == 0:
                Xc = np.vstack([X_maj, X_min])
                yc = np.hstack([np.zeros(len(X_maj)), np.ones(len(X_min))])
                sets.append((Xc, yc))
                continue

            bi = min(len(X_b), len(X_min) // 2) if len(X_b) > 0 else 0
            bs = X_b[np.random.choice(len(X_b), bi, replace=False)] if bi > 0 else np.empty(
                (0, X_maj.shape[1]))

            mi = len(X_min) - bi
            ms = X_m[np.random.choice(len(X_m), mi, replace=False)] if mi > 0 and len(
                X_m) > 0 else np.empty((0, X_maj.shape[1]))

            xis = X_min[np.random.choice(len(X_min), len(X_min), replace=True)]

            X_parts = [arr for arr in [bs, ms, xis] if arr.shape[0] > 0]
            y_parts = [
                np.zeros(bs.shape[0]) if bs.shape[0] > 0 else np.array([]),
                np.zeros(ms.shape[0]) if ms.shape[0] > 0 else np.array([]),
                np.ones(xis.shape[0])
            ]

            Xc = np.vstack(X_parts)
            yc = np.hstack(y_parts)

            assert Xc.shape[0] == yc.shape[0], f"Mismatch: {Xc.shape} vs {yc.shape}"
            print(f"Set shapes: X = {Xc.shape}, y = {yc.shape}")
            sets.append((Xc, yc))

        return sets


# -------------------------
# Multi-Stage Classifier
# -------------------------
class MultiStageClassifier:
    def __init__(self, random_seed=42):
        self.seed = random_seed
        self.first_layer = {}
        self.second_layer = None
        self._init_models()

    def _init_models(self):
        self.first_layer = {
            'lr': LogisticRegression(random_state=self.seed, max_iter=1000),
            'svm': SVC(random_state=self.seed, probability=True),
            'rf': RandomForestClassifier(random_state=self.seed),
            'xgb': xgb.XGBClassifier(random_state=self.seed, eval_metric='logloss'),
            'knn': KNeighborsClassifier(),
            'dnn': MLPClassifier(random_state=self.seed, max_iter=500)
        }
        self.params = {
            'lr': {'C': [0.1, 0.3, 1.0]},
            'svm': {'C': [0.1, 1.0], 'kernel': ['rbf', 'linear']},
            'rf': {'n_estimators': [60, 100], 'max_depth': [5, 7]},
            'xgb': {'max_depth': [3, 6], 'learning_rate': [0.01, 0.1]},
            'knn': {'n_neighbors': [3, 5]},
            'dnn': {'hidden_layer_sizes': [(50,), (100,)], 'learning_rate_init': [0.001, 0.01]}
        }

    def train_first(self, X, y):
        for name, model in self.first_layer.items():
            cv = GridSearchCV(
                model, self.params[name], cv=3, scoring='f1', n_jobs=-1)
            cv.fit(X, y)
            self.first_layer[name] = cv.best_estimator_

    def predict_first(self, X):
        preds, probs = {}, {}
        for name, m in self.first_layer.items():
            preds[name] = m.predict(X)
            probs[name] = m.predict_proba(X)[:, 1]
        return preds, probs

    def train_second(self, X, y):
        p, prob = self.predict_first(X)
        feat = np.hstack([np.column_stack([p[n], prob[n]]) for n in p])
        self.second_layer = LogisticRegression(
            random_state=self.seed).fit(feat, y)

    def predict(self, X):
        p, prob = self.predict_first(X)
        if self.second_layer:
            feat = np.hstack([np.column_stack([p[n], prob[n]]) for n in p])
            return self.second_layer.predict(feat), self.second_layer.predict_proba(feat)[:, 1]
        avg = np.mean([prob[n] for n in prob], axis=0)
        return (avg > 0.5).astype(int), avg

    def fit(self, X, y):
        self.train_first(X, y)
        self.train_second(X, y)


# -------------------------
# Fraud Evaluator
# -------------------------
class FraudDetectionEvaluator:
    def __init__(self, cost_rate=0.05):
        self.rate = cost_rate
        self.results = {}

    def cost_metrics(self, y_true, y_pred, amounts=None):
        if amounts is None:
            amounts = np.random.lognormal(3, 1.5, len(y_true))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        invest = np.sum(amounts[y_pred == 1]) * self.rate
        loss = np.sum(amounts[(y_true == 1) & (y_pred == 0)])
        total = invest + loss
        base = np.sum(amounts[y_true == 1])
        ratio = total / base if base > 0 else 0
        return invest, loss, total, base, ratio

    def evaluate(self, y_true, y_pred, y_prob, name):
        acc = np.mean(y_true == y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        fraud_r = recall_score(y_true, y_pred, pos_label=1)
        legit_r = recall_score(y_true, y_pred, pos_label=0)
        auc = roc_auc_score(y_true, y_prob)
        invest, loss, total, base, ratio = self.cost_metrics(
            y_true, y_pred, y_prob)
        self.results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fraud_recall": fraud_r,
            "legit_recall": legit_r,
            "auc": auc,
            "cost_ratio": ratio,
        }

    def compare(self):
        df = pd.DataFrame(self.results).T
        print(df)
        best = min(self.results, key=lambda n: self.results[n]["cost_ratio"])
        print(f"Best model: {best}")


# -------------------------
# Pipeline
# -------------------------
class CreditCardFraudDetectionPipeline:
    def __init__(self, random_seed=42):
        self.seed = random_seed
        self.pre = DataPreprocessor()
        self.sam = NovelSamplingTechnique(random_seed=self.seed)
        self.clf = MultiStageClassifier(random_seed=self.seed)
        self.eval = FraudDetectionEvaluator()

    def run_complete_pipeline(self, filepath=None, use_novel_sampling=True):
        df = self.pre.load_data(filepath)
        df = self.pre.handle_missing(df)
        df = self.pre.remove_outliers(df)

        X = df.drop("Class", axis=1).values
        y = df["Class"].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y)
        Xtr, Xte = self.pre.scale(Xtr, Xte)

        maj = Xtr[ytr == 0]
        min_ = Xtr[ytr == 1]

        if use_novel_sampling and len(min_) > 0:
            sets = self.sam.generate_sets(maj, min_)
            Xtr, ytr = sets[0]  # pick the first generated set
            print(
                f"After sampling: Xtr shape = {Xtr.shape}, ytr shape = {ytr.shape}")

        self.clf.fit(Xtr, ytr)
        yp, pp = self.clf.predict(Xte)
        self.eval.evaluate(yte, yp, pp, "Multi-Stage Framework")
        print("Pipeline completed.")
        self.eval.compare()


# -------------------------
# Run Experiment
# -------------------------
if __name__ == "__main__":
    pipeline = CreditCardFraudDetectionPipeline()
    print("=== Running pipeline with synthetic data ===")
    pipeline.run_complete_pipeline(filepath=None, use_novel_sampling=True)
