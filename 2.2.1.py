import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

X_train = np.load("data/classification/X_train.npy")
X_test  = np.load("data/classification/X_test.npy")
y_train = np.load("data/classification/y_train.npy")
y_test  = np.load("data/classification/y_test.npy")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = [
    (
        "LogisticRegression",
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, solver="lbfgs"))]),
        {
            "scaler": [StandardScaler(), RobustScaler()],
            "clf__C": np.logspace(-3, 2, 7),  # 0.001 → 100
            "clf__penalty": ["l2"],
        },
    ),
    (
        "SVC",
        Pipeline([("scaler", StandardScaler()), ("clf", SVC())]),
        {
            "scaler": [StandardScaler(), RobustScaler()],
            "clf__kernel": ["rbf"],
            "clf__C": [0.3, 1, 3, 10, 30, 100],
            "clf__gamma": ["scale", "auto", 1e-3, 3e-3, 1e-2, 3e-2],
        },
    ),
    (
        "KNN",
        Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
        {
            "scaler": [StandardScaler(), RobustScaler()],
            "clf__n_neighbors": [3,5,7,9,11,13,15,17,21,25,31],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],  # Manhattan vs Euclidien
        },
    ),
    (
        "MLP",
        Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(max_iter=800, random_state=42, early_stopping=True))]),
        {
            "scaler": [StandardScaler(), RobustScaler()],
            "clf__hidden_layer_sizes": [(100,), (150,), (200,), (150, 50)],
            "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "clf__activation": ["relu", "tanh"],
        },
    ),
    (
        "RandomForest",
        Pipeline([("clf", RandomForestClassifier(random_state=42))]),
        {
            "clf__n_estimators": [100, 200, 400],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
        },
    ),
]

best_name, best_estimator, best_cv = None, None, -np.inf

print("\nCross-validation (train only):")
for name, pipe, grid in candidates:
    search = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    search.fit(X_train, y_train)
    print(f"- {name}: best CV = {search.best_score_:.3f} | params = {search.best_params_}")
    if search.best_score_ > best_cv:
        best_cv = search.best_score_
        best_estimator = search.best_estimator_
        best_name = name

y_pred = best_estimator.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("\n================= FINAL TEST SCORE (single use) =================")
print(f"Selected: {best_name} | CV mean: {best_cv:.3f} | Test acc: {test_acc:.3f}")
print("=================================================================")
