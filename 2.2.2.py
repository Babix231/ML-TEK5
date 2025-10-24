import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import loguniform, randint
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

data_dir = Path("./data/regression")
X_train = np.load(data_dir / "X_train.npy")
X_test  = np.load(data_dir / "X_test.npy")
y_train = np.load(data_dir / "y_train.npy").ravel()
y_test  = np.load(data_dir / "y_test.npy").ravel()

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")

# Preprocessing
numeric_pre = Pipeline([
# Remplace les valeurs manquantes dans le dataset par la médiane de chaque colonne. Cela permet d’éviter les erreurs pendant l’entraînement des modèles quand il manque des données.
    ("imputer", SimpleImputer(strategy="median")),
# Normalise les données en leur donnant une moyenne de 0 et un écart-type de 1.
    ("scaler", StandardScaler())
])

# Model search spaces
spaces = {
    "Ridge": ("grid", Ridge(), {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    }),
    "Lasso": ("grid", Lasso(max_iter=10000), {
        "model__alpha": [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    }),
    "SVR": ("random", SVR(kernel="rbf"), {
        "model__C": loguniform(1e-2, 1e3),
        "model__gamma": loguniform(1e-4, 1e0),
        "model__epsilon": loguniform(1e-3, 1e-1)
    }, 40),
    "MLPRegressor": ("random", MLPRegressor(max_iter=1000, early_stopping=True, random_state=42), {
        "model__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
        "model__activation": ["relu", "tanh"],
        "model__alpha": loguniform(1e-6, 1e-2),
        "model__learning_rate_init": loguniform(1e-4, 1e-2)
    }, 40),
    "AdaBoostRegressor": ("random",
        AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=42),
        {
            "model__n_estimators": randint(50, 400),
            "model__learning_rate": loguniform(1e-3, 1.0),
            "model__estimator__max_depth": randint(1, 8),
            "model__estimator__min_samples_leaf": randint(1, 10)
        }, 40)
}

# 1 Tune on training set only (CV)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
best_model_name, best_estimator, best_cv_r2 = None, None, -np.inf
cv_results = []

for name, spec in spaces.items():
    kind = spec[0]
    model = spec[1]
    params = spec[2]

    pipe = Pipeline([
        ("prep", numeric_pre),
        ("model", model)
    ])

    print(f"\n=== Tuning {name} ===")
    if kind == "grid":
        search = GridSearchCV(pipe, param_grid=params, scoring="r2",
                              cv=cv, n_jobs=-1, refit=True, verbose=0)
    else:
        n_iter = spec[3]
        search = RandomizedSearchCV(pipe, param_distributions=params, n_iter=n_iter,
                                    scoring="r2", cv=cv, n_jobs=-1, random_state=42,
                                    refit=True, verbose=0)
    search.fit(X_train, y_train)
    print(f"{name} best CV R²: {search.best_score_:.3f}")
    print(f"{name} best params: {search.best_params_}")

    cv_results.append({"Model": name, "CV_R2": search.best_score_})
    if search.best_score_ > best_cv_r2:
        best_cv_r2 = search.best_score_
        best_model_name = name
        best_estimator = search.best_estimator_

# 2 Evaluate once on test set
print("\n================ FINAL TEST EVALUATION =================")
y_pred = best_estimator.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Best model (by CV): {best_model_name}")
print(f"CV mean R²: {best_cv_r2:.3f}")
print(f"TEST R²: {r2:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")

# 3 Save model + report
out_dir = Path("models")
out_dir.mkdir(exist_ok=True)
joblib.dump(best_estimator, out_dir / "best_model_2.2.2.joblib")

pd.DataFrame(cv_results).sort_values(by="CV_R2", ascending=False).to_csv(out_dir / "2.2.2_results.csv", index=False)
print(f"✔ Saved best model and CV results in {out_dir}/")
