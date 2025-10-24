import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

X = df.drop(columns=["Latitude", "Longitude"])
y_lat = df["Latitude"]
y_lon = df["Longitude"]

print(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")

# 2. Quick correlation analysis
corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix (features vs Latitude & Longitude)")
plt.show()

print("\nTop correlations with Latitude:")
print(corr["Latitude"].sort_values(ascending=False))
print("\nTop correlations with Longitude:")
print(corr["Longitude"].sort_values(ascending=False))

# 3. Split data and scale
X_train, X_test, y_lat_train, y_lat_test = train_test_split(
    X, y_lat, test_size=0.2, random_state=42
)
_, _, y_lon_train, y_lon_test = train_test_split(
    X, y_lon, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define regression models
models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42)
}

# 5. Train and evaluate models
for name, model in models.items():
    # Train model to predict Latitude
    model.fit(X_train_scaled, y_lat_train)
    y_lat_pred = model.predict(X_test_scaled)
    r2_lat = r2_score(y_lat_test, y_lat_pred)
    mae_lat = mean_absolute_error(y_lat_test, y_lat_pred)

    # Train model to predict Longitude
    model.fit(X_train_scaled, y_lon_train)
    y_lon_pred = model.predict(X_test_scaled)
    r2_lon = r2_score(y_lon_test, y_lon_pred)
    mae_lon = mean_absolute_error(y_lon_test, y_lon_pred)

    # Print evaluation results
    print(f"\n=== {name} ===")
    print(f"Latitude  : R² = {r2_lat:.3f} | MAE = {mae_lat:.3f}")
    print(f"Longitude : R² = {r2_lon:.3f} | MAE = {mae_lon:.3f}")

# 6. Visualization: Real vs Predicted Latitude
plt.figure(figsize=(6,6))
plt.scatter(y_lat_test, y_lat_pred, alpha=0.5, color="royalblue")
plt.plot([30,45],[30,45],"r--")
plt.xlabel("True Latitude")
plt.ylabel("Predicted Latitude")
plt.title("Latitude Prediction (Random Forest)")
plt.show()
