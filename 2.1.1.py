import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
n = 300

# Age (years, integer, 18–80)
age_years = np.random.randint(18, 81, size=n)

# Height (cm, float, weak correlation with age)
height_cm = np.random.normal(170, 10, n)
height_cm = np.clip(height_cm, 150, 200)

# Weight (kg, float, positively correlated with height)
bmi = np.random.normal(24, 4, n)  # IMC moyen 24
weight_kg = bmi * (height_cm / 100) ** 2

# Hours of exercise per week (float, negatively correlated with age)
hours_exercise_per_week = 6 - (age_years - 18) * 0.05 + np.random.normal(0, 1.5, n)
hours_exercise_per_week = np.clip(hours_exercise_per_week, 0, None)

# Systolic blood pressure (mmHg, ↑ with age, ↓ with exercise)
systolic_bp = 110 + (age_years - 18) * 0.6 - hours_exercise_per_week * 0.8 + np.random.normal(0, 7, n)

# Cholesterol (mg/dL, ↑ with age, ↓ with exercise)
cholesterol_mg_dl = 170 + (age_years - 18) * 0.9 - hours_exercise_per_week * 2.5 + np.random.normal(0, 12, n)

# Assemble DataFrame
df = pd.DataFrame({
    "age_years": age_years.astype(int),             # int
    "height_cm": height_cm.round(1),                # float
    "weight_kg": weight_kg.round(1),                # float
    "hours_exercise_per_week": hours_exercise_per_week.round(1), # float
    "systolic_bp": systolic_bp.round(1),            # float
    "cholesterol_mg_dl": cholesterol_mg_dl.round(1) # float
})

# Sauvegarde
df.to_csv("data/211/artificial_dataset.csv", index=False)

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Vérifications
print(df.head())
print("\nMeans:\n", df.mean(numeric_only=True))
print("\nStandard deviations:\n", df.std(numeric_only=True))