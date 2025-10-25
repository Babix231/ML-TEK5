import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# Charger le dataset
df = pd.read_csv("data/211/artificial_dataset.csv")

# Sélectionner uniquement les colonnes numériques
X = df.select_dtypes(include=[np.number]).values
cols = df.select_dtypes(include=[np.number]).columns

# ---- Métrique 1 : Euclidienne standardisée (toutes les colonnes à même importance) ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dist_matrix_euclidean = cdist(X_scaled, X_scaled, metric="euclidean")

# ---- Métrique 2 : Manhattan brute (garde l’effet des unités) ----
dist_matrix_manhattan = cdist(X, X, metric="cityblock")

def get_closest_and_farthest(dist_matrix):
    # Mettre la diagonale à NaN pour ignorer les distances à soi-même
    np.fill_diagonal(dist_matrix, np.nan)
    closest = np.unravel_index(np.nanargmin(dist_matrix), dist_matrix.shape)
    farthest = np.unravel_index(np.nanargmax(dist_matrix), dist_matrix.shape)
    return closest, farthest


# Résultats pour chaque métrique
closest_e, farthest_e = get_closest_and_farthest(dist_matrix_euclidean)
closest_m, farthest_m = get_closest_and_farthest(dist_matrix_manhattan)

def print_pair_info(name, pair, matrix):
    i, j = pair
    print(f"\n>>> {name}")
    print(f"Indices: {pair}, Distance = {matrix[i, j]:.3f}")
    print("Sample A:\n", df.iloc[i].to_dict())
    print("Sample B:\n", df.iloc[j].to_dict())

print("=== Métrique 1 : Euclidienne (standardisée) ===")
print_pair_info("Plus proches (Euclidienne)", closest_e, dist_matrix_euclidean)
print_pair_info("Plus éloignées (Euclidienne)", farthest_e, dist_matrix_euclidean)

print("\n=== Métrique 2 : Manhattan (brute, unités conservées) ===")
print_pair_info("Plus proches (Manhattan)", closest_m, dist_matrix_manhattan)
print_pair_info("Plus éloignées (Manhattan)", farthest_m, dist_matrix_manhattan)


