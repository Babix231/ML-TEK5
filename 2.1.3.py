import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- 1) Définition de Z ----
# X = taille en cm, Y = poids en kg
mu_X, sigma_X = 170, 10  # E[X]=170, Var[X]=100
mu_Y, sigma_Y = 70, 15   # E[Y]=70, Var[Y]=225
expected_value = np.array([mu_X, mu_Y])

# ---- 2) Échantillonnage ----
n_samples = 5000
X = np.random.normal(mu_X, sigma_X, n_samples)
Y = np.random.normal(mu_Y, sigma_Y, n_samples)
Z = np.vstack((X, Y)).T  # shape (n_samples, 2)

# ---- Graphique 1 : échantillons dans le plan (X,Y) ----
plt.figure(figsize=(6,6))
plt.scatter(Z[:500,0], Z[:500,1], alpha=0.5, label="Samples (X,Y)")
plt.scatter(*expected_value, c="red", marker="x", s=100, label="Expected value E[Z]")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Samples of Z = (Height, Weight)")
plt.legend()
plt.savefig("data/213/samples_scatter.png", dpi=300)
plt.close()

# ---- 3) Loi des grands nombres ----
empirical_means = np.cumsum(Z, axis=0) / np.arange(1, n_samples+1).reshape(-1,1)
distances = np.linalg.norm(empirical_means - expected_value, axis=1)

# ---- Graphique 2 : convergence ----
plt.figure(figsize=(8,5))
plt.plot(distances, label="|| Empirical mean - Expected value ||")
plt.axhline(0, color="red", linestyle="--", label="0 (convergence target)")
plt.xlabel("n (number of samples)")
plt.ylabel("Euclidean distance")
plt.title("Convergence of empirical mean to expected value (Law of Large Numbers)")
plt.legend()
plt.savefig("data/213/convergence_lln.png", dpi=300)
plt.close()
