import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# 1. Load data
X = np.load("data/dimensionality_reduction/data.npy")
y = np.load("data/dimensionality_reduction/labels.npy")

print("Data shape:", X.shape, "Labels shape:", y.shape)

# Standardize (important for PCA, Isomap, etc.)
X_scaled = StandardScaler().fit_transform(X)

# 2. Define methods
methods = {
    "PCA": PCA,
    "t-SNE": TSNE,
}
chosen_methods = ["PCA", "t-SNE"]

# 3. Reduce to 2D and 3D and plot
fig_num = 1
for name in chosen_methods:
    for dim in [2, 3]:
        print(f"\n→ {name} to {dim}D")

        if name == "PCA":
            reducer = PCA(n_components=dim, random_state=42)
        elif name == "t-SNE":
            reducer = TSNE(n_components=dim, perplexity=30, random_state=42)
        else:
            continue

        X_reduced = reducer.fit_transform(X_scaled)

        if dim == 2:
            plt.figure(fig_num, figsize=(6,5))
            plt.scatter(X_reduced[:,0], X_reduced[:,1],
                        c=y, cmap="coolwarm", s=25, alpha=0.8)
            plt.title(f"{name} - 2D projection")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.colorbar(label="Label (0=No tempest, 1=Tempest)")
            plt.tight_layout()
            plt.savefig(f"data/232/{name}_2D.png", dpi=300)
            plt.close()
        else:
            fig = plt.figure(fig_num, figsize=(6,5))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(X_reduced[:,0], X_reduced[:,1], X_reduced[:,2],
                            c=y, cmap="coolwarm", s=25, alpha=0.8)
            ax.set_title(f"{name} - 3D projection")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            fig.colorbar(sc, label="Label (0/1)")
            plt.tight_layout()
            plt.savefig(f"data/232/{name}_3D.png", dpi=300)
            plt.close()

        fig_num += 1

print("\n=== Prediction on reduced features ===")
for name in chosen_methods:
    for dim in [2, 3]:
        if name == "PCA":
            reducer = PCA(n_components=dim, random_state=42)
        elif name == "t-SNE":
            reducer = TSNE(n_components=dim, perplexity=30, random_state=42)
        else:
            continue

        X_reduced = reducer.fit_transform(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"{name} ({dim}D) prediction accuracy: {acc:.3f}")