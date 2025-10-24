import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def main():
    # 1. Load your CSV file
    path = "data/34/housing.csv"
    df = pd.read_csv(path)
    print("Dataset loaded successfully!")
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    # Display the first few rows of the dataframe
    print(df.head())
    # Display summary statistics and info
    print(df.describe())
    # Display info and check for missing values
    print(df.info())
    # Check for missing values
    print(df.isna().sum())
    df.drop(columns=['ocean_proximity'], inplace=True)

    def impute_knn(df):
        # separate dataframe into numerical/categorical
        ldf = df.select_dtypes(include=[np.number])  # select numerical columns in df
        ldf_putaside = df.select_dtypes(exclude=[np.number])  # select categorical columns in df
        # define columns w/ and w/o missing data
        cols_nan = ldf.columns[ldf.isna().any()].tolist()  # columns w/ nan
        cols_no_nan = ldf.columns.difference(cols_nan).values  # columns w/o nan

        for col in cols_nan:
            imp_test = ldf[ldf[col].isna()]  # indicies which have missing data will become our test set
            imp_train = ldf.dropna()  # all indicies which which have no missing data
            model = KNeighborsRegressor(n_neighbors=5)  # KNR Unsupervised Approach
            knr = model.fit(imp_train[cols_no_nan], imp_train[col])
            ldf.loc[df[col].isna(), col] = knr.predict(imp_test[cols_no_nan])

        return pd.concat([ldf, ldf_putaside], axis=1)

    df = impute_knn(df)
    print(df.isna().sum())
    print(df.info())

    # 2. Identify target column dynamically
    possible_targets = ["median_house_value"]
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    print(f"Target column detected: {target_col}")

    # 3. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop non-numeric columns (like 'ocean_proximity' if present)
    X = X.select_dtypes(include=[np.number])
    print(f"Using {X.shape[1]} numeric features: {list(X.columns)}")

    # 4. Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. PCA (2D)
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    explained_2d = pca_2d.explained_variance_ratio_

    print("\nPCA 2D explained variance ratio:", np.round(explained_2d, 3))
    print("Total variance captured:", np.round(np.sum(explained_2d), 3))

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        X_pca_2d[:, 0], X_pca_2d[:, 1],
        c=y, cmap='viridis', alpha=0.6, s=20
    )
    plt.colorbar(sc, label="Median House Value ($)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA 2D projection (colored by house value)")
    plt.tight_layout()
    plt.show()

    # 6. PCA (3D)
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    explained_3d = pca_3d.explained_variance_ratio_

    print("\nPCA 3D explained variance ratio:", np.round(explained_3d, 3))
    print("Total variance captured:", np.round(np.sum(explained_3d), 3))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(
        X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
        c=y, cmap="viridis", alpha=0.6, s=20
    )
    fig.colorbar(p, ax=ax, label="Median House Value ($)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 3D projection (colored by house value)")
    plt.tight_layout()
    plt.show()

    # 7. t-SNE (2D)
    print("\nRunning t-SNE.")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        random_state=42,
        init="random"
    )
    X_tsne_2d = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(7, 6))
    sc2 = plt.scatter(
        X_tsne_2d[:, 0], X_tsne_2d[:, 1],
        c=y, cmap="viridis", alpha=0.6, s=20
    )
    plt.colorbar(sc2, label="Median House Value ($)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.title("t-SNE 2D projection (colored by house value)")
    plt.tight_layout()
    plt.show()

    # 8. Correlation PCA components ↔ price
    comp_df = pd.DataFrame({
        "PCA1": X_pca_2d[:, 0],
        "PCA2": X_pca_2d[:, 1],
        "Price": y.values
    })
    corr_components = comp_df.corr()
    print("\nCorrelation between PCA components and target price:")
    print(corr_components)

    plt.figure(figsize=(4, 3))
    sns.heatmap(corr_components, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation between PCA components and house value")
    plt.tight_layout()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 2 Fit PCA again to full training set
    pca_full = PCA(n_components=3)
    X_train_pca = pca_full.fit_transform(X_train)
    X_test_pca = pca_full.transform(X_test)

    print(f"\nPCA transformed data shapes: Train={X_train_pca.shape}, Test={X_test_pca.shape}")
    print("Explained variance (3D):", np.round(pca_full.explained_variance_ratio_, 3))

    # 3 Train models on PCA data
    models = {
        "Ridge (PCA)": Ridge(alpha=1.0),
        "RandomForest (PCA)": RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=42
        )
    }

    for name, model in models.items():
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"\n{name} Results:")
        print(f"R²   = {r2:.3f}")
        print(f"RMSE = {rmse:.3f}")

    # 4 For comparison: train the same models on full data
    models_full = {
        "Ridge (Full)": Ridge(alpha=1.0),
        "RandomForest (Full)": RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=42
        )
    }

    for name, model in models_full.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"\n{name} Results (full data):")
        print(f"R²   = {r2:.3f}")
        print(f"RMSE = {rmse:.3f}")


if __name__ == "__main__":
    main()
