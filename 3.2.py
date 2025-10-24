import pandas as pd
import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge as BR, Ridge
import seaborn as sns

import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings('ignore')

df = pd.read_csv("data/32/housing.csv")

# Display the first few rows of the dataframe
print(df.head())
# Display summary statistics and info
print(df.describe())
# Display info and check for missing values
print(df.info())
# Check for missing values
print(df.isna().sum())
df.drop(columns=['ocean_proximity'],inplace=True)


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

df2 = impute_knn(df)
print(df2.isna().sum())
print(df2.info())

trdata,tedata = train_test_split(df2,test_size=0.3,random_state=43)

trdata.hist(bins=60, figsize=(15,9),color="purple")
# plt.show()

def corrMat(df, id=False):
    numeric_df = df.select_dtypes(include=[np.number])

    corr_mat = numeric_df.corr().round(2)

    f, ax = plt.subplots(figsize=(6, 6))
    mask = np.zeros_like(corr_mat, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        corr_mat,
        mask=mask,
        vmin=-1,
        vmax=1,
        center=0,
        cmap='plasma',
        square=False,
        lw=2,
        annot=True,
        cbar=False
    )
    # plt.show()

corrMat(trdata)

def plotTwo(df, lst):
    # load california from module, common for all plots
    cali = gpd.read_file(gplt.datasets.get_path('california_congressional_districts'))
    cali = cali.assign(area=cali.geometry.area)

    # Create a geopandas geometry feature; input dataframe should contain .longtitude, .latitude
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    proj = gcrs.AlbersEqualArea(central_latitude=37.16611, central_longitude=-119.44944)  # related to view

    ii = -1
    fig, ax = plt.subplots(1, 2, figsize=(21, 6), subplot_kw={'projection': proj})
    for i in lst:
        ii += 1
        tgdf = gdf.sort_values(by=i, ascending=True)
        gplt.polyplot(cali, projection=proj, ax=ax[ii])  # the module already has california
        gplt.pointplot(tgdf, ax=ax[ii], hue=i, cmap='plasma', legend=True, alpha=1.0, s=3)  #
        ax[ii].set_title(i)

    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.5)
    # plt.show()

plotTwo(trdata,['population','median_income'])
plotTwo(trdata,['housing_median_age','median_house_value'])
if 'geometry' in trdata.columns:
    del trdata['geometry']

maxval2 = trdata['median_house_value'].max()
trdata_upd = trdata[trdata['median_house_value'] != maxval2]
tedata_upd = tedata[tedata['median_house_value'] != maxval2]
trdata_upd.hist(bins=60, figsize=(15,9),color="purple")
# plt.show()


trdata_upd['diag_coord'] = (trdata_upd['longitude'] + trdata_upd['latitude'])
trdata_upd['bedperroom'] = trdata_upd['total_bedrooms']/trdata_upd['total_rooms']

corrMat(trdata_upd)

tedata_upd['diag_coord'] = (tedata_upd['longitude'] + tedata_upd['latitude'])
tedata_upd['bedperroom'] = tedata_upd['total_bedrooms']/tedata_upd['total_rooms']

plotTwo(trdata_upd,['diag_coord','median_house_value'])
plotTwo(trdata_upd,['bedperroom','median_house_value'])
if 'geometry' in trdata.columns:
    del trdata['geometry']

def modelEval(ldf, feature='median_house_value', model_id='dummy'):
    # Split feature/target variable
    y = ldf[feature].copy()
    X = ldf.copy()
    del X[feature]

    if model_id == 'dummy':
        model = DummyRegressor()
    if model_id == 'br':
        model = BR(verbose=False)
    if model_id == 'rf':
        model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2, random_state=42)

    cv_score = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))
    print("Model ID:", model_id)
    print("Scores:", cv_score)
    print("Mean:", cv_score.mean())
    print("std:", cv_score.std())
    print("-----")

modelEval(trdata,model_id='dummy')
modelEval(trdata,model_id='br')
modelEval(trdata_upd,model_id='br')
del trdata_upd['total_bedrooms']
del trdata_upd['total_rooms']
modelEval(trdata_upd,model_id='br')

def modelEval2(ldf, feature='median_house_value', model_id='dummy', scaling_id=False):
    # Given a dataframe, split feature/target variable
    y = ldf[feature].copy()
    X = ldf.copy()
    del X[feature]  # remove target variable

    for i in [2, 3]:
        if model_id == 'dummy':
            model = DummyRegressor()
        if model_id == 'br':
            model = BR(verbose=False)
        if model_id == 'rf':
            model = RandomForestRegressor(n_estimators=200,max_depth=20,min_samples_leaf=2,random_state=42)

        # Pick a Pipeline (Polynomial Feature Adjustment + Model)
        if not scaling_id:
            pipe = Pipeline(steps=[('poly', PolynomialFeatures(i)),
                                   ('model', model)])
        else:
            pipe = Pipeline(steps=[('scaler', StandardScaler()),
                                   ('poly', PolynomialFeatures(i)),
                                   ('model', model)])

        ''' Standard Cross Validation '''
        cv_score = np.sqrt(-cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error'))
        print("Scores:", cv_score.round(2))
        print("Mean:", cv_score.mean().round(2))
        print("std:", cv_score.std().round(2))
        print("-----")

modelEval2(trdata_upd,model_id='br',scaling_id=False)
modelEval2(trdata_upd,model_id='br',scaling_id=True)
modelEval(trdata_upd,model_id='rf')

X_train_full = trdata_upd.drop(columns=["median_house_value"])
y_train_full = trdata_upd["median_house_value"].values

# optional safety: drop any leftover non-numeric columns
X_train_full = X_train_full.select_dtypes(include=[np.number])

# -------------------------------------------------
# 2. Define models to compare
#    - Dummy baseline
#    - BayesianRidge (linear, probabilistic)
#    - RandomForestRegressor (non-linear)
#    - Polynomial Features + Ridge (to model interactions)
# -------------------------------------------------

models = {
    "Dummy": DummyRegressor(strategy="mean"),

    "BayesianRidge": BR(),

    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Poly2+Ridge": Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=1.0))
    ]),
}

# -------------------------------------------------
# 3. Cross-validation setup
# -------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# We'll compute:
# - RMSE (lower is better)
# - R²   (higher is better)

def cv_rmse(model, X, y, cv):
    # cross_val_score wants a scorer; we'll define negative RMSE and flip the sign
    neg_mse_scores = cross_val_score(
        model, X, y,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )
    # convert neg MSE -> RMSE
    rmse_scores = np.sqrt(-neg_mse_scores)
    return rmse_scores.mean(), rmse_scores.std()

def cv_r2(model, X, y, cv):
    r2_scores = cross_val_score(
        model, X, y,
        scoring="r2",
        cv=cv,
        n_jobs=-1
    )
    return r2_scores.mean(), r2_scores.std()

# -------------------------------------------------
# 4. Evaluate all models and collect metrics
# -------------------------------------------------
summary_rows = []

for name, model in models.items():
    mean_rmse, std_rmse = cv_rmse(model, X_train_full, y_train_full, cv)
    mean_r2, std_r2     = cv_r2(model, X_train_full, y_train_full, cv)

    summary_rows.append({
        "Model": name,
        "RMSE (mean)": mean_rmse,
        "RMSE (std)": std_rmse,
        "R2 (mean)": mean_r2,
        "R2 (std)": std_r2,
    })

    print(f"\n=== {name} ===")
    print(f"CV RMSE : {mean_rmse:.3f} ± {std_rmse:.3f}")
    print(f"CV R²   : {mean_r2:.3f} ± {std_r2:.3f}")

results_df = pd.DataFrame(summary_rows).sort_values(by="R2 (mean)", ascending=False)

print("\n######## Model Comparison (CV only, train data) ########")
print(results_df)

# -------------------------------------------------
# 5. Plot comparison (RMSE bars)
# -------------------------------------------------
plt.figure(figsize=(8,4))
plt.bar(results_df["Model"], results_df["RMSE (mean)"])
plt.ylabel("CV RMSE (lower is better)")
plt.title("Model performance comparison (5-fold CV)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()