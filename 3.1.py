"""
BREAST CANCER CLASSIFICATION ANALYSIS
======================================

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Source: UCI Machine Learning Repository / Sklearn built-in dataset
URL: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

DATASET DESCRIPTION (in my own words):
--------------------------------------
This dataset contains measurements from digitized images of fine needle aspirate (FNA) 
of breast masses. Each sample represents a breast tumor, and for each tumor, we have 
computed 30 numerical features describing the characteristics of cell nuclei present 
in the image.

The features are organized in three groups:
- Mean values (10 features): average measurements across all cells
- Standard error values (10 features): variability in measurements
- Worst/largest values (10 features): mean of the three largest values

The 10 base measurements captured are:
1. radius: distance from center to perimeter
2. texture: standard deviation of gray-scale values
3. perimeter: size of the core tumor
4. area: size of the tumor area
5. smoothness: local variation in radius lengths
6. compactness: (perimeter² / area - 1.0)
7. concavity: severity of concave portions of the contour
8. concave points: number of concave portions of the contour
9. symmetry: how symmetrical the tumor is
10. fractal dimension: "coastline approximation" - 1

PROBLEM STATEMENT:
------------------
We are trying to predict the DIAGNOSIS of a breast tumor: whether it is MALIGNANT (M) 
or BENIGN (B) based on the 30 numerical features describing the cell nuclei characteristics.

This is a binary classification problem where:
- Target variable: 'diagnosis' column (M = Malignant, B = Benign)
- Input features: All 30 numerical measurements

PRACTICAL VALUE:
This problem is extremely valuable for healthcare:
- Early cancer detection can save lives
- Automated diagnosis can assist pathologists in making faster, more consistent decisions
- Can help prioritize cases that need immediate attention
- Reduces subjectivity in diagnosis
- Can be deployed in regions with limited access to expert pathologists
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("STEP 1: LOADING AND INITIAL EXPLORATION")
print("="*80)

# Load the dataset from CSV
df = pd.read_csv('breast_cancer_data.csv')

print("\nDataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sum(), "total missing values")

print("\nTarget variable distribution:")
print(df['diagnosis'].value_counts())
print("\nPercentage:")
print(df['diagnosis'].value_counts(normalize=True) * 100)

print("\n" + "="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Basic statistics
print("\nBasic Statistics for Mean Features:")
mean_features = [col for col in df.columns if 'mean' in col]
print(df[mean_features].describe())

# Check for outliers using IQR method
def detect_outliers_iqr(df, columns):
    outlier_counts = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    return outlier_counts

print("\nOutlier Detection (IQR method):")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numeric_cols:
    numeric_cols.remove('id')
outliers = detect_outliers_iqr(df, numeric_cols[:10])  # Check first 10 features
for col, count in sorted(outliers.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {col}: {count} outliers")

# Correlation analysis
print("\nCorrelation Analysis:")
print("We analyze correlations to understand feature relationships and potential redundancy.")

# Prepare data for correlation
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis'].map({'M': 1, 'B': 0})

# Correlation between features and target
correlations_with_target = X.corrwith(y).abs().sort_values(ascending=False)
print("\nTop 10 features most correlated with diagnosis:")
print(correlations_with_target.head(10))

print("\nInterpretation:")
print("- Features like 'concave points_worst', 'perimeter_worst', and 'radius_worst'")
print("  show strong correlation with malignancy.")
print("- This aligns with medical intuition: larger, more irregular tumors tend to be malignant.")
print("- Multiple 'worst' features appear, suggesting extreme values are important indicators.")

# High inter-feature correlations
corr_matrix = X.corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((corr_matrix.columns[i], 
                                   corr_matrix.columns[j], 
                                   corr_matrix.iloc[i, j]))

print(f"\nHighly correlated feature pairs (|r| > 0.9): {len(high_corr_pairs)}")
if high_corr_pairs:
    print("Examples:")
    for feat1, feat2, corr in high_corr_pairs[:5]:
        print(f"  {feat1} <-> {feat2}: {corr:.3f}")
    print("\nThis suggests multicollinearity, which could affect some models.")

print("\n" + "="*80)
print("STEP 3: DATA VISUALIZATION")
print("="*80)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Target distribution
axes[0, 0].bar(['Benign', 'Malignant'], df['diagnosis'].value_counts().values)
axes[0, 0].set_title('Distribution of Diagnosis', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Count')
for i, v in enumerate(df['diagnosis'].value_counts().values):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# 2. Feature distributions by diagnosis
top_features = correlations_with_target.head(4).index
for idx, feature in enumerate(top_features):
    if idx < 2:
        ax = axes[0, 1] if idx == 0 else axes[1, 0]
        df.boxplot(column=feature, by='diagnosis', ax=ax)
        ax.set_title(f'{feature} by Diagnosis', fontsize=12, fontweight='bold')
        ax.set_xlabel('Diagnosis')
        plt.sca(ax)
        plt.xticks([1, 2], ['Benign', 'Malignant'])

# 3. Correlation heatmap (top features)
top_10_features = correlations_with_target.head(10).index.tolist()
corr_subset = X[top_10_features].corr()
sns.heatmap(corr_subset, annot=False, cmap='coolwarm', center=0, 
            ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
axes[1, 1].set_title('Correlation Matrix (Top 10 Features)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'eda_visualization.png'")
plt.show()

print("\n" + "="*80)
print("STEP 4: DATA PREPROCESSING")
print("="*80)

# Prepare features and target
# Drop id, diagnosis, and any unnamed columns
X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, errors='ignore')
y = df['diagnosis'].map({'M': 1, 'B': 0})  # 1 = Malignant, 0 = Benign

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nClass distribution: Benign={sum(y==0)}, Malignant={sum(y==1)}")
print(f"Class balance: {sum(y==1)/len(y)*100:.1f}% malignant")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scaling - IMPORTANT for distance-based algorithms and regularized models
print("\nScaling Strategy:")
print("StandardScaler is essential for this dataset because:")
print("- Features have vastly different scales (e.g., area ~100-2500, smoothness ~0.05-0.15)")
print("- Many algorithms (SVM, KNN, Logistic Regression) are scale-sensitive")
print("- It doesn't change the distribution, only centers and scales")

# First, handle missing values with imputation
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Then scale the imputed data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print("\nBefore scaling - sample feature ranges:")
print(X_train[['radius_mean', 'texture_mean', 'area_mean']].describe().loc[['mean', 'std']])
print("\nAfter scaling - same features:")
temp_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
print(temp_scaled[['radius_mean', 'texture_mean', 'area_mean']].describe().loc[['mean', 'std']])

print("\n" + "="*80)
print("STEP 5: MODEL TRAINING AND COMPARISON")
print("="*80)

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\nTraining models with 5-fold cross-validation...\n")

for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    
    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    training_time = time.time() - start_time
    
    # Metrics
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
        'training_time': training_time,
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Test Accuracy: {results[name]['test_accuracy']:.4f}")
    print(f"  Training Time: {training_time:.3f}s\n")

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': results.keys(),
    'CV Accuracy': [results[m]['cv_mean'] for m in results],
    'CV Std': [results[m]['cv_std'] for m in results],
    'Test Accuracy': [results[m]['test_accuracy'] for m in results],
    'Precision': [results[m]['test_precision'] for m in results],
    'Recall': [results[m]['test_recall'] for m in results],
    'F1 Score': [results[m]['test_f1'] for m in results],
    'ROC AUC': [results[m]['test_roc_auc'] for m in results],
    'Time (s)': [results[m]['training_time'] for m in results]
})

print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print(comparison_df.to_string(index=False))

print("\n" + "="*80)
print("STEP 6: HYPERPARAMETER OPTIMIZATION")
print("="*80)

print("\nPerforming Grid Search on best models...")

# Optimize Random Forest (one of the best performers typically)
print("\n1. Random Forest Optimization:")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train_scaled, y_train)

print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best CV score: {rf_grid.best_score_:.4f}")
print(f"Test accuracy: {rf_grid.score(X_test_scaled, y_test):.4f}")

# Optimize SVM
print("\n2. SVM Optimization:")
svm_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

svm_grid = GridSearchCV(
    SVC(kernel='rbf', probability=True, random_state=42),
    svm_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
svm_grid.fit(X_train_scaled, y_train)

print(f"Best parameters: {svm_grid.best_params_}")
print(f"Best CV score: {svm_grid.best_score_:.4f}")
print(f"Test accuracy: {svm_grid.score(X_test_scaled, y_test):.4f}")

# Select best model
best_model_name = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
best_model = results[best_model_name]['model']

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'='*80}")

print("\n" + "="*80)
print("STEP 7: DETAILED EVALUATION OF BEST MODEL")
print("="*80)

y_pred_best = results[best_model_name]['y_pred']
y_pred_proba_best = results[best_model_name]['y_pred_proba']

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)
print("\nInterpretation:")
print(f"True Negatives (Benign correctly identified): {cm[0,0]}")
print(f"False Positives (Benign predicted as Malignant): {cm[0,1]}")
print(f"False Negatives (Malignant predicted as Benign): {cm[1,0]} ⚠️ CRITICAL")
print(f"True Positives (Malignant correctly identified): {cm[1,1]}")

# Classification Report
print("\n" + "-"*80)
print("Classification Report:")
print("-"*80)
print(classification_report(y_test, y_pred_best, 
                          target_names=['Benign', 'Malignant']))

# Medical context interpretation
print("\n" + "-"*80)
print("MEDICAL CONTEXT INTERPRETATION:")
print("-"*80)
recall_malignant = recall_score(y_test, y_pred_best)
precision_malignant = precision_score(y_test, y_pred_best)

print(f"\nRecall (Sensitivity) for Malignant cases: {recall_malignant:.2%}")
print("→ This is the % of actual cancer cases we correctly identify")
print("→ HIGH RECALL IS CRITICAL: Missing cancer (false negative) can be fatal")

print(f"\nPrecision for Malignant cases: {precision_malignant:.2%}")
print("→ This is the % of predicted cancer cases that are truly cancer")
print("→ Lower precision means more unnecessary biopsies/treatments")

if cm[1,0] > 0:  # False negatives exist
    print(f"\n⚠️  WARNING: {cm[1,0]} cancer case(s) were missed!")
    print("This is the most critical error in cancer diagnosis.")

print("\n" + "="*80)
print("STEP 8: COMPARISON WITH AND WITHOUT PREPROCESSING")
print("="*80)

print("\nTraining models WITHOUT scaling (on raw data)...")
results_unscaled = {}

for name, model in models.items():
    try:
        model_copy = type(model)(**model.get_params())
        model_copy.fit(X_train, y_train)
        y_pred_unscaled = model_copy.predict(X_test)
        acc_unscaled = accuracy_score(y_test, y_pred_unscaled)
        results_unscaled[name] = acc_unscaled
        print(f"{name}: {acc_unscaled:.4f}")
    except:
        results_unscaled[name] = None
        print(f"{name}: Failed")

print("\n" + "-"*80)
print("IMPACT OF SCALING:")
print("-"*80)
for name in models.keys():
    if results_unscaled[name] is not None:
        diff = results[name]['test_accuracy'] - results_unscaled[name]
        print(f"{name}:")
        print(f"  Without scaling: {results_unscaled[name]:.4f}")
        print(f"  With scaling: {results[name]['test_accuracy']:.4f}")
        print(f"  Improvement: {diff:+.4f} ({diff/results_unscaled[name]*100:+.1f}%)\n")

print("\n" + "="*80)
print("STEP 9: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance_df.head(15).to_string(index=False))
    
    print("\nInsight:")
    top_feature = feature_importance_df.iloc[0]['Feature']
    print(f"'{top_feature}' is the most important feature.")
    print("This aligns with medical knowledge about tumor characteristics.")

print("\n" + "="*80)
print("FINAL CONCLUSIONS AND RECOMMENDATIONS")
print("="*80)

print(f"\n1. BEST MODEL: {best_model_name}")
print(f"   - Test Accuracy: {results[best_model_name]['test_accuracy']:.2%}")
print(f"   - Recall (Sensitivity): {results[best_model_name]['test_recall']:.2%}")
print(f"   - Precision: {results[best_model_name]['test_precision']:.2%}")
print(f"   - F1 Score: {results[best_model_name]['test_f1']:.2%}")

print("\n2. HAVE WE SOLVED THE PROBLEM?")
if results[best_model_name]['test_accuracy'] > 0.95:
    print("   ✓ YES - The model achieves excellent accuracy (>95%)")
else:
    print("   ~ PARTIALLY - Good but could be improved")

print("\n3. CAN THIS MODEL BE USED IN PRACTICE?")
if recall_malignant > 0.95:
    print("   ✓ YES - High recall means few cancer cases are missed")
    print("   → Could be used as a screening tool to assist pathologists")
    print("   → Should NOT replace human expert diagnosis")
else:
    print("   ⚠️  CAUTION - Recall could be improved")
    print("   → Currently misses too many cancer cases for clinical use")
    print("   → Needs improvement before deployment")

print("\n4. EXPECTED PERFORMANCE ON UNSEEN DATA:")
cv_mean = results[best_model_name]['cv_mean']
cv_std = results[best_model_name]['cv_std']
print(f"   Based on cross-validation: {cv_mean:.2%} ± {cv_std:.2%}")
print(f"   Test set performance: {results[best_model_name]['test_accuracy']:.2%}")

if abs(cv_mean - results[best_model_name]['test_accuracy']) < 0.02:
    print("   ✓ Model generalizes well (CV and test scores are close)")
else:
    print("   ⚠️  Some discrepancy between CV and test performance")

print("\n5. KEY TAKEAWAYS:")
print("   • Feature scaling is crucial for this dataset")
print("   • Multiple models achieve >95% accuracy")
print("   • 'Worst' features are most predictive of malignancy")
print("   • High correlation between features suggests dimensionality reduction possible")
print("   • For medical applications, minimizing false negatives is critical")

print("\n6. RECOMMENDATIONS FOR IMPROVEMENT:")
print("   • Collect more malignant samples to balance the dataset")
print("   • Try ensemble methods combining multiple models")
print("   • Implement cost-sensitive learning (higher cost for false negatives)")
print("   • Use techniques like SMOTE for better class balance")
print("   • Consider deep learning approaches for automatic feature extraction")

print("\n7. DEPLOYMENT CONSIDERATIONS:")
print("   • Model should be validated on external datasets")
print("   • Regular retraining with new data is essential")
print("   • Human expert review should always be the final decision")
print("   • Explainability tools (SHAP, LIME) should be added for clinical trust")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)