import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_sample_weight

# 1. DATA LOADING & PREP
print("🚀 Loading Dataset for Gender Subgroup Analysis...")
df = pd.read_csv('13Final_Thesis_Dataset.csv')
df = df.dropna(subset=['worthwhileness_rating'])

# City pre-processing (N=75 Threshold for stability)
threshold = 75 
city_counts = df['city'].value_counts()
small_cities = city_counts[city_counts < threshold].index
df['city'] = df['city'].apply(lambda x: 'Other_Small_City' if x in small_cities else x)

# Feature Engineering
df['Purpose_Leisure'] = df['purp_Leisure_Hobby'].fillna(0)

# Feature selection (Note: 'gender' is excluded from features as we split by it)
numerical_features = [
    'EFroutes_traffic_infrastructure', 'EFability_to_do_what_i_wanted', 
    'EFcomfort_pleasure', 'activity_count', 'education_level_numeric',
    'did_you_have_to_arrive', 'number_people_household',
    'Purpose_Leisure', 'leg_duration', 'act_Browsing', 'act_Accompanying', 'leg_distance'
] 

categorical_features = [
    'transport_category', 'age_range', 'city', 
    'marital_status_clean', 'weather_group', 
]
# 2. SUBGROUP ANALYSIS FUNCTION
def run_subgroup_pipeline(df_sub, label):
    print(f"\n🏗️ Training Final Research Model for: {label} (N={len(df_sub)})")
    
    X = df_sub[numerical_features + categorical_features].copy()
    y = df_sub['worthwhileness_rating']

    for col in categorical_features:
        X[col] = X[col].astype('category')

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Compute Weights
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # Training with "Golden" Stable Parameters
    model = xgb.XGBRegressor(
        objective='count:poisson',
        tree_method="hist",
        enable_categorical=True,
        n_estimators=1000,
        learning_rate=0.04,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=30,
        reg_lambda=10,
        reg_alpha=1,
        random_state=42
    )

    model.fit(X_train, y_train, sample_weight=weights)

    # Evaluation
    test_preds = model.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    
    # 1-Off Accuracy
    y_test_rounded = np.clip(np.round(test_preds), 1, 5)
    test_1off = np.mean(np.abs(y_test - y_test_rounded) <= 1)

    print(f"--- Results for {label} ---")
    print(f"R-Squared: {test_r2:.4f}")
    print(f"MAE:       {test_mae:.4f}")
    print(f"1-Off Acc: {test_1off:.2%}")

    # SHAP Analysis
    print(f"Generating SHAP for {label}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Plot 1: Global Importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"Global Feature Importance: {label}", fontsize=14)
    plt.xlabel("Average Impact on Worthwhileness (SHAP)")
    plt.tight_layout()
    plt.show()

    # Plot 2: Summary (Directional)
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary: {label} Directional Impact", fontsize=14)
    plt.tight_layout()
    plt.show()

# 3. EXECUTION
# Run for Male
df_male = df[df['gender'] == 'Male'].copy()
if not df_male.empty:
    run_subgroup_pipeline(df_male, "MALE")

# Run for Female
df_female = df[df['gender'] == 'Female'].copy()
if not df_female.empty:
    run_subgroup_pipeline(df_female, "FEMALE")
