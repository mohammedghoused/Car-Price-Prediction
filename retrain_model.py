import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def main():
    print("Loading cleaned data...")
    # Load the cleaned data
    try:
        car = pd.read_csv("Cleaned Car.csv")
    except FileNotFoundError:
        print("Error: Cleaned Car.csv not found.")
        sys.exit(1)

    # Drop index column if present
    if 'Unnamed: 0' in car.columns:
        car = car.drop(columns='Unnamed: 0')

    print(f"Data shape: {car.shape}")

    # Prepare features and target
    X = car.drop(columns='Price')
    y = car['Price']

    # Step 1: Split data into training and testing sets (80% train, 20% test)
    # Using a fixed random_state to avoid data leakage during split optimization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Define Preprocessing Steps
    # We use TargetEncoder for high-cardinality columns ('name', 'company')
    # and OneHotEncoder for low-cardinality columns ('fuel_type')
    categorical_target_cols = ['name', 'company']
    categorical_ohe_cols = ['fuel_type']
    numerical_cols = ['year', 'kms_driven']

    # Impute missing values with most frequent for categorical and median for numerical, then scale/encode
    cat_target_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('target_enc', TargetEncoder(target_type='continuous', random_state=42))
    ])

    cat_ohe_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat_target', cat_target_transformer, categorical_target_cols),
            ('cat_ohe', cat_ohe_transformer, categorical_ohe_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='drop'
    )

    # Step 3: Train and Evaluate Linear Regression Baseline (using One-Hot Encoding)
    print("\nTraining Linear Regression Baseline...")
    # Linear Regression works better with OneHotEncoder because it models category effects linearly
    lr_preprocessor = ColumnTransformer(
        transformers=[
            ('cat_ohe', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), ['name', 'company', 'fuel_type']),
            ('num', numerical_transformer, numerical_cols)
        ]
    )

    lr_pipeline = Pipeline(steps=[
        ('preprocessor', lr_preprocessor),
        ('regressor', LinearRegression())
    ])

    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)

    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    print(f"Linear Regression Results:")
    print(f"  R2 Score: {r2_lr:.4f}")
    print(f"  MAE     : INR {mae_lr:.2f}")
    print(f"  RMSE    : INR {rmse_lr:.2f}")

    # Step 4: Hyperparameter Tuning on Random Forest (using Target Encoding)
    print("\nTuning Random Forest Regressor using RandomizedSearchCV...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Parameter grid for tuning
    param_distributions = {
        'regressor__n_estimators': [50, 100, 150, 200, 300],
        'regressor__max_depth': [None, 8, 12, 16, 20, 24],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': [1.0, 0.8, 'sqrt', 'log2']
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=param_distributions,
        n_iter=50,
        cv=cv,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rf_search.fit(X_train, y_train)
    
    print(f"\nBest Random Forest Parameters found:")
    for param, val in rf_search.best_params_.items():
        print(f"  {param}: {val}")
    print(f"Best CV R2 Score: {rf_search.best_score_:.4f}")

    # Evaluate Tuned Random Forest on Test Set
    best_rf_pipeline = rf_search.best_estimator_
    y_pred_rf = best_rf_pipeline.predict(X_test)

    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    print(f"\nTuned Random Forest Results:")
    print(f"  R2 Score: {r2_rf:.4f}")
    print(f"  MAE     : INR {mae_rf:.2f}")
    print(f"  RMSE    : INR {rmse_rf:.2f}")

    # Step 5: Compare Results
    print("\n" + "="*40)
    print("MODEL COMPARISON (On Holdout Test Set)")
    print("="*40)
    print(f"{'Metric':<10} | {'Linear Regression':<20} | {'Random Forest (Tuned)':<20}")
    print("-"*60)
    print(f"{'R2 Score':<10} | {r2_lr:<20.4f} | {r2_rf:<20.4f}")
    print(f"{'MAE':<10} | INR {mae_lr:<18.2f} | INR {mae_rf:<18.2f}")
    print(f"{'RMSE':<10} | INR {rmse_lr:<18.2f} | INR {rmse_rf:<18.2f}")
    print("="*40)

    # Step 6: Select best model and save it
    # We will pick the one with the higher R2 score (or lower MAE/RMSE)
    if r2_rf > r2_lr:
        best_model = best_rf_pipeline
        model_name = "Tuned Random Forest"
    else:
        # Just in case LR performs better, but RF TE usually does
        best_model = lr_pipeline
        model_name = "Linear Regression"

    print(f"\nSaving the best model ({model_name}) to 'CarPriceModel.pkl'...")
    with open('CarPriceModel.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Best model saved successfully!")

    # Test prediction on a sample instance to verify compatibility
    print("\nTesting saved model prediction...")
    test_sample = pd.DataFrame([['Maruti Suzuki Swift', 'Maruti', 2008, 100, 'Petrol']], 
                               columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    pred_val = best_model.predict(test_sample)
    print(f"Sample Input: Maruti Suzuki Swift (Maruti, 2008, 100 km, Petrol)")
    print(f"Predicted Price: INR {pred_val[0]:.2f}")

if __name__ == '__main__':
    main()
