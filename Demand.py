import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------------------------- Load and Preprocess Data ----------------------------

def load_and_preprocess_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

    df = pd.read_csv(csv_path)

    required_cols = ['State_Name', 'Crop', 'Season', 'Production']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean the Production column
    df['Production'] = pd.to_numeric(df['Production'], errors='coerce')
    df.dropna(subset=['Production'], inplace=True)

    X = df[['State_Name', 'Crop', 'Season']]
    y = df['Production']
    return X, y

# ---------------------------- Train Model ----------------------------

def train_model(X, y):
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score

    categorical_features = ['State_Name', 'Crop', 'Season']

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [5, 10]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n=== Final Model Performance ===")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print("Best Params:", grid.best_params_)

    return best_model

# ---------------------------- Save Model ----------------------------

def save_model(model, filename="production_forecast_model.pkl"):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved as '{filename}'")

# ---------------------------- Predict Production ----------------------------

def predict_production(model):
    try:
        state = input("Enter State Name (e.g., Tamil Nadu, Punjab): ").strip()
        crop = input("Enter Crop Name (e.g., Rice, Wheat): ").strip()
        season = input("Enter Season (e.g., Kharif, Rabi): ").strip()
    except ValueError:
        print("❌ Invalid input. Make sure to enter correct values.")
        return

    input_df = pd.DataFrame([{
        'State_Name': state,
        'Crop': crop,
        'Season': season
    }])

    prediction = model.predict(input_df)[0]
    print("\n=== Crop Production Forecast ===")
    print(f"Predicted Demand: {prediction:.2f} tons")

# ---------------------------- Main ----------------------------

def main():
    csv_path = "AgrcultureDataset.csv"  # Update the path if needed

    try:
        X, y = load_and_preprocess_data(csv_path)
    except Exception as e:
        print("❌ Error:", e)
        return

    model = train_model(X, y)
    save_model(model)
    predict_production(model)

if __name__ == '__main__':
    main()
