import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ---------------------------- Data Loading and Preprocessing ----------------------------

def load_and_preprocess_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

    df = pd.read_csv(csv_path)

    required_cols = ['Nitrogen', 'Potassium', 'Phosphorous', 'PH', 'Soil', 'Crop']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    X = df[['Nitrogen', 'Potassium', 'Phosphorous', 'PH', 'Soil']]
    y = df['Crop']
    return X, y

# ---------------------------- Model Training ----------------------------

def train_model(X, y):
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Column types
    numeric_features = ['Nitrogen', 'Potassium', 'Phosphorous', 'PH']
    categorical_features = ['Soil']

    # Preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

    # Pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)

    # Train initial model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Initial Accuracy:", accuracy_score(y_test, y_pred))

    # Hyperparameter tuning (optional)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10],
        'classifier__min_samples_split': [2, 4]
    }

    grid = GridSearchCV(clf, param_grid, cv=3, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Final evaluation
    y_pred_final = best_model.predict(X_test)
    print("\n=== Final Model Performance ===")
    print(classification_report(y_test, y_pred_final))
    print("Best Params:", grid.best_params_)
    print("Final Accuracy:", accuracy_score(y_test, y_pred_final))

    return best_model, label_encoder

# ---------------------------- Save & Load Model ----------------------------

def save_model(model, encoder, filename="crop_classifier.pkl"):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump((model, encoder), f)
    print(f"\n✅ Model saved as '{filename}'")

def load_model(filename):
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)

# ---------------------------- Prediction Interface ----------------------------

def predict_crop(model, encoder):
    try:
        nitrogen = float(input("Enter Nitrogen value: "))
        potassium = float(input("Enter Potassium value: "))
        phosphorous = float(input("Enter Phosphorous value: "))
        ph = float(input("Enter pH value: "))
        soil_type = input("Enter Soil Type: ").strip()
    except ValueError:
        print("❌ Invalid input. Ensure all numerical values are correct.")
        return

    # Create DataFrame for input
    input_data = pd.DataFrame([{
        'Nitrogen': nitrogen,
        'Potassium': potassium,
        'Phosphorous': phosphorous,
        'PH': ph,
        'Soil': soil_type
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    predicted_label = encoder.inverse_transform([prediction])[0]

    print("\n=== Crop Recommendation ===")
    print(f"Recommended Crop: {predicted_label}")

# ---------------------------- Main ----------------------------

def main():
    csv_path = "fertilizer_recommendation_dataset.csv"  # Still using the same file

    try:
        X, y = load_and_preprocess_data(csv_path)
    except Exception as e:
        print("❌ Error:", e)
        return

    model, label_encoder = train_model(X, y)
    save_model(model, label_encoder)
    predict_crop(model, label_encoder)

if __name__ == '__main__':
    main()
