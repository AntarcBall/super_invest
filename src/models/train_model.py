import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.models.prepare_data import prepare_training_data
from src.features.feature_builder import add_fundamental_features, add_technical_features, add_macro_features
from src.data.data_loader import get_stock_data

def train_lgbm_model(X, y):
    """
    Trains a LightGBM model and evaluates its performance.

    Args:
        X (pd.DataFrame): The feature set.
        y (pd.Series): The target variable.

    Returns:
        lgb.LGBMClassifier: The trained LightGBM model.
    """
    # Time-based split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Initialize and train the model
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == '__main__':
    TICKER = 'MSFT'
    START = '2022-01-01'
    END = '2024-01-01'
    
    data = get_stock_data(TICKER, START, END)
    
    if not data.empty:
        # Build features
        data = add_fundamental_features(data, TICKER)
        data = add_technical_features(data)
        data = add_macro_features(data)
        
        # Prepare training data
        X, y = prepare_training_data(data, horizon=5, threshold=0.02)
        
        # Train the model
        trained_model = train_lgbm_model(X, y)
        print("\nModel training complete.")
