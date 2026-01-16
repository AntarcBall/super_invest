import pandas as pd

def validate_dataframe_for_lgbm(df: pd.DataFrame, df_name: str):
    """
    Checks a DataFrame for non-numeric columns and raises a descriptive error if any are found.
    This serves as a final validation gate before passing data to a LightGBM model.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        df_name (str): The name of the DataFrame (e.g., 'X_train') for the error message.

    Raises:
        ValueError: If any non-numeric (object, category, etc.) columns are found.
    """
    print(f"--- Validating DataFrame: '{df_name}' ---")
    non_numeric_cols = df.select_dtypes(exclude=['number', 'bool']).columns
    
    if not non_numeric_cols.empty:
        error_message = (
            f"Validation failed for '{df_name}'. Found non-numeric columns: {list(non_numeric_cols)}. "
            f"LightGBM requires all features to be int, float, or bool. "
            f"Please ensure one-hot encoding is correctly applied and the original categorical columns are dropped."
        )
        print(f"ERROR: {error_message}")
        print("Problematic dtypes:")
        print(df[non_numeric_cols].dtypes)
        raise ValueError(error_message)
    
    print(f"OK: '{df_name}' is clean. All columns are numeric.")

