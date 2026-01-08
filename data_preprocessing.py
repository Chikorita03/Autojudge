"""
data_preprocessing.py

NOTE:
This module is intentionally lightweight.
All file loading is handled directly in app.py.
During inference (Streamlit app), this module operates ONLY on DataFrames.
"""

def preprocess_data(df):
    """
    Preprocess dataframe for inference.
    Safe for Streamlit usage (no file I/O).
    """
    # Basic safety cleaning
    return df.fillna("")
