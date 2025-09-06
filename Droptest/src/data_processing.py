import pandas as pd
import os

def get_project_root():
    """Get the project root directory path."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_data():
    """Load the dataset from the data directory."""
    data_path = os.path.join(get_project_root(), 'data', 'student_dropout_high_accuracy.csv')
    return pd.read_csv(data_path)

def preprocess_data(df):
    """Preprocess data by encoding categorical variables and scaling numerical ones."""
    
    # Binary encoding for simple categorical features
    binary_mapping = {
        'School': {'GP': 0, 'MS': 1},
        'Gender': {'F': 0, 'M': 1},
        'Address': {'R': 0, 'U': 1},
        'Family_Size': {'LE3': 0, 'GT3': 1},
        'Parental_Status': {'T': 0, 'A': 1},
        'School_Support': {'no': 0, 'yes': 1},
        'Family_Support': {'no': 0, 'yes': 1},
        'Extra_Paid_Class': {'no': 0, 'yes': 1},
        'Extra_Curricular_Activities': {'no': 0, 'yes': 1},
        'Attended_Nursery': {'no': 0, 'yes': 1},
        'Wants_Higher_Education': {'no': 0, 'yes': 1},
        'Internet_Access': {'no': 0, 'yes': 1},
        'In_Relationship': {'no': 0, 'yes': 1},
    }
    df.replace(binary_mapping, inplace=True)

    # One-hot encoding for multi-class categorical features
    categorical_cols = ['Mother_Job', 'Father_Job', 'Reason_for_Choosing_School', 'Guardian']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df
