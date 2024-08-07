import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Handle missing values and normalization if needed
    df = df.dropna()
    df.loc[:, 'title'] = df['title'].str.lower().str.strip()
    return df

def display_data(file_path):
    # Load the provided data
    job_titles_df = pd.read_csv(file_path)
    # Display the first few rows of the data
    return job_titles_df.head()
