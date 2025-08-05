import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def load_data(path: str) -> pd.DataFrame:
    try:
        #print(f"DEBUG: Checking if file exists at {path}: {os.path.exists(path)}")
        df = pd.read_csv(path,on_bad_lines='skip',encoding='latin-1')
        print(df.head(2))
        return df
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse the CSV file from {path}.")
        print(e)
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading the data.")
        print(e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Renaming columns for clarity and dropping unnecessary ones
        df = df[['v1','v2']].copy()
        df.rename(columns = {"v1":"Target", "v2":"Text"}, inplace = True)
        #df.drop(columns=[ 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True)
        
        #Label encode the Target and use it as y
        label_encoder = preprocessing.LabelEncoder()
        df['Target'] = label_encoder.fit_transform(df["Target"])
        
        return df
    except KeyError as e:
        print(f"Error: Missing column {e} in the dataframe.")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred during preprocessing.")
        print(e)
        raise
    
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise

def main():
    try:
        # Get absolute path to project root
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # gets path to src/
        PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))  # go one level up

        # Path to the CSV
        #csv_path = os.path.join(PROJECT_ROOT, "spam.csv")
        csv_path = "https://raw.githubusercontent.com/raunakravi084/Spam-classifier/main/spam.csv"
        print(f"Trying to open CSV at: {os.path.abspath(csv_path)}")
        df = load_data(path=csv_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)
        save_data(train_data, test_data, data_path='data')
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to complete the data ingestion process.")

if __name__ == '__main__':
    main()
