import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
import yaml

log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(file_path : str)->dict:
    try:
        with open(file_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug("parameters recieved from %s",file_path)
        return params
    except FileNotFoundError as e:
        logger.error("file not fount at the location : %s",file_path)
        raise
    except yaml.error.YAMLError as e:
        logger.error("yaml error: %s",e)
        raise
    except Exception as e:
        logger.error("unexpected error occured while loading parameters %s", e)
        raise


def load_data(data_url : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("data loaded successfully from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file %s",e)
        raise
    except Exception as e:
        logger.error("An unexpected error occured %s",e)
        raise

def preprocess_data(df : pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"], inplace=True)
        df.rename(columns={"v1":"target", "v2":"text"}, inplace=True)
        logger.debug("data preproccessed successfully")
        return df

    except KeyError as e:
        logger.error("Missing column in the dataframe %s",e)
        raise
    
    except Exception as e:
        logger.error("An unexpected error occured %s",e)
        raise

def save_data(train_data : pd.DataFrame, test_data:pd.DataFrame, data_path:str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test_data.csv'), index=False)
        logger.debug("data saved successfully")
    
    except Exception as e:
        logger.error("An unexpected error occured while saving the data %s", e)
        raise

def main():
    try:
        params = load_params(file_path="params.yaml")
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2
        data_url = "https://raw.githubusercontent.com/Deb272004/spam-detection/refs/heads/main/experiments/spam.csv"
        df = load_data(data_url=data_url)
        final_df = preprocess_data(df=df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state= 2)
        save_data(train_data,test_data, data_path="./data")

    except Exception as e:
        logger.error("Failed to complete the data ingestion process %s",e)
        print(f"error {e}")


if __name__ == '__main__':
    main()
        
