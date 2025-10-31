import pandas as pd
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import numpy as np
import yaml

log_dir = 'log'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

fomatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(fomatter)
file_handler.setFormatter(fomatter)

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


def load_data(file_path : str)->pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug("data loaded successfully %s",file_path)
        return df
    
    except pd.errors.ParserError as e:
        logger.error("failed to parse the data %s",e)
        raise

    except Exception as e:
        logger.error("failed to load the data %s",e)
        raise

def apply_tfidf(train_data : pd.DataFrame, test_data: pd.DataFrame, max_features : int)->tuple:

    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        x_train = train_data['text'].values
        y_train = train_data['target'].values
        x_test = test_data['text'].values
        y_test = test_data['target'].values

        
        x_train_idf = vectorizer.fit_transform(x_train)
        x_test_idf = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_idf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_idf.toarray())
        test_df['label'] = y_test

        logger.debug("TF-IDF applied and data transformed")

        return train_df, test_df
    
    except Exception as e:
        logger.error("error during TFIDF transformation %s",e)
        raise


def save_data(df:pd.DataFrame, file_path:str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug("File saved at %s",file_path)
    except Exception as e:
        logger.error("an unexpected error occured while saving data %s",e)
        raise


def main():
    try:
        params = load_params(file_path="params.yaml")
        max_feature = params['feature_engineering']['max_features']

        train_data = load_data("./data/intermediate/train_processed.csv")
        test_data = load_data("./data/intermediate/test_processed.csv")

        train_df , test_df = apply_tfidf(train_data,test_data,max_feature)

        save_data(train_df, os.path.join("./data","processed","train_tfidf.csv"))
        save_data(test_df, os.path.join("./data","processed","test_tfidf.csv"))

    except Exception as e:
        logger.error("failed to complete feature engineering process %s",e)
        print(f"error : {e}")


if __name__ == "__main__":
    main()

