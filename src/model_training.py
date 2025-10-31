import os
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import yaml



log_dir = 'log'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')

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

def load_data(file_path : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("file loaded successfully %s with shape %s",file_path,df.shape)
        return df
    
    except pd.errors.ParserError as e:
        logger.error("problem in parsing the file %s",e)

    except FileNotFoundError as e:
        logger.error("file not found %s",e)

    
    except Exception as e:
        logger.error("An unexpected error occured %s",e)
        print(f"error {e}")


def model_training(x_train:np.ndarray , y_train : np.ndarray, params : dict)->RandomForestClassifier:
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("the number of sample in x_train must be same with y_train")
        logger.debug("initialising randomforestclassifier with the param %s",params)

        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        clf.fit(x_train,y_train)
        logger.debug("model training done successfully")
        return clf
    
    except ValueError as e:
        logger.error("value error during model training %s",e)
        raise
    except Exception as e:
        logger.error("unexpected error occured during model training : %s",e)
        raise

def save_model(model, file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug("model saved successfully as %s",file_path)
    except FileNotFoundError as e:
        logger.error("file not found %s",e)
        raise
    except Exception as e:
        logger.error("unexpected error occured while saving the model %s",e)
        raise


def main():
    try:
        params = load_params(file_path="params.yaml")['model_training']
        
    
        training_data = load_data('./data/processed/train_tfidf.csv')
        x_train = training_data.iloc[ : , :-1].values
        y_train = training_data.iloc[ : , -1].values

        clf = model_training(x_train, y_train, params)

        model_save_path = "model/model.pkl"

        save_model(clf, model_save_path)
        logger.debug("model saved successfully ")

    except Exception as e:
        logger.error("failed to complete model building process")
        raise




if __name__ == "__main__":
    main()


