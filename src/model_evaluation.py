import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import json
import logging
import pickle
import yaml
from dvclive import Live

log_dir = 'log'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')

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



def load_model(file_path : str):
    try:
        with open(file_path,'rb') as file:
            model = pickle.load(file)
        logger.debug("model loaded from %s",file_path)
        return model
    except FileNotFoundError as e:
        logger.error("file not found")
        raise
    except Exception as e:
        logger.error("unexpected error occured while loading the model from %s",file_path)
        print(f"error : {e}")
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

def evaluate_model(clf, x_test : np.ndarray, y_test : np.ndarray) -> dict:
    try:
        
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred_proba)

        metrices_dict = {
            'accuracy_score':accuracy,
            'precision_score':precision,
            'recall':recall,
            'auc':auc
        }
        
        logger.debug("model evaluation metrices calculated sucessfully")
        return metrices_dict

    except Exception as e:
        logger.error("error occured during model evatuation %s",e)
        raise


def save_metrices(metrices:dict,file_path : str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'w') as file:
            json.dump(metrices,file,indent=4)
        logger.debug("metrices saves successfull at %s",file_path)

    except Exception as e:
        logger.error("error occured during svaing the metrices %s",e)
        raise

def main():
    try:
        params = load_params(file_path="params.yaml")
        clf = load_model("./model/model.pkl")
        test_data = load_data("./data/processed/test_tfidf.csv")

        x_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values

        metrices = evaluate_model(clf, x_test,y_test)

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy_scroe : ',accuracy_score(y_test,y_test))
            live.log_metric('precision_score : ',precision_score(y_test,y_test))
            live.log_metric('recall_score : ',recall_score(y_test,y_test))

            live.log_params(params)

        save_metrices(metrices, "reports/metrices.json")

    except Exception as e:
        logger.error("Error occured while completing model evluation process : %s",e)
        print(f"error{e}")


if __name__ == "__main__":
    main()







