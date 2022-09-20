import joblib
import pandas as pd
import numpy as np
import os
import pickle
import xgboost
from sklearn.metrics import f1_score


import config
import dataset

import train

def predict(test, direction):

    if (direction == 'In') or (direction == 'Out'):
        
        preds = []
        for i in range(5):

            xgb = joblib.load(os.path.join(config.model_path,f'xgb_{direction}_{i}.pkl'))
            xgb_pred = xgb.predict(xgboost.DMatrix(test))

            lgb = joblib.load(os.path.join(config.model_path,f'lgb_{direction}_{i}.pkl'))
            lgb_pred = lgb.predict(test)

            preds.append((xgb_pred + lgb_pred) / 2)
        
        return np.array(preds).mean(axis=0)

        


    else:

        preds = []
        
        for i in range(5):
            print(i)
            lgb = joblib.load(os.path.join(config.model_path,f'lgb_categories_{i}.pkl'))
        
            y_pred = lgb.predict_proba(test)
    
            preds.append(y_pred)

        preds = np.array(preds).mean(axis=0)
        return preds



def run():
    
    train_in_txns, train_out_txns = dataset.get_in_out_txns(config.path, config.training_file)


    with open(os.path.join(config.model_path,'user_id.pkl'),'rb') as output:
        le_dict_user = pickle.load(output)

    
    x_test = train_out_txns.query(f"month == {7}").drop(['Out_sum'],axis=1)
    #x_test = pd.read_csv('../input/test/x_test_out_txns.csv')

    

    

    x_test['user_id'] = x_test['user_id'].apply(lambda x: le_dict_user.get(x, 'NA'))


    
    preds = predict(x_test, 'Out')
    
    x_test['Predictions'] = preds

    

    expense_predictions = pd.DataFrame({'user_id':x_test['user_id'],'month':x_test['month'], 'Predictions':preds})
    
    expense_predictions.to_csv(os.path.join(config.test_path,'expense_predictions.csv'), index=False)


    x_test = train_in_txns.query(f"month == {7}").drop(['In_sum'],axis=1)
    #x_test = pd.read_csv('../input/test/x_test_in_txns.csv')

    

    #print(y_test)

    x_test['user_id'] = x_test['user_id'].apply(lambda x: le_dict_user.get(x, 'NA'))


    
    preds = predict(x_test, 'In')
    
    x_test['Predictions'] = preds

    

    income_predictions = pd.DataFrame({'user_id':x_test['user_id'],'month':x_test['month'], 'Predictions':preds})

    income_predictions.to_csv(os.path.join(config.test_path,'income_predictions.csv'), index=False)

    
    categories  = dataset.prepare_categories(config.path, config.training_file)


    with open(os.path.join(config.model_path,'user_id.pkl'),'rb') as output:
        le_dict_user = pickle.load(output)

    
    with open(os.path.join(config.model_path,'categories.pkl'),'rb') as output:
        le_dict_category = pickle.load(output)

    
    x_test = categories.query(f"month == {7}").drop(['categories'],axis=1)
    
    #x_test = pd.read_csv('../input/test/x_test_categories.csv')

    

    

    x_test['user_id'] = x_test['user_id'].apply(lambda x: le_dict_user.get(x, 'NA'))

    


    preds = predict(x_test, 'category')

    
    preds = np.concatenate([x_test[['user_id','month']].values, preds],axis=1)

    categories_predictions = pd.DataFrame(preds, columns = ['user_id','month'] + list(le_dict_category.keys())[:-1])

    #doing the below step to convert column wise to row wise
    categories_predictions = categories_predictions.melt(id_vars=['user_id','month'], var_name="Predictions",value_name="percentage")

    

    
    
    categories_predictions.to_csv(os.path.join(config.test_path,'categories_predictions.csv'), index=False)



if __name__ == "__main__":
    run()

    










