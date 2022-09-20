import pandas as pd
import numpy as np
import os
import pickle
import sys

import glob

import config
import dataset


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

if __name__ ==  "__main__":

    '''
    Process the test file to compare with 
    predictions.
    
    '''


    test_file_name = [os.path.basename(x) for x in glob.glob(os.path.join(config.private_path,'*testing.csv*'))][0]

    print('Testing with the filename ',test_file_name)

    test_in_txns, test_out_txns = dataset.get_in_out_txns(config.private_path,test_file_name)

    categories  = dataset.prepare_categories(config.private_path,test_file_name)

    

    with open(os.path.join(config.model_path,'user_id.pkl'),'rb') as output:
        le_dict_user = pickle.load(output)

    
    


    ### apply transformations for test set
    test_in_txns['user_id'] = test_in_txns['user_id'].apply(lambda x: le_dict_user.get(x, 'NA'))
    test_out_txns['user_id'] = test_out_txns['user_id'].apply(lambda x: le_dict_user.get(x, 'NA'))
    categories['user_id'] = categories['user_id'].apply(lambda x: le_dict_user.get(x, 'NA'))
    
    categories_total = categories.groupby(['user_id','month','categories']).agg({'Total': np.sum})

    categories_actual = categories_total.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()



    ###read predictions from the models
    income_preds = pd.read_csv(os.path.join(config.test_path,'income_predictions.csv'))
    expense_preds = pd.read_csv(os.path.join(config.test_path,'expense_predictions.csv'))
    categories_preds = pd.read_csv(os.path.join(config.test_path,'categories_predictions.csv'))
    categories_preds = categories_preds.groupby(['user_id','month','Predictions']).agg({'percentage': np.average}).reset_index()


    ####combine testfile and predictions on user_id and month to compare
    compare_income = test_in_txns.merge(income_preds, on=['user_id','month'], how='left')
    compare_income.rename(columns={'In_sum':'Actual'}, inplace=True)
    compare_income[['user_id','month','Actual','Predictions']].to_csv(os.path.join(config.private_path, 'compare_income.csv'), index=False)
    

    print('RMSPE on Income predictions is ',rmspe(compare_income['Actual'], compare_income['Predictions']) * 100)

    compare_expense = test_out_txns.merge(expense_preds, on=['user_id','month'], how='left')
    compare_expense.rename(columns={'Out_sum':'Actual'}, inplace=True)
    compare_expense[['user_id','month','Actual','Predictions']].to_csv(os.path.join(config.private_path, 'compare_expense.csv'), index=False)
    
    print('RMSPE on Expense predictions is ',rmspe(compare_expense['Actual'], compare_expense['Predictions'])* 100)

    
    ## these two csv files are used in web app
    categories_actual.to_csv(os.path.join(config.private_path, 'categories_actual.csv'), index=False)
    categories_preds.to_csv(os.path.join(config.private_path, 'categories_preds.csv'), index=False)


    

    















