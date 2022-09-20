import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb

from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import pickle
import joblib
import warnings

warnings.filterwarnings('ignore')


import dataset
import config
from preprocessing import preprocess


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

# Function to early stop with root mean squared percentage error
def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False

def feval_rmspe_xgb(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred)












def income_expense_training(X, Y, x_test,direction):

    

    


    with open(os.path.join(config.model_path,'user_id.pkl'),'rb') as output:
        le_dict_user = pickle.load(output)

    
    X['user_id'] = X['user_id'].apply(lambda x: le_dict_user.get(x, 'NA'))

    x_test['user_id'] = x_test['user_id'].apply(lambda x: le_dict_user.get(x, 'NA'))


    oof_predictions = np.zeros(X.shape[0])
    test_predictions = np.zeros(x_test.shape[0])

    group = GroupKFold(n_splits=5)

    group_values = X['month']

    for fold, (trn_ind, val_ind) in enumerate(group.split(X, Y, groups=group_values)):

        print(f'Training fold {fold + 1}')
        x_train, x_val = X.iloc[trn_ind], X.iloc[val_ind]
        y_train, y_val = Y.iloc[trn_ind], Y.iloc[val_ind]
    
        
        train_dataset = lgb.Dataset(x_train, y_train,  categorical_feature = ['turbine_id'])
        val_dataset = lgb.Dataset(x_val, y_val,  categorical_feature = ['user_id','month'])
    
    
        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(x_val, y_val)
    
    
    
        

        watchlist = [(dtrain, 'train'), (dtest, 'test')]

    

    
        model = lgb.train(params = config.params_lgb, 
                    train_set = train_dataset, 
                    valid_sets =  [train_dataset,val_dataset], 
                    num_boost_round = 600, 
                    early_stopping_rounds = 50, 
                    verbose_eval = 500,
                    feval = feval_rmspe)
    
    
    
        xgb_model = xgb.train(config.params_xgb, dtrain,  500,evals = watchlist,
                      early_stopping_rounds = 50, feval = feval_rmspe_xgb, verbose_eval = 500)
    
    
        

        
    

        
    
        lgb_val = model.predict(x_val) 
        xgb_val = xgb_model.predict(xgb.DMatrix(x_val))
        
        joblib.dump(model,os.path.join(config.model_path,f'lgb_{direction}_{fold}.pkl'))

        joblib.dump(xgb_model,os.path.join(config.model_path,f'xgb_{direction}_{fold}.pkl'))
    
        oof_predictions[val_ind] =   (lgb_val + xgb_val ) / 2
        
        lgb_pred = model.predict(x_test) / 5
        xgb_pred = xgb_model.predict(xgb.DMatrix(x_test)) / 5
        
        test_predictions += (lgb_pred + xgb_pred ) / 2



    return oof_predictions, test_predictions



def txn_categories_training(X,Y, x_test, y_test):

    with open(os.path.join(config.model_path,'user_id.pkl'),'rb') as output:
        le_dict_user = pickle.load(output)

    
    with open(os.path.join(config.model_path,'categories.pkl'),'rb') as output:
        le_dict_category = pickle.load(output)


    X['user_id'] = X['user_id'].apply(lambda x: le_dict_user.get(x, 'NA'))

    x_test['user_id'] = x_test['user_id'].apply(lambda x: le_dict_user.get(x, 'NA'))

    Y['categories'] = Y['categories'].apply(lambda x: le_dict_category.get(x, 'NA'))

    y_test['categories'] = y_test['categories'].apply(lambda x: le_dict_category.get(x, 'NA'))
    



    oof_predictions = np.zeros(X.shape[0])
    

    group = GroupKFold(n_splits=5)

    group_values = X['month']

    scores = []
 

    for fold, (trn_ind, val_ind) in enumerate(group.split(X, Y, groups=group_values)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = X.iloc[trn_ind], X.iloc[val_ind]
        y_train, y_val = Y.iloc[trn_ind], Y.iloc[val_ind]
    
    
        lgb = LGBMClassifier(**config.params)
    
        lgb.fit(x_train, y_train,eval_set=[(x_train,y_train),(x_val,y_val)], verbose = 500, early_stopping_rounds=67, 
                categorical_feature=['user_id','month'])
    
        y_val = lgb.predict_proba(x_val)
    
        vals = [np.argmax(line) for line in y_val]
    
        oof_predictions[val_ind] = vals
    
        y_pred = lgb.predict_proba(x_test)
    
        preds = [np.argmax(line) for line in y_pred]
    
        #print(set(preds))

        joblib.dump(lgb,os.path.join(config.model_path,f'lgb_categories_{fold}.pkl'))

        score = f1_score(y_test,preds, average='micro')
        print(score)
        scores.append(score)




    return oof_predictions, scores


def run():
    
    
    preprocess()

    train_in_txns, train_out_txns = dataset.get_in_out_txns(config.path, config.training_file)

    print("########## started training for income #################")
    
    hold_out_month = train_in_txns['month'].max()

    
    X = train_in_txns.query(f"month != {hold_out_month}").drop(['In_sum'],axis=1)
    Y = train_in_txns.query(f"month != {hold_out_month}")['In_sum']

    x_test = train_in_txns.query(f"month == {hold_out_month}").drop(['In_sum'],axis=1)

    y_test = train_in_txns.query(f"month == {hold_out_month}")['In_sum']

    


    oof_predictions, test_predictions = income_expense_training(X,Y,x_test, 'In')


    print("out of fold predictions for expesnes ",rmspe(Y,oof_predictions))

    print("RMSPE for holdout set ", rmspe(y_test, test_predictions))



    print("############# training completed for income##########")
    

    
    print("########## started training for expenses#################")

    hold_out_month = train_out_txns['month'].max()

    X = train_out_txns.query(f"month != {hold_out_month}").drop(['Out_sum'],axis=1)
    Y = train_out_txns.query(f"month != {hold_out_month}")['Out_sum']

    x_test = train_out_txns.query(f"month == {hold_out_month}").drop(['Out_sum'],axis=1)

    y_test = train_out_txns.query(f"month == {hold_out_month}")['Out_sum']

    oof_predictions, test_predictions = income_expense_training(X,Y,x_test, 'Out')


    print("out of fold predictions for expesnes ",rmspe(Y,oof_predictions))

    print("RMSPE for holdout set ", rmspe(y_test, test_predictions))


    print("############# training completed for expenses###########")

    


    
    print("########## started training for categories#################")

    


    categories  = dataset.prepare_categories(config.path, config.training_file)

    hold_out_month = categories['month'].max()

    

    X = categories.query(f"month != {hold_out_month}").drop(['categories'],axis=1)

    Y = categories.query(f"month != {hold_out_month}")[['categories']]

    x_test = categories.query(f"month == {hold_out_month}").drop(['categories'],axis=1)

    y_test = categories.query(f"month == {hold_out_month}")[['categories']]


    
    
    
    oof_predictions, scores = txn_categories_training(X,Y,x_test,y_test)

    print("out of fold predictions for categories is",f1_score(Y,oof_predictions, average='micro'))

    print("F1  score for hold out dataset is ",np.mean(scores))

    print("########## completed training for categories#################")




if __name__ == "__main__":

    run()



    

    