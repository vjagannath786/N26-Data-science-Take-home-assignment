

path = '../input/train'
training_file = '2016-09-19_79351_training.csv'

test_path = '../input/test'

private_path = '../input/private'

seed = 2021

model_path = '../models'



params_lgb = {
        'num_iterations': 1000,
        'learning_rate': 0.05,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 64,        
        'random_state':seed,
        'bagging_fraction': 0.5,
        'feature_fraction': 0.5,
        'seed':seed
    }


params_xgb = {'max_depth': 3, 
            'n_estimators': 500, 
            'objective': 'reg:squarederror', 
            'subsample': 0.8, 
            'colsample_bytree': 0.85, 
            'learning_rate': 0.05, 
            'seed': seed}


##### parameters for spending categories
params = {
    'num_iterations': 1000,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',    
    'metric': 'multi_logloss',
    'learning_rate': 0.01,
    'max_depth': 10,
    'num_leaves': 100,
    'min_data_in_leaf': 50,
    'num_class':17,
    'lambda_l2':0.3,
    'min_split_gain' : 0.5,
    'max_bin':100,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'bagging_freq': 70,
    'random_seed':seed,
    'is_unbalance':True,
    'verbose':-1
    }  
