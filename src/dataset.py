import pandas as pd
import numpy as np
import os
import config
import sys




def count_unique(series):
    return len(np.unique(series))








def create_direction_feature(data, direction):


    
    target_df = pd.DataFrame()

    create_feature_dict = {
        'amount_n26_currency': [np.sum, np.mean,np.min, np.max, np.median, np.var, np.std],
        'direction': [np.size],
        'txn_explanations': [count_unique]        
    }

    target_df = data.query(f"direction == '{direction}'").groupby(['user_id','month']).agg(create_feature_dict).reset_index()
    
    target_df.columns = ['user_id','month',f'{direction}_sum',f'{direction}_mean',f'{direction}_min',f'{direction}_max', f'{direction}_median',
    f'{direction}_var',f'{direction}_std','unique_txns',f'{direction}_count']
    
    return target_df




def get_in_out_txns(path, file_name):

    train = pd.read_csv(os.path.join(path,file_name))
    mcc_cats = pd.read_csv(os.path.join(path,'mcc_group_definition.csv'))
    txn_types = pd.read_csv(os.path.join(path, 'transaction_types.csv'))

    txn_types.rename(columns={'explanation':'txn_explanations'}, inplace=True)

    train = train.merge(mcc_cats, on='mcc_group',how='left')

    train = train.merge(txn_types, left_on='transaction_type', right_on='type', how='left')


    train.drop(['type'],axis=1, inplace=True)


    train['transaction_date'] = pd.to_datetime(train['transaction_date'])

    train['month'] = train['transaction_date'].apply(lambda x: x.month)

    train['day_of_week'] = train['transaction_date'].apply(lambda x: x.dayofweek)

    train['day_of_month'] = train['transaction_date'].apply(lambda x: x.day)

    train_in_txns = create_direction_feature(train, "In")

    train_out_txns = create_direction_feature(train, "Out")


    return train_in_txns, train_out_txns


def prepare_categories(path, file_name):


    train = pd.read_csv(os.path.join(path,file_name))
    mcc_cats = pd.read_csv(os.path.join(path,'mcc_group_definition.csv'))
    txn_types = pd.read_csv(os.path.join(path, 'transaction_types.csv'))

    txn_types.rename(columns={'explanation':'txn_explanations'}, inplace=True)
    
    train = train.merge(mcc_cats, on='mcc_group',how='left')
    train = train.merge(txn_types, left_on='transaction_type', right_on='type', how='left')
    train.drop(['type'],axis=1, inplace=True)




    train['transaction_date'] = pd.to_datetime(train['transaction_date'])
    train['month'] = train['transaction_date'].apply(lambda x: x.month)
    train['day_of_week'] = train['transaction_date'].apply(lambda x: x.dayofweek)
    train['day_of_month'] = train['transaction_date'].apply(lambda x: x.day)
    

    txn_category_amount = train.query("direction=='Out'").groupby(['user_id','month','explanation','day_of_week','day_of_month'])['amount_n26_currency'].\
    agg([np.sum, np.mean,np.min, np.max, np.median]).reset_index()

    txn_categories = train.query("direction=='Out'").groupby(['user_id','month','day_of_week','day_of_month'])['explanation'].\
    value_counts().reset_index(name='count')

    txn_categories = txn_categories.merge(txn_category_amount, on=['user_id','month','explanation',
    'day_of_week','day_of_month'],how='left')

    txn_categories.rename(columns={'explanation':'categories','sum':'Total','amin':'min','amax':'max'},inplace=True)

    return txn_categories



def prepare_test_in_out_txns(data, direction, month):
    target_df = pd.DataFrame()

    create_feature_dict = {
        f'{direction}_sum' : [np.mean, np.min, np.max, np.median, np.var,np.std],
        'unique_txns':[np.min],
        f'{direction}_count' : [np.min]
    }

    target_df = data.groupby(['user_id']).agg(create_feature_dict).reset_index()

    target_df['month'] = month

    target_df.columns = ['user_id',f'{direction}_mean',f'{direction}_min',f'{direction}_max', f'{direction}_median',
    f'{direction}_var',f'{direction}_std','unique_txns',f'{direction}_count','month']

    target_df = target_df[['user_id','month',f'{direction}_mean',f'{direction}_min',f'{direction}_max', f'{direction}_median',
    f'{direction}_var',f'{direction}_std','unique_txns',f'{direction}_count']]
    
    return target_df



def prepare_test_categories(data,month):
    
    target_df = pd.DataFrame()

    create_feature_dict_min = {
        'day_of_week': [np.min],
        'day_of_month': [np.min],
        'Total': [np.sum,np.mean,np.min,np.max, np.median],
        'count' : [np.min],

    }


    create_feature_dict_max = {
        'day_of_week': [np.max],
        'day_of_month': [np.max],
        'Total': [np.sum,np.mean,np.min,np.max, np.median],
        'count' : [np.max],

    }


    target_df_min = data.groupby(['user_id']).agg(create_feature_dict_min).reset_index()
    target_df_min['month'] = month

    #print(target_df_min.head())
    target_df_min.columns = ['user_id','day_of_week','day_of_month','Total','mean','min','max','median','count','month']


    target_df_max = data.groupby(['user_id']).agg(create_feature_dict_max).reset_index()
    target_df_max['month'] = month
    target_df_max.columns = ['user_id','day_of_week','day_of_month','Total','mean','min','max','median','count','month']


    target_df = pd.concat([target_df_min,target_df_max],axis=0)

    target_df = target_df[['user_id','month','day_of_week','day_of_month','Total','mean','min','max','median','count']]


    
    return target_df








if __name__ == "__main__":

    
    test_month = int(sys.argv[1])

    train_in_txns, train_out_txns = get_in_out_txns(config.path, config.training_file)


    


    test_in_txns = prepare_test_in_out_txns(train_in_txns, 'In', test_month)

    test_out_txns = prepare_test_in_out_txns(train_out_txns, 'Out', test_month)


    
    
    test_in_txns.to_csv(os.path.join(config.test_path,'x_test_in_txns.csv'), index=False)
    
    test_out_txns.to_csv(os.path.join(config.test_path,'x_test_out_txns.csv'), index=False)


    categories  = prepare_categories(config.path, config.training_file)

    


    test_categories = prepare_test_categories(categories,test_month)


    test_categories.to_csv(os.path.join(config.test_path,'x_test_categories.csv'), index=False)

    








