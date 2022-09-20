import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly_express as px

import os
import config

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = 'N26 Data science task'

compare_income = pd.read_csv(os.path.join(config.private_path,'train_income.csv'))
compare_income = compare_income[['user_id','month','In_sum']]

compare_income['Predictions'] = compare_income['In_sum']

compare_income.columns = ['user_id','month','Actual','Predictions']


compare_expense = pd.read_csv(os.path.join(config.private_path,'train_outcome.csv'))
compare_expense = compare_expense[['user_id','month','Out_sum']]

compare_expense['Predictions'] = compare_expense['Out_sum']

compare_expense.columns = ['user_id','month','Actual','Predictions']

incomes = pd.read_csv(os.path.join(config.private_path,'compare_income.csv'))
expense = pd.read_csv(os.path.join(config.private_path,'compare_expense.csv'))


incomes = pd.concat([compare_income, incomes],axis=0)

incomes = incomes.groupby(['user_id','month']).agg({'Actual': np.sum, 'Predictions':np.sum}).reset_index()


expense  = pd.concat([compare_expense, expense], axis=0)

expense = expense.groupby(['user_id','month']).agg({'Actual': np.sum, 'Predictions':np.sum}).reset_index()

categories_actual = pd.read_csv(os.path.join(config.private_path,'categories_actual.csv'))
categories_preds = pd.read_csv(os.path.join(config.private_path,'categories_preds.csv'))





row = html.Div([

    html.P('Select the user id from list of 10k users from the slider'),

    dcc.Slider(
        id='my-slider',
        min=1,
        max=10000,
        step=1,
        value=6453,
        marks= {
            1 : {'label': '1'},
            5000 : {'label': '5k'},
            10000 : {'label': '10k'}

        }
    ),
    

    dbc.Row([
        dbc.Col(dcc.Graph(id='income'),width=6),
        dbc.Col(dcc.Graph(id='expense'),width=6),

    ], ),

    
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='cat_actual'),width=6),
        dbc.Col(dcc.Graph(id='cat_pred'),width=6),

    ]),

])

app.layout = row



@app.callback(
    [dash.dependencies.Output('income', 'figure'),
    dash.dependencies.Output('expense', 'figure'),
    dash.dependencies.Output('cat_actual', 'figure'),
    dash.dependencies.Output('cat_pred', 'figure'),],
    
    [dash.dependencies.Input('my-slider', 'value')])
def update_output(value):
    
    print(incomes.query(f"user_id == {value}"))

    print(expense.query(f"user_id == {value}"))



    fig1 = px.line(data_frame= incomes.query(f'user_id == {value}'), x='month', 
                         y =['Actual','Predictions'])

    fig2 = px.line(data_frame= expense.query(f'user_id == {value}'), x='month',
                         y =['Actual','Predictions'])
    
    '''
    fig1.update_traces(patch={"line": {"color": "black", "width": 4}})
    fig1.update_traces(patch={"line": {"color": "black", "width": 4, "dash": 'dot'}}, selector={"legendgroup": "7"})

    fig2.update_traces(patch={"line": {"color": "black", "width": 4}})
    fig2.update_traces(patch={"line": {"color": "black", "width": 4, "dash": 'dot'}}, selector={"legendgroup": "7"})
    ''' 


    cat_actual = px.pie(categories_actual.query(f"user_id == {value}"), names='categories', values='Total', title='Actual categories')

    cat_preds = px.pie(categories_preds.query(f"user_id == {value}"), names='Predictions', values='percentage', title='predicted categories')

    return fig1, fig2, cat_actual, cat_preds




if __name__ == '__main__':
    app.run_server(debug=True)

