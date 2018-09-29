#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:38:20 2018

@author: herman
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
from dash.dependencies import Input, Output, State
import json
from pyfin import Portfolio, TFSA, DI, RA
import plotly.graph_objs as go
from plotly import tools
import pandas as pd
import numpy as np

app = dash.Dash()
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

colours = {'text': '#236AB3'}

app.layout = html.Div(children=[
    html.H1(children='PyFin',
            style={'textAlign':'center',
                   'color': colours['text']}),

    html.H3(children='''
        Intelligent Personal Finance, by Invoke Analytics
        
    ''',
            style={'textAlign':'center',
                   'color': colours['text'],
                   'font-family':'Libre Baskerville',
                   'font-style':'italic'}),

    html.Div(children='''      
        PyFin is a tool which helps individuals decide how to split their\
        retirement savings given South African tax rates. It considers how much\
        money is already in those accounts, how future savings can be allocated,\
        and how they grow and are taxed now and at withdrawal. It then devises\
        a plan that will maximise a person's mean income after tax during retirement.\
        The output of the tool is a report with a possible allocation plan for\
        the rest of this tax year. It also provides the projections into the\
        future.
        
        NOTE:
        - This is the Alpha version, made available for feedback. So it isn't\
        pretty, and it isn't necessarily trustworthy yet.
        - If you see any problems, or there are features you think would\
        enhance PyFin for most users, please send as an email.
        - You can read the White Paper about how PyFin works here:\
        https://gitlab.com/invokeanalytics/pyfin/blob/master/PyFin%20White%20Paper/Pyfin%20White%20Paper.pdf
    ''',
            style={'font-family':'Open Sans'}),
    html.Div(id='basic_portfolio_info', className='form', children=[             
        html.H2(children="Basic Portfolio Information",
                    style={'textAlign':'left',
                           'font-family':'Maven Pro'}),
        html.Div(children=[
            html.Label('Name: ', style={'font-family':'Open Sans'}),                 
            dcc.Input(id='input_name',
                placeholder='Herman',
                type='text',
                value='Herman')],
                style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label('Date of Birth: ', style={'font-family':'Open Sans'}),
            dcc.DatePickerSingle(id='input_dob',
                date=dt(1987, 2, 5))],
                style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label('Annual Income Before Tax: ', style={'font-family':'Open Sans'}),    
            dcc.Input(id='input_ibt',
                placeholder='650000',
                type='text',
                value='650000')],
            style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label('Monthly Expenses: ', style={'font-family':'Open Sans'}),
            dcc.Input(id='input_expenses',
                placeholder='240000',
                type='text',
                value='240000'
                )],
            style={'margin': '20px 0px'}),
        html.Div(children=[
                html.Label('Expected Retirement Age: ', style={'font-family':'Open Sans'}),
                dcc.Input(id='input_era',
                    placeholder='65',
                    type='text',
                    value='65'
                    )],
                style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label('Life Expectancy: ', style={'font-family':'Open Sans'}),
            dcc.Input(id='input_le',
                placeholder='95',
                type='text',
                value='95'
                )],
            style={'margin': '20px 0px'}),
        html.H3('Medical Expenses', style={'font-family':'Open Sans'}),
        html.Div(children=[
            html.Label('Number of dependants: ', style={'font-family':'Open Sans'}),
            dcc.Input(id='input_ma_dependants',
                placeholder='2',
                type='text',
                value='2'
                )],
            style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label('Monthly Medical Aid Contributions: ', style={'font-family':'Open Sans'}),
            dcc.Input(id='input_ma_contr',
                placeholder='2200',
                type='text',
                value='2200'
                )],
            style={'margin': '20px 0px'}),
        html.Div(children=[        
            html.Label('Annual out-of-pocket medical expenses: ', style={'font-family':'Open Sans'}),
            dcc.Input(id='input_medical_expenses',
                placeholder='7000',
                type='text',
                value='7000'
                )],
            style={'margin': '20px 0px'}),
        html.H3('Retirement payout strategy', style={'font-family':'Open Sans'}),
        dcc.RadioItems(id='radio_strategy',
            options=[
                {'label': 'Optimal', 'value': 'optimal'},
                {'label': 'Safe', 'value': 'safe'},
            ],
            value='optimal')
        ]),

    #html.Div(id='my-div'),
    html.H1('Investments'),
    html.H2('Pension Funds / Retirement Annuities'),
    html.Button(id='button_expand_ra',
                n_clicks=0,
        children=['Add an RA'],
        style={'margin': '20px 0px'},
        value='0'
    ),        
    html.Div(id='RA',
             children=[
        html.H2(children="First Pension Fund / Retirement Annuity",
                    style={'textAlign':'left',
                           'font-family':'Maven Pro'}),
        html.Div(children=[
        html.Div(children=[
            html.Label('Description: ', style={'font-family':'Open Sans'}),                 
            dcc.Input(id='input_ra_name',
                placeholder='Enter the RA name here',
                type='text',
                value='RA')],
                style={'margin': '20px 0px'}),
            html.Label('Current capital amount: ', style={'font-family':'Open Sans'}),                 
            dcc.Input(id='input_ra_initial',
                placeholder='0',
                type='text',
                value='15000')],
                style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label('Year-to-date contributions: ', style={'font-family':'Open Sans'}),    
            dcc.Input(id='input_ra_ytd',
                placeholder='0',
                type='text',
                value='4000')],
            style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label('Annualized Growth: ', style={'font-family':'Open Sans'}),
            dcc.Input(id='input_ra_growth',
                placeholder='0',
                type='text',
                value='9.73'
                )],
            style={'margin': '20px 0px'}),
        html.Div(children=[
                html.Label("Expected Living Annuity Growth. (Leave as-is if you're not sure)", style={'font-family':'Open Sans'}),
                dcc.Input(id='input_la_growth',
                    placeholder='5.5',
                    type='text',
                    value='8.5'
                    )],
                style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label("Payout Fraction at Retirement. Leave as-is if you're not sure. Otherwise, e.g. 0.3 would be 30%", style={'font-family':'Open Sans'}),
            dcc.Input(id='input_payout_fraction',
                placeholder='0',
                type='text',
                value='0'),
                      
        html.Div(children=[
            html.Label('Current monthly contribution: ', style={'font-family':'Open Sans'}),                 
            dcc.Input(id='input_ra_contr',
                placeholder='Amount',
                type='text',
                value='10000')],
                style={'margin': '20px 0px'}),
                      ])
        ]),
        html.H2('Tax Free Savings Accounts'),
        html.Label('Do you have a Tax Free Savings Account?'),

        html.Button(id='button_expand_tfsa',
                    n_clicks=0,
                    style={'margin': '20px 0px'},
                   children=['Add a TFSA']), 
        html.Div(id='TFSA',
             children=[
        html.H2(children="First Tax Free Savings Account",
                    style={'textAlign':'left',
                           'font-family':'Maven Pro'}),
        html.Div(children=[
        html.Div(children=[
            html.Label('Description: ', style={'font-family':'Open Sans'}),                 
            dcc.Input(id='input_tfsa_name',
                placeholder='Enter the TFSA name here',
                type='text',
                value='TFSA')],
                style={'margin': '20px 0px'}),
                      
            html.Label('Current capital amount: ', style={'font-family':'Open Sans'}),                 
            dcc.Input(id='input_tfsa_initial',
                placeholder='0',
                type='text',
                value='33000')],
                style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label('Year-to-date contributions: ', style={'font-family':'Open Sans'}),    
            dcc.Input(id='input_tfsa_ytd',
                placeholder='0',
                type='text',
                value='33000')],
            style={'margin': '20px 0px'}),
        html.Div(children=[
                html.Label('Total contributions to date: ', style={'font-family':'Open Sans'}),
                dcc.Input(id='input_tfsa_ctd',
                    placeholder='0',
                    type='text',
                    value='33000'
                    )],
                style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label('Annualized Growth: ', style={'font-family':'Open Sans'}),
            dcc.Input(id='input_tfsa_growth',
                placeholder='9.73',
                type='text',
                value='9.73'
                )],
            style={'margin': '20px 0px'}),
                      
        html.Div(children=[
            html.Label('Current monthly contribution: ', style={'font-family':'Open Sans'}),                 
            dcc.Input(id='input_tfsa_contr',
                placeholder='Amount',
                type='text',
                value='0')],
                style={'margin': '20px 0px'}),
        ]),
        html.H2('Discretionary Investments'),
        html.Label('Do you have discretionary investments such as unit trusts or exchange traded funds?'),

        html.Button(id='button_expand_di',
                    n_clicks=0,
                    style={'margin': '20px 0px'},
                   children=['Add a DI']),  
        html.Div(id='DI',
             children=[
        html.H2(children="First Discretionary Investment",
                    style={'textAlign':'left',
                           'font-family':'Maven Pro'}),
        html.Div(children=[
            html.Label('Description: ', style={'font-family':'Open Sans'}),                 
            dcc.Input(id='input_di_name',
                placeholder='Enter the DI name here',
                type='text',
                value='DI')],
                style={'margin': '20px 0px'}),

        html.Div(children=[
            html.Label('Current capital amount: ', style={'font-family':'Open Sans'}),                 
            dcc.Input(id='input_di_initial',
                placeholder='0',
                type='text',
                value='100000')],
                style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label('Annualized growth: ', style={'font-family':'Open Sans'}),
            dcc.Input(id='input_di_growth',
                placeholder='0',
                type='text',
                value='9.73'
                )],
            style={'margin': '20px 0px'}),
        html.Div(children=[
            html.Label('Amount of current capital which is capital gains', style={'font-family':'Open Sans'}),
            dcc.Input(id='input_di_cg',
                placeholder='0',
                type='text',
                value='50000'
                )],
            style={'margin': '20px 0px'}),
            
        html.Div(children=[
            html.Label('Current monthly contribution: ', style={'font-family':'Open Sans'}),                 
            dcc.Input(id='input_di_contr',
                placeholder='Amount',
                type='text',
                value='1000')],
                style={'margin': '20px 0px'}),

        ]),
                      
    html.Div(children=[html.Button(id='calculate-button',
                n_clicks=0,
                children='Calculate!',
                style={'margin': '20px 0px',
                       'display': 'none'})]),
    html.Div(id='data-div',
             style={'display':'none'}),
    html.Div(id='iat-display',
             children=[],
             style={'margin': '20px 0px'}),
    html.Div(id='output_iat'),
    html.Div(id='output_capital'),   
    html.Div(id='output_withdr'),
    html.Div(id='output_contr'),

    ])
    
@app.callback(Output(component_id='RA', component_property='style'),
    [Input(component_id='button_expand_ra', component_property='n_clicks')])
def has_ra(has_ra):
    if int(has_ra):
        return 
    else:
        return {'display':'none'}
    
@app.callback(Output(component_id='TFSA', component_property='style'),
    [Input(component_id='button_expand_tfsa', component_property='n_clicks')])
def has_tfsa(has_tfsa):
    if int(has_tfsa):
        return 
    else:
        return {'display':'none'}
    
@app.callback(Output(component_id='DI', component_property='style'),
    [Input(component_id='button_expand_di', component_property='n_clicks')])
def has_di(has_di):
    if int(has_di):
        return 
    else:
        return {'display':'none'}

@app.callback(Output('calculate-button', 'style'),
              [Input('button_expand_ra', 'n_clicks'),
               Input('button_expand_tfsa', 'n_clicks'),
               Input('button_expand_di', 'n_clicks')])
def displayCalculateButton(ra, tfsa, di):
    if any([int(ra), int(tfsa), int(di)]):
        return
    else:
        return {'margin': '20px 0px',
                       'display': 'none'}

@app.callback(Output('data-div', 'children'),
              [Input('calculate-button', 'n_clicks')],
              [State(component_id='input_name', component_property='value'),
             State(component_id='input_dob', component_property='date'),
             State(component_id='input_ibt', component_property='value'),
             State(component_id='input_expenses', component_property='value'),
             State(component_id='input_era', component_property='value'),
             State(component_id='input_le', component_property='value'),
             State(component_id='input_ma_dependants', component_property='value'),
             State(component_id='input_ma_contr', component_property='value'),
             State(component_id='input_medical_expenses', component_property='value'),
             State(component_id='radio_strategy', component_property='value'),            
             
             State(component_id='input_ra_name', component_property='value'),
             State(component_id='input_ra_initial', component_property='value'),
             State(component_id='input_ra_ytd', component_property='value'),
             State(component_id='input_ra_growth', component_property='value'),
             State(component_id='input_la_growth', component_property='value'),
             State(component_id='input_payout_fraction', component_property='value'),
             State(component_id='input_ra_contr', component_property='value'),             
             
             State(component_id='input_tfsa_name', component_property='value'),
             State(component_id='input_tfsa_initial', component_property='value'),
             State(component_id='input_tfsa_ytd', component_property='value'),
             State(component_id='input_tfsa_ctd', component_property='value'),
             State(component_id='input_tfsa_growth', component_property='value'),
             State(component_id='input_tfsa_contr', component_property='value'),
             State(component_id='input_di_name', component_property='value'),
             State(component_id='input_di_initial', component_property='value'),
             State(component_id='input_di_growth', component_property='value'),
             State(component_id='input_di_cg', component_property='value'),
             State(component_id='input_di_contr', component_property='value')])
def calculate(  n_clicks,
                input_name,
                input_dob,
                input_ibt,
                input_expenses,
                input_era,
                input_le,
                input_ma_dependants,
                input_ma_contr,
                input_ma_expenses,
                radio_strategy,
                
                input_ra_name,
                input_ra_initial,
                input_ra_ytd,
                input_ra_growth,
                input_la_growth,
                input_payout_fraction,
                input_ra_contr,
                
                input_tfsa_name,
                input_tfsa_initial,
                input_tfsa_ytd,
                input_tfsa_ctd,
                input_tfsa_growth,
                input_tfsa_contr,
                
                input_di_name,
                input_di_initial,
                input_di_growth,
                input_di_cg,
                input_di_contr):

    if n_clicks > 0:
        p = Portfolio(dob=input_dob,
                      ibt=int(input_ibt),
                      expenses=int(input_expenses)*12,
                      monthly_med_aid_contr=int(input_ma_contr),
                      ma_dependents=int(input_ma_dependants),
                      medical_expenses=int(input_ma_expenses),
                      era=int(input_era),
                      le=int(input_le),
                      strategy=radio_strategy)   
        
        ra = RA(initial=float(input_ra_initial),
                        ra_growth=float(input_ra_growth),
                        la_growth=float(input_la_growth),
                        ytd=float(input_ra_ytd),
                        dob=input_dob,
                        le=int(input_le),
                        era=int(input_era),
                        payout_fraction=float(input_payout_fraction))
        di = DI(initial=float(input_di_initial),
                        growth=float(input_di_growth),
                        dob=input_dob,
                        le=int(input_le),
                        era=int(input_era))
        tfsa = TFSA(initial=float(input_tfsa_initial),
                        growth=float(input_tfsa_growth),
                        ytd=float(input_tfsa_ytd),
                        ctd=float(input_tfsa_ctd),
                        dob=input_dob,
                        le=int(input_le),
                        era=int(input_era))
        
        p.addInvestment(input_ra_name, ra)
        p.addInvestment(input_tfsa_name, tfsa)
        p.addInvestment(input_di_name, di)
        
        contr_TFSA = pd.Series(index=tfsa.df.index, name='contr',
                       data=np.ones(p.df.shape[0])*float(input_ra_contr))
        contr_DI = pd.Series(index=tfsa.df.index, name='contr',
                             data=np.ones(p.df.shape[0])*float(input_di_contr))
        contr_RA = pd.Series(index=tfsa.df.index, name='contr',
                             data=np.ones(p.df.shape[0])*float(input_ra_contr))
        
        contr_TFSA.loc[p.retirement_date:] = 0
        contr_DI.loc[p.retirement_date:] = 0
        contr_RA.loc[p.retirement_date:] = 0
        
        ra.calculateOptimalWithdrawal(contr_RA)
        tfsa.calculateOptimalWithdrawal(contr_TFSA)
        di.calculateOptimalWithdrawal(contr_DI)
        
        p.calculate()     
        return json.dumps(p.df.to_json())
        
@app.callback(Output('output_withdr', 'children'),
              [Input('calculate-button', 'n_clicks'),
               Input('data-div', 'children')])
def plot_withdr(n_clicks, data_json):   
        df = pd.read_json(json.loads(data_json))
        return dcc.Graph(id='withdrawals-graph',
                        figure={
                            'data': [
                                {'x': df.index, 'y': df.withdrawals_RA, 'type': 'line', 'name': 'RA'},
                                {'x': df.index, 'y': df.withdrawals_TFSA, 'type': 'line', 'name': u'TFSA'},
                                {'x': df.index, 'y': df.withdrawals_DI, 'type': 'line', 'name': u'DI'},
                            ],
                            'layout': {
                                'title': 'Withdrawals'
                            }
                        }
                    )

@app.callback(Output('output_contr', 'children'),
              [Input('calculate-button', 'n_clicks'),
              Input('data-div', 'children')])
def plot_contr(n_clicks, data_json):
    
    df = pd.read_json(json.loads(data_json))    
    return dcc.Graph(id='contributions-graph',
                    figure={
                        'data': [
                            {'x': df.index, 'y': df.contr_RA, 'type': 'line', 'name': 'RA'},
                            {'x': df.index, 'y': df.contr_TFSA, 'type': 'line', 'name': u'TFSA'},
                            {'x': df.index, 'y': df.contr_DI, 'type': 'line', 'name': u'DI'},
                        ],
                        'layout': {
                            'title': 'Contributions'
                        }
                    }
                )  
@app.callback(Output('output_capital', 'children'),
              [Input('calculate-button', 'n_clicks'),
              Input('data-div', 'children')])    
def plot_capital(n_clicks, data_json):
    
    df = pd.read_json(json.loads(data_json))    
    return dcc.Graph(id='capital-graph',
                    figure={
                        'data': [
                            {'x': df.index, 'y': df.capital_RA, 'type': 'line', 'name': 'RA'},
                            {'x': df.index, 'y': df.capital_TFSA, 'type': 'line', 'name': u'TFSA'},
                            {'x': df.index, 'y': df.capital_DI, 'type': 'line', 'name': u'DI'},
                        ],
                        'layout': {
                            'title': 'Capital'
                        }
                    }
                )  
@app.callback(Output('output_iat', 'children'),
              [Input('calculate-button', 'n_clicks'),
              Input('data-div', 'children')])    
def plot_iat(n_clicks, data_json):
    
    df = pd.read_json(json.loads(data_json))
    return dcc.Graph(id='iat-graph',
                    figure={
                        'data': [
                            {'x': df.index, 'y': df['iat'], 'type': 'bar', 'name': 'IAT'},
                        ],
                        'layout': {
                            'title': 'Income After Tax'
                        }
                    }
                )  

@app.callback(Output('iat-display', 'children'),
              [Input('calculate-button', 'n_clicks'),
              Input('data-div', 'children')])    
def display_iat(n_clicks, data_json):
    df = pd.read_json(json.loads(data_json))
    ret_date = df.loc[df.withdrawals_total >0].index[0]
    mean_ret_iat = df.loc[ret_date:, 'iat'].mean()
    mean_working_iat = df.loc[:ret_date, 'iat'].mean()
    if n_clicks > 0:
        if mean_ret_iat > mean_working_iat:
            return html.H1("""Your average annual income after tax during retirement, in today's money, is:R{:,}.\n
                           
                           This is R{:,} more per year than your current income after tax.""".format(round(mean_ret_iat, 2), round(mean_ret_iat - mean_working_iat, 2)))
        else:
            return html.H1("""Your average annual income after tax during retirement, in today's money, is:\nR{:,}.\n
                           
                           This is R{:,} less per year than your current income after tax.""".format(round(mean_ret_iat, 2), round(mean_working_iat - mean_ret_iat, 2)))

if __name__ == '__main__':
    app.run_server(debug=True)
  
'''
ras = {}
dis = {}
tfsas = {}

if len(data_dict.investments.keys()) > 0:
    for i in data_dict.keys():
        if 'ra' in i:
            ras[i] = RA(initial=data_dict['investments'][i]['initial'],
                    ra_growth=data_dict['investments'][i]['ra_growth'],
                    la_growth=data_dict['investments'][i]['la_growth'],
                    ytd=data_dict['investments'][i]['ytd'],
                    dob=data_dict['investments']['portfolio']['dob'],
                    le=data_dict['investments']['portfolio']['le'],
                    era=data_dict['investments']['portfolio']['era'],
                    payout_fraction=data_dict['investments'][i]['payout_fraction'])
        elif 'di' in i:
            dis[i] = DI(initial=data_dict['investments'][i]['initial'],
                    growth=data_dict['investments'][i]['growth'],
                    dob=data_dict['investments']['portfolio']['dob'],
                    le=data_dict['investments']['portfolio']['le'],
                    era=data_dict['investments']['portfolio']['era'])
        elif 'tfsa' in i:
            tfsas[i] = TFSA(initial=data_dict['investments'][i]['initial'],
                    growth=data_dict['investments'][i]['growth'],
                    ytd=data_dict['investments'][i]['ytd'],
                    ctd=data_dict['investments'][i]['ctd'],
                    dob=data_dict['investments']['portfolio']['dob'],
                    le=data_dict['investments']['portfolio']['le'],
                    era=data_dict['investments']['portfolio']['era'])

if len(ras.keys()):
    for name, obj in ras.items():
        p.addInvestment(name, obj)
if len(tfsas.keys()):
    for name, obj in tfsas.items():
        p.addInvestment(name, obj)
if len(dis.keys()):
    for name, obj in dis.items():
        p.addInvestment(name, obj)
'''

'''
@app.callback(Output(component_id='data-div', component_property='children'),
              [Input(component_id='calculate-button', component_property='n_clicks')],
    [State(component_id='input_ra_name', component_property='value'),
     State(component_id='input_ra_initial', component_property='value'),
     State(component_id='input_ra_ytd', component_property='value'),
     State(component_id='input_ra_growth', component_property='value'),
     State(component_id='input_la_growth', component_property='value'),
     State(component_id='input_payout_fraction', component_property='value'),
     State(component_id='input_ra_contr', component_property='value'),
     State('data-div', 'children')])
def instantiatePortfolio(name,
                 initial,
                 ytd,
                 ra_growth,
                 la_growth,
                 payout_fraction,
                 ra_contr,
                 data_list):
    data_dict['portfolio'] = {'p_name': name,
                    'dob': dob,
                    'expenses': expenses,
                    'era': era,
                     'le': le,
                     'ma_contr': ma_contr,
                     'ma_dependants': ma_dependants,
                     'medical_expenses': medical_expenses,
                     'strategy': strategy}
    
    return data_list + [json.dumps(data_dict)]


@app.callback(Output(component_id='data-div', component_property='children'),
              [Input(component_id='calculate-button', component_property='n_clicks')],
    [State(component_id='input_ra_name', component_property='value'),
     State(component_id='input_ra_initial', component_property='value'),
     State(component_id='input_ra_ytd', component_property='value'),
     State(component_id='input_ra_growth', component_property='value'),
     State(component_id='input_la_growth', component_property='value'),
     State(component_id='input_payout_fraction', component_property='value'),
     State(component_id='input_ra_contr', component_property='value'),
     State('data-div', 'children')])
def store_RA(name,
                 initial,
                 ytd,
                 ra_growth,
                 la_growth,
                 payout_fraction,
                 ra_contr,
                 data_list):

        data_dict['investments']['ra'] = {'ra_name': name,
                'ra_initial': initial,
                 'ra_ytd': ytd,
                 'ra_ra_growth': ra_growth,
                 'ra_la_growth': la_growth,
                 'ra_payout_fraction': payout_fraction,
                 'ra_contr': ra_contr}
        return data_list + [json.dumps(data_dict)]
   
@app.callback(Output(component_id='data-div', component_property='children'),
              [Input(component_id='calculate-button', component_property='n_clicks')],
    [State(component_id='input_tfsa_name', component_property='value'),
     State(component_id='input_tfsa_initial', component_property='value'),
     State(component_id='input_tfsa_ytd', component_property='value'),
     State(component_id='input_tfsa_ctd', component_property='value'),
     State(component_id='input_tfsa_growth', component_property='value'),
     State(component_id='input_tfsa_contr', component_property='value'),
     State(component_id='data-div', component_property='children')
     ]
)

def store_TFSA(name,
                     initial,
                     ytd,
                     ctd,
                     growth,
                     contr,
                     data_list):
    data_dict={}
    data_dict['investments']['tfsa'] = {'name': name,
            'initial': initial,
            'ytd': ytd,
            'ctd': ctd,
            'growth': growth,
            'contr': contr}
    return data_list + [json.dumps(data_dict)]
    
@app.callback(Output(component_id='data-div', component_property='children'),
              [Input(component_id='calculate-button', component_property='n_clicks')],
    [State(component_id='input_di_name', component_property='value'),
     State(component_id='input_di_initial', component_property='value'),
     State(component_id='input_di_growth', component_property='value'),
     State(component_id='input_di_cg', component_property='value'),
     State(component_id='input_di_contr', component_property='value'),
     State(component_id='data-div', component_property='children'),
     ])
def store_DI(click,
                 name,
                 initial,
                 growth,
                 cg,
                 contr,
                 data_list):
            
        data_dict = {}
        data_dict['investments']['di']={'name': name,
                 'initial': initial,
                 'growth': growth,
                 'cg': cg,
                 'contr': contr}
        return data_list + [json.dumps(data_dict)]
'''     