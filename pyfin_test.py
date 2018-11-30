#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:57:23 2018

@author: herman
"""

from pyfin import Portfolio, Person, TFSA, DI, RA
import pandas as pd
import numpy as np

#%% Portfolio
herman = Person(dob='1987-02-05',
              ibt=50000,
              expenses=19000,
              monthly_med_aid_contr=2583.18,
              ma_dependants=2,
              medical_expenses=9000,
              era=65,
              le=95,
              strategy='optimal')

p = Portfolio(herman)
df_p = p.df


tfsa = TFSA(herman, 
            initial=0,
            growth=15,
            ctd=0,
            ytd=33000)

ra = RA(herman,
        initial=50000,
        ra_growth=15,
        la_growth=15,
        ytd=2500,
        payout_fraction=0.3)

di = DI(herman,
        initial=0,
        growth=15,
        cg_to_date=0)

contr_TFSA = pd.Series(index=tfsa.df.index, name='contr',
                       data=8674/12*np.ones(tfsa.df.shape[0]))
contr_DI = pd.Series(index=tfsa.df.index, name='contr',
                     data=9000*np.ones(di.df.shape[0]))
contr_RA = pd.Series(index=tfsa.df.index, name='contr',
                     data=13750*12*np.ones(ra.df.shape[0]))
withdrawals_TFSA = pd.Series(index=tfsa.df.index,
                        name='withdrawals',
                        data=0*np.ones(tfsa.df.shape[0]))
withdrawals_DI = pd.Series(index=tfsa.df.index,
                        name='withdrawals',
                        data=50000*np.ones(tfsa.df.shape[0]))
withdrawals_RA = pd.Series(index=tfsa.df.index,
                        name='withdrawals',
                        data=7000*np.ones(tfsa.df.shape[0]))

contr_TFSA.iloc[16:] = 0
contr_DI.loc[p.first_fy_after_retirement:] = 0
contr_RA.loc[p.first_fy_after_retirement:] = 0

withdrawals_DI.loc[p.df.index[0]:p.retirement_fy_end] = 0
withdrawals_RA.loc[p.df.index[0]:p.last_working_year] = 0
withdrawals_TFSA.loc[p.df.index[0]:] = 0


ra.calculateOptimalWithdrawal(contr_RA)
tfsa.calculateOptimalWithdrawal(contr_TFSA)
di.calculateOptimalWithdrawal(contr_DI)

p.addInvestment('RA', ra)
p.addInvestment('DI', di)
p.addInvestment('TFSA', tfsa)
p.calculate()

df_di = di.df
df_ra = ra.df
df_tfsa = tfsa.df
df_p = p.df
#p.plot()
print('Mean IAT, current contributions: R', round(df_p.loc[p.retirement_fy_end:, 'iat'].mean()/12, 2))


#%%
p.optimize(reduced_expenses=True)
df_p = p.df
print('Average monthly IAT during retirement:', round(p.df.loc[p.retirement_fy_end:, 'iat'].mean()/12))
#%% Hyperparameter optimization

from skopt import gp_minimize
params = [[0.0, 2.0], #  cognitive parameter (weight of personal best)
           [0.0, 2.0], #  social parameter (weight of swarm best)
           [0.0, 1.0], #  initial velocity
           [0.0, 2.0], #  inertia
           [0, 10], 
           [1, 2],#  Distance function (Minkowski p-norm). 1 for abs, 2 for Euclidean
           [-1.0, 0.0],
           [0.0, 1.0]]  

res = gp_minimize(p.optimizeParams,
                  params,
                  acq_func="EI",      # the acquisition function
                  n_calls=100,         # the number of evaluations of f 
                  n_random_starts=4, # the number of random initialization points
                  verbose=True,
                  n_jobs=1)  

#%%
'''
for count, i in enumerate(self.investments.keys()):
    self.contr.loc[:self.last_working_date, count] = self.savable_income/self.size
    self.withdrawals.loc[self.retirement_fy_end:, count] = self.taxable_ibt/self.size/3


scenario = np.concatenate((self.contr.values, self.withdrawals.values), axis=1)
bounds = ()
#  Create bounds. Contributions only during working months, withdrawals only during retirement

for i in range(self.size):
    for j in range(self.number_working_years):
        bounds += ((0, self.savable_income),)
    for j in range(self.number_retirement_years):
        bounds += ((0, 0),)
for i in range(self.size):
    for j in range(self.number_working_years):
        bounds += ((0, 0),)
    for j in range(self.number_retirement_years):
        bounds += ((0, None),)
        
self.bounds = bounds
self.sc1 = scenario.reshape(scenario.size, order='F')
#cons = ({'type': 'eq', 'fun': lambda x: self.constraint(x, i)},)

self.res = spm.minimize(self.objective,
            method='TNC',
           x0=scenario.reshape(scenario.size, order='F'),
           bounds=bounds)


solution_1d = self.res.x
'''