#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:57:23 2018

@author: herman
"""

from pyfin import Portfolio, TFSA, DI, RA
import pandas as pd
import numpy as np

#%% Portfolio

p = Portfolio(dob='1987-02-05',
              ibt=70000*12,
              expenses=19000,
              monthly_med_aid_contr=2583.18,
              ma_dependents=2,
              medical_expenses=9000,
              era=65,
              le=95,
              strategy='optimal')

tfsa = TFSA(initial=0,
            growth=15,
            ytd=0,
            ctd=0,
            dob='1987-02-05',
            era=65,
            le=95)

ra = RA(initial=50000,
        ra_growth=12,
        la_growth=10,
        ytd=2500,
        dob='1987-02-05',
        le=95,
        era=65,
        payout_fraction=0)

di = DI(initial=0,
        growth=15,
        dob='1987-02-05',
        era=65,
        le=95)

contr_TFSA = pd.Series(index=tfsa.df.index, name='contr',
                       data=269649*np.ones(tfsa.df.shape[0]))
contr_DI = pd.Series(index=tfsa.df.index, name='contr',
                     data=269649*np.ones(tfsa.df.shape[0]))
contr_RA = pd.Series(index=tfsa.df.index, name='contr',
                     data=16500*np.ones(tfsa.df.shape[0]))
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

contr_DI.loc[p.retirement_date:] = 0
contr_RA.loc[p.retirement_date:] = 0

withdrawals_DI.loc[p.df.index[0]:p.retirement_date] = 0
withdrawals_RA.loc[p.df.index[0]:p.retirement_date] = 0
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
print('Mean IAT: R', df_p.loc[p.first_retirement_date:, 'iat'].mean())
#%%
p.optimize()
df_p = p.df
print('Average monthly IAT during retirement:', round(p.df.loc[p.first_retirement_date:, 'iat'].mean()/12))

'''
for count, i in enumerate(self.investments.keys()):
    self.contr.loc[:self.last_working_date, count] = self.savable_income/self.size
    self.withdrawals.loc[self.first_retirement_date:, count] = self.taxable_ibt/self.size/3


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