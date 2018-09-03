#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:53:46 2018

@author: herman
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#from deap import base, creator, tools, algorithms
import time

#import numba
#import multiprocessing

class Portfolio(object):
    
    def __init__(self,
                 dob,
                 ibt,
                 expenses,
                 ma_dependents,
                 medical_expenses,
                 monthly_med_aid_contr,
                 era=65,
                 le=95,
                 strategy='optimal',
                 optimizer='PSO',
                 inflation=5.5,
                 uif=True):
        '''
        Portfolio object, combining all investments.
        ------
        Parameters:
        dob:                str. Date of Birth, in format "YYYY-MM-DD"
        ibt:                int. Annual income before tax
        expenses:           float. Expenses before tax, monthly
        ma_dependents:      int. Number of medical aid dependants, including self.
        medical_expenses:   float. Annual out-of-pocket medical expenses
        era:                int. Expected Retirement Age.
        le:                 int. life expectancy.
        uif:                bool. Whether Unemployment Insurance Fund contributions are applicable.
        '''
        
        self.dob = pd.to_datetime(dob).date()
        #assert(isinstance(ibt, int))
        self.taxable_ibt = ibt
        #assert(isinstance(expenses, float) or isinstance(expenses, int))
        self.expenses = expenses
        #assert(isinstance(ma_dependents, int))
        self.ma_dependents = ma_dependents
        #assert(isinstance(era, int))
        self.era = era
        #assert(isinstance(le, int))
        self.le = le
        #assert(isinstance(medical_expenses, float) or isinstance(medical_expenses, int))
        self.medical_expenses = medical_expenses
        self.ra_payouts = 0
        self.size = 0
        self.age = pd.datetime.today().date() - self.dob
        self.retirement_date = pd.datetime(self.dob.year + era, self.dob.month, self.dob.day)
        self.optimizer = optimizer
        if pd.datetime.today()>self.retirement_date:
            raise AttributeError('This calculator only works pre-retirement. You have specified a retirement date in the past.')
        
        self.strategy = strategy
        self.inflation = inflation/100
        self.monthly_med_aid_contr = monthly_med_aid_contr
        
        self.uif_contr = 0
        if uif == True:
            self.uif_contr = min(148.72*12, 0.01*ibt)
        
        self.investments = {}
        self.ra_list = []
        self.tfsa_list = []
        self.di_list = []
        self.investment_names = []
        self.max_ra_growth = 0
        self.max_tfsa_growth = 0
        self.max_di_growth = 0
        self.max_ra_name = ''
        self.max_tfsa_name = ''
        self.max_di_name = ''        
        
        self.df = pd.DataFrame(index=pd.DatetimeIndex(start=pd.datetime.today().date(),
                                                      end=pd.datetime(self.dob.year + le, self.dob.month, self.dob.day),
                                                      freq='A-FEB'),
                                columns=['taxable_ibt',
                                        'capital_gains',
                                        'age',
                                        'date',
                                        'iat',
                                        'withdrawals_total',
                                        'withdrawals_RA',
                                        'withdrawals_TFSA',
                                        'withdrawals_DI',
                                        'contr_RA',
                                        'contr_DI',
                                        'contr_TFSA',
                                        'contr_total',
                                        'contr_total_at',
                                        'savable_iat'])

        self.df.loc[:, ['taxable_ibt',
                        'capital_gains',
                        'age',
                        'date',
                        'withdrawals_total',
                        'withdrawals_RA',
                        'withdrawals_TFSA',
                        'withdrawals_DI',
                        'contr_RA',
                        'contr_DI',
                        'contr_TFSA',
                        'contr_total',
                        'contr_total_at',
                        'savable_iat',
                        'iat']] = 0
        
        self.df.loc[:, 'medical_expenses'] = medical_expenses

        self.df.loc[:, 'medical_expenses'] = medical_expenses
        self.df.loc[:, 'date'] = self.df.index
        
        self.this_year = self.df.index[0]
        self.last_working_date = self.df.loc[self.df.index<self.retirement_date].index[-1]
        self.df.loc[:self.last_working_date, 'iat'] = -self.uif_contr

        self.first_retirement_date = self.df.loc[self.df.index>=self.retirement_date].index[0]
        self.number_working_years = self.df.loc[self.df.index<self.last_working_date].shape[0]
        self.number_retirement_years = self.df.loc[self.df.index>=self.last_working_date].shape[0]
        self.df.loc[:self.last_working_date, 'taxable_ibt'] = self.taxable_ibt
        self.df.loc[:, 'age'] = (self.df.index - pd.Timestamp(self.dob)).days/365.25
        
    def addInvestment(self, name, investment):  
        
        '''
        Adds an investment object to the portfolio. Saved to a dictionary under
        the key 'name'.
        ------
        Parameters:
        name:           str or list. Name of investment.
        investments:    monpy Investment. Investment object. Can be an RA,
                        TFSA, or DI.
        '''        
     
        if isinstance(name, str):
            self.investments[name] = investment
            self.investment_names += [name]
            self.size +=1
            if investment.type == 'RA':
                self.ra_list += [name]
                if investment.ra_growth > self.max_ra_growth:  # working only with ra growth here, should probably include la growth at some stage.
                    self.max_ra_growth = investment.ra_growth
                    self.max_ra_name = name
            elif investment.type == 'TFSA':
                self.tfsa_list += [name]
                if investment.growth > self.max_tfsa_growth:
                    self.max_tfsa_growth = investment.growth
                    self.max_tfsa_name = name
            elif investment.type == 'DI':
                self.di_list += [name]
                if investment.growth > self.max_di_growth:
                    self.max_di_growth = investment.growth
                    self.max_di_name = name
            else:
                print('Type for {} not recognised'.format(name)) 
                
        elif isinstance(name, list):
            for name, count in enumerate(name):
                self.size += 1
                self.investments[name] = investment[count]
                self.investment_names += [name]
                if investment[count].type == 'RA':
                    self.ra_list += [name]
                elif investment[count].type == 'TFSA':
                    self.tfsa_list += [name]
                elif investment[count].type == 'DI':
                    self.di_list += [name]
                else:
                    print('Type for {} not recognised'.format(name))
                    
    def optimize(self):
        
        '''
        Optimises the investment allocations of the portfolio over time.
        '''
        
        time1 = time.time()
        #import scipy.optimize as spm
        
        print('Mean post-retirement IAT, tax efficient plan, TFSA priority:', self.calculateTaxEfficientTFSAFirstIAT())
        print('Mean post-retirement IAT tax efficient plan, RA priority:', self.calculateTaxEfficientRAFirstIAT())

          
        self.contr = pd.DataFrame(index=self.df.index,
                                     columns=np.arange(0, self.size))

        if self.optimizer == 'GA':
            self.pop_size = 100
            self.ngen = 20
            self.GA()
            self.solution = self.fractionsToRands(self.reshape(self.best_ind))

        if self.optimizer == 'PSO':
            
            self.pso()
            self.solution = self.fractionsToRands(self.reshape(self.best_ind))
            self.ra_payouts = self.best_ind[0]
        for count, key in enumerate(self.investments.keys()):
            self.contr.loc[:, key] = self.solution[:, count]
            #self.contr.rename(columns={count:i}, inplace=True)
            self.investments[key].calculateOptimalWithdrawal(self.contr[key], self.strategy)
            
        self.calculate()
        print('Duration:', (time.time() - time1)/60, 'min')
        self.plot()
               
    #@numba.jit
    def reshape(self, ind):
        arr_ind = np.array(ind[1:])
        '''
        Fortran-like indexing. That means that [1, 2, 3, 4, 5, 6] becomes
        [[1, 4],
        [2, 5],
        [3, 6]]
        instead of:
        [[1, 2]
        [3, 4]'
        [5, 6]]
        '''
        return arr_ind.reshape(int(arr_ind.size/(self.size)), self.size, order='F')
    
    def NPV(self, amount, date):
        n = date.year - self.this_year.year + 1
        return amount/(1 + self.inflation)**n
    
    def FV(self, amount, date):
        n = date.year - self.this_year.year + 1
        return amount*(1 + self.inflation)**n    
            
    #@numba.jit
    def fractionsToRands(self, ind):
        
        '''
        converts fractions saved into Rands saved. It does so for all years at once.
        ------
        Parameters:     
        ind:            ndarray. Numpy array of size [self.number_working_years + number_retirement_years, self.size]
        ------
        Returns:        ndarray. Same shape as input. Just with Rand values.
        '''
        contr = np.zeros_like(ind) #  exclude RA payout fraction (first item in arr)
        ra_contr = self.taxable_ibt*ind[:, :len(self.ra_list)].sum(axis=1)
        tax = np.zeros(len(contr))
        for i, year in enumerate(self.df.index[:]):
            taxSeries = pd.Series({'taxable_ibt': self.taxable_ibt,
                            'contr_RA': ra_contr[i],
                            'capital_gains': 0,
                            'medical_expenses': self.medical_expenses})
            taxSeries.name = year
            tax[i] = self.totalTax(taxSeries)
        
        savable_income = np.maximum(0, self.taxable_ibt - ra_contr - tax - self.uif_contr - self.expenses)
        savable_income[self.number_working_years:] = np.maximum(0, savable_income[self.number_working_years] - self.uif_contr)
        mask = np.ones_like(savable_income)
        mask[savable_income <= 0] = 0
        contr[:, :len(self.ra_list)] = mask[:, None]*self.taxable_ibt*np.array(ind[:, :len(self.ra_list)])
        contr[:, len(self.ra_list):] = savable_income[:, None]*np.array(ind[:, len(self.ra_list):])
        return contr

    def pso_objectiveSwarm(self, swarm):
        '''
        Objective Function for optimization. 
        ------
        Parameters:
        scenario_1d:    ndarray. 1D Numpy array. This is a reshaped form of an
                        array of dimension [working months + retirement months, 2*portfolio size]
                        where the values are fractions. The first len(self.ra_list)
                        columns are retirement annuities. These fractions are 
                        fractions of the total income before tax allocated to the
                        retirement annuity. The rest of the columns are from
                        income after tax, specifically the savable income;
                        IAT - expenses.
        ------
        Returns:        float. The mean income after tax during the retirement 
                        period.                        
        '''
        results = 1e5*np.ones(swarm.position.shape[0])
        swarm = self.rebalancePSO(swarm)
        pos = swarm.position
        
        for i in range(pos.shape[0]):
            self.pso_objective(pos[i, :])

        
        return results
    
    def pso_objective(self, pos):
            fracs = self.reshape(pos)
            ra_payout_frac = pos[0]
            scenario = self.fractionsToRands(fracs)
            self.contr.loc[:, self.contr.columns] = scenario
            for count, j in enumerate(self.investments.keys()):
                self.investments[j].calculateOptimalWithdrawal(self.contr.loc[:, count],
                                                                self.strategy,
                                                               ra_payout_frac)
            self.calculate()
            self.df.loc[self.df['iat']==0, 'iat'] = -self.taxable_ibt*100
            result = 0
            if self.strategy == 'optimal':
                result = round(-self.df.loc[self.retirement_date:, 'iat'].mean(), 2) #+ penalty_oversaved, 2)
            elif self.strategy == 'safe':
                result = round(-self.df.loc[self.retirement_date:, 'iat'].mean(), 2) #+ penalty_oversaved, 2)
            return result
        

    def calculate(self):
        
        '''
        Calculates the income, taxable income, tax, etc. for the whole portfolio
        '''
        #  Zero all relevant columns so that the amounts don't build up over 
        #  different function calls
        withdrawals_colnames = ['withdrawals_' + i for i in self.investments.keys()]
        contr_colnames = ['contr_' + i for i in self.investments.keys()]
        zeroed_cols = ['taxable_ibt',
        'capital_gains',
        'contr_total',
        'contr_total_at',
        'savable_iat',
        'withdrawals_total',
        'iat'] + withdrawals_colnames + contr_colnames
        self.df.loc[:, zeroed_cols] = 0
                        
        self.df.loc[:self.last_working_date, 'iat'] = -self.uif_contr
        self.df.loc[:self.last_working_date, 'taxable_ibt'] = self.taxable_ibt
        
        for i in self.ra_list:
            self.df['taxable_ibt'] += self.investments[i].df['withdrawals']
            self.df['withdrawals_total'] += self.investments[i].df['withdrawals']
            self.df['withdrawals_RA'] += self.investments[i].df['withdrawals']
            self.df['contr_RA'] += self.investments[i].df['contr']
            self.df['capital_RA'] = self.investments[i].df['capital']
            self.df['contr_total'] += self.investments[i].df['contr']

            #self.ra_payouts += self.investments[i].payout
            self.df.loc[self.first_retirement_date, 'taxable_ibt'] += self.CGTRA(self.investments[i].payout)
        
        for count, i in enumerate(self.di_list):
            if count == 0:  #  Allocate RA lump sums to first DI. Can build more
                            #  intelligent functionality later.
                self.investments[i].df.loc[self.first_retirement_date, 'contr'] += self.ra_payouts - self.CGTRA(self.ra_payouts)
                self.investments[i].recalculateOptimalWithdrawal()
            self.df['capital_gains'] += self.investments[i].df['withdrawal_cg']
            self.df['iat'] += self.investments[i].df['withdrawals']
            self.df['contr_DI'] += self.investments[i].df['contr']
            self.df['withdrawals_total'] += self.investments[i].df['withdrawals']
            self.df['withdrawals_DI'] += self.investments[i].df['withdrawals']
            self.df['capital_DI'] = self.investments[i].df['capital']
            self.df['contr_total'] += self.investments[i].df['contr']
            self.df['contr_total_at'] += self.investments[i].df['contr']

        for i in self.tfsa_list:
            self.df['iat'] += self.investments[i].df['withdrawals']
            self.df['contr_TFSA'] += self.investments[i].df['contr']
            self.df['withdrawals_total'] += self.investments[i].df['withdrawals']
            self.df['withdrawals_TFSA'] += self.investments[i].df['withdrawals']
            self.df['capital_TFSA'] = self.investments[i].df['capital']
            self.df['contr_total'] += self.investments[i].df['contr']
            self.df['contr_total_at'] += self.investments[i].df['contr']

        self.df['it'] = self.df.apply(self.totalTax, axis=1)
        self.df['iat'] = self.df['iat'] + self.df['taxable_ibt'] - self.df['contr_RA'] - self.df['it']
        self.df['savable_iat'] = self.df['iat'] - self.expenses
        
    def totalTax(self, s):
        
        '''
        Calculates total income tax.
        ------
        Parameters:
        s:          Pandas Series. Containing columns capital_gains, 
                    contr_RA, taxable_ibt
        ------
        Returns:
        tax:                float. tax payable in a particular year
        '''
        
        age = (s.name - pd.Timestamp(self.dob)).days/365.25
        taxable_income = 0
        if s.name < self.retirement_date:
            if s.contr_RA <= 0.275*s.taxable_ibt and s.contr_RA <= 350000:
                taxable_income = s.taxable_ibt + self.taxableCapitalGains(s.capital_gains, s.name) - s.contr_RA
            elif s.contr_RA > 0.275*s.taxable_ibt and s.contr_RA < 350000:
                taxable_income = s.taxable_ibt - s.taxable_ibt*0.275 + self.taxableCapitalGains(s.capital_gains, s.name)
            else:
                taxable_income = s.taxable_ibt - 350000 + self.taxableCapitalGains(s.capital_gains, s.name)

        if s.name >= self.retirement_date:
            if age < 65:
                taxable_income = max(0, s.taxable_ibt + self.taxableCapitalGains(s.capital_gains, s.name))
            elif age < 75:
                taxable_income = max(0, s.taxable_ibt + self.taxableCapitalGains(s.capital_gains, s.name) - 121000)
            else:
                taxable_income = max(0, s.taxable_ibt + self.taxableCapitalGains(s.capital_gains, s.name) - 135300)                
        
        tax = self.incomeTax(taxable_income, age)
        
        self.tax_credit_ma = self.taxCreditMa(self.monthly_med_aid_contr,
                                              self.ma_dependents,
                                              s.medical_expenses,
                                              taxable_income,
                                              age)    
        return max(0, tax - self.tax_credit_ma)
    
    def taxableCapitalGains(self, amount, year):
        return self.NPV(0.4*max(0, amount - self.FV(40000, year)), year)
    
    #@numba.jit
    def incomeTax(self, taxable_income, age=64):
        
        '''
        Calculates tax according to income tax brackets.
        ------
        Parameters:
        taxable income      float. Income before tax, but possibly after 
                            RA contributions have been deducted.
        ------
        Returns:
        income tax as float.
        '''        
        if age < 65:
            rebate = 14067
        elif age < 75:
            rebate = 14067 + 7713
        else:
            rebate = 14067 + 7713 + 2574
            
        
        if taxable_income <= 78150:
            return 0
        if taxable_income <= 195850:
             return  0.18*(taxable_income) - rebate
        elif taxable_income <= 305850:
             return  35253 + ((taxable_income) - 195850)*0.26 - rebate
        elif taxable_income <= 423300:
             return  63853 + (taxable_income - 305850)*0.31 - rebate
        elif taxable_income <= 555600:
             return  100263 + (taxable_income - 423300)*0.36 - rebate
        elif taxable_income <= 708310:
             return  147891 + (taxable_income - 555600)*0.39 - rebate
        elif taxable_income <= 1500000:
             return  207448 + (taxable_income - 708310)*0.41 - rebate
        elif taxable_income >= 1500000:
             return  532041 + (taxable_income - 1500000)*0.45 - rebate
    
    #@numba.jit
    def taxCreditMa(self, 
                    monthly_med_aid_contr, 
                    ma_dependents,
                    medical_expenses,
                    taxable_income,
                    age):
        
        if age > 65:
            if ma_dependents <=2:
                ma_d_total = ma_dependents*310*12
            else:
                ma_d_total = 620*12 + 12*(ma_dependents - 2)*209
            
            tax_credit_ma = ma_d_total\
                                + 0.33*max(0, monthly_med_aid_contr*12 - 3*ma_d_total)\
                                + 0.33*max(0, medical_expenses)
        else:
            if ma_dependents <=2:
                ma_d_total = ma_dependents*310
            else:
                ma_d_total = 12*620 + 12*(ma_dependents - 2)*209
            
            tax_credit_ma = ma_d_total \
                                + 0.25*max(0, medical_expenses - 0.075*taxable_income)\
                                + 0.25*max(0, monthly_med_aid_contr*12 - ma_d_total*4)
        return tax_credit_ma
    
    #@numba.jit        
    def CGTRA(self, lump_sum):

        lump_sum_FV = self.FV(lump_sum, self.retirement_date)
        if lump_sum_FV < self.FV(500000, self.retirement_date):
            return 0
        elif lump_sum_FV < self.FV(700000, self.retirement_date):
            return self.NPV((lump_sum_FV - self.FV(500000, self.retirement_date))*0.18, self.retirement_date)
        elif lump_sum_FV < self.FV(1050000, self.retirement_date):
            return self.NPV(self.FV(36000, self.retirement_date) + (lump_sum_FV - self.FV(700000, self.retirement_date))*0.27, self.retirement_date)
        elif lump_sum_FV >= self.FV(1050000, self.retirement_date):
            return self.NPV(self.FV(130500, self.retirement_date) + (lump_sum_FV - self.FV(1050000, self.retirement_date))*0.36, self.retirement_date)
        
    def plot(self):
        plt.figure(1)
        index = [x.strftime('%Y-%M-%d') for x in self.df.index.date]
        plt.plot(index, self.df['withdrawals_TFSA'], label='TFSA')
        plt.plot(index, self.df['withdrawals_RA'], label='RA')
        plt.plot(index, self.df['withdrawals_DI'], label='DI')
        plt.title('Withdrawals')
        plt.xlabel('Dates')
        plt.ylabel('Amount [R]')
        plt.xticks(rotation=90)
        plt.legend()
        
        plt.figure(2)
        plt.plot(index, self.df['contr_TFSA'], label='TFSA')
        plt.plot(index, self.df['contr_RA'], label='RA')
        plt.plot(index, self.df['contr_DI'], label='DI')
        plt.xlabel('Dates')
        plt.ylabel('Amount [R]')
        plt.xticks(rotation=90)
        plt.legend()        
        plt.title('Contributions')
        
        plt.figure(3)
        plt.plot(index, self.df['capital_TFSA'], label='TFSA')
        plt.plot(index, self.df['capital_RA'], label='RA')
        plt.plot(index, self.df['capital_DI'], label='DI')
        plt.xlabel('Dates')
        plt.ylabel('Amount [R]')
        plt.xticks(rotation=90)
        plt.legend()        
        plt.title('Capital')
        
        plt.figure(4)
        plt.ylim(0, self.df['iat'].max()*1.05)
        plt.bar(index, self.df['iat'])
        plt.title('Income After Tax')
        plt.xticks(rotation=90)
        #axes = plt.gca()
        #axes.set_ylim([0, self.df['iat'].max()*1.05])
            
    #@numba.jit
    def randomIndividual(self, factor):
        
        contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
        ra_payout = np.random.random()*0.3
        for i in range(self.number_working_years):
            #  Generate RA allocations to sum to anything up to 27.5%:
            ra = np.random.dirichlet(factor*np.ones(len(self.ra_list)))*np.random.beta(1, 3.9)
            #  Generate other allocations to sum to one:
            others = np.random.dirichlet(factor*np.ones(self.size - len(self.ra_list)))
            contr[i, :] = np.concatenate([ra, others])
        #return self.convertPercentagesToRands(contr)
        #return np.array([self.convertPercentagesToRands(i) for i in contr])
        reshaped = contr.reshape(contr.size, order='F')
        
        return np.insert(reshaped, 0, ra_payout)

    def randomConstantIndividual(self, factor):
        
        '''
        Initialize Random combination, apply to all working years
        ------
        Parameters:
        factor:     float. Dirichlet distribution factor. For low numbers (<1)
                    the Dirichlet distribution will allocate almost exclusively
                    to one column, making the rest close to zero. For high
                    numbers (>10), allocations will be about equal.
        '''
        ra_payout = np.random.random()*0.3
        contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
        #ra = np.random.dirichlet(factor*np.ones(len(self.ra_list)))*np.random.triangular(0, 0.275, 0.5)
        ra = np.random.dirichlet(factor*np.ones(len(self.ra_list)))*np.random.beta(1, 3.9)
        #  Generate other allocations to sum to one:
        if sum(ra) > 0.275:
            ra = 0.275*ra/sum(ra)
        others = np.random.dirichlet(factor*np.ones(self.size - len(self.ra_list)))
        plan = np.concatenate([ra, others])
        for i in range(self.number_working_years):
            contr[i] = plan
        #contr = np.array([plan if i < self.number_working_years else np.zeros(self.size) for i in range(len(contr))])
        return np.insert(contr.reshape(contr.size, order='F'), 0, ra_payout)
    
    def calculateTaxEfficientRAFirstIAT(self):
        self.contr = pd.DataFrame(index=self.df.index,
                                  columns=self.investment_names)

        solution = self.fractionsToRands(self.reshape(self.taxEfficientIndividualRAFirst()))
        self.ra_payouts = self.taxEfficientIndividualRAFirst()[0]
        for count, key in enumerate(self.investments.keys()):
            self.contr.loc[:, key] = solution[:, count]
            self.investments[key].calculateOptimalWithdrawal(self.contr[key], self.strategy)
        self.calculate()
        return round(self.df.loc[self.first_retirement_date:, 'iat'].mean()/12)

    def calculateTaxEfficientTFSAFirstIAT(self):
        
        self.contr = pd.DataFrame(index=self.df.index,
                                  columns=np.arange(0, self.size))

        
        solution = self.fractionsToRands(self.reshape(self.taxEfficientIndividualTFSAFirst()))
        self.ra_payouts = self.taxEfficientIndividualTFSAFirst()[0]
        for count, key in enumerate(self.investments.keys()):
            self.contr.loc[:, key] = solution[:, count]
            self.investments[key].calculateOptimalWithdrawal(self.contr[key], self.strategy)
        self.calculate()
        return round(self.df.loc[self.first_retirement_date:, 'iat'].mean()/12)
        
    
    def taxEfficientIndividualRAFirst(self):
        '''
        Allocate up to 27.5% (depending on savable income) to RAs, then R33 000 
        to TFSAs, and the rest to DIs
        '''
        #  Create blank individual
        contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
        for i in range(self.number_working_years):
            #  Allocate RA saving:
            if len(self.ra_list) > 0:
                ra_frac = 0
                savable_income = 1
                #  Find RA allocation, up to 27.5%:
                while ra_frac < 0.275 and savable_income > 0:      
                        ra_frac += 0.001
                        ra_contr = self.taxable_ibt*ra_frac
                        tax = self.incomeTax(self.taxable_ibt - ra_contr, age=self.df.loc[self.df.index[0], 'age'])
                        savable_income = np.maximum(0, self.taxable_ibt - ra_contr - tax - self.uif_contr - self.expenses)
    
                #  If there is only one RA, allocate to it.
                if len(self.ra_list) == 1:
                    contr[i, 0] = ra_frac
                elif len(self.ra_list) > 1: #  Else allocate to max growth RA
                    contr[i, self.investment_names.index[self.max_ra_name]] = ra_frac
                    
            # Calculate TFSA
            if savable_income >= 33000:
                tfsa_frac = 33000/savable_income #  TFSA as % of savable income
            else:
                tfsa_frac = 1       # 100% of savable income   
            contr[i, self.investment_names.index(self.max_tfsa_name)] = tfsa_frac
    
            #  Calculate and allocate DI
            if tfsa_frac < 1:
                contr[i, self.investment_names.index(self.max_di_name)] = 1 - tfsa_frac
            
        #contr_full = np.array([contr[0, :] if i < self.number_working_years else np.zeros(self.size) for i in range(len(contr))])
        
        ra_payout = np.random.random()*0.3
        return np.insert(contr.reshape(contr.size, order='F'), 0, ra_payout)

    def taxEfficientIndividualTFSAFirst(self):
        '''
        Allocate up to R33 000 to TFSAs, then up to 27.5% to RA, and the rest
        to DIs
        '''
        #  Create blank individual
        contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
        #  Loop through every working year, calculating contributions
        for i in range(self.number_working_years):
            #  Allocate RA saving, given 33000 to TFSA:
            if len(self.ra_list) > 0:
                savable_income = 1
                tfsa_contr = 0
                while tfsa_contr < 33000 and savable_income > 0:
                    tfsa_contr += 100
                    tax = self.incomeTax(self.taxable_ibt, age=self.df.loc[self.df.index[0], 'age'])
                    savable_income = np.maximum(0, self.taxable_ibt - tax - self.uif_contr - 33000 - self.expenses)
                
                if savable_income < 0:
                    tfsa_contr -= 100
                savable_income = np.maximum(0, self.taxable_ibt - tax - self.uif_contr - tfsa_contr - self.expenses)

                tfsa_frac = tfsa_contr/savable_income
                contr[i, self.investment_names.index(self.max_tfsa_name)] = tfsa_frac

                #  Find RA allocation, up to 27.5%:
                ra_frac = 0
                while ra_frac < 0.275 and savable_income > 0:      
                        ra_frac += 0.001
                        ra_contr = self.taxable_ibt*ra_frac
                        tax = self.incomeTax(self.taxable_ibt - ra_contr, age=self.df.loc[self.df.index[0], 'age'])
                        savable_income = np.maximum(0, self.taxable_ibt - ra_contr - tax - self.uif_contr - self.expenses - tfsa_contr)
    
                #  If there is only one RA, allocate to it.
                if len(self.ra_list) == 1:
                    contr[i, 0] = ra_frac
                elif len(self.ra_list) > 1: #  Else allocate to max growth RA
                    contr[i, self.investment_names.index[self.max_ra_name]] = ra_frac
                    
    
            #  Calculate and allocate DI
            if tfsa_frac < 1:
                contr[i, self.investment_names.index(self.max_di_name)] = 1 - tfsa_frac
                    
        ra_payout = np.random.random()*0.3
        return np.insert(contr.reshape(contr.size, order='F'), 0, ra_payout)

    def rebalancePSO(self, swarm):
        pos = swarm.position
        for i in range(pos.shape[0]):
            ind_1d = pos[i, :]
            ra_payout = pos[i, 0]
            #  Find all indices where allocations are out of bounds:
            idx_negative = np.where(ind_1d < 0)[0]
            ind_1d[idx_negative] = 0
            idx_g1 = np.where(ind_1d > 1)[0]
            ind_1d[idx_g1] = 1
            if ra_payout > 0.3:
                ra_payout = 0.3
                swarm.velocity[i, 0] = 0
            elif ra_payout < 0:
                ra_payout = 0
                swarm.velocity[i, 0] = 0
            
            #  if the velocities of the investments are in the wrong
            #  directions, make them zero (they "hit the wall" in that dim):
            swarm.velocity[i, idx_negative] = np.maximum(0, swarm.velocity[i, idx_negative])
            swarm.velocity[i, idx_g1] = np.minimum(0, swarm.velocity[i, idx_g1])
            #  Reshape to intelligible form for specific calcs:
            ind = self.reshape(ind_1d)
            ras = ind[:, :len(self.ra_list)]
            #  If ras are so high (still < 1) that savable income is negative,
            #  reduce ra contributions by 1%.
            ras_sum = ras.sum(axis=1)
            for j in range(len(ras[:, 0])):
                savable_income = -1
                
            #  If savable_income < 0, adjust ras downwards. At the moment it
            #  adjusts all positive RAs downwards by 1%. 
                while savable_income < 0 and ras[j, :].any() > 0:                   
                    tax = self.incomeTax(self.taxable_ibt - ras_sum[j]*self.taxable_ibt)
                    if i <= self.number_working_years:
                        savable_income = self.taxable_ibt - tax - self.uif_contr - self.expenses*12 - ras_sum[j]*self.taxable_ibt
                    else:
                        savable_income = self.taxable_ibt - tax - self.expenses*12 - ras_sum[j]*self.taxable_ibt
                    ras[j, :] = np.maximum(0, ras[j, :] - 0.01) #  subtract 1% if ras[i, ?] > 0
                    
            others = ind[:, len(self.ra_list):]
            others = others/others.sum(axis=1)[:, None] # normalize
            others[others==np.inf] = 0
            others[others!=others] = 0 #  NaNs
            ind[:, :len(self.ra_list)] = ras
            ind[:, len(self.ra_list):] = others
            
            #  Assign back to pos
            pos[i, :] = np.concatenate([np.array([ra_payout]), ind.reshape(ind.size, order='F')])
        swarm.position = pos
        return swarm
 
    def checkBounds(self):
        
        '''
        Checks if asset allocations are still within bounds, and corrects
        if need be.
        '''
        def decorator(func):

            def wrapper(*args, **kwargs):
                offspring = func(*args, **kwargs)
                for child in offspring:
                    ind, ra_payout = self.rebalance(child)
                    child = creator.Individual(np.insert(ind.reshape(ind.size), 0, ra_payout)),
                #print('ending checkbounds')
                return offspring
            return wrapper
        return decorator
    
    def pso(self):
        
        '''
        Uses Particle Swarm Optimization to find optimal investment strategy.
        '''      
        
        import pyswarms as ps
        
        #  Create bounds. Contributions only during working months, withdrawals only during retirement
        min_bounds = np.zeros(1 + self.size*(self.number_working_years + self.number_retirement_years))
        max_bounds = np.zeros(1 + self.size*(self.number_working_years + self.number_retirement_years))
        tax = self.incomeTax(self.taxable_ibt, age=self.df.loc[self.df.index[0], 'age'])
        savable_income = self.taxable_ibt - tax - self.uif_contr - self.expenses    
        #  Find all columns in the dataframe containing 'capital'. This will be
        #  used for determining max withdrawal bounds.
        capital_cols = [i for i in self.df.columns.tolist() if 'capital' in i]
        capital_cols.remove('capital_gains')
        max_withdrawal = self.df.loc[self.first_retirement_date, capital_cols].max()
        #  bounds on lump sum withdrawal:
        min_bounds[0] = 0
        max_bounds[0] = 0.3
        #  bounds on contributions:
        index = 1 #  index for persistence over multiple loops
        for i in range(self.size): 
            #  up to savable income during working years
            for j in range(self.number_working_years):
                min_bounds[index] = 0
                max_bounds[index] = 1#savable_income
                index += 1
            #  No contributions during retirement
            for j in range(self.number_retirement_years):
                min_bounds[index] = 0
                max_bounds[index] = 1e-5
                index += 1           
                
        #  No bounds on withdrawals because we do not guess withdrawals. They are
        #  calculated.
        bounds = (min_bounds, max_bounds)
        self.bounds = bounds        
        
        n_particles = 10
        dimensions = min_bounds.size
        factor_list = np.geomspace(1/20, 100, 30)
        iterations = 10
        tolerance = 1e-2 #  Stopping criterion: improvement per iteration
        print_interval = 1   
        w_max = 0.8
        w_min = 0.05
        options = {'c1': 2, #  cognitive parameter (weight of personal best)
                   'c2': 2, #  social parameter (weight of swarm best)
                   'v': 0, #  initial velocity
                   'w': 0.1, #  inertia
                   'k': 2, #  Number of neighbours. Ring topology seems popular
                   'p': 2}  #  Distance function (Minkowski p-norm). 1 for abs, 2 for Euclidean
        topology = ps.backend.topology.Star()
                
        lst_init_pos = [None]*n_particles
        lst_init_pos[0] = self.randomConstantIndividual(np.random.choice(factor_list)).T
        lst_init_pos[1] = self.taxEfficientIndividualRAFirst().T
        lst_init_pos[2] = self.taxEfficientIndividualTFSAFirst().T
        for i in range(3, n_particles - 2, 2):
            lst_init_pos[i-1] = self.randomConstantIndividual(np.random.choice(factor_list)).T
            lst_init_pos[i] = self.randomIndividual(np.random.choice(factor_list)).T
        lst_init_pos[-1] = self.taxEfficientIndividualRAFirst().T
        lst_init_pos[-2] = self.taxEfficientIndividualTFSAFirst().T
         
        init_pos = np.zeros([len(lst_init_pos), len(lst_init_pos[0])])
        for i in range(n_particles):
            init_pos[i,:] = lst_init_pos[i]        
        
        self.init_pos = init_pos
        
        for i, ind in enumerate(lst_init_pos):
            if (ind <= max_bounds).all():
                pass
            else:
                print('Individual {} exceeds max bounds'.format(i))
                print((ind <= max_bounds))
            if (ind>= min_bounds).all():
                pass
            else:
                print('Individual {} smaller than min bounds'.format(i))
                print((ind >= min_bounds))
              
        self.myswarm = ps.backend.generators.create_swarm(n_particles,
                     init_pos=init_pos,
                     options=options,
                     dimensions=dimensions,
                     #bounds=bounds,
                     #clamp=(-0.2, 0.2)
                     )
        
        improvement = 100
        previous_cost = -1
        counter = 0
        for i in range(iterations):
        #while improvement > tolerance:
            counter += 1
            max_iter = max(50, counter)
            #self.myswarm.options['w'] = w_max - (w_max - w_min)*np.exp(-(counter/(max_iter/10)))
            #self.myswarm.options['w'] = min(w_min + counter*(w_max - w_min)/100, w_max)
            #self.myswarm.options['k'] = min(n_particles, np.ceil(counter/2))
            #  Update personal bests
            # Compute cost for current position and personal best
            self.myswarm.current_cost = self.pso_objectiveSwarm(self.myswarm)
            pbest_cost = np.zeros(n_particles)
            for i in range(n_particles):
                pbest_cost[i] = self.pso_objective(self.myswarm.pbest_pos[i,:])
            self.myswarm.pbest_cost = pbest_cost
            self.myswarm.pbest_pos, self.myswarm.pbest_cost = ps.backend.operators.compute_pbest(
                self.myswarm
            )
            # Update gbest from neighborhood
            self.myswarm.best_pos, self.myswarm.best_cost = topology.compute_gbest(self.myswarm,
#                options['p'], options['k']
            )
        
            if i%print_interval==0:
                print('Iteration: {} | best cost: {:.0f} | Improvement: {:2f}'.format(counter, self.myswarm.best_cost, improvement))
            self.myswarm.velocity = topology.compute_velocity(self.myswarm)
            self.myswarm.position = topology.compute_position(self.myswarm)
            self.myswarm = self.rebalancePSO(self.myswarm)
            
            improvement = self.myswarm.best_cost/previous_cost - 1
            previous_cost = self.myswarm.best_cost
        print('The best cost: {:.4f}'.format(self.myswarm.best_cost))
        print('The best position found by our swarm is: {}'.format(self.myswarm.best_pos))           
        
        self.best_ind, cost = self.myswarm.best_pos, self.myswarm.best_cost

    def determineRAContr(self,
                         ibt,
                         RA_monthly_contr=0,
                         age=64,
                         uif=True):
        
        '''
        Convenience function calculating how much your RA contr can be to the 
        nearest R100, given your income before tax and your expenses.
        This calculation assumes that everything is allocated to the RA, and 
        does not do the optimization calculation.
        ------
        Parameters:
        ibt:        Annual Income Before Tax 
        '''
        
        if RA_monthly_contr == 0:
            RA_annual_contr = 0
            iat = 0
            surplus = 1
            while surplus > 0:
                RA_annual_contr += 100
                ibt_ara = ibt - RA_annual_contr            
                iat = ibt_ara - self.incomeTax(ibt_ara, age) - self.uif_contr
                surplus = iat - self.expenses
            
            RA_annual_contr -= 100
            ibt_ara = ibt - RA_annual_contr            
            iat = ibt_ara - self.incomeTax(ibt_ara, age) - self.uif_contr
            surplus = iat - self.expenses
            RA_monthly_contr = RA_annual_contr/12
            iat_monthly = iat/12
            print('RA Debit order: \t\t\t\tR', round(RA_monthly_contr, 2))
            print('Monthly IAT: \t\t\t\t\tR', round(iat_monthly))
            print('Max tax free RA contr (27.5% of IBT) = \t\tR', round(0.275*ibt/12, 2))
            print('Total earned per month, incl. RA: \t\tR', round(RA_annual_contr/12 + iat_monthly, 2))
            print('Total monthly tax = \t\t\t\tR', round(self.incomeTax(ibt_ara, age)/12, 2))
            print('Total annual RA contr', RA_annual_contr)
        else:
            RA_annual_contr = RA_monthly_contr*12
            ibt_ara = ibt - RA_annual_contr            
            iat = ibt_ara - self.incomeTax(ibt_ara, age) - self.uif_contr
            surplus = iat - self.expenses           
            iat_monthly = iat/12
            print('RA Debit order: \t\t\t\tR', round(RA_monthly_contr, 2))
            print('Annual taxable income: \t\t\t\tR', round(ibt_ara))
            print('Monthly IAT: \t\t\t\t\tR', round(iat_monthly))
            print('Max tax free RA contr (27.5% of IBT) = \t\tR', round(0.275*ibt/12, 2))
            print('Total earned per month, incl. RA: \t\tR', round(RA_annual_contr/12 + iat_monthly, 2))
            print('Total monthly tax = \t\t\t\tR', round(self.incomeTax(ibt_ara, age)/12, 2))
            print('Total annual RA contr\t\t\t\tR', round(RA_annual_contr, 2))

'''
class Investment(object):

    
    def __init__(self,
                 initial,
                 growth):
        
        Investment object
        ------
        Parameters:
        initial:    float. Investment value at present time.
        growth:     float. Annual growth rate of investment in percentage
                    points, i.e. 5 is 5%.
        
        self.initial = initial
        self.growth = growth/100
        self.monthly_growth = 10**(np.log10(1 + self.growth)/12) - 1
        
'''   
class TFSA(object):
    
    def __init__(self,
                 initial,
                 dob,
                 ytd,
                 ctd,
                 era=65,
                 le=95,
                 growth=18,
                 inflation=5.5):
        
        '''
        Tax-Free Savings Account object.
        ------
        Parameters:
        ytd:    float. Year-to-date contribution, according to the tax year.
        ctd:    float. Total contr to date. 
        growth: float. Annualized growth rate of investment. E.g. if 10 if 10%.
                If not specified, the average annualized growth rate of the JSE
                over a rolling window of similar length to the investment
                horizon is used.
        '''
        self.initial = initial
        self.growth = growth/100
        self.monthly_growth = 10**(np.log10(1 + self.growth)/12) - 1
        
        self.type = 'TFSA'        
        #Investment.__init__(self, initial, growth)
        self.ctd = ctd
        self.ytd = ytd

        self.dob = pd.to_datetime(dob).date() 

        self.df = pd.DataFrame(index=pd.DatetimeIndex(start=pd.datetime.today().date(),
                                                      end=pd.datetime(self.dob.year + le, self.dob.month, self.dob.day),
                                                      freq='A-FEB'),
                               columns=['capital',
                                        'YTD contr',
                                        'Total contr',
                                        'withdrawals',
                                        'contr'])
            
        self.df.loc[:, ['capital',
                        'YTD contr',
                        'Total contr',
                        'withdrawals',
                        'contr']] = 0
    
        self.df.loc[self.df.index[0], 'capital'] = self.initial
        self.df.loc[self.df.index[0], 'YTD contr'] = self.ytd
        self.df.loc[self.df.index[0], 'Total contr'] = self.ctd  
        
        self.overall_growth = growth/100
        
        self.inflation = inflation/100
        #  In real terms:

        self.retirement_date = pd.datetime(self.dob.year + era, self.dob.month, self.dob.day)
        self.last_working_date = self.df.loc[self.df.index<=self.retirement_date].index[-1]
        self.first_retirement_date = self.df.loc[self.df.index>=self.retirement_date].index[0]
        
        self.retirement_date = pd.datetime(self.dob.year + era, self.dob.month, self.dob.day)
        self.last_working_date = self.df.loc[self.df.index<=self.retirement_date].index[-1]
        self.first_retirement_date = self.df.loc[self.df.index>=self.retirement_date].index[0]
        
        if growth == 0:
            jse = pd.read_csv('JSE_returns.csv', index_col=0)
            size = min(2017 - 1974, self.retirement_date.year - pd.datetime.today().year)
            lst = []
            for i in range(jse.shape[0] - size):
                investment = 1
                for j in jse['return'].iloc[i:i+size]:
                    investment *= 1 + j
                growth = 10**((1/size)*np.log10(investment)) - 1
                lst += [growth]
            self.growth = (1 + np.mean(lst))/(1 + self.inflation) - 1
        else:
            self.growth = (1 + self.overall_growth)/(1 + self.inflation) - 1
           
    def calculateOptimalWithdrawal(self,
                                   contr,
                                   strategy='optimal',
                                   ra_payout_frac=0): #  dummy, for Portfolio calling RA.
        
        '''
        Calculates growth of tax free savings
        ------
        Parameters:
        contr:  DataFrame. Dataframe, indexed by year from today to 
                        retirement age, with contr.
        withdrawals:    DataFrame. Dataframe, indexed by year from today to 
                        retirement age, with withdrawls.
        '''
        self.df.loc[:, ['capital',
                'YTD contr',
                'Total contr',
                'withdrawals',
                'contr']] = 0
                                
        self.df.loc[:, 'contr'] = contr                            
        self.calculate()            
        # Determine capped contributions before optimising withdrawals
        c = self.df.loc[self.first_retirement_date, 'capital']
        drawdown = 0.04
        if strategy == 'optimal':
            capital_at_le = 1e7
            arr = self.df.loc[self.last_working_date:, ['capital']].values
            while capital_at_le > 0:
                drawdown += 0.001
                capital_at_le = self._calculateQuick(arr, self.growth, c*drawdown)
                #print(capital_at_le)
            drawdown -= 0.001
        
        self.df.loc[self.first_retirement_date:, 'withdrawals'] = drawdown*c
        
        self.df.loc[:, ['capital',
        'YTD contr',
        'Total contr',
        'withdrawals',
        'contr']] = 0
        self.df.loc[self.first_retirement_date:, 'withdrawals'] = drawdown*c
        self.df.loc[:, 'contr'] = contr  
        self.calculate()
        
    #@numba.jit
    def _calculateQuick(self, arr, growth, withdrawal):
        
        '''Quick version of calculate, simply for determining optimal drawdown rate.
        Does not calculate all variables of interest like calculate() does.
        
        Parameters:
        arr:            numpy array. First column is capital, second column is 
                        contributions.
        growth:         float. Annual growth rate of portfolio
        withdrawal:     float. Annual (fixed) withdrawal amount.
        '''
        for i in range(1, len(arr)): 
            arr[i] = self.calculateCapitalAnnualized(arr[i - 1],
                                                        0,
                                                        withdrawal,
                                                        growth)
        return arr[-1]  
        
    def calculateCapitalAnnualized(self,
                                   capital,
                                   contributions,
                                   withdrawals,
                                   growth,
                                   installments=12):
        
        '''
        Calculates actual annual capital growth considering monthly
        contributions and withdrawals.
        -------
        Parameters:
        capital:        float. Capital at beginning of year
        contributions:  float. Total annual contributions. Assumed equal
                        monthly.
        withdrawals:    float. Total annual withdrawals, Assumed equal monthly.
        growth:         float. Annual growth expressed as fraction. 
                        E.g. 5% = 0.05.
        installments:   Number of period in over which calculation should be
                        done. Usually 12, except for first year.        
        '''
        capital_calc = capital
        monthly_growth = 10**(np.log10(1 + growth)/12) - 1

        #print(contributions)
        contr = contributions/installments
        withdr = withdrawals/installments
        for i in range(0, installments):
            capital_calc = capital_calc*(1 + monthly_growth) + contr - withdr
        return max(0, capital_calc)

    def calculateLastYearWithdrawals(self, 
                                     capital,
                                     contributions,
                                     withdrawals,
                                     growth,
                                     installments=12):
        
        '''
        Calculates last year's withdrawals when capital runs out
        -------
        Parameters:
        capital:        float. Capital at beginning of year
        contributions:  float. Total annual contributions. Assumed equal
                        monthly.
        withdrawals:    float. Total annual withdrawals, Assumed equal monthly.
        growth:         float. Annual growth expressed as fraction. 
                        E.g. 5% = 0.05.
        installments:   Number of period in over which calculation should be
                        done. Usually 12, except for first year.        
        '''
        monthly_growth = 10**(np.log10(1 + growth)/12) - 1

        withdr_total = 0
        contr = contributions/installments
        withdr = withdrawals/installments
        capital = capital*(1 + monthly_growth) + contr - withdr
        withdr_total += withdr
        while capital > 0:
            capital = capital*(1 + monthly_growth) + contr - withdr
            withdr_total += withdr
        withdr_total += capital
        return withdr        

        
    def calculate(self):
        self.df.loc[:, 'capital'] = 0
        self.df.loc[self.df.index[0], 'capital'] = self.initial
        self.df.loc[self.df.index[0], 'YTD contr'] = self.ytd + self.df.loc[self.df.index[0], 'contr']
        self.df.loc[self.df.index[0], 'Total contr'] = self.ctd + self.df.loc[self.df.index[0], 'contr']

        previous_year = self.df.index[0] 
        contr = self.contrAfterTax(self.df.loc[previous_year, 'contr'],
                                   self.df.loc[previous_year, 'YTD contr'],
                                   self.df.loc[previous_year, 'Total contr'])
        
        this_month = pd.datetime.today().month
        if this_month > 2:
            self.df.loc[previous_year, 'capital'] = self.calculateCapitalAnnualized(self.df.loc[previous_year, 'capital'],
                                                                            contr,
                                                                            self.df.loc[previous_year, 'withdrawals'], 
                                                                            self.growth,
                                                                            installments=13 - this_month)
        else:
            self.df.loc[previous_year, 'capital'] = self.calculateCapitalAnnualized(self.df.loc[previous_year, 'capital'],
                                                                            contr,
                                                                            self.df.loc[previous_year, 'withdrawals'], 
                                                                            self.growth,
                                                                            installments=3 - this_month)
            
            
        self.df.loc[self.df.index[0], 'capital'] += contr
        
        for year in self.df.index[1:]:
            self.df.loc[year, 'Total contr'] = self.df.loc[previous_year, 'Total contr'] + self.df.loc[year, 'contr']
            contr = self.contrAfterTax(self.df.loc[year, 'contr'],
                                       self.df.loc[year, 'YTD contr'],
                                       self.df.loc[year, 'Total contr'])
            self.df.loc[year, 'capital'] = self.calculateCapitalAnnualized(self.df.loc[previous_year, 'capital'],
                                                                            contr,
                                                                            self.df.loc[year, 'withdrawals'], 
                                                                            self.growth)
            
            #  Check if there is enough money in the account
            if self.df.loc[year, 'capital'] == 0:
                self.df.loc[year, 'withdrawals'] = self.calculateLastYearWithdrawals(self.df.loc[previous_year, 'capital'],
                                                                            contr,
                                                                            self.df.loc[year, 'withdrawals'], 
                                                                            self.growth)
            '''
            if self.df.loc[year, 'withdrawals'] <= self.df.loc[year, 'capital']:
                self.df.loc[year, 'capital'] -= self.df.loc[year, 'withdrawals']
            else:
                self.df.loc[year, 'withdrawals'] = self.df.loc[year, 'capital']
                self.df.loc[year, 'capital'] = 0     
            '''
            previous_year = year    
    
    def contrAfterTax(self, contr, ytd_contr, total_contr):
        
        '''
        Returns the contribution after tax, depending on whether the year to
        date and total contribution limits have been exceeded.
        ------
        Parameters:
        contr:          float. Contribution before tax.
        ytd_contr:      float. Year-to-date contributions before current contr.
        total_contr:    float. Total contributions 
        ------
        Returns:
        contributions after tax.
        '''
        contr_func = contr
        ytd_excess = 33000 - (ytd_contr + contr_func)
        total_excess = 500000 - (total_contr + contr_func)
        if ytd_excess >= 0 and total_excess >= 0:
            return contr_func
        if total_contr >= 500000:
            return 0.6*contr_func
        amount_exceeded_total = 0
        if total_excess < 0 or ytd_excess < 0:
            if total_excess > 0:
                total_excess = 0
            if ytd_excess > 0:
                ytd_excess = 0
            if total_excess < ytd_excess:
                return abs(amount_exceeded_total)*0.6 + contr_func + total_excess
            else:
                return contr_func + ytd_excess + 0.6*abs(ytd_excess)
          
            
class RA(object):

    
    #  TODO: Add fees
    
    def __init__(self,
                 initial,
                 dob,
                 era,
                 le,
                 ytd,
                 ra_growth=9.73,
                 la_growth=0,
                 payout_fraction=1/3,
                 inflation=5.5,
                 cg_to_date=0):
        
        '''
        Retirement Annuity object. Assumes that the RA is converted to a living
        annuity upon retirement.
        ------
        Parameters:
        initial:            float. Value of RA at present time.
        ra_growth:          float. Growth rate of RA in percentage points. i.e. 13 = 13%.
                            The default value is set at 13.73% - 4% fees, 
                            which is the average of the annualised balanced 
                            fund growths since inception for Allan Gray, Foord, 
                            Old Mutual, Sanlam, Discovery, Absa, and 
                            Coronation. The 4% fees is set to such a high
                            number because if a person does not know the growth
                            rate of the RA, they are probably paying high
                            fees as well.
        la_growth:          float. Growth rate of LA (after payout) in percentage points. i.e. 13 = 13%.
                            assigned a value of inflation + 1% if left unspecified.
        dob:                Pandas Datetime object. Date of birth.
        retirement_date:    Pandas Datetime object. Date of retirement, when RA is 
                            converted to living annuity.
        le:                 int. Life expectancy.
        ytd:                float. Contribution to RA, tax year to date.
        '''       
        
        #Investment.__init__(self, initial, ra_growth)
        self.initial = initial
        self.type = 'RA'
        self.dob = pd.to_datetime(dob).date()
        self.ra_growth_overall = ra_growth/100
        if la_growth == 0:
            self.la_growth_overall = (inflation + 1)/100
        else:
            self.la_growth_overall = la_growth/100

        self.cg_to_date = cg_to_date
        self.inflation = inflation/100
        #  In real terms:
        self.ra_growth = (1 + self.ra_growth_overall)/(1 + self.inflation) - 1
        self.la_growth = (1 + self.la_growth_overall)/(1 + self.inflation) - 1

        self.monthly_la_growth = 10**(np.log10(1 + self.ra_growth)/12) - 1
        self.monthly_ra_growth = 10**(np.log10(1 + self.la_growth)/12) - 1
        self.payout_fraction = payout_fraction
        self.retirement_date = pd.datetime(self.dob.year + era, self.dob.month, self.dob.day)
        self.df = pd.DataFrame(index=pd.DatetimeIndex(start=pd.datetime.today().date(),
                                          end=pd.datetime(self.dob.year + le, self.dob.month, self.dob.day),
                                          freq='A-FEB'),
                   columns=['capital',
                            'YTD contr',
                            'withdrawals',
                            'contr'])
            
        self.df.loc[:, ['capital',
                            'YTD contr',
                            'withdrawals',
                            'contr']] = 0
        self.df.loc[self.df.index[0], 'capital'] = self.initial
        self.df.loc[self.df.index[0], 'YTD contr'] = ytd
        self.first_retirement_date = self.df.loc[self.df.index>=self.retirement_date].index[0]

    def calculateOptimalWithdrawal(self,
                                   contr,
                                   strategy='optimal',
                                   payout_fraction=None):
        if payout_fraction is None: 
            self.growthBeforeRetirement(contr, self.payout_fraction)
        else:
            self.growthBeforeRetirement(contr, payout_fraction)
        self.growthAfterRetirementOptimalWithdrawal(contr, strategy)

    def calculateCapitalAnnualized(self,
                                   capital,
                                   contributions,
                                   withdrawals,
                                   growth,
                                   installments=12):
        
        '''
        Calculates actual annual capital growth considering monthly
        contributions and withdrawals.
        -------
        Parameters:
        capital:        float. Capital at beginning of year
        contributions:  float. Total annual contributions. Assumed equal
                        monthly.
        withdrawals:    float. Total annual withdrawals, Assumed equal monthly.
        growth:         float. Annual growth expressed as fraction. 
                        E.g. 5% = 0.05.
        installments:   Number of period in over which calculation should be
                        done. Usually 12, except for first year.        
        '''
        capital_calc = capital
        monthly_growth = 10**(np.log10(1 + growth)/12) - 1
        contr = contributions/installments
        withdr = withdrawals/installments
        for i in range(0, installments):
            if capital_calc > 0:
                capital_calc = capital_calc*(1 + monthly_growth) + contr - withdr            
            else:
                return 
        return capital_calc
        
    def growthBeforeRetirement(self,
                               contr,
                               payout_fraction=None):
        
        if payout_fraction is None:
            payout_fraction = self.payout_fraction
            
        previous_year = self.df.index[0]
        self.df['contr'] = contr    
        
        for year in self.df.loc[self.df.index[0]:self.first_retirement_date].index[1:]:
            self.df.loc[year, 'capital'] = self.calculateCapitalAnnualized(self.df.loc[previous_year, 'capital'],
                                                                            self.df.loc[year, 'contr'],
                                                                            0,
                                                                            self.ra_growth)
            self.last_ra_year = year
            previous_year = year
        capital_at_retirement = self.df.loc[self.last_ra_year, 'capital']
        self.payout = capital_at_retirement*payout_fraction

        self.df.loc[self.first_retirement_date, 'capital'] = capital_at_retirement*(1 - payout_fraction)

    def growthAfterRetirementOptimalWithdrawal(self,
                                               contr,
                                               strategy='optimal',
                                               ra_payout_frac=0): #  dummy, for when Portfolio calls RA.
        
        '''
        Calculates growth of a retirement annuity
        ------
        Parameters:
        contr:          DataFrame. Dataframe, indexed by year from today to 
                        retirement age, with contributions.
        withdrawals:    DataFrame. Dataframe, indexed by year from today to 
                        retirement age, with withdrawls.
        '''
        self.df['contr'] = contr    
        self.df['withdrawals'] = 0  
        
        c = self.df.loc[self.first_retirement_date, 'capital']
        drawdown = 0.04
        
        if strategy == 'optimal':
            capital_at_le = 1e7
            
            arr = self.df.loc[self.df.index >= self.first_retirement_date, 'capital'].values
            while capital_at_le > 0 and drawdown < 0.175 and 0.175*capital_at_le > drawdown*c:
                drawdown = min(drawdown + 0.001, 0.175)
                withdrawal = drawdown*c
                capital_at_le = self._growthAfterRetirementQuick(arr, self.la_growth, withdrawal)

        self.df.loc[self.first_retirement_date:, 'withdrawals'] = drawdown*c
        self.growthAfterRetirement(contr)
        
    def growthAfterRetirement(self, contr):
        previous_year = self.first_retirement_date

        for year in self.df.loc[self.first_retirement_date:].index[1:]:            
            if self.df.loc[year, 'withdrawals'] <= self.df.loc[previous_year, 'capital']:
                if self.df.loc[year, 'withdrawals'] < 0.025*self.df.loc[previous_year, 'capital']:
                    self.df.loc[year, 'withdrawals'] = 0.025*self.df.loc[previous_year, 'capital']
                elif self.df.loc[year, 'withdrawals'] > 0.175*self.df.loc[previous_year, 'capital']:
                    self.df.loc[year, 'withdrawals'] = 0.175*self.df.loc[previous_year, 'capital']
                #self.df.loc[year, 'capital'] = max(0, self.df.loc[previous_year, 'capital']*(1 + self.la_growth) - self.df.loc[year, 'withdrawals'])
                self.df.loc[year, 'capital'] = self.calculateCapitalAnnualized(self.df.loc[previous_year, 'capital'],
                                                                               self.df.loc[year, 'contr'],
                                                                                   self.df.loc[year, 'withdrawals'],
                                                                            self.la_growth)
            
            else:
                self.df.loc[year, 'withdrawals'] = self.df.loc[year, 'capital']
                self.df.loc[year, 'capital'] = 0

            previous_year = year 
    
    #@numba.jit      
    def _growthAfterRetirementQuick(self, arr, growth, withdrawal):
        
            for i in range(1, len(arr)):  
                if withdrawal < 0.025*arr[i - 1]:
                    withdrawal_i = 0.025*arr[i - 1]
                elif withdrawal > 0.175*arr[i - 1]:
                    withdrawal_i = 0.175*arr[i - 1]
                else:
                    withdrawal_i = withdrawal
                arr[i] = self.calculateCapitalAnnualized(arr[i - 1],
                                                           0,
                                                           withdrawal_i,
                                                           growth)
            return arr[-1]  
        
        
class DI(object):
    
    
    '''
    Discretionary Investment object.
    '''
    
    def __init__(self,
                 initial,
                 dob,
                 era,
                 le,
                 growth=18,
                 inflation=5.5,
                 cg_to_date=0):
        
        #Investment.__init__(self, initial, growth)
        self.initial = initial
        self.growth = growth/100
        self.monthly_growth = 10**(np.log10(1 + self.growth)/12) - 1
        self.type = 'DI'
        self.dob = pd.to_datetime(dob).date()   
        self.df = pd.DataFrame(index=pd.DatetimeIndex(start=pd.datetime.today().date(),
                                          end=pd.datetime(self.dob.year + le, self.dob.month, self.dob.day),
                                          freq='A-FEB'),
                   columns=['capital',
                            'capital_gain',
                            'contr',
                            'withdrawals',
                            'withdrawal_cg'])
            
        self.df.loc[:, ['capital',
                        'contr',
                        'capital_gain',
                        'withdrawals',
                        'withdrawal_cg']] = 0
        self.df.loc[self.df.index[0], 'capital'] = self.initial
        self.inflation = inflation/100
        self.overall_growth = growth/100
        #In real terms:
        self.cg_to_date = cg_to_date
        self.monthly_growth = 10**(np.log10(1 + self.growth)/12) - 1
        self.retirement_date = pd.datetime(self.dob.year + era, self.dob.month, self.dob.day)
        self.first_retirement_date = self.df.loc[self.df.index>=self.retirement_date].index[0]
        self.last_working_date = self.df.loc[self.df.index<self.retirement_date].index[-1]

        if growth == 0:
            jse = pd.read_csv('JSE_returns.csv', index_col=0)
            size = min(2017 - 1974, self.retirement_date.year - pd.datetime.today().year)
            lst = []
            for i in range(jse.shape[0] - size):
                investment = 1
                for j in jse['return'].iloc[i:i+size]:
                    investment *= 1 + j
                growth = 10**((1/size)*np.log10(investment)) - 1
                lst += [growth]
            self.growth = (1 + np.mean(lst))/(1 + self.inflation) - 1
        else:
            self.growth = (1 + self.growth)/(1 + self.inflation) - 1
            
    def _calculateQuick(self, arr, growth, withdrawal):
        
        '''Quick version of calculate, simply for determining optimal drawdown rate.
        Does not calculate all variables of interest like calculate() does.
        
        Parameters:
        arr:            numpy array. First column is capital, second column is 
                        contributions.
        growth:         float. Annual growth rate of portfolio
        withdrawal:     float. Annual (fixed) withdrawal amount.
        '''
        for i in range(1, len(arr)): 
            arr[i], b, c = self.calculateCapitalAnnualized(arr[i - 1],
                                                        0,
                                                        0,
                                                        withdrawal,
                                                        growth)
        return arr[-1]  
    

    def calculateCapitalAnnualized(self,
                                   capital,
                                   capital_gains,
                                   contributions,
                                   withdrawals,
                                   growth,
                                   installments=12):
        
        '''
        Calculates actual annual capital growth considering monthly
        contributions and withdrawals.
        -------
        Parameters:
        capital:        float. Capital at beginning of year
        contributions:  float. Total annual contributions. Assumed equal
                        monthly.
        withdrawals:    float. Total annual withdrawals, Assumed equal monthly.
        growth:         float. Annual growth expressed as fraction. 
                        E.g. 5% = 0.05.
        installments:   Number of period in over which calculation should be
                        done. Usually 12, except for first year.        
        '''
        capital_calc = capital
        capital_gains_calc = capital_gains
        monthly_growth = 10**(np.log10(1 + growth)/12) - 1       
       
        cg_growth = (1 + self.inflation)*(growth + 1) - 1 #converting back from real terms
        annual_cg_growth = 10**(np.log10(1 + cg_growth)/12) - 1
        monthly_cg_growth = 10**(np.log10(1 + annual_cg_growth)/12) - 1

        contr = contributions/installments
        withdr = withdrawals/installments
        withdrawal_cg = 0
        withdrawal_cg_incr = 0
        
        for i in range(0, installments):
            if capital_calc > 0:
                capital_gains_calc = capital_gains_calc + capital*monthly_cg_growth
                withdrawal_cg_incr = withdrawals*(capital_gains_calc/capital)
                capital_gains_calc -= withdrawal_cg_incr
                withdrawal_cg += withdrawal_cg_incr
                capital_calc = capital_calc*(1 + monthly_growth) + contr - withdr            
            else:
                return 0, capital_gains_calc, withdrawal_cg
            
        return capital_calc, capital_gains_calc, withdrawal_cg 
    
    def calculateLastYearWithdrawals(self, 
                                     capital,
                                     capital_gains,
                                     contributions,
                                     withdrawals,
                                     growth,
                                     installments=12):
        
        '''
        Calculates last year's withdrawals when capital runs out
        -------
        Parameters:
        capital:        float. Capital at beginning of year
        contributions:  float. Total annual contributions. Assumed equal
                        monthly.
        withdrawals:    float. Total annual withdrawals, Assumed equal monthly.
        growth:         float. Annual growth expressed as fraction. 
                        E.g. 5% = 0.05.
        installments:   Number of period in over which calculation should be
                        done. Usually 12, except for first year.        
        '''
        monthly_growth = 10**(np.log10(1 + growth)/12) - 1
        cg_growth = (1 + self.inflation)*(growth + 1) - 1 #converting back from real terms
        annual_cg_growth = 10**(np.log10(1 + cg_growth)/12) - 1
        monthly_cg_growth = 10**(np.log10(1 + annual_cg_growth)/12) - 1

        cg_growth = (1 + self.inflation)*(growth + 1) - 1 #converting back from real terms
        annual_cg_growth = 10**(np.log10(1 + cg_growth)/12) - 1
        monthly_cg_growth = 10**(np.log10(1 + annual_cg_growth)/12) - 1
        withdr_total = 0
        withdrawal_cg = 0
        capital_gains_calc = 0
        withdr = withdrawals/installments
        capital = capital*(1 + monthly_growth) - withdr # + contr
        withdr_total += withdr
        print('withdrawals', withdrawals)
        while capital > 0:
            capital_gains_calc = capital_gains_calc + capital*monthly_cg_growth
            if withdrawals < capital:
                print('capital', capital)
                print('capital gains calc', capital_gains_calc)
                withdrawal_cg_incr = withdrawals*capital_gains_calc/capital
                capital_gains_calc -= withdrawal_cg_incr
                withdrawal_cg += withdrawal_cg_incr            
                capital = capital*(1 + monthly_growth) - withdr
            else:
                withdrawal_cg += capital_gains_calc
                withdr_total += capital
                capital = 0
            withdrawals += 100
        return withdr_total, withdrawal_cg

    def calculateOptimalWithdrawal(self, 
                                   contr,
                                   strategy='optimal',
                                   ra_payout_frac=0): # dummy variable for RA.
        
        self.df.loc[:, 'contr'] = contr    
        self.df.loc[:, 'withdrawals'] = 0
        
        self.recalculateOptimalWithdrawal()
        
    def recalculateOptimalWithdrawal(self, strategy='optimal'):
        
        '''
        Calculates optimal withdrawal figure so that capital lasts exactly as 
        long as life expectancy. This is in inherently risky strategy if life
        expectancy is underestimated.
        '''
        self.calculate()
        c = self.df.loc[self.first_retirement_date, 'capital']
        drawdown = 0.04
        if strategy == 'optimal':
            capital_at_le = 1e7
            arr = self.df.loc[self.last_working_date:, 'capital'].values
            while capital_at_le > 0:
                drawdown += 0.001
                capital_at_le = self._calculateQuick(arr, self.growth, drawdown*c)
            drawdown -= 0.001

        self.df.loc[self.first_retirement_date:, 'withdrawals'] = drawdown*c
        self.calculate()

    def calculate(self):
        
        previous_year = self.df.index[0]
        self.df.loc[:, 'capital'] = 0.001
        if self.initial > 0:
            self.df.loc[previous_year, 'capital'] = self.initial
        else:
            self.df.loc[previous_year, 'capital'] = 0.001

            self.df.loc[previous_year, 'capital'],\
            self.df.loc[previous_year, 'capital_gain'],\
            self.df.loc[previous_year, 'withdrawal_cg'] = self.calculateCapitalAnnualized(self.df.loc[previous_year, 'capital'],
                                                                           self.df.loc[previous_year, 'capital_gain'],
                                                                           self.df.loc[previous_year, 'contr'],
                                                                           self.df.loc[previous_year, 'withdrawals'],
                                                                           self.growth)

        for year in self.df.index[1:]:
            self.df.loc[year, 'capital'],\
            self.df.loc[year, 'capital_gain'],\
            self.df.loc[year, 'withdrawal_cg'] = self.calculateCapitalAnnualized(self.df.loc[previous_year, 'capital'],
                                                                           self.df.loc[previous_year, 'capital_gain'],
                                                                           self.df.loc[year, 'contr'],
                                                                           self.df.loc[year, 'withdrawals'],
                                                                           self.growth)

            if self.df.loc[year, 'capital'] == 0:
                self.df.loc[year, 'withdrawals'],\
                #self.df.loc[year, 'withdrawal_cg'] = self.calculateLastYearWithdrawals(self.df.loc[previous_year, 'capital'],
                #                                                           self.df.loc[previous_year, 'capital_gain'],
                #                                                            self.df.loc[year, 'contr'],
                #                                                            self.df.loc[year, 'withdrawals'], 
                #                                                            self.growth)
                self.df.loc[year, 'capital_gain'] = 0
                
            previous_year = year

"""
    def GA(self):
        '''
        Uses Genetic Algorithm to find optimal investment strategy.
        '''
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        #pool = multiprocessing.Pool()
        #toolbox.register("map", pool.map)
        toolbox.register("individual_guess", self.initIndividual, creator.Individual)
        toolbox.register("population_guess", self.initPopulation, list, toolbox.individual_guess)
        
        population = toolbox.population_guess()

        toolbox.register("evaluate", self.fitness)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutatePyfin, indpb=0.9, number_changed=10)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        toolbox.decorate("mate", self.checkBounds())
        toolbox.decorate("mutate", self.checkBounds())
        
        def main(): #  This is necessary to enable multiprocessing.
            pop = toolbox.population_guess()        
            hof = tools.HallOfFame(maxsize=20)        
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)        
            MU = int(self.pop_size)
            LAMBDA = int(self.pop_size)
            CXPB = 0.7
            MUTPB = 0.2
            pop, logbook = algorithms.eaMuPlusLambda(pop, 
                                                     toolbox, 
                                                     MU, 
                                                     LAMBDA, 
                                                     CXPB, 
                                                     MUTPB, 
                                                     self.ngen,
                                                     stats,
                                                     halloffame=hof,
                                                     verbose=True)
            #print('before best_ind')
            best_ind = tools.selBest(pop, 1)[0]

            print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
            self.best_ind = best_ind
            return pop, stats, hof, logbook, best_ind
                        
        if __name__ == "__main__":
            pop, stats, hof, logbook, n_best = main()
            
        gen = logbook.select("gen")
        fit_mins = [logbook[i]['min'] for i in range(len(logbook))]
        size_avgs = [logbook[i]['avg'] for i in range(len(logbook))]
        stds = [logbook[i]['std'] for i in range(len(logbook))]    
        plt.figure(0)
        plt.rcParams['font.family'] = "serif"
        plt.rcParams['font.size'] = 14
        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")
        
        ax2 = ax1.twinx()
        line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
        ax2.set_ylabel("Size", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")
        
        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="best")
       
    def mutatePyfin(self, individual, indpb, number_changed):
        
        '''
        Mutation function of Genetic Algorithm.
        '''
        ind_with_ra_payout = np.array(individual)
        ind = ind_with_ra_payout[1:]
        ra_payout = ind_with_ra_payout[0]
        reshaped = ind.reshape(int(ind.size/self.size), self.size, order='F')
        for i in range(number_changed):
            if indpb < np.random.random():
                    year = np.random.randint(0, self.number_working_years)
                    position = np.random.randint(0, self.size)
                    reshaped[year, position] = max(0, reshaped[year, position]*np.random.normal(1, 0.5/3)) 
        if indpb < np.random.random():
            ra_payout = ra_payout + np.random.random()*0.1 - 0.05
        return creator.Individual(np.insert(reshaped.reshape(reshaped.size, order='F'), 0, ra_payout)),
        
    def initPopulation(self, pcls, ind_init):
        
        ind_list = [np.ones([self.number_working_years,
                             self.size]) for i in range(self.pop_size)]
        ind_list=[]
        contr = self.taxEfficientIndividual()
        ind_list += [contr.reshape(contr.size, order='F')]   
        factor_list = np.geomspace(1/20, 100, 30)
        for _ in range(30):
            contr = self.randomIndividual(np.random.choice(factor_list))
            ind_list += [contr.reshape(contr.size, order='F')]            
            contr = self.taxEfficientIndividual()
            ind_list += [contr.reshape(contr.size, order='F')]    
            contr = self.randomConstantIndividual(np.random.choice(factor_list))
            ind_list += [contr.reshape(contr.size, order='F')]
            contr = self.taxEfficientIndividual()
            ind_list += [contr.reshape(contr.size, order='F')]   
            
        #  Equally divided portions:
        contr = np.zeros([self.number_working_years + self.number_retirement_years + 1, self.size])        
        for i in range(self.number_working_years):
            contr[i, :] = 1/self.size*np.array([1]*self.size)
                        
        ind_list += [np.insert(contr.reshape(contr.size, order='F'),
                               0,
                               np.random.random()*0.3)]
        contr = self.taxEfficientIndividual()
        ind_list += [contr.reshape(contr.size, order='F')]           
        
        self.ind_list = ind_list
        return (pcls(ind_init(i) for i in ind_list))    
    
    def rebalance(self, child):
            ind = self.reshape(child)
            ras = ind[:, :len(self.ra_list)]
            #  If ras are so high that savable income is negative,
            #  reduce ra contributions by 1%.
            for i in range(len(ras[:, 0])):
                savable_income = -1
                ras_sum = ras.sum(axis=1)
                while savable_income < 0 and ras_sum[i] > 0:
                    ras_sum = ras.sum(axis=1)
                    tax = self.incomeTax(self.taxable_ibt - ras_sum[i]*self.taxable_ibt)
                    if i <= self.number_working_years:
                        savable_income = self.taxable_ibt - tax - self.uif_contr - self.expenses*12 - ras_sum[i]*self.taxable_ibt
                    else:
                        savable_income = self.taxable_ibt - tax - self.expenses*12 - ras_sum[i]*self.taxable_ibt
                    ras[i, :] -= 0.01
                #print('savable_income', savable_income)
                if ras[i, :] < 0:
                    ras[i, :] = 0
                else:
                    ras[i, :] += 0.01  
                #print('ras[i,:]', ras[i, :])

            others = ind[:, len(self.ra_list):]
            others = others/others.sum(axis=1)[:, None] # normalize
            others[others==np.inf] = 0
            others[others!=others] = 0 #  NaNs
            ra_payout = np.clip(child[0], 0, 0.3)
            ind[:, :len(self.ra_list)] = np.abs(ras)
            ind[:, len(self.ra_list):] = np.abs(others)
            return ind, ra_payout
            
    def fitness(self, scenario_1d, verbose=False):
        
        '''
        Genetic Algorithm Fitness function. Just a wrapper casting the result of
        the objective function as a tuple, which is what DEAP needs.
        '''
        ret_tuple = self.objective(scenario_1d),
        #print('after objective')
        print(ret_tuple)
        return 
        
    def initIndividual(self, icls, content):
        
        return icls(content)

    def objective(self, individual):
        '''
        Objective Function for optimization. 
        ------
        Parameters:
        scenario_1d:    ndarray. 1D Numpy array. This is a reshaped form of an
                        array of dimension [working months + retirement months, 2*portfolio size]
                        where the values are fractions. The first len(self.ra_list)
                        columns are retirement annuities. These fractions are 
                        fractions of the total income before tax allocated to the
                        retirement annuity. The rest of the columns are from
                        income after tax, specifically the savable income;
                        IAT - expenses.
        strategy:       XXXXXXXXXXXXXXXXXXXXXX
        ------
        Returns:        float. The mean income after tax during the retirement 
                        period.
                        
        '''
        self.individual = individual
        fracs, ra_payout_frac = self.rebalance(individual)
        scenario = self.fractionsToRands(fracs)
        self.contr.loc[:, self.contr.columns] = scenario
        for count, i in enumerate(self.investments.keys()):
            self.investments[i].calculateOptimalWithdrawal(self.contr.loc[:, count],
                                                            self.strategy,
                                                            ra_payout_frac)
        self.calculate()
        self.df.loc[self.df['iat']==0, 'iat'] = -self.taxable_ibt*100
        if self.strategy == 'optimal':
            return round(-self.df.loc[self.retirement_date:, 'iat'].mean(), 2) #+ penalty_oversaved, 2)
        elif self.strategy == 'safe':
            return round(-self.df.loc[self.retirement_date:, 'iat'].mean(), 2) #+ penalty_oversaved, 2)
    

"""