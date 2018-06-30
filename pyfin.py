#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:53:46 2018

@author: herman
"""
### TODO:
#  Bou inflasie in
#  Bou spreiblad vermoë in
#  fooie?
#  Bou plan evalueringsvermoë in (sonder optimering)
#  Maak skoon, comment, privatiseer funksies en veranderlikes.

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import time

#import numba
import multiprocessing

class Portfolio(object):
    
    def __init__(self,
                 dob,
                 ibt,
                 expenses,
                 ma_dependents,
                 medical_expenses,
                 era,
                 le,
                 strategy='optimal'):
        '''
        Portfolio object, combining all investments.
        ------
        Parameters:
        dob:                str. Date of Birth, in format "YYYY-MM-DD"
        ibt:                int. Annual income before tax
        expenses:           float. Expenses before tax
        ma_dependents:      int. Number of medical aid dependants, including self.
        medical_expenses:   float. Annual out-of-pocket medical expenses
        era:                int. Expected Retirement Age.
        le:                 int. life expectancy.
        '''
        
        self.dob = pd.to_datetime(dob).date()
        assert(isinstance(ibt, int))
        self.taxable_ibt = ibt
        assert(isinstance(expenses, float) or isinstance(expenses, int))
        self.expenses = expenses
        assert(isinstance(ma_dependents, int))
        self.ma_dependents = ma_dependents
        assert(isinstance(era, int))
        self.era = era
        assert(isinstance(le, int))
        self.le = le
        assert(isinstance(medical_expenses, float) or isinstance(medical_expenses, int))
        self.medical_expenses = medical_expenses
        self.ra_payouts = 0
        self.size = 0
        self.age = pd.datetime.today().date() - self.dob
        self.retirement_date = pd.datetime(self.dob.year + era, self.dob.month, self.dob.day)
        self.strategy = strategy
        
        self.investments = {}
        self.ra_list = []
        self.tfsa_list = []
        self.di_list = []
        
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
                                        'contr_total'])

        self.df.loc[:, ['taxable_ibt',
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
                        'contr_total']] = 0
                        
        self.df.loc[:, 'age'] = (self.df.index - pd.Timestamp(self.dob)).days/365.25
        self.df.loc[:, 'medical_expenses'] = medical_expenses
        
        self.df.loc[:, 'medical_expenses'] = medical_expenses
        self.df.loc[:, 'date'] = self.df.index
        
        self.this_year = self.df.index[0]
        self.last_working_date = self.df.loc[self.df.index<self.retirement_date].index[-1]
        self.first_retirement_date = self.df.loc[self.df.index<self.retirement_date].index[-1]
        self.number_working_years = self.df.loc[self.df.index<self.last_working_date].shape[0]
        self.number_retirement_years = self.df.loc[self.df.index>=self.last_working_date].shape[0]
        self.df.loc[:self.last_working_date, 'taxable_ibt'] = self.taxable_ibt

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
            self.size +=1
            if investment.type == 'RA':
                self.ra_list += [name]
            elif investment.type == 'TFSA':
                self.tfsa_list += [name]
            elif investment.type == 'DI':
                self.di_list += [name]
            else:
                print('Type for {} not recognised'.format(name)) 
                
        elif isinstance(name, list):
            for name, count in enumerate(name):
                self.size += 1
                self.investments[name] = investment[count]
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
        ------
        Parameters:
        strategy:           str. 'optimal', 'safe'. The method by which the 
                            withdrawals are calculated. If 'optimal' is chosen, 
                            the withdrawal rate resulting in zero capital at 
                            life expectancy is found for each investment. This 
                            is the maximum sustainable annual amount that can 
                            be withdrawn from the investment. If 'safe' is 
                            selected, a drawdown of 4% is used - proven to be 
                            sustainable indefinitely.
        '''
        
        time1 = time.time()
        import scipy.optimize as spm
        self.savable_income = self.taxable_ibt*0.725 - self.expenses
          
        self.contr = pd.DataFrame(index=self.df.index,
                                     columns=np.arange(0, self.size))

        self.pop_size = 100
        self.ngen = 20
        self.GA()
        self.solution = self.fractionsToRands(self.reshape(self.best_ind))

        for count, i in enumerate(self.investments.keys()):
            self.contr.loc[:, count] = self.solution[:, count]
            self.contr.rename(columns={count:i}, inplace=True)
            self.investments[i].calculateOptimalWithdrawal(self.contr[i], self.strategy)
            
        self.calculate()
        print('Duration:', (time.time() - time1)/60, 'min')
        self.plot()
               
        
    #@numba.jit
    def reshape(self, ind):
        arr_ind = np.array(ind[1:])
        return arr_ind.reshape(int(arr_ind.size/(self.size)), self.size)
    
    #@numba.jit
    def fractionsToRands(self, ind):
        
        '''
        converts fractions saved into Rands saved.
        ------
        Parameters:     
        ind:            ndarray. Numpy array of size [self.number_working_years + number_retirement_years, self.size]
        ------
        Returns:        ndarray. Same shape as input. Just with Rand values.
        '''
        #print('ind\n', ind)
        #print('==========')
        contr = np.zeros_like(ind[1:]) #  exclude RA payout fraction (first item in arr)
        ra_contr = self.taxable_ibt*ind[1:, :len(self.ra_list)].sum(axis=1)
        tax = np.zeros(len(contr))
        for i, year in enumerate(self.df.index[1:]):
            tax[i] = self.incomeTax(self.taxable_ibt - ra_contr[i], age=self.df.loc[year, 'age'])
        savable_income = np.maximum(0, self.taxable_ibt - ra_contr - tax - self.expenses)
        contr[:, :len(self.ra_list)] = self.taxable_ibt*np.array(ind[1:, :len(self.ra_list)])
        contr[:, len(self.ra_list):] = savable_income[:, None]*np.array(ind[1:, len(self.ra_list):])
        #print('contr\n', contr)
        return contr

        
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
        #  Reshape array:
        #print('individual shape', np.array(individual).shape)
        scenario = self.fractionsToRands(self.reshape(individual))
        #print('scenario shape', scenario.shape)
        ra_payout_frac = individual[0]
        self.contr.loc[:, self.contr.columns] = scenario
        #self.withdrawals.loc[:, self.withdrawals.columns] = scenario[:, self.size:self.size*2]
        for count, i in enumerate(self.investments.keys()):
            self.investments[i].calculateOptimalWithdrawal(self.contr.loc[:, count],
                                                            self.strategy,
                                                            ra_payout_frac)
        #print('before calculate')
        self.calculate()
        
        #  Penalise IAT==0 in any year
        #print('before penalty')
        self.df.loc[self.df['iat']==0, 'iat'] = -self.taxable_ibt*100
        #  Penalise algorithm by the difference between max and min iat during retirement:
        
        #penalty_chebychev = self.df.loc[self.first_retirement_date:, 'iat'].max() - self.df.loc[self.first_retirement_date:, 'iat'].min()
        #print('Chebychev', penalty_chebychev)
        #  Penalise algorithm by if sum of contributions are larger than savable income:
        #  Penalise the algorithm for leaving money in the accounts at death:
        #penalty_left_over = self.df.loc[self.df.index[-1], ['capital_RA', 'capital_TFSA', 'capital_DI']].sum()
        #print('Saved surplus', penalty_oversaved)
        #print(-self.df.loc[self.retirement_date:, 'iat'].sum() + penalty_chebychev + penalty_oversaved + penalty_left_over)
        #penalty_stdev = self.df.loc[self.first_retirement_date:, 'iat'].std()
        #return round(-self.df.loc[self.retirement_date:, 'iat'].sum() + penalty_chebychev + penalty_oversaved + penalty_left_over + penalty_stdev, 2)
       
        if self.strategy == 'optimal':
            #print('before saved_surplus')
            #saved_surplus = self.df.loc[:self.last_working_date, 'iat'] - self.expenses - self.df.loc[:self.last_working_date, 'contr_total']
            #penalty_oversaved = saved_surplus.loc[saved_surplus<0].sum()**2
            #print('after saved surplus')
            return round(-self.df.loc[self.retirement_date:, 'iat'].mean(), 2) #+ penalty_oversaved, 2)
        elif self.strategy == 'safe':
            #saved_surplus = self.df.loc[:self.last_working_date, 'iat'] - self.expenses - self.df.loc[:self.last_working_date, 'contr_total']
            #penalty_oversaved = saved_surplus.loc[saved_surplus<0].sum()**2
            return round(-self.df.loc[self.retirement_date:, 'iat'].mean(), 2) #+ penalty_oversaved, 2)

    def calculate(self):
        
        '''
        Calculates the income, taxable income, tax, etc. for the whole portfolio
        '''
        
        self.df.loc[:, ['taxable_ibt',
         'capital_gains',
         'iat',
        'withdrawals_total',
        'withdrawals_RA',
        'withdrawals_TFSA',
        'withdrawals_DI',
        'contr_RA',
        'contr_DI',
        'contr_TFSA',
        'contr_total']] = 0
        self.ra_payouts = 0      
        self.df.loc[:self.last_working_date, 'taxable_ibt'] = self.taxable_ibt
        
        for i in self.ra_list:
            self.df['taxable_ibt'] += self.investments[i].df['withdrawals']
            self.df['withdrawals_total'] += self.investments[i].df['withdrawals']
            self.df['withdrawals_RA'] += self.investments[i].df['withdrawals']
            self.df['contr_RA'] += self.investments[i].df['contr']
            self.df['capital_RA'] = self.investments[i].df['capital']
            self.df['contr_total'] += self.investments[i].df['contr']

            #self.df['income'] = self.df['income'] + df['withdrawals']
            self.ra_payouts += self.investments[i].payout
            #  Check whether this works:
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


        for i in self.tfsa_list:
            self.df['iat'] += self.investments[i].df['withdrawals']
            self.df['contr_TFSA'] += self.investments[i].df['contr']
            self.df['withdrawals_total'] += self.investments[i].df['withdrawals']
            self.df['withdrawals_TFSA'] += self.investments[i].df['withdrawals']
            self.df['capital_TFSA'] = self.investments[i].df['capital']
            self.df['contr_total'] += self.investments[i].df['contr']


        self.df['it'] = self.df.apply(self.totalTax, axis=1)
        #self.df['it'] = self.df['taxable_income'].map(self.IT)
        self.df['iat'] = self.df['iat'] + self.df['taxable_ibt'] - self.df['it']
        
    def totalTax(self, s):
        
        '''
        Calculates total income tax.
        ------
        Parameters:
        s:          Pandas Series. Containing columns capital_gains, income, RA_contributions
        ------
        Returns:
        tax:                float. tax payable in a particular year
        '''
        
        age = (s.name - pd.Timestamp(self.dob)).days/365.25
        taxable_income = 0
        if s.name < self.retirement_date:
            if s.contr_RA <= 0.275*s.taxable_ibt and s.contr_RA <=350000:
                taxable_income = s.taxable_ibt + 0.18*max(0, s.capital_gains - 40000) - s.contr_RA
            elif s.contr_RA > 0.275*s.taxable_ibt and s.contr_RA < 350000:
                taxable_income = s.taxable_ibt - s.taxable_ibt*0.275 + 0.18*max(0, s.capital_gains - 40000)
            else:
                taxable_income = s.taxable_ibt - 350000 + 0.18*max(0, s.capital_gains - 40000)

        if s.name >= self.retirement_date:
            if age < 65:
                taxable_income = max(0, s.taxable_ibt + 0.18*max(0, s.capital_gains - 40000) - 78150)
            elif age < 75:
                taxable_income = max(0, s.taxable_ibt + 0.18*max(0, s.capital_gains - 40000) - 121000)
            else:
                taxable_income = max(0, s.taxable_ibt + 0.18*max(0, s.capital_gains - 40000) - 135300)                
        
        tax = self.incomeTax(taxable_income, age)
        
        self.taxCreditMa(s.medical_expenses, taxable_income, age)    
        return max(0, tax - self.tax_credit_ma)
    
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
    def taxCreditMa(self, medical_expenses, taxable_income, age):
        if age > 65:
            if self.ma_dependents <=2:
                ma_d_total = self.ma_dependents*310
            else:
                ma_d_total = 620 + (self.ma_dependents - 2)*209
            
            self.tax_credit_ma = 0.33*(medical_expenses - 3*ma_d_total)
        else:
            if self.ma_dependents <=2:
                ma_d_total = self.ma_dependents*310
            else:
                ma_d_total = 620 + (self.ma_dependents - 2)*209
            
            if medical_expenses > 0.075*taxable_income:
                self.tax_credit_ma = ma_d_total + 0.25*(self.medical_expenses - 0.075*taxable_income)
            else:
                self.tax_credit_ma = ma_d_total
    
    #@numba.jit        
    def CGTRA(self, lump_sum):
        '''
        Calculates the Capital Gains Tax on Retirement Annuity lump sum payment
        '''
        
        '''
        Lump Sum Tax Benefits - not sure why this is different from the below?
        
        if lump_sum < 25000:
            return 0
        elif lump_sum <= 660000:
            return lump_sum*0.18
        elif lump_sum <= 990000:
            return 114300 + (lump_sum - 660000)*0.27
        elif lump_sum > 990000:
            return 203400 + (lump_sum - 990000)*0.36
        '''
        if lump_sum < 500000:
            return 0
        elif lump_sum < 700000:
            return (lump_sum - 500000)*0.18
        elif lump_sum < 1050000:
            return 36000 + (lump_sum - 700000)*0.27
        elif lump_sum >= 1050000:
            return 130500 + (lump_sum - 1050000)*0.36
        
    def plot(self):
        plt.figure(1)
        index = [x.strftime('%Y-%M-%d') for x in self.df.index.date]
        plt.plot(index, self.df['withdrawals_TFSA'], label='TFSA')
        plt.plot(index, self.df['withdrawals_RA'], label='RA')
        plt.plot(index, self.df['withdrawals_DI'], label='DI')
        plt.title('Withdrawals during Retirement')
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
        plt.title('Contributions while working')
        
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
        plt.plot(index, self.df['iat'])
        plt.title('Income After Tax')
        plt.xticks(rotation=90)
        
    def fitness(self, scenario_1d, verbose = False):
        
        '''
        Genetic Algorithm Fitness function. Just a wrapper casting the result of
        the objective function as a tuple, which is what DEAP needs.
        '''
        ret_tuple = self.objective(scenario_1d),
        #print('after objective')
        #print(ret_tuple)
        return ret_tuple
        
    def initIndividual(self, icls, content):
        
        return icls(content)
    
    #@numba.jit
    def randomIndividual(self, factor):
        
        contr = np.zeros([1 + self.number_working_years + self.number_retirement_years, self.size])
        ra_payout = np.random.random()*0.3
        for i in range(self.number_working_years):
            #  Generate RA allocations to sum to anything up to 27.5%:
            ra = np.random.dirichlet(factor*np.ones(len(self.ra_list)))*np.random.beta(1, 3.9)
            #  Generate other allocations to sum to one:
            others = np.random.dirichlet(factor*np.ones(self.size - len(self.ra_list)))
            contr[i, :] = np.concatenate([ra, others])
        #return self.convertPercentagesToRands(contr)
        #return np.array([self.convertPercentagesToRands(i) for i in contr])
        reshaped = contr.reshape(contr.size)
        
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
        contr = np.zeros([1 + self.number_working_years + self.number_retirement_years, self.size])
        #ra = np.random.dirichlet(factor*np.ones(len(self.ra_list)))*np.random.triangular(0, 0.275, 0.5)
        ra = np.random.dirichlet(factor*np.ones(len(self.ra_list)))*np.random.beta(1, 3.9)
        #  Generate other allocations to sum to one:
        others = np.random.dirichlet(factor*np.ones(self.size - len(self.ra_list)))
        plan = np.concatenate([ra, others])
        contr = np.array([plan if i < self.number_working_years else np.zeros(self.size) for i in range(len(contr))])
        #contr[:self.number_working_years, :] = np.array([self.convertPercentagesToRands(a) for i in range(self.number_working_years)])
        return np.insert(np.array(contr), 0, ra_payout)
        
    
    def initPopulation(self, pcls, ind_init):
        
        ind_list = [np.ones([self.number_working_years,
                             self.size]) for i in range(self.pop_size)]
        ind_list=[]
        for j in np.geomspace(1/20, 100, 30):            
            contr = self.randomIndividual(j)
            ind_list += [contr.reshape(contr.size)]            
        
        for j in np.geomspace(1/20, 100, 30):            
            contr = self.randomConstantIndividual(j)
            ind_list += [contr.reshape(contr.size)]
            
        #  Equally divided portions:
        contr = np.zeros([self.number_working_years + self.number_retirement_years + 1, self.size])        
        for i in range(self.number_working_years):
            contr[i, :] = 1/self.size*np.array([1]*self.size)
                        
        ind_list += [np.insert(contr.reshape(contr.size),
                               0,
                               np.random.random()*0.3)]
        
        self.ind_list = ind_list
        return (pcls(ind_init(i) for i in ind_list))
    
    def checkBounds(self):
        
        '''
        Checks if asset allocations are still within bounds, and corrects
        if need be.
        '''
        def decorator(func):

            def wrapper(*args, **kwargs):
                offspring = func(*args, **kwargs)
                for child in offspring:
                    #print('in decorator, ind shape:', np.array(child).shape)
                    ind = self.reshape(child)
                    #print('ind shape', ind.shape)
                    ras = ind[:, :len(self.ra_list)]
                    others = ind[:, len(self.ra_list):]
                    if ras.sum(axis=1).any() > 0.275:
                        for i in range(len(ras)):
                            if ras[i, :].sum() > 0.275: # normalize
                                ras[i, :] = 0.275*ras[i, :]/ras[i, :].sum()
                    others[others==0] = 0.01
                    others = others/others.sum(axis=1)[:, None] # normalize
                    ra_payout = np.clip(child[0], 0, 0.3)
                    ind[:, :len(self.ra_list)] = np.abs(ras)
                    ind[:, len(self.ra_list):] = np.abs(others)  
                    child = creator.Individual(np.insert(ind.reshape(ind.size), 0, ra_payout)),
                #print('ending checkbounds')
                return offspring
            return wrapper
        return decorator
        
    def GA(self):
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        #pool = multiprocessing.Pool()
        #toolbox.register("map", pool.map)
        toolbox.register("individual_guess", self.initIndividual, creator.Individual)
        toolbox.register("population_guess", self.initPopulation, list, toolbox.individual_guess)
        
        population = toolbox.population_guess()

        toolbox.register("evaluate", self.fitness)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", self.mutatePyfin, indpb=0.9, number_changed=10)
        toolbox.register("select", tools.selTournament, tournsize=5)
        
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
    
        plt.rcParams['font.family'] = "serif"
        plt.rcParams['font.size'] = 14
        plt.figure(0)
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
        
        plt.show()
        
    def mutatePyfin(self, individual, indpb, number_changed):
        
        '''
        Mutation function of Genetic Algorithm.
        '''
        #print('in mutate')
        ind_with_ra_payout = np.array(individual)
        ind = ind_with_ra_payout[1:]
        ra_payout = ind_with_ra_payout[0]
        reshaped = ind.reshape(int(ind.size/self.size), self.size)
        for i in range(number_changed):
            if indpb < np.random.random():
                    year = np.random.randint(0, self.number_working_years)
                    position = np.random.randint(0, self.size)
                    reshaped[year, position] = max(0, reshaped[year, position]*np.random.normal(1, 0.5/3)) 
        if indpb < np.random.random():
            ra_payout = ra_payout + np.random.random()*0.1 - 0.05
        #print('finished with mutate')
        #print('reshaped shape', reshaped.shape)
        #print('reshaped after reshape', reshaped.reshape(reshaped.size).shape)
        #print('concatted', np.insert(reshaped.reshape(reshaped.size), 0, ra_payout).shape)

        return creator.Individual(np.insert(reshaped.reshape(reshaped.size), 0, ra_payout)),
        
    def determineRAContr(self, ibt, RA_monthly_contr=0, age=64):
        
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
                iat = ibt_ara - self.incomeTax(ibt_ara, age)
                surplus = iat - self.expenses
            
            RA_annual_contr -= 100
            ibt_ara = ibt - RA_annual_contr            
            iat = ibt_ara - self.incomeTax(ibt_ara, age)
            surplus = iat - self.expenses
            RA_monthly_contr = RA_annual_contr/12
            iat_monthly = iat/12
            print('RA Debit order: \t\t\t\tR', round(RA_monthly_contr, 2))
            print('Monthly IAT: \t\t\t\t\tR', round(iat_monthly))
            print('Max tax free RA contr (27.5% of IBT) = \t\tR', round(0.275*ibt/12, 2))
            print('Total earned per month, incl. RA: \t\tR', round(RA_annual_contr/12 + iat_monthly, 2))
            print('Total monthly tax = \t\t\t\tR', round(self.incomeTax(ibt_ara, age)/12, 2))

        else:
            RA_annual_contr = RA_monthly_contr*12
            ibt_ara = ibt - RA_annual_contr            
            iat = ibt_ara - self.incomeTax(ibt_ara, age)
            surplus = iat - self.expenses           
            iat_monthly = iat/12
            print('RA Debit order: \t\t\t\tR', round(RA_monthly_contr, 2))
            print('Annual taxable income: \t\t\t\tR', round(ibt_ara))
            print('Monthly IAT: \t\t\t\t\tR', round(iat_monthly))
            print('Max tax free RA contr (27.5% of IBT) = \t\tR', round(0.275*ibt/12, 2))
            print('Total earned per month, incl. RA: \t\tR', round(RA_annual_contr/12 + iat_monthly, 2))
            print('Total monthly tax = \t\t\t\tR', round(self.incomeTax(ibt_ara, age)/12, 2))


class Investment(object):

    
    def __init__(self,
                 initial,
                 growth):
        '''
        Investment object
        ------
        Parameters:
        initial:    float. Investment value at present time.
        growth:     float. Annual growth rate of investment in percentage
                    points, i.e. 5 is 5%.
        '''
        self.initial = initial
        self.growth = growth/100
        self.monthly_growth = 10**(np.log10(1 + self.growth)/12) - 1
        
   
class TFSA(Investment):
    
    def __init__(self,
                 initial,
                 growth,
                 dob,
                 ytd,
                 ctd,
                 era,
                 le):
        
        '''
        Tax-Free Savings Account object.
        ------
        Parameters:
        ytd:    float. Year-to-date contribution, according to the tax year.
        ctd:    float. Total contr to date. 
        '''
        
        self.type = 'TFSA'        
        Investment.__init__(self, initial, growth)
        self.ctd = ctd
        self.dob = pd.to_datetime(dob).date()   
        self.ytd = ytd


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
        
        self.growth = growth/100

        self.retirement_date = pd.datetime(self.dob.year + era, self.dob.month, self.dob.day)
        self.last_working_date = self.df.loc[self.df.index<=self.retirement_date].index[-1]
        self.first_retirement_date = self.df.loc[self.df.index>=self.retirement_date].index[0]
           
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
        #self.df.loc[self.df.index[0], 'capital'] = self.initial
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
          
            
class RA(Investment):

    
    #  TODO: Add fees
    
    def __init__(self,
                 initial,
                 ra_growth,
                 la_growth,
                 dob,
                 era,
                 le,
                 ytd,
                 payout_fraction=1/3):
        
        '''
        Retirement Annuity object. Assumes that the RA is converted to a living
        annuity upon retirement.
        ------
        Parameters:
        initial:            float. Value of RA at present time.
        ra_growth:          float. Growth rate of RA in percentage points. i.e. 13 = 13%.
        la_growth:          float. Growth rate of LA (after payout) in percentage points. i.e. 13 = 13%.
        dob:                Pandas Datetime object. Date of birth.
        retirement_date:    Pandas Datetime object. Date of retirement, when RA is 
                            converted to living annuity.
        le:                 int. Life expectancy.
        ytd:                float. Contribution to RA, tax year to date.
        '''        
        
        Investment.__init__(self, initial, ra_growth)
        self.type = 'RA'
        self.dob = pd.to_datetime(dob).date()
        self.ra_growth = ra_growth/100
        self.la_growth = la_growth/100
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
        #print(contributions)
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
            while capital_at_le > 0 and drawdown*c <= 0.175*capital_at_le:
                drawdown += 0.001
                withdrawal = drawdown*c
                capital_at_le = self._growthAfterRetirementQuick(arr, self.la_growth, withdrawal)
        self.df.loc[self.first_retirement_date:, 'withdrawals'] = drawdown*c
        self.growthAfterRetirement(contr)

        
    def growthAfterRetirement(self, contr):
        previous_year = self.first_retirement_date

        for year in self.df.loc[self.first_retirement_date:].index[1:]:            
            if self.df.loc[year, 'withdrawals'] <= self.df.loc[previous_year, 'capital']:
                if self.df.loc[year, 'withdrawals'] < 0.025*self.df.loc[previous_year, 'capital']:
                    #print('Adjusting RA withdrawals: too low.')
                    self.df.loc[year, 'withdrawals'] = 0.025*self.df.loc[previous_year, 'capital']
                elif self.df.loc[year, 'withdrawals'] > 0.175*self.df.loc[previous_year, 'capital']:
                    #print('Adjusting withdrawals: too high')
                    self.df.loc[year, 'withdrawals'] = 0.175*self.df.loc[previous_year, 'capital']
                self.df.loc[year, 'capital'] = max(0, self.df.loc[previous_year, 'capital']*(1 + self.la_growth) - self.df.loc[year, 'withdrawals'])
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
        
        
class DI(Investment):
    
    
    '''
    Discretionary Investment object.
    '''
    #  TODO: Add fees
    
    #@numba.jit

    def __init__(self, initial, growth, dob, era, le):
        
        Investment.__init__(self, initial, growth)
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
        self.growth = growth/100
        self.monthly_growth = 10**(np.log10(1 + self.growth)/12) - 1
        self.retirement_date = pd.datetime(self.dob.year + era, self.dob.month, self.dob.day)
        self.first_retirement_date = self.df.loc[self.df.index>=self.retirement_date].index[0]
        self.last_working_date = self.df.loc[self.df.index<self.retirement_date].index[-1]

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
        contr = contributions/installments
        withdr = withdrawals/installments
        withdrawal_cg = 0
        withdrawal_cg_incr = 0
        for i in range(0, installments):
            if capital_calc > 0:
                capital_gains_calc = capital_gains_calc + capital*monthly_growth
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

        withdr_total = 0
        withdrawal_cg = 0
        capital_gains_calc = 0
        #contr = contributions/installments
        withdr = withdrawals/installments
        capital = capital*(1 + monthly_growth) - withdr # + contr
        withdr_total += withdr
        print('withdrawals', withdrawals)
        while capital > 0:
            capital_gains_calc = capital_gains_calc + capital*monthly_growth
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
                
            '''
            if self.df.loc[year, 'withdrawals'] <= self.df.loc[year, 'capital']:
                self.df.loc[year, 'capital'] -=  self.df.loc[year, 'withdrawals']             
                self.df.loc[year, 'capital_gain'] = self.df.loc[previous_year, 'capital_gain'] + self.df.loc[previous_year, 'capital']*(self.growth)
                if self.df.loc[year, 'capital'] > 0:
                    self.df.loc[year, 'withdrawal_cg'] = self.df.loc[year, 'withdrawals']*(self.df.loc[year, 'capital_gain']/self.df.loc[year, 'capital'])
                    self.df.loc[year, 'capital_gain'] -= self.df.loc[year, 'withdrawal_cg']
                else:
                    self.df.loc[year, 'withdrawal_cg'] = 0
            else:
                self.df.loc[year, 'withdrawals'] = self.df.loc[year, 'capital']
                self.df.loc[year, 'withdrawal_cg'] = self.df.loc[year, 'capital_gain']
                self.df.loc[year, 'capital'] = 0
            '''
            previous_year = year  
            
#%% Portfolio

p = Portfolio(dob='1987-02-05',
              ibt=27375*12,
              expenses=16000*12,
              ma_dependents=2,
              medical_expenses=12000,
              era=65,
              le=95,
              strategy='optimal')

tfsa = TFSA(initial=100000,
            growth=7,
            ytd=0,
            ctd=10000,
            dob='1987-02-05',
            era=65,
            le=95)

ra = RA(initial=50000,
        ra_growth=7,
        la_growth=5,
        ytd=2500,
        dob='1987-02-05',
        le=95,
        era=65,
        payout_fraction=0)

di = DI(initial=10000,
        growth=5,
        dob='1987-02-05',
        era=65,
        le=95)

contr_TFSA = pd.Series(index=tfsa.df.index, name='contr',
                       data=33000*np.ones(tfsa.df.shape[0]))
contr_DI = pd.Series(index=tfsa.df.index, name='contr',
                     data=5000*np.ones(tfsa.df.shape[0]))
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
self.sc1 = scenario.reshape(scenario.size)
#cons = ({'type': 'eq', 'fun': lambda x: self.constraint(x, i)},)

self.res = spm.minimize(self.objective,
            method='TNC',
           x0=scenario.reshape(scenario.size),
           bounds=bounds)


solution_1d = self.res.x
'''