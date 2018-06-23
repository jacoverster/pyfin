#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:53:46 2018

@author: herman
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import time

class Portfolio(object):
    
    def __init__(self,
                 dob,
                 ibt,
                 expenses,
                 ma_dependents,
                 medical_expenses,
                 era,
                 le):
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
        import scipy.optimize as spm
        self.savable_income = self.taxable_ibt*0.725 - self.expenses
        
        self.withdrawals = pd.DataFrame(np.zeros([self.df.shape[0],
                                                     len(self.investments)]),
                                        index=self.df.index,
                                        columns=np.arange(0, len(p.investments)))
            
        self.contr = pd.DataFrame(np.zeros_like(self.withdrawals),
                                     index=self.df.index,
                                     columns=np.arange(0, len(p.investments)))
        '''
        for count, i in enumerate(self.investments.keys()):
            self.contr.loc[:self.last_working_date, count] = self.savable_income/self.size
            self.withdrawals.loc[self.first_retirement_date::, count] = self.taxable_ibt/self.size/3
        
        
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
        self.pop_size = 100
        self.ngen = 50
        self.GA()
        solution_1d = np.array(self.best_ind)
        self.solution = solution_1d.reshape(int(solution_1d.size/(self.size*2)), self.size*2)

        for count, i in enumerate(self.investments.keys()):
            self.contr.loc[:, count] = self.solution[:, count]
            self.withdrawals.loc[:, count + self.size] = self.solution[:, count*2]
            self.contr.rename(columns={count:i}, inplace=True)
            self.withdrawals.rename(columns={count:i}, inplace=True)
            
            self.investments[i].calculate(self.contr[i], self.withdrawals[i])
            
        self.calculate()
        self.plot()
        
        #  Variable bounds: 
        #  [0, savable_income]
        #  Make withdrawals zero during working time and contr zero during retirement.
        #  Constraints: 
        #  sum of all variables = savable income.
    '''    
    Removed because the constraint is recursive. Rather added penalty to objective function.
    def constraint(self, x, i):
        if i<=self.size:
            return x.reshape(int(x.size/(self.size*2)), self.size*2)[:, 0:self.size].sum(axis=1)[i] - self.savable_income
        else:
            return 0
    '''   
        
    def objective(self, individual):
        '''
        Objective Function for optimization. 
        ------
        Parameters:
        scenario_1d:    ndarray. 1D Numpy array. This is a reshaped form of an
                        array of dimension [working months + retirement months, 2*portfolio size]
                        where the first half of the columns are contributions
                        and the second half of the columns are withdrawals.
        '''
        #  Reshape array:
        scenario_1d = np.array(individual)
        scenario = scenario_1d.reshape(int(scenario_1d.size/(self.size*2)), self.size*2)
        #di = 
        #np.insert(scenario, 1, di, axis=1)
        self.contr.loc[:, self.contr.columns] = scenario[:, 0:self.size]
        self.withdrawals.loc[:, self.withdrawals.columns] = scenario[:, self.size:self.size*2]
        for count, i in enumerate(self.investments.keys()):
            self.investments[i].calculate(self.contr.loc[:, count], self.withdrawals.loc[:, count])
        
        self.calculate()
        
        #  Penalise IAT==0 in any year
        self.df.loc[self.df['iat']==0, 'iat'] = -self.taxable_ibt*100
        #  Penalise algorithm by the difference between max and min iat during retirement:
        
        penalty_chebychev = self.df.loc[self.first_retirement_date::, 'iat'].max() - self.df.loc[self.first_retirement_date::, 'iat'].min()
        #print('Chebychev', penalty_chebychev)
        #  Penalise algorithm by if sum of contributions are larger than savable income:
        saved_surplus = self.df.loc[:self.last_working_date, 'iat'] - self.expenses - self.df.loc[:self.last_working_date, 'contr_total']
        penalty_oversaved = saved_surplus.loc[saved_surplus<0].sum()
        #  Penalise the algorithm for leaving money in the accounts at death:
        penalty_left_over = self.df.loc[self.df.index[-1], ['capital_RA', 'capital_TFSA', 'capital_DI']].sum()
        #print('Saved surplus', penalty_oversaved)
        #print(-self.df.loc[self.retirement_date::, 'iat'].sum() + penalty_chebychev + penalty_oversaved + penalty_left_over)
        penalty_stdev = self.df.loc[self.first_retirement_date::, 'iat'].std()
        #return round(-self.df.loc[self.retirement_date::, 'iat'].sum() + penalty_chebychev + penalty_oversaved + penalty_left_over + penalty_stdev, 2)
        return round(-self.df.loc[self.retirement_date::, 'iat'].mean() + penalty_chebychev + penalty_stdev, 2)

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
                self.investments[i].recalculate()
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
            if s.contr_RA <= 0.275*s.taxable_ibt:
                taxable_income = s.taxable_ibt + 0.18*max(0, s.capital_gains - 40000) - s.contr_RA
            else:
                taxable_income = s.taxable_ibt - s.taxable_ibt*0.275 + 0.18*max(0, s.capital_gains - 40000)
                
        if s.name >= self.retirement_date:
            if age < 65:
                taxable_income = max(0, s.taxable_ibt + 0.18*max(0, s.capital_gains - 40000) - 78150)
            elif age < 75:
                taxable_income = max(0, s.taxable_ibt + 0.18*max(0, s.capital_gains - 40000) - 121000)
            else:
                taxable_income = max(0, s.taxable_ibt + 0.18*max(0, s.capital_gains - 40000) - 135300)                
        
        #  Income tax:       
        #print('taxable income:', taxable_income)
        if taxable_income <= 78150:
            tax = 0
        if taxable_income <= 195850:
             tax =  0.18*taxable_income
        elif taxable_income <= 305850:
             tax =  35253 + (taxable_income - 195850)*0.26
        elif taxable_income <= 423300:
             tax =  63853 + (taxable_income - 305850)*0.31
        elif taxable_income <= 555600:
             tax =  100263 + (taxable_income - 305850)*0.36
        elif taxable_income <= 708310:
             tax =  147891 + (taxable_income - 555600)*0.39
        elif taxable_income <= 1500000:
             tax =  207448 + (taxable_income - 708310)*0.41
        elif taxable_income >=1500000:
             tax =  532041 + (taxable_income - 1500000)*0.45
        
        self.taxCreditMa(s.medical_expenses, taxable_income, age)    
        return max(0, tax - self.tax_credit_ma)
    
    
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

                
    def taxableIncomeRetirement(self, income, capital_gains, age):
        '''
        Calculates taxable income after retirement
        '''
        
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
        plt.plot(index, self.df['contr_RA'], label='RA')
        plt.plot(index, self.df['contr_DI'], label='DI')
        plt.plot(index, self.df['contr_TFSA'], label='TFSA')
        plt.xlabel('Dates')
        plt.ylabel('Amount [R]')
        plt.xticks(rotation=90)
        plt.legend()        
        plt.title('Contributions while working')
        
        plt.figure(3)
        plt.plot(index, self.df['capital_RA'], label='RA')
        plt.plot(index, self.df['capital_DI'], label='DI')
        plt.plot(index, self.df['capital_TFSA'], label='TFSA')
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
        
        return self.objective(scenario_1d),
        
    def initIndividual(self, icls, content):
        
        return icls(content)

    def initPopulation(self, pcls, ind_init):
        
        ind_list = [np.ones([self.number_working_years + self.number_retirement_years,
                             self.size*2]) for i in range(self.pop_size)]
        
        #  Generate random contribution and withdrawal matrices
        ind_list=[0]*self.pop_size
        contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
        withdrawals = np.zeros_like(contr)
        
        for j in range(50):
            contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
            withdrawals = np.zeros_like(contr)
            
            for i in range(self.number_working_years):
                a = np.random.random(self.size - 1)
                while sum(a) > 1:
                    a = np.random.random(self.size - 1)
                contr[i, :] = self.savable_income*np.concatenate([a, (1 - sum(a), )])
            
            for i in np.arange(self.number_working_years, self.number_working_years + self.number_retirement_years):
                a = np.random.random(self.size - 1)
                while sum(a) > 1:
                    a = np.random.random(self.size - 1)
                withdrawals[i, :] = self.savable_income/3*np.concatenate([a, (1 - sum(a), )])

            overall = np.concatenate([contr, withdrawals], axis=1)
            overall = overall.reshape(overall.size)
            ind_list[j] = overall

        
        for count, j in enumerate(np.arange(len(ind_list), len(ind_list) + 10)):
            contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
            withdrawals = np.zeros_like(contr)
            
            for i in range(self.number_working_years):
                contr[i, :] = self.savable_income/3*np.array([1]*self.size)
            
            for i in np.arange(self.number_working_years, self.number_working_years + self.number_retirement_years):
                withdrawals[i, :] = self.savable_income*(count/10)*np.array([1]*self.size)

            
            overall = np.concatenate([contr, withdrawals], axis=1)
            overall = overall.reshape(overall.size)
            ind_list += [overall]
            
        for j in range(10):
            contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
            withdrawals = np.zeros_like(contr)
            
            a = np.random.random(self.size - 1)
            while sum(a) > 1:
                a = np.random.random(self.size - 1)
            contr[::self.number_working_years, :] = self.savable_income*np.concatenate([a, (1 - sum(a), )])
        
            a = np.random.random(self.size - 1)
            while sum(a) > 1:
                a = np.random.random(self.size - 1)
            withdrawals[self.number_working_years::, :] = self.savable_income*np.concatenate([a, (1 - sum(a), )])

            overall = np.concatenate([contr, withdrawals], axis=1)
            overall = overall.reshape(overall.size)
            ind_list += [overall]
        
        
        for count, j in enumerate(np.arange(len(ind_list), len(ind_list) + 10)):
            contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
            withdrawals = np.zeros_like(contr)
            
            for i in range(self.number_working_years):
                contr[i, :] = self.savable_income/3*np.array([1]*self.size)
            
            for i in np.arange(self.number_working_years, self.number_working_years + self.number_retirement_years):
                withdrawals[i, :] = self.savable_income*(count/10)*np.array([1]*self.size)

            
            overall = np.concatenate([contr, withdrawals], axis=1)
            overall = overall.reshape(overall.size)
            ind_list += [overall]
            
        for j in range(self.size):
            contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
            withdrawals = np.zeros_like(contr)
            
            for i in range(self.number_working_years):
                distr = self.savable_income*np.array([1]*self.size)
                distr[j] = 0
                contr[i, :] = distr
            
            for i in np.arange(self.number_working_years, self.number_working_years + self.number_retirement_years):
                distr = self.savable_income*np.array([1]*self.size)
                distr[min(self.size - 1, self.size - j)] = 0
                withdrawals[i, :] = self.savable_income/3*(count/10)*np.array([1]*self.size)

            overall = np.concatenate([contr, withdrawals], axis=1)
            overall = overall.reshape(overall.size)
            ind_list += [overall]

        for j in range(self.size - 1):
            contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
            withdrawals = np.zeros_like(contr)
            
            for i in range(self.number_working_years):
                distr = self.savable_income*np.array([0]*self.size)
                distr[j] = 1
                contr[i, :] = distr
            
            for i in np.arange(self.number_working_years, self.number_working_years + self.number_retirement_years):
                distr = self.savable_income*np.array([0]*self.size)
                distr[min(self.size - 1, self.size - j)] = 1
                withdrawals[i, :] = self.savable_income/3*(count/10)*np.array([1]*self.size)
            
            overall = np.concatenate([contr, withdrawals], axis=1)
            overall = overall.reshape(overall.size)
            ind_list += [overall]
        
        return (pcls(ind_init(i) for i in ind_list))
        
    def GA(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        toolbox.register("individual_guess", self.initIndividual, creator.Individual)
        toolbox.register("population_guess", self.initPopulation, list, toolbox.individual_guess)
        
        population = toolbox.population_guess()

        toolbox.register("evaluate", self.fitness)
        toolbox.register("mate", tools.cxUniform, indpb=0.7)
        toolbox.register("mutate", self.mutatePyfin, indpb=1, number_changed=3)
        toolbox.register("select", tools.selTournament, tournsize=3)

        def main():
            pop = toolbox.population_guess()        
            hof = tools.HallOfFame(maxsize=20)        
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)        
            MU = int(self.pop_size)
            LAMBDA = int(self.pop_size)
            CXPB = 0.5
            MUTPB = 0.5
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
        plt.figure(1)
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
        
        for i in range(number_changed):
            if indpb > np.random.random():
                    position = np.random.randint(0, len(individual))
                    individual[position] *= np.random.uniform(0, 2) 

        return individual,
        
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
        
class TFSA(Investment):
    
    def __init__(self,
                 initial,
                 growth,
                 dob,
                 ytd,
                 ctd,
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
                                        'capital_gains',
                                        'withdrawals',
                                        'contr'])
            
        self.df.loc[:, ['capital',
                        'YTD contr',
                        'Total contr',
                        'capital_gains',
                        'withdrawals',
                        'contr']] = 0
    
        self.df.loc[self.df.index[0], 'capital'] = self.initial
        self.df.loc[self.df.index[0], 'YTD contr'] = self.ytd
        self.df.loc[self.df.index[0], 'Total contr'] = self.ctd  
        
        self.df.loc[:, 'capital_gains'] = 0
        self.growth = growth/100
        
    def calculate(self, contr, withdrawals):
        
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
        self.df.loc[:, 'contr'] = contr    
        self.df.loc[:, 'withdrawals'] = withdrawals
        previous_year=self.df.index[0]        
        
        self.df.loc[:, ['capital',
                'YTD contr',
                'Total contr',
                'capital_gains',
                'withdrawals',
                'contr']] = 0
        
        for count, year in enumerate(self.df.index):
        
            self.df.loc[self.df.index[0], 'capital'] = self.initial
            self.df.loc[self.df.index[0], 'YTD contr'] = self.ytd
            self.df.loc[self.df.index[0], 'Total contr'] = self.ctd 
                
            if count == 0:
                self.df.loc[year, 'capital'] += self.df.loc[year, 'contr']                
                self.df.loc[year, 'Total contr'] = self.df.loc[year, 'Total contr'] + self.df.loc[year, 'contr']

            elif count > 0:
                #  Check for violation of TFSA terms, and add tax by subtracting 40%.
                current_margin = max(0, 33000 - self.df.loc[previous_year, 'YTD contr'])
                if (self.df.loc[previous_year, 'Total contr'] > 500000 and self.df.loc[year, 'contr'] >0) or (year.year!=3 and self.df.loc[year, 'contr'] + current_margin > 33000 and self.df.loc[year, 'contr']>0):
                    self.df.loc[year, 'contr'] = 0.6*(self.df.loc[year, 'contr'] - current_margin) + current_margin

                self.df.loc[year, 'capital'] = self.df.loc[previous_year, 'capital']*(1 + self.growth)
                self.df.loc[year, 'capital'] += self.df.loc[year, 'contr']              

                #  Check if there is enough money in the account
                self.df.loc[year, 'Total contr'] = self.df.loc[previous_year, 'Total contr'] + self.df.loc[year, 'contr']
                if self.df.loc[year, 'withdrawals'] <= self.df.loc[year, 'capital']:
                    self.df.loc[year, 'capital'] -= self.df.loc[year, 'withdrawals']
                else:
                    self.df.loc[year, 'withdrawals'] = self.df.loc[year, 'capital']
                    self.df.loc[year, 'capital'] = 0

            previous_year = year
       
class RA(Investment):

    #  TODO: Add fees
    
    def __init__(self,
                 initial,
                 ra_growth,
                 la_growth,
                 dob,
                 era,
                 le,
                 ytd):
        
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
        
    def calculate(self, contr, withdrawals):
        self.growthBeforeRetirement(contr, withdrawals)
        self.growthAfterRetirement(contr, withdrawals)
        
    def growthBeforeRetirement(self, contr, withdrawals):
        
        previous_year = self.df.index[0]
        self.df['contr'] = contr    
        self.df['withdrawals'] = withdrawals
        
        for count, year in enumerate(self.df.loc[self.df.index<self.retirement_date].index):
            if count > 0:
                self.df.loc[year, 'capital'] = self.df.loc[previous_year, 'capital']*(1 + self.growth) + self.df.loc[year, 'contr']
                if year.year == 3:
                    self.df.loc[year, 'YTD contr'] = self.df.loc[year, 'contr']
                else:
                    self.df.loc[year, 'YTD contr'] = self.df.loc[previous_year, 'YTD contr'] + self.df.loc[year, 'contr']
            self.last_ra_year = year
            previous_year = year
        capital_at_retirement = self.df.loc[self.last_ra_year, 'capital']
        self.payout = capital_at_retirement/3
        #  Build in functionality so that it can be reinvested into LA or into
        #  DI. 
        #  Also, CG needs to be taxed with income tax in that year.
        #  Also,  
        self.reinvestment = capital_at_retirement*(2/3)
        
    def growthAfterRetirement(self, contr, withdrawals):
        
        '''
        Calculates growth of a retirement annuity
        ------
        Parameters:
        contr:          DataFrame. Dataframe, indexed by year from today to 
                        retirement age, with contributions.
        withdrawals:    DataFrame. Dataframe, indexed by year from today to 
                        retirement age, with withdrawls.
        '''
        previous_year = self.last_ra_year
        self.df['contr'] = contr    
        self.df['withdrawals'] = withdrawals        
        
        for count, year in enumerate(self.df.loc[self.df.index>=self.retirement_date].index):
            if count == 0:
                self.df.loc[year, 'capital'] = self.reinvestment
            
            if count > 0:
                self.df.loc[year, 'capital'] = self.df.loc[previous_year, 'capital']*(1 + self.growth)
                if self.df.loc[year, 'withdrawals'] <= self.df.loc[year, 'capital']:
                    if self.df.loc[year, 'withdrawals'] < 0.025*self.df.loc[year, 'capital']:
                        #print('Adjusting RA withdrawals: too low.')
                        self.df.loc[year, 'withdrawals'] = 0.025*self.df.loc[year, 'capital']
                    elif self.df.loc[year, 'withdrawals'] > 0.175*self.df.loc[year, 'capital']:
                        #print('Adjusting withdrawals: too high')
                        self.df.loc[year, 'withdrawals'] = 0.175*self.df.loc[year, 'capital']
                    self.df.loc[year, 'capital'] = max(0, self.df.loc[previous_year, 'capital']*(1 + self.growth) - self.df.loc[year, 'withdrawals'])
                    self.df.loc[year, 'withdrawals'] = self.df.loc[year, 'withdrawals']
                else:
                    self.df.loc[year, 'withdrawals'] = 0
                    self.df.loc[year, 'capital'] = 0

            previous_year = year 
            
        
    '''        
    def lumpSumWithdrawal(self, amount, date):
        if self.df.capital.loc[date] < 247500:
            
        elif amount > (1/3)*self.df.capital.loc[date]:
                self.df.capital.loc[date] -= (1/3)*self.df.capital.loc[date]
                return (1/3)*self.df.capital.loc[date]
        else:
            self.df.capital.loc[date] -= amount
            return amount
     '''      
class DI(Investment):
    '''
    Discretionary Investment object.
    '''
    #  TODO: Add fees
    
    def __init__(self, initial, growth, dob, le):
        
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
        
    def calculate(self, contr, withdrawals):
        
        '''
        Calculates growth of a retirement annuity
        ------
        Parameters:
        contr:  DataFrame. Dataframe, indexed by year from today to 
                        retirement age, with contr.
        withdrawals:    DataFrame. Dataframe, indexed by year from today to 
                        retirement age, with withdrawls.
        '''
        self.df.loc[:, 'contr'] = contr    
        self.df.loc[:, 'withdrawals'] = withdrawals
        self.recalculate()

    def recalculate(self):
        previous_year=self.df.index[0]
        for count, year in enumerate(self.df.index):
            if count > 0:
                self.df.loc[year, 'capital'] = self.df.loc[previous_year, 'capital']*(1 + self.growth)
                self.df.loc[year, 'capital'] += self.df.loc[year, 'contr'] - self.df.loc[year, 'withdrawals']             
                self.df.loc[year, 'capital_gain'] = self.df.loc[previous_year, 'capital_gain'] + self.df.loc[previous_year, 'capital']*(self.growth)
                if self.df.loc[year, 'withdrawals'] <= self.df.loc[year, 'capital']:
                    if self.df.loc[year, 'capital'] > 0:
                        self.df.loc[year, 'withdrawal_cg'] = self.df.loc[year, 'withdrawals']*(self.df.loc[year, 'capital_gain']/self.df.loc[year, 'capital'])
                        self.df.loc[year, 'capital_gain'] -= self.df.loc[year, 'withdrawal_cg']
                    else:
                        self.df.loc[year, 'withdrawal_cg'] = 0
                else:
                    self.df.loc[year, 'withdrawal'] = self.df.loc[year, 'capital']
                    self.df.loc[year, 'capital'] = 0
            previous_year = year  


#%% Portfolio

p = Portfolio(dob='1987-02-05',
              ibt=60000*12,
              expenses=24000*12,
              ma_dependents=2,
              medical_expenses=12000,
              era=65,
              le=95)
tfsa = TFSA(initial=10000,
            growth=7,
            ytd=0,
            ctd=10000,
            dob='1987-02-05',
            le=95)

ra = RA(initial=16000,
        ra_growth=5,
        la_growth=5,
        ytd=2500,
        dob='1987-02-05',
        le=95,
        era=65)

di = DI(initial=10000,
        growth=7,
        dob='1987-02-05',
        le=95)

contr_TFSA = pd.Series(index=tfsa.df.index, name='contr',
                       data=2500*np.ones(tfsa.df.shape[0]))
contr_DI = pd.Series(index=tfsa.df.index, name='contr',
                     data=5000*np.ones(tfsa.df.shape[0]))
contr_RA = pd.Series(index=tfsa.df.index, name='contr',
                     data=16500*np.ones(tfsa.df.shape[0]))
withdrawals_TFSA = pd.Series(index=tfsa.df.index,
                        name='withdrawals',
                        data=12000*np.ones(tfsa.df.shape[0]))
withdrawals_DI = pd.Series(index=tfsa.df.index,
                        name='withdrawals',
                        data=50000*np.ones(tfsa.df.shape[0]))
withdrawals_RA = pd.Series(index=tfsa.df.index,
                        name='withdrawals',
                        data=7000*np.ones(tfsa.df.shape[0]))

contr_TFSA.iloc[200::] = 0
contr_DI.loc[p.retirement_date::] = 0
contr_RA.loc[p.retirement_date::] = 0

withdrawals_DI.loc[p.df.index[0]:p.retirement_date] = 0
withdrawals_RA.loc[p.df.index[0]:p.retirement_date] = 0
withdrawals_TFSA.loc[p.df.index[0]:p.retirement_date] = 0


ra.calculate(contr_RA, withdrawals_RA)
tfsa.calculate(contr_TFSA, withdrawals_TFSA)
di.calculate(contr_DI, withdrawals_DI)

p.addInvestment('RA', ra)
p.addInvestment('DI', di)
p.addInvestment('TFSA', tfsa)
p.calculate()

df_di = di.df
df_ra = ra.df
df_tfsa = tfsa.df
df_p = p.df

p.optimize()
df_p = p.df
print('Average monthly IAT during retirement:', round(p.df.loc[p.first_retirement_date::, 'iat'].mean()/12))
