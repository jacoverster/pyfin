#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:53:46 2018

@author: Invoke Analytics
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import time
import pyswarms as ps
from abc import ABCMeta, abstractmethod
from copy import copy

class TaxableEntity(metaclass=ABCMeta):
    
    
    @abstractmethod
    def __init__(self, person=None):
        
        '''
        Class to define tax constants in one place.
        '''
        
        #  Tax law constants:
        self.TFSA_ANN_LIMIT = 33000
        self.TFSA_ANN_LIMIT_EXC_TAX = 0.4
        self.TFSA_TOTAL_LIMIT = 500000
        self.CGTRA_LIM1 = 500000
        self.CGTRA_LIM18 = 700000
        self.CGTRA_CONST18 = 36000
        self.CGTRA_LIM36 = 1050000
        self.CGTRA_CONST_36 = 130500
        self.RA_MAX_WITHDR = 0.175
        self.RA_MIN_WITHDR = 0.025
        self.RA_MAX_PERC = 0.275
        self.RA_ANN_LIMIT = 350000
        self.TAX_0THRESH = 78150
        self.TAX_18PERC_THRESH = 195850
        self.TAX_26PERC_THRESH = 305850
        self.TAX_31PERC_THRESH = 423300
        self.TAX_36PERC_THRESH = 555600
        self.TAX_39PERC_THRESH = 708310
        self.TAX_41PERC_THRESH = 1500000
        self.TAX_26CONST = 35253
        self.TAX_31CONST = 63853
        self.TAX_36CONST = 100263
        self.TAX_39CONST = 147891
        self.TAX_41CONST = 207448
        self.TAX_45CONST = 532041
        self.REBATE_UNDER65 = 14067
        self.REBATE6575 = 7713
        self.REBATE_75PLUS = 2574
        self.MA_CREDIT_LE2 = 310
        self.MA_CREDIT_G2 = 209
        self.MA_INCL_RATE_GE65 = 0.33
        self.MA_INCL_RATE = 0.25
        self.MA_TAXABLE_INC_PERC = 0.075
        self.CGT_6575 = 121000
        self.CGT_GE75 = 135300
        self.CGT_INCL_RATE = 0.4
        self.CGT_EXEMPTION= 40000   
        #>>> Remember to change UIF_MONTHLY in person when updating constants.
        
        self.RA_MIN_WITHDR_MONTHLY = 10**(np.log10(1 + self.RA_MIN_WITHDR)/12) - 1
        self.RA_MAX_WITHDR_MONTHLY = 10**(np.log10(1 + self.RA_MAX_WITHDR)/12) - 1
        
        #  If person is passed, create some extra infrastructure:
        if person:
            #  copy all the attrs of person to self:
            self.__dict__.update(**person.__dict__)
            
            self.df = pd.DataFrame(index=pd.DatetimeIndex(start=pd.datetime.today().date(),
                                                          end=pd.datetime(self.dob.year + self.le, self.dob.month, self.dob.day),
                                                          freq='A-FEB'),
                                   columns=['capital',
                                            'withdrawals',
                                            'contr'])     
                
            self.df.loc[:, 'date'] = self.df.index.copy()
            self.this_year = self.df.index[0]
            self.last_working_year = self.df.loc[self.df.index<self.retirement_date].index[-1]
            
            if self.retirement_date.month >= 3: #  if your birthday is in the new FY, move up by a year.
                self.retirement_fy_end = self.df.loc[self.df.index>=self.retirement_date].index[1]
                self.first_fy_after_retirement = self.df.loc[self.df.index>=self.retirement_date].index[2]
                self.ret_year_installments = 15 - self.retirement_date.month
            else: #  if your birthday is in Jan or Feb, keep FY.
                 self.retirement_fy_end = self.df.loc[self.df.index>=self.retirement_date].index[0]
                 self.first_fy_after_retirement = self.df.loc[self.df.index>=self.retirement_date].index[1]
                 self.ret_year_installments = 3 - self.retirement_date.month
            self.number_working_years = self.df.loc[self.df.index<self.retirement_date].shape[0]
            self.number_retirement_years = self.df.loc[self.df.index>=self.retirement_date].shape[0]
       
class Person(TaxableEntity):
    
    
    def __init__(self,
                 dob,
                 ibt,
                 expenses,
                 ma_dependants,
                 medical_expenses,
                 monthly_med_aid_contr,
                 era=65,
                 le=95,
                 strategy='optimal',
                 inflation=5.5,
                 uif=True):
        
        '''
        Person class, with data about a person. Creates dataframe around these
        personal attributes. Person is a natural person, not a company etc.
        
        ------
        Parameters:
        dob:                    str. Date of Birth, in format "YYYY-MM-DD"
        ibt:                    int. Monthly income before tax
        expenses:               float. Expenses before tax, monthly
        ma_dependants:          int. Number of medical aid dependants, including self.
        medical_expenses:       float. Annual out-of-pocket medical expenses
        monthly_med_aid_contr:  float. Monthly contribution to medical aid.
        era:                    int. Expected Retirement Age.
        le:                     int. life expectancy.
        uif:                    bool. Whether Unemployment Insurance Fund contributions are applicable.
        strategy:               str. Whether to follow the 'safe' or the 'optimal' withdrawal strategy.
        inflation:              float. inflation rate in percentage points. e.g. 5 = 5% inflation
        uif:                    float. Whether or not to consider Unemployment Insurance Fund contrs
        '''
        
        self.UIF_MONTHLY = 148.72        
        
        self.ibt = ibt
        self.dob = pd.to_datetime(dob).date()
        self.taxable_ibt = ibt*12
        self.expenses = expenses*12
        self.ma_dependants = ma_dependants
        self.era = era
        self.le = le
        self.medical_expenses = medical_expenses
        self.age = pd.datetime.today().date() - self.dob
        self.retirement_date = pd.datetime(self.dob.year + era, self.dob.month, self.dob.day)
        if pd.datetime.today() > self.retirement_date:
            raise AttributeError('This calculator only works pre-retirement. You have specified a retirement date in the past.')
        
        self.strategy = strategy
        self.inflation = inflation/100
        self.monthly_med_aid_contr = monthly_med_aid_contr
        
        self.uif_contr = 0
        if uif == True:
            self.uif_contr = min(self.UIF_MONTHLY*12, 0.01*ibt*12)

    def __repr__(self):
        
        return f'''{self.__class__.__name__}
                Date of birth: {self.dob}
                Age: {self.age}
                Income before tax: {self.ibt}
                Expenses: {self.expenses}
                
                Medical Aid dependants: {self.ma_dependants}
                Monthly Medical Aid Contribution: {self.monthly_med_aid_contr}
                Expected Retirement Age: {self.era}
                
                Life expectancy: {self.le}
                Strategy: {self.strategy}
                Inflation: {self.inflation}
                '''
    
class Portfolio(TaxableEntity):
        
    def __init__(self, person):
       
        '''
        Portfolio class creates the infrastructure for a collection of investments.
        '''
        
        super().__init__(person)
        self.person=person

        self.investments = {}
        self.ra_list = []
        self.tfsa_list = []
        self.di_list = []
        self.investment_names = []
        self.ra_payout_fracs = []
        self.max_ra_growth = 0
        self.max_tfsa_growth = 0
        self.max_di_growth = 0
        self.max_ra_name = ''
        self.max_tfsa_name = ''
        self.max_di_name = ''        
        self.size = 0
        self.ra_payouts = 0

        #  create dataframe according to metadata
        self.df = pd.DataFrame(index=pd.DatetimeIndex(start=pd.datetime.today().date(),
                                                      end=pd.datetime(person.dob.year + self.le, self.dob.month, self.dob.day),
                                                      freq='A-FEB'),
                                columns=['taxable_ibt',
                                         'expenses',
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
                                        'savable_iat',
                                        'medical_expenses',
                                        'monthly_med_aid_contr',
                                        'ma_dependants'])

        self.df.loc[:, ['taxable_ibt',
                        'expenses',
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
                        'iat',
                        'medical_expenses',
                        'monthly_med_aid_contr',
                        'ma_dependants']] = 0
        
        self.df.loc[:, 'medical_expenses'] = self.medical_expenses


        try:
            self.readInputExcel()
        except FileNotFoundError:
            self.df.loc[:self.first_fy_after_retirement, 'taxable_ibt'] = person.taxable_ibt
            
        self.df.loc[:, 'age'] = (self.df.index - pd.Timestamp(self.dob)).days/365.25
        self.df.loc[:self.first_fy_after_retirement, 'iat'] -= self.uif_contr

    def __repr__(self):
        return f'''{self.__class__.__name__}
    Person: {self.person}
    Investments: {self.investments}
    '''

    def addInvestment(self, name, investment):  
        
        '''
        Adds an investment object to the portfolio. Saved to a dictionary under
        the key 'name'.
        ------
        Parameters:
        name:           str or list. Name of investment.
        investment:     obj. Investment object. Can be an RA, TFSA, or DI.
        '''        
     
        if isinstance(name, str):
            self.investments[name] = investment
            self.investment_names += [name]
            self.size +=1
            if investment.type == 'RA':
                self.ra_list += [name]
                self.ra_payout_fracs += [investment.payout_fraction]
                if (self.number_working_years*investment.ra_growth + 
                    self.number_retirement_years*investment.la_growth)/self.df.shape[0] > self.max_ra_growth:
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
    
    def genInputExcel(self):
        df = pd.DataFrame(index=self.df.index.year,
                          columns=['Monthly IBT',
                                   'Monthly Expenses',
                                   'Monthly medical aid contribution',	
                                   'Medical Aid dependants'])
            
        df.index.rename('Financial year', inplace=True)
        df.iloc[0] = [self.taxable_ibt,
               self.expenses/12,
               self.monthly_med_aid_contr,
               self.ma_dependants]
        df.to_excel('input.xlsx')
        
    def readInputExcel(self):
        df_input = pd.read_excel('input.xlsx',
                           index_col=0)
        
        df_input.index = self.df.index.copy()        
        if (df_input.loc[:self.retirement_fy_end, 'Monthly IBT'] < df_input.loc[:self.retirement_fy_end, 'Monthly Expenses']).any():
            raise AttributeError('Income smaller than expenses')
        self.df.loc[:, 'taxable_ibt'] = df_input['Monthly IBT']*12
        self.df.loc[:, 'expenses'] = df_input['Monthly Expenses']*12
        self.df.loc[:, 'monthly_med_aid_contr'] = df_input['Monthly medical aid contribution']*12
        self.df.loc[:, 'ma_dependants'] = df_input['Medical Aid dependants']
                
    def optimize(self, reduced_expenses=False):
        
        '''
        Optimizes the investment allocations of the portfolio over time by using
        a Particle Swarm Optimization algorithm.
        
        This function contains two subfunctions: solutions, and determineOptimalSolution.
        This was done to decrease code repetition.
        '''
        
        def solutions():
            
            '''
            This function determines the solutions to the PSO, TFSA, and RA
            plans
            '''
        
            time1 = time.time()
            #import scipy.optimize as spm
            tfsa_priority_plan = self.calculateTaxEfficientTFSAFirstIAT()
            ra_priority_plan = self.calculateTaxEfficientRAFirstIAT()
    
              
            self.contr = pd.DataFrame(index=self.df.index,
                                         columns=self.investment_names)
    
                
            best_ind, cost = self.pso()
            print('Duration:', (time.time() - time1)/60, 'min')
            self.best_ind = best_ind
            solution, ra_payouts = self.fractionsToRands(self.reshape(best_ind)) 
            self.solution = solution
            for count, key in enumerate(self.investment_names):
                self.contr.loc[:, key] = solution[:, count]
                self.investments[key].calculateOptimalWithdrawal(self.contr[key], self.strategy)
                
            self.calculate()
            pso_plan = round(self.df.loc[self.retirement_fy_end:, 'iat'].mean()/12, 2)       
            return pso_plan, tfsa_priority_plan, ra_priority_plan            

        def determineOptimalPlan(plan_tup):
            
            '''
            This function evaluates the solutions from the PSO, TFSA, and RA
            plans, and selects the best one. It also assigns the best plan to
            the object's dataframe and plots the solution.
            '''
            
            pso_plan, tfsa_priority_plan, ra_priority_plan = plan_tup[0], plan_tup[1], plan_tup[2]
            if pso_plan > ra_priority_plan and pso_plan > tfsa_priority_plan:
                best_plan = pso_plan
                print('The PSO plan is the best, with a mean post-retirement IAT of R', pso_plan)
                print('RA:', ra_priority_plan, 'Percentage improvement:', round(100*(pso_plan/ra_priority_plan - 1), 2))
                print('TFSA:', tfsa_priority_plan, 'Percentage improvement:', round(100*(pso_plan/tfsa_priority_plan - 1), 2))
                self.plot()
            elif int(tfsa_priority_plan) == (tfsa_priority_plan):
                best_plan = tfsa_priority_plan
                print('The TFSA and RA priority plans are equal, with a mean post-retirement IAT of R', tfsa_priority_plan)
                solution, ra_payouts = self.fractionsToRands(self.reshape(self.taxEfficientPosTFSAFirst())) 
                self.solution = solution
                for count, key in enumerate(self.investment_names):
                    self.contr.loc[:, key] = solution[:, count]
                    self.investments[key].calculateOptimalWithdrawal(self.contr[key], self.strategy)
                self.calculate()
                self.plot()
            elif ra_priority_plan > pso_plan and ra_priority_plan > tfsa_priority_plan:
                best_plan = ra_priority_plan
                print('The RA priority plan is the best, with a mean post-retirement IAT of R', ra_priority_plan)
                print('PSO:', pso_plan)
                print('TFSA:', tfsa_priority_plan)            
                solution, ra_payouts = self.fractionsToRands(self.reshape(self.taxEfficientPositionRAFirst())) 
                self.solution = solution
                for count, key in enumerate(self.investment_names):
                    self.contr.loc[:, key] = solution[:, count]
                    self.investments[key].calculateOptimalWithdrawal(self.contr[key], self.strategy)
                self.calculate()
                self.plot()            
            elif tfsa_priority_plan > pso_plan and tfsa_priority_plan > ra_priority_plan:
                best_plan = tfsa_priority_plan
                print('The TFSA priority plan is the best, with a mean post-retirement IAT of R', tfsa_priority_plan)
                print('PSO:', pso_plan)
                print('RA:', tfsa_priority_plan)
                solution, ra_payouts = self.fractionsToRands(self.reshape(self.taxEfficientPosTFSAFirst())) 
                self.solution = solution
                for count, key in enumerate(self.investment_names):
                    self.contr.loc[:, key] = solution[:, count]
                    self.investments[key].calculateOptimalWithdrawal(self.contr[key], self.strategy)
                self.calculate()
                self.plot()  
            return best_plan
            
        expenses = 0
        if reduced_expenses:
            best_plan = determineOptimalPlan(solutions())            
            #  Now calculate with reduced expenses
            expenses = self.df.expenses
            self.df.loc[:, 'expenses'] = self.df.loc[:, 'expenses']*0.9
            # Recalculate at reduced expenses:
            best_plan_reduced = determineOptimalPlan(solutions())
            self.df.loc[:, 'expenses'] = expenses
            print(f'''If you reduce your expenses by 10%, you can increase your\
                  post-retirement income after tax by  {round((best_plan_reduced/best_plan - 1), 2)*100}%''')
        else:
            best_plan = determineOptimalPlan(solutions())


    def optimizeParams(self, params):
        
        '''
        Optimizes the investment allocations of the portfolio over time by using
        a Particle Swarm Optimization algorithm.
        
        This function contains two subfunctions: solutions, and determineOptimalSolution.
        This was done to decrease code repetition.
        '''
  
   
        time1 = time.time()
        #import scipy.optimize as spm
        tfsa_priority_plan = self.calculateTaxEfficientTFSAFirstIAT()
        ra_priority_plan = self.calculateTaxEfficientRAFirstIAT()

          
        self.contr = pd.DataFrame(index=self.df.index,
                                     columns=self.investment_names)

            
        best_ind, cost = self.pso(params)
        print('Duration:', (time.time() - time1)/60, 'min')
        self.best_ind = best_ind
        solution, ra_payouts = self.fractionsToRands(self.reshape(best_ind)) 
        self.solution = solution
        for count, key in enumerate(self.investment_names):
            self.contr.loc[:, key] = solution[:, count]
            self.investments[key].calculateOptimalWithdrawal(self.contr[key], self.strategy)
            
        self.calculate()
        pso_plan = round(self.df.loc[self.retirement_fy_end:, 'iat'].mean()/12, 2)       
        return -pso_plan

    #@numba.jit
    def reshape(self, pos):
        '''
        Reshapes 1D array of contributions for all investments during working
        years to a n_working_years x n_investments array. Uses Fortran-style
        indexing. That means that [1, 2, 3, 4, 5, 6] becomes
        [[1, 4],
        [2, 5],
        [3, 6]]
        instead of:
        [[1, 2]
        [3, 4]'
        [5, 6]]
        ------
        Parameters:
        pos:        ndarray. 1D Numpy array
        ------
        Returns:
        ndarray of shape n_working_years x n_investments, in Fortran indexing.
        '''
        if len(self.ra_list):
            arr_pos = np.array(pos[len(self.ra_list):]) # disregard RA lump sum withdrawal figures.
        else:
            arr_pos = np.array(pos[1:])
        if len(self.ra_list):
            return arr_pos.reshape(int(arr_pos.size/self.size), self.size, order='F'), pos[:len(self.ra_list)]
        else:
            return arr_pos.reshape(int(arr_pos.size/self.size), self.size, order='F'), None

    def NPV(self, amount, date):
        '''
        Nett present value conversion from future date to present value.
        ------
        Parameters:
        amount:     float. Amount to be converted.
        date:       datetime obj. Date in future (or past) from which to convert.
        '''
        n = date.year - self.this_year.year + 1
        return amount/(1 + self.inflation)**n
    
    def FV(self, amount, date):
        '''
        Future value conversion. Converts value in today's money to what it would
        be at some date in the future, considering inflation.
        ------
        amount:     float. Amount to be converted.
        date:       datetime obj. Date in future 
        ------
        Returns:
        Future value of present amount.
        '''
        
        n = date.year - self.this_year.year + 1
        return amount*(1 + self.inflation)**n    
            
    #@numba.jit
    def fractionsToRands(self, tup, verbose=False):
        
        '''
        converts fractions saved into Rands saved. It does so for all years at once.
        ------
        Parameters:     
        tup:                tuple. Tuple containing ind as the first element
                            and ra_payout_frac as the second.
                            ind: ndarray. Numpy array of size [self.number_working_years + number_retirement_years, self.size]
                            ra_payout_frac: list. List of payout fractions for all RAs
        ------
        Returns:        ndarray. Same shape as input. Just with Rand values.
        '''
        ind = tup[0]
        ra_payout_frac = tup[1]
        contr = np.zeros_like(ind) #  exclude RA payout fraction (first item in arr)
        if len(self.ra_list):
            ra_contr = self.df.taxable_ibt*ind[:, :len(self.ra_list)].sum(axis=1)
        else:
            ra_contr = np.array(np.zeros(self.df.shape[0]))
        tax = np.zeros(len(contr))
        for i, year in enumerate(self.df.index):
            taxSeries = pd.Series({'taxable_ibt': self.df.loc[year,'taxable_ibt'],
                            'contr_RA': ra_contr[i],
                            'capital_gains': 0,
                            'medical_expenses': self.df.loc[year, 'medical_expenses']})
            taxSeries.name = year
            tax[i] = self.totalTax(taxSeries)
            if verbose: 
                print('taxSeries', year, '\n', taxSeries, '\n',)
                print('Tax:', tax[i], '\n')
                      
        savable_income = np.maximum(0, self.df.taxable_ibt - ra_contr - tax - self.uif_contr - self.df.expenses)
        if verbose:
            print('Tax:\n', tax)
            print('savable income:\n', savable_income)
        #savable_income[self.number_working_years:] = np.maximum(0, savable_income[self.number_working_years] - self.uif_contr)
        mask = np.ones_like(savable_income)
        mask[savable_income <= 0] = 0
        self.ind = ind
        self.mask = mask
        contr[:, :len(self.ra_list)] = mask.reshape(-1,1)*self.df.loc[:, 'taxable_ibt'].values.reshape(-1,1)*ind[:, :len(self.ra_list)]
        contr[:, len(self.ra_list):] = savable_income[:, None]*np.array(ind[:, len(self.ra_list):])
        if len(self.ra_list) == 0 and ra_payout_frac is None:
            return contr, 0       
        elif len(self.ra_list) == len(ra_payout_frac):
            ra_payout = [0]*len(self.ra_list)
            
            for i, ra_name in enumerate(self.ra_list):
                self.investments[ra_name].growthBeforeRetirement(contr[:, i], ra_payout_frac[i])
                ra_payout[i] = self.investments[ra_name].df.loc[self.last_working_year, 'capital']*self.ra_payout_fracs[i]
            return contr, ra_payout
        else:
            raise AttributeError('RA list length {} does not match payout_frac length {}'.format(len(self.ra_list), len(ra_payout_frac)))
        

    def pso_objectiveSwarm(self, swarm):
        '''
        Objective Function for optimization. Calculates whole swarm at once.
        ------
        Parameters:
        swarm:    pyswarms swarm obj. 
        ------
        Returns:        float. The mean income after tax during the retirement 
                        period.                        
        '''
        results = np.inf*np.ones(swarm.position.shape[0])
        swarm = self.rebalancePSO(swarm)
        pos = swarm.position
        
        for i in range(pos.shape[0]):
            results[i] = self.pso_objective(pos[i, :])        
        return results
    
    def pso_objective(self, pos):
        
        '''
        Objective function for optimisation. Calculates cost of a given position.
        ------
        Parameters:
        pos:    ndarray. 1D array of position. 
        ------
        Returns:        float. The mean income after tax during the retirement 
                        period for specific position.                    
        
        '''
        
        fracs, ra_payout_frac = self.reshape(pos)
        scenario, ra_payouts = self.fractionsToRands((fracs, ra_payout_frac))
        self.contr.loc[:, self.contr.columns] = scenario
        for count, name in enumerate(self.investments.keys()):
            self.investments[name].calculateOptimalWithdrawal(self.contr.loc[:, name],
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
        and stores it in the object's dataframe.
        '''
        #  Zero all relevant columns so that the amounts don't build up over 
        #  different function calls
        
        if len(self.di_list) == 0:
            self.addInvestment('DI', DI(self.person,
                                        0,
                                        12,
                                        0))
        
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
    
        try:
            self.readInputExcel()
        except FileNotFoundError:
            self.df.loc[:self.first_fy_after_retirement, 'taxable_ibt'] = self.taxable_ibt
                        
        self.df.loc[:self.first_fy_after_retirement, 'iat'] = -self.uif_contr

        
        ra_payouts = 0
        for i in self.ra_list:
            self.df['taxable_ibt'] += self.investments[i].df['withdrawals']
            self.df['withdrawals_total'] += self.investments[i].df['withdrawals']
            self.df['withdrawals_RA'] += self.investments[i].df['withdrawals']
            self.df['contr_RA'] += self.investments[i].df['contr']
            self.df['capital_RA'] = self.investments[i].df['capital']
            self.df['contr_total'] += self.investments[i].df['contr']
            ra_payouts += self.investments[i].payout
        
        self.df.loc[self.retirement_fy_end, 'taxable_ibt'] += self.CGTRA(ra_payouts)
        self.ra_payouts = ra_payouts
        
        self.investments[self.max_di_name].ra_lump_sum = ra_payouts
        self.investments[self.max_di_name].recalculateOptimalWithdrawal()
        
        for count, i in enumerate(self.di_list):
            self.df['capital_gains'] += self.investments[i].df['withdrawal_cg']
            self.df['iat'] += self.investments[i].df['withdrawals']
            self.df['contr_DI'] += self.investments[i].df['contr']
            self.df['withdrawals_total'] += self.investments[i].df['withdrawals']
            self.df['withdrawals_DI'] += self.investments[i].df['withdrawals']
            self.df['capital_DI'] = self.investments[i].df['capital'].copy()
            self.df['contr_total'] += self.investments[i].df['contr']
            self.df['contr_total_at'] += self.investments[i].df['contr']

        for i in self.tfsa_list:
            self.df['iat'] += self.investments[i].df['withdrawals']
            self.df['contr_TFSA'] += self.investments[i].df['contr']
            self.df['withdrawals_total'] += self.investments[i].df['withdrawals']
            self.df['withdrawals_TFSA'] += self.investments[i].df['withdrawals']
            self.df['capital_TFSA'] = self.investments[i].df['capital'].copy()
            self.df['contr_total'] += self.investments[i].df['contr']
            self.df['contr_total_at'] += self.investments[i].df['contr']

        self.df['it'] = self.df.apply(self.totalTax, axis=1)
        self.df['iat'] = self.df['iat'] + self.df['taxable_ibt'] - self.df['contr_RA'] - self.df['it']
        self.df['savable_iat'] = self.df['iat'] - self.df.expenses
        
    def totalTax(self, s, verbose=False):
        
        '''
        Calculates total income tax. The name of the Dataframe s is the year
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
            if verbose: print('Pre-retirement')
            if s.contr_RA <= self.RA_MAX_PERC*s.taxable_ibt and s.contr_RA <= self.RA_ANN_LIMIT:
                if verbose: print('RA contr within percentage limits')
                taxable_income = s.taxable_ibt + self.taxableCapitalGains(s.capital_gains, s.name) - s.contr_RA
            elif s.contr_RA > self.RA_MAX_PERC*s.taxable_ibt and s.contr_RA < self.RA_ANN_LIMIT:
                if verbose: print('RA outside percentage limits')
                taxable_income = s.taxable_ibt - s.taxable_ibt*self.RA_MAX_PERC + self.taxableCapitalGains(s.capital_gains, s.name)
            else:
                if verbose: print('Only a portion of RA contr tax free')
                taxable_income = s.taxable_ibt - self.RA_ANN_LIMIT + self.taxableCapitalGains(s.capital_gains, s.name)
        else:
            if verbose: print('Post-retirement')
            if age < 65:
                if verbose: print('Younger than 65')
                taxable_income = max(0, s.taxable_ibt + self.taxableCapitalGains(s.capital_gains, s.name))
            elif age < 75:
                if verbose: print('65-75')
                taxable_income = max(0, s.taxable_ibt + self.taxableCapitalGains(s.capital_gains, s.name) - self.CGT_6575)
            else:
                if verbose: print('>75')
                taxable_income = max(0, s.taxable_ibt + self.taxableCapitalGains(s.capital_gains, s.name) - self.CGT_GE75)                
        
        if verbose:
            print('Taxable income:', taxable_income)
            print('age', age)
        tax = self.incomeTax(taxable_income, age, verbose)
        if verbose: print('Tax inside totalTax:', tax)
        self.tax_credit_ma = self.taxCreditMa(self.df.loc[s.name, 'monthly_med_aid_contr'],
                                              self.df.loc[s.name, 'ma_dependants'],
                                              self.df.loc[s.name, 'medical_expenses'],
                                              taxable_income,
                                              age, 
                                              verbose)  
        if verbose: print(tax - self.tax_credit_ma)
        return max(0, tax - self.tax_credit_ma)    
    
    def taxableCapitalGains(self, amount, year):
        
        return self.NPV(self.CGT_INCL_RATE*max(0, amount - self.FV(self.CGT_EXEMPTION, year)), year)
    
    #@numba.jit
    def incomeTax(self, taxable_income, age=64, verbose=False):
        
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
            rebate = self.REBATE_UNDER65
            if verbose: print('<65, rebate=', rebate)
        elif age < 75:
            rebate = self.REBATE_UNDER65 + self.REBATE6575
            if verbose: print('65-75, rebate=', rebate)

        else:
            rebate = self.REBATE_UNDER65 + self.REBATE6575 + self.REBATE_75PLUS
            if verbose: print('>75, rebate=', rebate)

        
        if taxable_income <= self.TAX_0THRESH:
            if verbose: print('taxable_income< tax threshold, returning 0')
            return 0
        if taxable_income <= self.TAX_18PERC_THRESH:
            if verbose: print('taxable_income in 18% bracket')
            return  0.18*(taxable_income) - rebate
        elif taxable_income <= self.TAX_26PERC_THRESH:
            if verbose: print('taxable_income in 26% bracket')
            return  self.TAX_26CONST + ((taxable_income) - self.TAX_18PERC_THRESH)*0.26 - rebate
        elif taxable_income <= self.TAX_31PERC_THRESH:
            if verbose: print('taxable_income in 31% bracket')
            return  self.TAX_31CONST + (taxable_income - self.TAX_26PERC_THRESH)*0.31 - rebate
        elif taxable_income <= self.TAX_36PERC_THRESH:
            if verbose: print('taxable_income in 36% bracket')
            return  self.TAX_36CONST + (taxable_income - self.TAX_31PERC_THRESH)*0.36 - rebate
        elif taxable_income <= self.TAX_39PERC_THRESH:
            if verbose: print('taxable_income in 39% bracket')
            return  self.TAX_39CONST + (taxable_income - self.TAX_36PERC_THRESH)*0.39 - rebate
        elif taxable_income <= self.TAX_41PERC_THRESH:
            if verbose: print('taxable_income in 41% bracket')
            return  self.TAX_41CONST + (taxable_income - self.TAX_39PERC_THRESH)*0.41 - rebate
        elif taxable_income >= self.TAX_41PERC_THRESH:
            if verbose: print('taxable_income in 45% bracket')
            return  self.TAX_45CONST + (taxable_income - self.TAX_41PERC_THRESH)*0.45 - rebate
    
    #@numba.jit
    def taxCreditMa(self, 
                    med_aid_contr, 
                    ma_dependants,
                    medical_expenses,
                    taxable_income,
                    age,
                    verbose):
        
        '''
        Calculates the tax credit due for medical aid contributions.
        ------
        Parameters:
        med_aid_contr:              float. Annual medical aid contribution.
        ma_dependants:              int. Number of dependants.
        medical expenses:           float. Annual eligible medical expenses not 
                                    covered by the medical aid.
        taxable_income:             float. Taxable income. Income after RA
                                    deduction.
        age:                        int. Age of person under consideration.
        ------
        Returns:
        float. Tax credit due to medical expenses.
        '''
        if age > 65:
            if ma_dependants <=2:
                ma_d_total = ma_dependants*self.MA_CREDIT_LE2*12
            else:
                ma_d_total = self.MA_CREDIT_LE2*2*12 + 12*(ma_dependants - 2)*self.MA_CREDIT_G2
            tax_credit_ma = ma_d_total\
                                + self.MA_INCL_RATE_GE65*max(0, med_aid_contr - 3*ma_d_total)\
                                + self.MA_INCL_RATE_GE65*max(0, medical_expenses)
        else:
            if ma_dependants <=2:
                ma_d_total = ma_dependants*self.MA_CREDIT_LE2
            else:
                ma_d_total = 12*self.MA_CREDIT_LE2*2 + 12*(ma_dependants - 2)*self.MA_CREDIT_G2
            
            tax_credit_ma = ma_d_total \
                                + self.MA_INCL_RATE*max(0, medical_expenses - self.MA_TAXABLE_INC_PERC*taxable_income)\
                                + self.MA_INCL_RATE*max(0, med_aid_contr - ma_d_total*4)
        
        if verbose: 
            print('ma_d_total', ma_d_total)
            print('Tax credit MA', tax_credit_ma)

        return tax_credit_ma
    
    #@numba.jit        
    def CGTRA(self, lump_sum):
        
        '''
        Calculates taxable capital gains on RA lump sum withdrawals
        ------
        Parameters:
        lump_sum:       float. Lump sum amount to be withdrawn.
        ------
        Returns:
        Taxable amount of capital gains of lump sum withdrawal.
        '''
        lump_sum_FV = self.FV(lump_sum, self.retirement_date)
        if lump_sum_FV < self.FV(self.CGTRA_LIM1, self.retirement_date):
            return 0
        elif lump_sum_FV < self.FV(self.CGTRA_LIM18, self.retirement_date):
            return self.NPV((lump_sum_FV - self.FV(self.CGTRA_LIM1, self.retirement_date))*0.18, self.retirement_date)
        elif lump_sum_FV < self.FV(self.CGTRA_LIM36, self.retirement_date):
            return self.NPV(self.FV(self.CGTRA_CONST18, self.retirement_date) + (lump_sum_FV - self.FV(self.CGTRA_LIM18, self.retirement_date))*0.27, self.retirement_date)
        elif lump_sum_FV >= self.FV(self.CGTRA_LIM36, self.retirement_date):
            return self.NPV(self.FV(self.CGTRA_CONST_36, self.retirement_date) + (lump_sum_FV - self.FV(self.CGTRA_LIM36, self.retirement_date))*0.36, self.retirement_date)
        
    def plot(self):
        '''
        Plots graphs of income after tax, contributions, withdrawals, and
        capital changes over time between present date and life expectancy date.
        '''
        
        plt.figure(1)
        index = [x.strftime('%Y-%M-%d') for x in self.df.index.date]
        if len(self.tfsa_list):
            plt.plot(index, self.df['withdrawals_TFSA'], label='TFSA')
        if len(self.ra_list):
            plt.plot(index, self.df['withdrawals_RA'], label='RA')
        plt.plot(index, self.df['withdrawals_DI'], label='DI')
        plt.title('Withdrawals')
        plt.xlabel('Dates')
        plt.ylabel('Amount [R]')
        plt.xticks(rotation=90)
        plt.legend()
        
        plt.figure(2)
        if len(self.tfsa_list):
            plt.plot(index, self.df['contr_TFSA'], label='TFSA')
        if len(self.ra_list):
            plt.plot(index, self.df['contr_RA'], label='RA')
        plt.plot(index, self.df['contr_DI'], label='DI')
        plt.xlabel('Dates')
        plt.ylabel('Amount [R]')
        plt.xticks(rotation=90)
        plt.legend()        
        plt.title('Contributions')
        
        plt.figure(3)
        if len(self.tfsa_list):
            plt.plot(index, self.df['capital_TFSA'], label='TFSA')
        if len(self.ra_list):
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
    def randomPosition(self, factor):
        '''
        Generates random starting position for a particle according to a 
        Dirichlet distribution, so that the savings allocations for every year
        adds up to one.
        ------
        Parameters:
        factor:     float. Dirichlet distribution factor. For low numbers (<1)
                    the Dirichlet distribution will allocate almost exclusively
                    to one column, making the rest close to zero. For high
                    numbers (>10), allocations will be about equal.
        ------
        Returns:
        ndarray. 1D numpy array of position, according to Fortran indexing.
        '''
        
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

    def randomConstantPosition(self, factor):
        
        '''
        Initialize Random combination, apply to all working years
        ------
        Parameters:
        factor:     float. Dirichlet distribution factor. For low numbers (<1)
                    the Dirichlet distribution will allocate almost exclusively
                    to one column, making the rest close to zero. For high
                    numbers (>10), allocations will be about equal.
        ------
        Returns:
        ndarray. 1D array of position ordered by Fortran indexing.
        '''
        ra_payout = np.random.random()*0.3
        contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
        #ra = np.random.dirichlet(factor*np.ones(len(self.ra_list)))*np.random.triangular(0, self.RA_MAX_PERC, 0.5)
        ra = np.random.dirichlet(factor*np.ones(len(self.ra_list)))*np.random.beta(1, 3.9)
        #  Generate other allocations to sum to one:
        if sum(ra) > self.RA_MAX_PERC:
            ra = self.RA_MAX_PERC*ra/sum(ra)
        others = np.random.dirichlet(factor*np.ones(self.size - len(self.ra_list)))
        plan = np.concatenate([ra, others])
        for i in range(self.number_working_years):
            contr[i] = plan
        #contr = np.array([plan if i < self.number_working_years else np.zeros(self.size) for i in range(len(contr))])
        return np.insert(contr.reshape(contr.size, order='F'), 0, ra_payout)
    
    def calculateTaxEfficientRAFirstIAT(self):
        self.contr = pd.DataFrame(index=self.df.index,
                                  columns=self.investment_names)

        solution, self.ra_payouts = self.fractionsToRands(self.reshape(self.taxEfficientPositionRAFirst()))
        for count, key in enumerate(self.investments.keys()):
            self.contr.loc[:, key] = solution[:, count]
            self.investments[key].calculateOptimalWithdrawal(self.contr[key], self.strategy)
        self.calculate()
        return round(self.df.loc[self.retirement_fy_end:, 'iat'].mean()/12)

    def calculateTaxEfficientTFSAFirstIAT(self):
        '''
        Generates a tax efficient position with TFSA priority, and calculates
        the mean post-retirement IAT.
        ------
        Returns:
        float. Mean income after tax during retirement.
        '''
        self.contr = pd.DataFrame(index=self.df.index,
                                  columns=self.investment_names)

        ind = self.taxEfficientPosTFSAFirst()
        self.ind = ind
        solution, ra_payouts = self.fractionsToRands(self.reshape(ind))
        for count, key in enumerate(self.ra_list):
            self.investments[key].payout = ra_payouts[count]
        for count, key in enumerate(self.investments.keys()):
            self.contr.loc[:, key] = solution[:, count]
            self.investments[key].calculateOptimalWithdrawal(self.contr[key], self.strategy)

        self.calculate()
        return round(self.df.loc[self.retirement_fy_end:, 'iat'].mean()/12)
        
    
    def taxEfficientPositionRAFirst(self):
        '''
        Allocate up to 27.5%  to RAs, then R33 000 
        to TFSAs, and the rest to DIs (depending on savable income)
        ------
        Returns:
        ndarray. 1D array of tax efficient position, using Fortran indexing.
        '''
        #  Create blank individual
        contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
        ra_frac = np.zeros(self.number_working_years)
        ra_contr = np.zeros(self.number_working_years)
        tfsa_frac = 0
        ra_limit = np.minimum(self.RA_MAX_PERC*self.df.taxable_ibt, self.RA_ANN_LIMIT)
        
        for i in range(self.number_working_years):
            #  Allocate RA saving:
            if len(self.ra_list) > 0:
                savable_income = 1
                #  Find RA allocation, up to 27.5%:

                while ra_contr[i] < ra_limit[i] and savable_income > 0:      
                        ra_frac[i] += 0.001
                        ra_contr[i] = self.df.loc[self.df.index[i],'taxable_ibt']*ra_frac[i]
                        tax = self.incomeTax(self.df.loc[self.df.index[i], 'taxable_ibt'] - ra_contr[i], age=self.df.loc[self.df.index[0], 'age'])
                        taxSeries = pd.Series({'taxable_ibt': self.df.loc[self.df.index[i],'taxable_ibt'],
                                        'contr_RA': ra_contr[i],
                                        'capital_gains': 0,
                                        'medical_expenses': self.df.loc[self.df.index[i], 'medical_expenses']})
                        taxSeries.name = self.df.index[i]             
                        tax = self.totalTax(taxSeries)
                        savable_income = np.maximum(0, self.df.loc[self.df.index[i], 'taxable_ibt'] - ra_contr[i] - tax - self.uif_contr - self.df.loc[self.df.index[i], 'expenses'])
    
                #  If there is only one RA, allocate to it.
                if len(self.ra_list) == 1:
                    contr[i, 0] = ra_frac[i]
                elif len(self.ra_list) > 1: #  Else allocate to max growth RA
                    contr[i, self.investment_names.index[self.max_ra_name]] = ra_frac[i]
                    
                # Calculate TFSA
                if len(self.tfsa_list):
                    if savable_income >= self.TFSA_ANN_LIMIT:
                        tfsa_frac = self.TFSA_ANN_LIMIT/savable_income #  TFSA as % of savable income
                    else:
                        tfsa_frac = 1       # 100% of savable income   
                    contr[i, self.investment_names.index(self.max_tfsa_name)] = tfsa_frac
        
                #  Calculate and allocate DI
                if tfsa_frac < 1:
                    contr[i, self.investment_names.index(self.max_di_name)] = 1 - tfsa_frac
            
        #contr_full = np.array([contr[0, :] if i < self.number_working_years else np.zeros(self.size) for i in range(len(contr))])
        if len(self.ra_list):
            self.investments[self.ra_list[0]].growthBeforeRetirement(contr) 
            ra_retirement_capital = self.investments[self.ra_list[0]].df.loc[self.last_working_year, 'capital'].copy()
            ra_payout = min(self.CGTRA_LIM1/ra_retirement_capital, ra_retirement_capital*0.3/ra_retirement_capital)
        else:
            ra_payout = np.array([0])
        return np.insert(contr.reshape(contr.size, order='F'), 0, ra_payout)

    def taxEfficientPosTFSAFirst(self):
        '''
        Allocate up to R33 000 to TFSAs, then up to 27.5% to RA, and the rest
        to DIs (depending on savable income)
        ------
        Returns:
        ndarray. 1D array of tax efficient position, using Fortran indexing.
        '''
        #  Create blank individual
        contr = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
        #  Loop through every working year, calculating contributions
        ra_frac = np.zeros(self.number_working_years)
        ra_contr = np.zeros(self.number_working_years)
        tfsa_frac = np.zeros(self.number_working_years)
        tfsa_contr = np.zeros(self.number_working_years)
        ra_limit = np.minimum(self.RA_MAX_PERC*self.df.taxable_ibt, self.RA_ANN_LIMIT)
        for i in range(self.number_working_years):
            #  Allocate TFSA first:
            if len(self.tfsa_list) > 0 and tfsa_contr[0:i].sum() < self.TFSA_TOTAL_LIMIT:
                savable_income_after_tfsa = 1
                savable_income = 1
                #print('tfsa contr', tfsa_contr[i])
                while tfsa_contr[i] < self.TFSA_ANN_LIMIT and savable_income_after_tfsa > 0:
                    tfsa_contr[i] += 100
                    #tax = self.incomeTax(self.df.loc[self.df.index[i],'taxable_ibt'], age=self.df.loc[self.df.index[0], 'age'])
                    taxSeries = pd.Series({'taxable_ibt': self.df.loc[self.df.index[i],'taxable_ibt'],
                                    'contr_RA': 0,
                                    'capital_gains': 0,
                                    'medical_expenses': self.df.loc[self.df.index[i], 'medical_expenses']})
                    taxSeries.name = self.df.index[i]
                    tax = self.totalTax(taxSeries)                    
                    savable_income = np.maximum(0, self.df.loc[self.df.index[i], 'taxable_ibt'] - tax - self.uif_contr - self.df.loc[self.df.index[i], 'expenses']) 
                    savable_income_after_tfsa = savable_income - tfsa_contr[i]
                    #print('Savable income after tfsa', savable_income_after_tfsa)
                if savable_income_after_tfsa < 0:
                    tfsa_contr[i] -= 100

                #print('savable income', savable_income)
                #print('ra_frac[i]', ra_frac[i])

                #  Now, before calculating the fraction, first determine the
                #  RA contribution, since this contribution affects the savable
                #  income, which is the denominator in the fraction of the
                #  tfsa_frac figure.
                
                #  Find RA allocation, up to 27.5%:
                income_after_saving = savable_income_after_tfsa
                ra_contr[i] = 0
                while ra_contr[i] < ra_limit[i] and income_after_saving > 0:      
                        ra_frac[i] += 0.001
                        ra_contr[i] = self.df.loc[self.df.index[i],'taxable_ibt']*ra_frac[i]
                        taxSeries = pd.Series({'taxable_ibt': self.df.loc[self.df.index[i],'taxable_ibt'],
                                        'contr_RA': ra_contr[i],
                                        'capital_gains': 0,
                                        'medical_expenses': self.df.loc[self.df.index[i], 'medical_expenses']})
                        taxSeries.name = self.df.index[i]
                        tax = self.totalTax(taxSeries)                           
                        income_after_saving = np.maximum(0, self.df.loc[self.df.index[i], 'taxable_ibt'] - ra_contr[i] - tax - self.uif_contr - self.df.loc[self.df.index[i], 'expenses'] - tfsa_contr[i])
                
                #  If there is only one RA, allocate to it.
                if len(self.ra_list) == 1:
                    contr[i, 0] = ra_frac[i]
                elif len(self.ra_list) > 1: #  Else allocate to max growth RA
                    contr[i, self.investment_names.index[self.max_ra_name]] = ra_frac[i]

                savable_income = np.maximum(0, self.df.loc[self.df.index[i],'taxable_ibt'] - ra_contr[i] - tax - self.uif_contr - self.df.loc[self.df.index[i], 'expenses'])
                tfsa_frac[i] = tfsa_contr[i]/savable_income
                contr[i, self.investment_names.index(self.max_tfsa_name)] = tfsa_frac[i]
                    
            #  Calculate and allocate DI
            if tfsa_frac[i] < 1:
                contr[i, self.investment_names.index(self.max_di_name)] = 1 - tfsa_frac[i]
                    
        if len(self.ra_list):
            self.investments[self.ra_list[0]].growthBeforeRetirement(contr)
            ra_retirement_capital = self.investments[self.ra_list[0]].df.loc[self.last_working_year, 'capital'].copy()
            ra_payout = min(self.CGTRA_LIM1/ra_retirement_capital, ra_retirement_capital*0.3/ra_retirement_capital)
        else:
            ra_payout = np.array([0])
        return np.insert(contr.reshape(contr.size, order='F'), 0, ra_payout)

    def rebalancePSO(self, swarm):
        '''
        Rebalanced portfolio allocations. As the particle moves through space,
        it will likely end up in a position where the fractional contributions
        do not sum to unity. This routine corrects to position to a viable one.
        It does this for all positions in the swarm.
        ------
        Parameters:
        swarm:      pyswarms swarm object.
        
        Returns:
        ------
        swarm, with rebalanced positions.
        '''
        pos = swarm.position
        for i in range(pos.shape[0]):
            ind_1d = pos[i, :]
            ra_payout = pos[i, 0]
            #  Find all indices where allocations are out of bounds:
            idx_negative = np.where(ind_1d < 0)[0]
            ind_1d[idx_negative] = 0
            idx_g1 = np.where(ind_1d > 1)[0]
            ind_1d[idx_g1] = 1
            
            #  Stop particle dead when it reaches a boundary.
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
            ind = self.reshape(ind_1d)[0]
            ras = ind[:, :len(self.ra_list)]
            #  If ras are so high (still < 1) that savable income is negative,
            #  reduce ra contributions by 1%.
            ras_sum = ras.sum(axis=1)
            if len(self.ra_list):
               for j in range(len(ras[:, 0])):
                    savable_income = np.array([-1, -1])
                
            #  If savable_income < 0, adjust ras downwards. At the moment it
            #  adjusts all positive RAs downwards by 1%. 
                    while savable_income.any() < 0 and any(ras[j, :]) > 0:                   
                        tax = self.incomeTax(self.df.loc[self.df.index[j],'taxable_ibt'] - ras_sum[j]*self.df.loc[self.df.index[j],'taxable_ibt'])
                        if i <= self.number_working_years:
                            savable_income = self.df.loc[self.df.index[j], 'taxable_ibt'] - tax - self.uif_contr - self.df.expenses - ras_sum[j]*self.df.loc[self.df.index[j], 'taxable_ibt']
                        else:
                            savable_income = self.df.loc[self.df.index[j], 'taxable_ibt'] - tax - self.df.expenses - ras_sum[j]*self.df.loc[self.df.index[j], 'taxable_ibt']
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
 
    def pso(self, params=[]):
        
        '''
        Uses Particle Swarm Optimization to find optimal investment strategy.
        '''      

        #  Create bounds. Contributions only during working years, withdrawals only during retirement
        #min_bounds = np.zeros(1 + self.size*(self.number_working_years + self.number_retirement_years))
        #max_bounds = np.zeros(1 + self.size*(self.number_working_years + self.number_retirement_years))
        #tax = self.incomeTax(self.df.loc[:, 'taxable_ibt'], age=self.df.loc[:, 'age'])
        #savable_income = self.df.loc[:,'taxable_ibt'] - tax - self.uif_contr - self.df.expenses    
        #  Find all columns in the dataframe containing 'capital'. This will be
        #  used for determining max withdrawal bounds.
        capital_cols = [i for i in self.df.columns.tolist() if 'capital' in i]
        capital_cols.remove('capital_gains')
        #max_withdrawal = self.df.loc[self.retirement_fy_end, capital_cols].max()
        #  bounds on lump sum withdrawal:
        #min_bounds[0] = 0
        #max_bounds[0] = 0.3
        #  bounds on contributions:
        index = 0 #  index for persistence over multiple loops
            #  up to savable income during working years
            
        #        for j in range(self.number_working_years):
        #            for i in range(self.size): 
        #                min_bounds[index] = 0
        #                max_bounds[index] = 1 #savable_income
        #                index += 1
        #            #  No contributions during retirement
        #
        #        for j in range(self.number_retirement_years):
        #            for i in range(self.size): 
        #                min_bounds[index] = 0
        #                max_bounds[index] = 1e-5
        #                index += 1           
                
        min_bounds = np.zeros([self.number_working_years + self.number_retirement_years, self.size])
        min_bounds = np.concatenate([np.zeros(len(self.ra_list)), min_bounds.reshape(min_bounds.size,  order='F')])
        max_bounds = np.ones([self.number_working_years + self.number_retirement_years, self.size])
        max_bounds[self.number_working_years:,:] = 1e-5
        max_bounds = np.concatenate([0.3*np.ones(len(self.ra_list)), max_bounds.reshape(max_bounds.size, order='F')])
        
        
        #  No bounds on withdrawals because we do not guess withdrawals. They are
        #  calculated.    
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        n_particles = int(max(20, len(self.investment_names)*self.number_working_years/4))
        if n_particles%2 != 0:
            n_particles += 1
        dimensions = min_bounds.size
        factor_list = np.geomspace(1/20, 100, 30)
        #iterations = 10
        tolerance = 2.5e-3 #  Stopping criterion: improvement per iteration
        print_interval = 1 
        options = {}
        clamp = (-0.5, 0.2)
        if not len(params):
            options = {'c1': 1.680603393245492, #  cognitive parameter (weight of personal best)
                       'c2': 1.8521484727363389, #  social parameter (weight of swarm best)
                       'v': 0.6191471730049908, #  initial velocity
                       'w': 0.4133545009700662, #  inertia
                       'k': 9, #  Number of neighbours. Ring topology seems popular
                       'p': 2}  #  Distance function (Minkowski p-norm). 1 for abs, 2 for Euclidean
        else:
            options = {'c1': params[0], #  cognitive parameter (weight of personal best)
                       'c2': params[1], #  social parameter (weight of swarm best)
                       'v': params[2], #  initial velocity
                       'w': params[3], #  inertia
                       'k': params[4], #  Number of neighbours. Ring topology seems popular
                       'p': params[5]}  #  Distance function (Minkowski p-norm). 1 for abs, 2 for Euclidean
            clamp = (-params[6], params[7])
        topology = ps.backend.topology.Star()
                
        lst_init_pos = [None]*n_particles
        lst_init_pos[0] = self.randomConstantPosition(np.random.choice(factor_list)).T
        #  Handle cases where people don't have RAs or TFSAs:
        if len(self.ra_list):
            lst_init_pos[1] = self.taxEfficientPositionRAFirst().T

        else:
            lst_init_pos[1] = self.randomConstantPosition(np.random.choice(factor_list)).T

        if len(self.tfsa_list):
            lst_init_pos[2] = self.taxEfficientPosTFSAFirst().T

        else:
            lst_init_pos[2] = self.randomConstantPosition(np.random.choice(factor_list)).T
        
        for i in range(3, n_particles, 2):
            lst_init_pos[i-1] = self.randomConstantPosition(np.random.choice(factor_list)).T
            lst_init_pos[i] = self.randomPosition(np.random.choice(factor_list)).T
        
        #  Convert to numpy array. Not efficient, but gets it in the correct
        #  format.
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
                     #bounds=(min_bounds, max_bounds),
                     options=options,
                     dimensions=dimensions,
                     clamp=clamp
                     )
        
        #improvement = [100, 100, 100]
        improvement = 100
        previous_cost = -1
        counter = 0
        pbest_cost = np.zeros(n_particles)
        while improvement > tolerance:

            counter += 1
            max_iter = max(50, counter)
            #  Update personal bests
            # Compute cost for current position and personal best
            self.myswarm.current_cost = self.pso_objectiveSwarm(self.myswarm)
            for i, pos in enumerate(self.myswarm.position):
                if (pos >= self.max_bounds).any():
                    indices = pos <= self.max_bounds
                    pos[indices] = self.max_bounds[indices]
                    self.myswarm.position[i] = pos
                    self.myswarm.current_cost[i] = self.pso_objective(pos)
                    self.myswarm.velocity[i][indices] = 0
            for i in range(n_particles):
                pbest_cost[i] = self.pso_objective(self.myswarm.pbest_pos[i,:])

            self.myswarm.pbest_cost = pbest_cost
            self.myswarm.pbest_pos, self.myswarm.pbest_cost = ps.backend.operators.compute_pbest(
                self.myswarm)
            self.myswarm.current_cost = self.pso_objectiveSwarm(self.myswarm)
            pbest_cost = np.zeros(n_particles)                    
            # Update gbest from neighborhood
            self.myswarm.best_pos, self.myswarm.best_cost = topology.compute_gbest(self.myswarm)#,
                #options['p'], options['k'])
            
            #improvement[1:] = improvement[0:2]
            #improvement[0] = self.myswarm.best_cost - previous_cost
            improvement = self.myswarm.best_cost/previous_cost - 1     
            if i%print_interval==0:
                print('Iteration: {} | best cost: {:.3f} | Improvement: {:2f}'.format(counter, self.myswarm.best_cost, improvement))
            self.myswarm.velocity = topology.compute_velocity(self.myswarm)
            self.myswarm.position = topology.compute_position(self.myswarm)
            self.myswarm = self.rebalancePSO(self.myswarm)
            
            previous_cost = self.myswarm.best_cost    

        topology = ps.backend.topology.Star()
        improvement = 100
        self.myswarm.options['k'] = n_particles
        print('MOVING TO STAR TOPOLOGY')
        while improvement > tolerance:
            counter += 1
            max_iter = max(50, counter)
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
            self.myswarm.best_pos, self.myswarm.best_cost = topology.compute_gbest(self.myswarm)#,
                #options['p'], options['k'])
            
            #improvement[1:] = improvement[0:2]
            #improvement[0] = self.myswarm.best_cost - previous_cost
            improvement = self.myswarm.best_cost/previous_cost - 1     
            if i%print_interval==0:
                print('Iteration: {} | best cost: {:.3f} | Improvement: {:2f}'.format(counter, self.myswarm.best_cost, improvement))
            self.myswarm.velocity = topology.compute_velocity(self.myswarm)
            self.myswarm.position = topology.compute_position(self.myswarm)
            self.myswarm = self.rebalancePSO(self.myswarm)
            
            previous_cost = self.myswarm.best_cost    

        
        return self.myswarm.best_pos, self.myswarm.best_cost

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
                surplus = iat - self.df.expenses
            
            RA_annual_contr -= 100
            ibt_ara = ibt - RA_annual_contr            
            iat = ibt_ara - self.incomeTax(ibt_ara, age) - self.uif_contr
            surplus = iat - self.df.expenses
            RA_monthly_contr = RA_annual_contr/12
            iat_monthly = iat/12
            print('RA Debit order: \t\t\t\tR', round(RA_monthly_contr, 2))
            print('Monthly IAT: \t\t\t\t\tR', round(iat_monthly))
            print('Max tax free RA contr (27.5% of IBT) = \t\tR', round(self.RA_MAX_PERC*ibt/12, 2))
            print('Total earned per month, incl. RA: \t\tR', round(RA_annual_contr/12 + iat_monthly, 2))
            print('Total monthly tax = \t\t\t\tR', round(self.incomeTax(ibt_ara, age)/12, 2))
            print('Total annual RA contr', RA_annual_contr)
        else:
            RA_annual_contr = RA_monthly_contr*12
            ibt_ara = ibt - RA_annual_contr            
            iat = ibt_ara - self.incomeTax(ibt_ara, age) - self.uif_contr
            surplus = iat - self.df.expenses           
            iat_monthly = iat/12
            print('RA Debit order: \t\t\t\tR', round(RA_monthly_contr, 2))
            print('Annual taxable income: \t\t\t\tR', round(ibt_ara))
            print('Monthly IAT: \t\t\t\t\tR', round(iat_monthly))
            print('Max tax free RA contr (27.5% of IBT) = \t\tR', round(self.RA_MAX_PERC*ibt/12, 2))
            print('Total earned per month, incl. RA: \t\tR', round(RA_annual_contr/12 + iat_monthly, 2))
            print('Total monthly tax = \t\t\t\tR', round(self.incomeTax(ibt_ara, age)/12, 2))
            print('Total annual RA contr\t\t\t\tR', round(RA_annual_contr, 2))

class TFSA(TaxableEntity):
    
    def __init__(self,
                 person,
                 initial,
                 ytd,
                 ctd,
                 growth=0):
        
        '''
        Tax-Free Savings Account object.
        ------
        Parameters:
        initial:    float. Amount in investment at present.
        ytd:        float. Year to date contributions. (tax year)
        ctd:        float. Contributions to date (all years)
        ytd:        float. Year-to-date contribution, according to the tax year.
        ctd:        float. Total contr to date. 
        growth:     float. Annualized growth rate of investment. E.g. if 10 if 10%.
                    If not specified, the average annualized growth rate of the JSE
                    over a rolling window of similar length to the investment
                    horizon is used.
        
        '''
        super().__init__(person)
        self.initial = initial
        self.growth = growth/100
        self.monthly_growth = 10**(np.log10(1 + self.growth)/12) - 1
        
        self.type = 'TFSA'
        #Investment.__init__(self, initial, growth)
        self.ctd = ctd
        self.ytd = ytd
        self.df = self.df.join(pd.DataFrame(index=self.df,
                                                   columns=['Total contr', 'YTD contr'],
                                                   data=np.zeros([self.df.index.size, 2])))
        self.df.loc[:, ['capital',
                        'withdrawals',
                        'contr']] = 0
    
        self.df.loc[self.df.index[0], 'capital'] = copy(self.initial)
        self.df.loc[self.df.index[0], 'YTD contr'] = copy(self.ytd)
        self.df.loc[self.df.index[0], 'Total contr'] = copy(self.ctd)
        
        self.overall_growth = growth/100
        
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
    
    
    def __repr__(self):
        return f'''{self.__class__.__name__}
                Initial: {self.initial}
                Growth: {self.growth}
                Capital at retirement: {self.df.loc[self.retirement_fy_end, 'capital']}
                '''
    
    def calculateOptimalWithdrawal(self,
                                   contr,
                                   ra_payout_frac=0): #  dummy, for Portfolio calling RA.
        
        '''
        Determines the capital series with an optimal withdrawal rate such 
        that a constant amount can be withdrawn every year during retirement 
        and the capital runs out in the year after the life expectancy date.
        This is in inherently risky strategy if life expectancy is 
        underestimated.
        ------
        Parameters:
        contr:          DataFrame. Dataframe, indexed by year from today to 
                            retirement age, with contr.
        strategy:       str. Choose between 'optimal' withdrawals (0 capital at life expectancy) and 'safe' (4% withdrawal rate)
        ra_payout_frac: Dummy variable. Do not assign.
        '''
        
        self.df.loc[:, ['capital',
                'YTD contr',
                'Total contr',
                'withdrawals',
                'contr']] = 0
                                
        self.df.loc[:, 'contr'] = contr      
        self.df.loc[self.retirement_fy_end, 'contr'] = self.df.loc[self.retirement_fy_end, 'contr']*self.ret_year_installments/12                      
        self.calculate()            
        # Determine capped contributions before optimising withdrawals
        c = self.df.loc[self.retirement_fy_end, 'capital'].copy()
        drawdown = 0.04
        if self.strategy == 'optimal':
            capital_at_le = np.inf
            arr = self.df.loc[self.retirement_date:, ['capital']].values.copy()
            while capital_at_le > 0:
                arr[0] = self.df.loc[self.retirement_fy_end, 'capital'] - (12 - self.ret_year_installments)/12*c*drawdown
                drawdown += 0.001
                capital_at_le = self._calculateQuick(arr, self.growth, c*drawdown)
                #print(capital_at_le)
            drawdown -= 0.001
        elif self.strategy == 'safe':
            drawdown = 0.04
        
        self.df.loc[self.retirement_fy_end:, 'withdrawals'] = drawdown*c
        
        self.df.loc[:, ['capital',
        'YTD contr',
        'Total contr',
        'withdrawals',
        'contr']] = 0
        self.df.loc[self.retirement_fy_end:, 'withdrawals'] = drawdown*c
        self.df.loc[self.retirement_fy_end, 'withdrawals'] = (12 - self.ret_year_installments)/12*c*drawdown
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
        Calculates the withdrawal rate such that withdrawals can be kept
        constant during retirement, and capital runs in year after life expectancy.
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
        
        '''
        Calculates the investment movements for a TFSA account, given
        metadata speficied.
        '''
        self.df.loc[:, 'capital'] = 0
        self.df.loc[self.df.index[0], 'capital'] = copy(self.initial)
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
        ytd_excess = self.TFSA_ANN_LIMIT - (ytd_contr + contr_func)
        total_excess = self.TFSA_TOTAL_LIMIT - (total_contr + contr_func)
        if ytd_excess >= 0 and total_excess >= 0:
            return contr_func
        if total_contr >= self.TFSA_TOTAL_LIMIT:
            return (1 - self.TFSA_ANN_LIMIT_EXC_TAX)*contr_func
        amount_exceeded_total = 0
        if total_excess < 0 or ytd_excess < 0:
            if total_excess > 0:
                total_excess = 0
            if ytd_excess > 0:
                ytd_excess = 0
            if total_excess < ytd_excess:
                return abs(amount_exceeded_total)*(1 - self.TFSA_ANN_LIMIT_EXC_TAX) + contr_func + total_excess
            else:
                return contr_func + ytd_excess + (1 - self.TFSA_ANN_LIMIT_EXC_TAX)*abs(ytd_excess)
          
            
class RA(TaxableEntity):  
    
    def __init__(self,
                 person,
                 initial,
                 ytd,
                 ra_growth=9.73,
                 la_growth=0,
                 payout_fraction=1/3,
                 cg_to_date=0):
        
        '''
        Retirement Annuity object. Assumes that the RA is converted to a living
        annuity upon retirement.
        ------
        Parameters:
        initial:            float. Value of RA at present time.
        ytd:                float. Year to date contributions.
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
        payout_fraction:    float. Fraction of RA capital to be paid out as a
                            lump sum at retirement. Max 0.3
        cg_to_date:         Capital Gains to date. For an account which has
                            been open for a number of years, a significant
                            portion of the "initial" or current capital amount,
                            may be capital gains.
        '''       

        #Investment.__init__(self, initial, ra_growth)
        super().__init__(person)
        self.initial = initial
        self.type = 'RA'
        self.ra_growth_overall = ra_growth/100
        if la_growth == 0:
            self.la_growth_overall = (inflation + 1)/100
        else:
            self.la_growth_overall = la_growth/100

        self.cg_to_date = cg_to_date
        #  In real terms:
        self.ra_growth = (1 + self.ra_growth_overall)/(1 + self.inflation) - 1
        self.la_growth = (1 + self.la_growth_overall)/(1 + self.inflation) - 1

        self.monthly_la_growth = 10**(np.log10(1 + self.ra_growth)/12) - 1
        self.monthly_ra_growth = 10**(np.log10(1 + self.la_growth)/12) - 1
        self.payout_fraction = payout_fraction

        self.df = self.df.join(pd.DataFrame(index=self.df,
                                                   columns=['YTD contr'],
                                                   data=np.zeros(self.df.index.size)))
        self.df.loc[:, ['capital',
                            'withdrawals',
                            'contr']] = 0
        self.df.loc[self.df.index[0], 'capital'] = copy(self.initial)
        self.df.loc[self.df.index[0], 'YTD contr'] = copy(ytd)
        
    def __repr__(self):
        return f'''{self.__class__.__name__}
                Initial: {self.initial}
                Capital at retirement: {self.df.loc[self.retirement_fy_end, 'capital']}
                '''
    
        
    def calculateOptimalWithdrawal(self,
                                   contr,
                                   payout_fraction=None):
        
        '''
        Determines the capital series with an optimal withdrawal rate such 
        that a constant amount can be withdrawn every year during retirement 
        and the capital runs out in the year after the life expectancy date.
        This is in inherently risky strategy if life expectancy is 
        underestimated.
        ------
        Parameters:
        contr:          DataFrame. Dataframe, indexed by year from today to 
                            retirement age, with contr.
        strategy:       str. Choose between 'optimal' withdrawals (0 capital at life expectancy) and 'safe' (4% withdrawal rate)
        ra_payout_frac: float. Fraction of RA capital to pay out at retirement. 
                        Leave as None if this amount needs to be determined.
        '''
        
        self.growthBeforeRetirement(contr, self.payout_fraction)
        self.growthAfterRetirementOptimalWithdrawal(contr)

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
        #print(f'contr {contr}')
        #print(f'withdr {withdr}')
        #print(f'monthly growth {monthly_growth}')
        #print(f'capital calc {capital_calc}')
        #print(f'installments {installments}')
        if installments > 1:
            for i in range(0, installments):
                if capital_calc > 0:
                    capital_calc = capital_calc*(1 + monthly_growth) + contr - withdr    
                else:
                    return 0
            return capital_calc

        elif capital_calc > 0:
            return capital_calc*(1 + monthly_growth) + contr - withdr            
        else:
            return 0
        
       
    def growthBeforeRetirement(self,
                               contr,
                               payout_fraction=None):
        
        '''
        Growth before and after retirement is different, since RA and LA
        growth rates will differ. Therefore, growth before and after retirement
        is calculated in separate functions.
        ------
        Parameters:
        contr:              Series. Pandas series of contributions between 
                            current year and life expectancy year. 
                            Contributions are expected to be zero during 
                            retirement years.
                            
        payout_fraction:    Fraction of capital at retirement to be paid out as
                            a lump sum. Max 0.3. Leave as None if unspecified.
        
        '''
        
        if payout_fraction is None:
            payout_fraction = copy(self.payout_fraction)
        previous_year = self.df.index[0]
        self.df['contr'] = contr    
        self.df.loc[self.retirement_fy_end, 'contr'] = self.df.loc[self.retirement_fy_end, 'contr']*self.ret_year_installments/12
        
        for year in self.df.loc[self.df.index[0] : self.last_working_year].index[1:]:
            self.df.loc[year, 'capital'] = self.calculateCapitalAnnualized(capital=self.df.loc[previous_year, 'capital'],
                                                                           contributions=self.df.loc[year, 'contr'],
                                                                            withdrawals=0,
                                                                            growth=self.ra_growth)
            previous_year = year
        
    def growthAfterRetirementOptimalWithdrawal(self,
                                               contr,
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

        self.df['withdrawals'] = 0  
        
        c = self.df.loc[self.retirement_fy_end, 'capital'].copy()
        drawdown = 0.04
        monthly_growth_ra = 10**(np.log10(1 + self.ra_growth)/12) - 1
        monthly_growth_la = 10**(np.log10(1 + self.la_growth)/12) - 1

        
        monthly_contr_ret_year = contr.loc[self.retirement_fy_end]*self.ret_year_installments/12
        
        
        withdrawals = np.zeros(self.number_retirement_years)
        if self.strategy == 'optimal':
            capital_at_le = np.inf
            
            arr = self.df.loc[self.retirement_fy_end:, 'capital'].values.copy()
            while capital_at_le > 0 and drawdown < self.RA_MAX_WITHDR and capital_at_le > drawdown*c:
                drawdown = min(drawdown + 0.001, self.RA_MAX_WITHDR)
                capital_calc = self.df.loc[self.last_working_year, 'capital'].copy()        
                if self.ret_year_installments > 1:
                    for i in range(0, self.ret_year_installments):
                        capital_calc = capital_calc*(1 + monthly_growth_ra) + monthly_contr_ret_year  
                withdrawals[:] = drawdown*capital_calc
                self.payout = self.payout_fraction*capital_calc
                capital_calc -= self.payout
                
                withdr = withdrawals[0]*(12 - self.ret_year_installments)/12
                if 12-self.ret_year_installments > 1:
                    for i in range(self.ret_year_installments, 12):
                        if capital_calc > 0:
                            capital_calc = capital_calc*(1 + monthly_growth_la) - withdr
                        else:
                            capital_calc = 0                        
                arr[0] = capital_calc    
                capital_at_le = self._growthAfterRetirementQuick(arr, self.la_growth, withdrawals)
            #  Redo at last drawdown rate that worked:
            drawdown -= 0.001            
            capital_calc = self.df.loc[self.last_working_year, 'capital'].copy()        
            if self.ret_year_installments >= 1:
                for i in range(0, self.ret_year_installments):
                    if capital_calc > 0:
                        capital_calc = capital_calc*(1 + monthly_growth_ra) + monthly_contr_ret_year  
                    else:
                        capital_calc = 0
            withdrawal = drawdown*capital_calc
            self.payout = self.payout_fraction*capital_calc
            capital_calc -= self.payout
            withdr = withdrawal*(12 - self.ret_year_installments)/12   
            if 12-self.ret_year_installments >= 1:
                for i in range(0, 12-self.ret_year_installments):
                    if capital_calc > 0:
                        
                        capital_calc = capital_calc*(1 + monthly_growth_la) - min(self.RA_MAX_WITHDR_MONTHLY*capital_calc, 
                                                                                   max(withdr, 
                                                                                      self.RA_MIN_WITHDR_MONTHLY*capital_calc))     

            self.df.loc[self.retirement_fy_end, 'withdrawals'] = withdr
            self.df.loc[self.retirement_fy_end, 'capital'] = capital_calc
            self.df.loc[self.first_fy_after_retirement:, 'withdrawals'] = withdrawal

        elif self.strategy == 'safe':
            drawdown = 0.04


        previous_year = self.retirement_fy_end
        for year in self.df.loc[self.first_fy_after_retirement:].index:   
            if self.df.loc[year, 'withdrawals'] <= self.df.loc[previous_year, 'capital'] and self.df.loc[previous_year, 'capital'] > 0:
                    self.df.loc[year, 'withdrawals'] = min(self.RA_MAX_WITHDR*self.df.loc[previous_year, 'capital'], 
                                                           max(self.df.loc[year, 'withdrawals'].copy(), 
                                                               self.RA_MIN_WITHDR*self.df.loc[previous_year, 'capital']))
                    #self.df.loc[year, 'capital'] = max(0, self.df.loc[previous_year, 'capital']*(1 + self.la_growth) - self.df.loc[year, 'withdrawals'])
                    self.df.loc[year, 'capital'] = self.calculateCapitalAnnualized(self.df.loc[previous_year, 'capital'],
                                                                                   contr.loc[year],
                                                                                   self.df.loc[year, 'withdrawals'],
                                                                                   self.la_growth)
            
            else:
                self.df.loc[year, 'withdrawals'] = self.df.loc[year, 'capital'].copy()
                self.df.loc[year, 'capital'] = 0

            previous_year = year 
            

        #self.growthAfterRetirement(contr)

    #@numba.jit      
    def _growthAfterRetirementQuick(self, arr, growth, withdrawal):
        
        '''Quick version of growthAfterRetirement, simply for determining 
        optimal drawdown rate. Does not calculate all variables of interest 
        like calculate() does.
        
        Parameters:
        arr:            numpy array. First column is capital, second column is 
                        contributions.
        growth:         float. Annual growth rate of portfolio
        withdrawal:     float. Annual (fixed) withdrawal amount.
        '''        
        
        for i in range(1, len(arr)):  
            withdrawal_i = min(self.RA_MAX_WITHDR_MONTHLY*arr[i - 1], max(withdrawal[i], 
                                                                   self.RA_MIN_WITHDR_MONTHLY*arr[i - 1]))
            
            arr[i] = self.calculateCapitalAnnualized(arr[i - 1],
                                                       0,
                                                       withdrawal_i,
                                                       growth)
        return arr[-1]  
        
        
class DI(TaxableEntity):
    
    
    '''
    Discretionary Investment object.
    ------
    Parameters:
    initial:            float. Value of RA at present time.
    growth:             float. Annualized growth rate of investment in
                        percentage points. E.g. if 10 if 10%.
                        If not specified, the average annualized growth rate of the JSE
                        over a rolling window of similar length to the investment
                        horizon is used.    la_growth:          float. Growth rate of LA (after payout) in percentage points. i.e. 13 = 13%.
                        assigned a value of inflation + 1% if left unspecified.
    cg_to_date:         Capital Gains to date. For an account which has
                        been open for a number of years, a significant
                        portion of the "initial" or current capital amount,
                        may be capital gains.
    '''
    

    
    def __init__(self,
                 person,
                 initial,
                 growth=0,
                 cg_to_date=0):
       
        super().__init__(person)
        
        self.initial = initial
        self.growth = growth/100
        self.monthly_growth = 10**(np.log10(1 + self.growth)/12) - 1
        self.ra_lump_sum = 0
        self.type = 'DI'

        self.df = self.df.join(pd.DataFrame(index=self.df,
                                                   columns=['capital_gain', 'withdrawal_cg'],
                                                   data=np.zeros([self.df.index.size, 2])))            
        self.df.loc[:, ['capital',
                        'contr',
                        'withdrawals',]] = 0
                        
        self.df.loc[self.df.index[0], 'capital'] = copy(self.initial)
        self.df.loc[self.df.index[0], 'capital_gain'] = copy(cg_to_date)

        self.overall_growth = growth/100
        #In real terms:
        self.cg_to_date = cg_to_date
        self.monthly_growth = 10**(np.log10(1 + self.growth)/12) - 1

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
            
    def __repr__(self):
        return f'''{self.__class__.__name__}
                Initial: {self.initial}
                Growth: {self.growth}
                Capital at retirement: {self.df.loc[self.retirement_fy_end, 'capital']}
                '''
    
            
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
            #print('arr[i]', arr[i])
            #print('i', i)
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
        capital_gains:  float. capital gains to date.
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
        
        if capital_gains > 0: #  Filter out early to save time
            for i in range(0, installments):
                if capital_calc > 0:
                    capital_gains_calc = capital_gains_calc + capital*monthly_cg_growth
                    withdrawal_cg_incr = withdrawals*(capital_gains_calc/capital)
                    capital_gains_calc -= withdrawal_cg_incr
                    withdrawal_cg += withdrawal_cg_incr
                    capital_calc = capital_calc*(1 + monthly_growth) + contr - withdr            
                else:
                    return 0, capital_gains_calc, withdrawal_cg
        else:
            for i in range(0, installments):
                if capital_calc > 0:
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
        capital_gains:  float. capital gains to date.
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
        while capital > 0:
            capital_gains_calc = capital_gains_calc + capital*monthly_cg_growth
            if withdrawals < capital:
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
                                   ra_payout_frac=0): # dummy variable for RA.
        
        '''
        This function is named for consistency, so that Portfolio can call it
        consistently for different objects.
        
        ------
        Parameters:
        contr:          DataFrame. Dataframe, indexed by year from today to 
                            retirement age, with contr.
        strategy:       str. Choose between 'optimal' withdrawals (0 capital at life expectancy) and 'safe' (4% withdrawal rate)
        ra_payout_frac: Dummy variable. Do not assign.
        '''
        self.df.loc[:, 'contr'] = contr  
        self.df.loc[self.retirement_fy_end, 'contr'] = self.df.loc[self.retirement_fy_end, 'contr']*self.ret_year_installments/12
        self.df.loc[:, 'withdrawals'] = 0
        self.recalculateOptimalWithdrawal()
        
    def recalculateOptimalWithdrawal(self):
        
        '''
        Calculates optimal withdrawal figure so that capital lasts exactly as 
        long as life expectancy. This is in inherently risky strategy if life
        expectancy is underestimated.
        ------
        Parameters:
        strategy:       str. Select between options of "safe" and "optimal".
        '''
        
        self.calculate()
        c = self.df.loc[self.retirement_fy_end, 'capital'].copy()
        #print('c', c)
        drawdown = 0.04
        arr = np.zeros(self.number_retirement_years)
        if self.strategy == 'optimal':
            capital_at_le = np.inf
            
            while capital_at_le > 0:
                arr[0] = self.df.loc[self.retirement_fy_end, 'capital']
                arr[1:] = self.df.loc[self.first_fy_after_retirement:, 'capital'].values - c*drawdown
                drawdown += 0.001
                capital_at_le = self._calculateQuick(arr,
                                                     self.growth,
                                                     drawdown*c)
            drawdown -= 0.001
            self.drawdown = drawdown
            #print('capital_at_le', capital_at_le)
        elif self.strategy == 'safe':
            drawdown = 0.04

        #print('drawdown', drawdown)
        self.df.loc[self.retirement_fy_end:, 'withdrawals'] = drawdown*c
        self.df.loc[self.retirement_fy_end, 'withdrawals'] = (12 - self.ret_year_installments)/12*c*drawdown
        self.calculate()

    def calculate(self):

        '''
        Calculates the investment movements for a DI account, given
        metadata speficied.
        '''

        previous_year = self.df.index[0]
        self.df.loc[self.df.index[0], 'capital_gain'] = copy(self.cg_to_date)
        self.df.loc[:, 'capital'] = 0.001
        if self.initial > 0:
            self.df.loc[previous_year, 'capital'] = copy(self.initial)
        else:
            self.df.loc[previous_year, 'capital'] = 0.001
            self.df.loc[previous_year, 'capital'],\
            self.df.loc[previous_year, 'capital_gain'],\
            self.df.loc[previous_year, 'withdrawal_cg'] = self.calculateCapitalAnnualized(capital=self.df.loc[previous_year, 'capital'],
                                                                           capital_gains=self.df.loc[previous_year, 'capital_gain'],
                                                                           contributions=self.df.loc[previous_year, 'contr'],
                                                                           withdrawals=self.df.loc[previous_year, 'withdrawals'],
                                                                           growth=self.growth)

        for year in self.df.index[1:self.number_working_years]:
            self.df.loc[year, 'capital'],\
            self.df.loc[year, 'capital_gain'],\
            self.df.loc[year, 'withdrawal_cg'] = self.calculateCapitalAnnualized(capital=self.df.loc[previous_year, 'capital'],
                                                                           capital_gains=self.df.loc[previous_year, 'capital_gain'],
                                                                           contributions=self.df.loc[year, 'contr'],
                                                                           withdrawals=self.df.loc[year, 'withdrawals'],
                                                                           growth=self.growth)

            if self.df.loc[year, 'capital'] == 0:
                self.df.loc[year, 'withdrawals'] = 0
                self.df.loc[year, 'capital_gain'] = 0
            previous_year = year
        
        #  Retirement year:
        self.df.loc[self.retirement_fy_end, 'capital'],\
        self.df.loc[self.retirement_fy_end, 'capital_gain'],\
        self.df.loc[self.retirement_fy_end, 'withdrawal_cg'] = self.calculateCapitalAnnualized(capital=self.df.loc[previous_year, 'capital'],
                                                                       capital_gains=self.df.loc[previous_year, 'capital_gain'],
                                                                       contributions=self.df.loc[year, 'contr'],
                                                                       withdrawals=self.df.loc[year, 'withdrawals'],
                                                                       growth=self.growth,
                                                                       installments=self.ret_year_installments)
        #  Add lump sum at end of retirement year
        self.df.loc[self.retirement_fy_end, 'capital'] += self.ra_lump_sum        
        
        previous_year = copy(self.retirement_fy_end)
        #  Post-retirement:
        for year in self.df.index[self.number_working_years:]:
            self.df.loc[year, 'capital'],\
            self.df.loc[year, 'capital_gain'],\
            self.df.loc[year, 'withdrawal_cg'] = self.calculateCapitalAnnualized(capital=self.df.loc[previous_year, 'capital'],
                                                                           capital_gains=self.df.loc[previous_year, 'capital_gain'],
                                                                           contributions=self.df.loc[year, 'contr'],
                                                                           withdrawals=self.df.loc[year, 'withdrawals'],
                                                                           growth=self.growth)

            if self.df.loc[year, 'capital'] == 0:
                self.df.loc[year, 'withdrawals'] = 0
                self.df.loc[year, 'capital_gain'] = 0
            previous_year = year

