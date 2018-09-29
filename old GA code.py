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

	------------

if self.optimizer == 'GA':
    self.pop_size = 100
    self.ngen = 20
    self.GA()
    self.solution = self.fractionsToRands(self.reshape(self.best_ind))
        
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
    

"""
