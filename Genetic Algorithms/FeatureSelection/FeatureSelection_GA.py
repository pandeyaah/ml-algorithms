import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import random


def rounder(num):
    num = str(num)
    l = num.split(".")
    res = int(l[0])
    if int(l[1][0]) >= 5:
        res += 1
    return res


class FitnessCalc:
    """
    The Fitness Calculator. Less of a class but implemented as one.
    The fitness function is simple Classification Accuracy

    Static Properties
    -----------------
    1. max_fitness: stores the maximum fitness achieved
    2. solution: the solution chromosome
    3. data: data to evaluate
    4. target_feat: target_feat
    5. eval_method: cross validation or train_test split
    6. num_folds: if cv eval method is used, then supplies num of folds
    7. model: Classifier to evaluate

    Static Methods
    --------------
    1. set_params(): used to set parameters of Fitness Calculator
    2. calculate_fitness(): to calculate fitness of a population
    3. get_max_fitness(): get max fitness score

    """
    max_fitness = 0.0
    solution = None
    data = None
    target_feat = None
    eval_method = 'cv'
    num_folds = 0
    model = None

    # def __init__(self, chromosome_len, data, target_feat, fitness_metric=None,
    #              eval_method='cv', num_folds=5):
    #     if not chromosome_len:
    #         ValueError("Chromosome Length not provided")
    #     else:
    #         FitnessCalc.solution = np.ones((chromosome_len, ), dtype=np.int8)
    #         FitnessCalc.max_fitness = FitnessCalc.calculate_fitness(FitnessCalc.solution)
    #         FitnessCalc.data = data
    #         FitnessCalc.target_feat = target_feat
    #         if fitness_metric:
    #             FitnessCalc.fitness_metric = fitness_metric
    #         else:
    #             FitnessCalc.fitness_metric = accuracy_score
    #         if eval_method != 'cv' or eval_method != 'train_test_split':
    #             ValueError("Incorrect eval_method Parameter. Should be either 'cv' or 'train_test_split'")
    #         FitnessCalc.eval_method = eval_method
    #         if eval_method == 'cv':
    #             FitnessCalc.num_folds = num_folds

    @staticmethod
    def set_params(chromosome_len, data, target_feat,
                   eval_method='cv', num_folds=5, model=LogisticRegression()):
        """
        Used to set the parameters used by FitnessCalc internally
        Parameters
        ----------
        chromosome_len: integer.
            determines the length of the chromosome (solution) of each of the Individuals of the Population

        data: array like. shape (m x chromosomelen)
            data to run feature selection upon

        target_feat: target feature for the dataset

        eval_method: string. Optional(default='cv')
            'cv'-> cross validation
            'train_test_split' -> Train and test split type Validation

        num_folds: integer. optional(default=5)
            if eval_method is 'cv', determines no of folds.

        model: optional. Default=LogisticRegression
            classifier model to check feature performance on
        """
        FitnessCalc.solution = np.ones((chromosome_len, ), dtype=np.int8)
        FitnessCalc.data = data
        FitnessCalc.target_feat = target_feat
        # if fitness_metric:
        #    FitnessCalc.fitness_metric = fitness_metric
        # else:
        #    FitnessCalc.fitness_metric = accuracy_score
        if eval_method != 'cv' or eval_method != 'train_test_split':
            ValueError(
                "Incorrect eval_method Parameter. Should be either 'cv' or 'train_test_split'")
        FitnessCalc.eval_method = eval_method
        if eval_method == 'cv':
            FitnessCalc.num_folds = num_folds
        FitnessCalc.model = model
        FitnessCalc.max_fitness = FitnessCalc.calculate_fitness(
            FitnessCalc.solution)

    @staticmethod
    def calculate_fitness(solution):
        """
        To Calculate fitness of a given solution (chromosome)
        Parameters
        ----------
        solution: chromosome to evaluate

        Return
        ------
        fitness_score: Fitness score of the solution. (Basically Classification Accuracy)
            higher the better
        """
        features = FitnessCalc.data.columns
        features = features.difference([FitnessCalc.target_feat])

        to_use = []
        for x in range(solution.size):
            if solution[x] == 1:
                to_use.append(features[x])
        # y = FitnessCalc.data[FitnessCalc.target_feat].values
        fitness_score = 0.0
        clf = FitnessCalc.model
        if FitnessCalc.eval_method == 'cv':
            skf = StratifiedKFold(FitnessCalc.data[FitnessCalc.target_feat].values,
                                  n_folds=FitnessCalc.num_folds, random_state=42)

            for train_index, test_index in skf:
                visible_train = FitnessCalc.data[
                    to_use].iloc[train_index].values
                visible_y = FitnessCalc.data[
                    FitnessCalc.target_feat].iloc[train_index].values
                blind_train = FitnessCalc.data[to_use].iloc[test_index].values
                blind_y = FitnessCalc.data[
                    FitnessCalc.target_feat].iloc[test_index].values

                clf.fit(visible_train, visible_y)
                preds = clf.predict(blind_train)
                acc = accuracy_score(blind_y, preds)
                fitness_score += acc
            fitness_score /= FitnessCalc.num_folds

        else:
            X_train, X_test, y_train, y_test = train_test_split(FitnessCalc.data[to_use].values,
                                                                FitnessCalc.data[
                                                                    FitnessCalc.target_feat].values,
                                                                test_size=0.20)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            fitness_score = accuracy_score(y_test, preds)

        if fitness_score >= FitnessCalc.max_fitness:
            FitnessCalc.max_fitness = fitness_score

        return fitness_score

    @staticmethod
    def get_max_fitness():
        """
        Returns max fitness recorded

        Return
        ------
        FitnessCalc.max_fitness
        """
        return FitnessCalc.max_fitness


class Individual(object):
    """
    Class that represents an entity in the GA, which is in turn the representative of the solution
    in the form of Chromosomes

    Private Properties
    ------------------
    1. chromosome: The actual solution bit encoded
    2. fitness: The fitness of the given individual

    Public Methods
    --------------
    1. Constructor *__init__* (): Constructs the Individual
    2. generate_individual(): generates the individual
    3. size(): returns the chromosome size
    4. get_gene(): returns an allele at an index
    5. set_gene(): sets the allele at an index
    6. get_fitness(): gets the fitness for the current individual
    7. get_get_chromosome(): get the entire chromosome (solution) represented by the individual
    """

    def __init__(self, chromosome_len):
        """
        Initialize the Individual
        Parameter
        --------
        chromosome_len: integer
            length of the chromosome
        """
        if not chromosome_len:
            ValueError("Chromosome Length not provided")

        else:
            self.__chromosome = np.zeros((chromosome_len,), dtype=np.int8)

        self.__fitness = 0.0

    def generate_individual(self):
        """
        Generates the individual by assigning random alleles
        """
        self.__chromosome = np.random.randint(2, size=self.size())

    def size(self):
        """
        Returns the size of the chromosome
        Return
        ------
        chromosome.size: size of the chromosome
        """
        return self.__chromosome.size

    def get_gene(self, index):
        """
        Returns gene at a specific index (zero based)
        Parameter
        ---------
        index: integer.

        Return
        ------
        gene at index
        """
        return self.__chromosome[index]

    def set_gene(self, index, value):
        """
        Sets the gene at a specific index
        Parameters
        ---------
        index: integer.
        value: integer {0,1}
        """
        self.__chromosome[index] = value
        self.__fitness = 0

    def get_fitness(self):
        """
        Get fitness of the Individual

        Return
        ------
        fitness : float
        """
        if self.__fitness == 0:
            self.__fitness = FitnessCalc.calculate_fitness(self.__chromosome)
        return self.__fitness

    def get_chromosome(self):
        """
        Get the entire chromosome of the Individual
        Return
        ------
        chromosome_str: string. The string representation of the Individual's Chromosome
        """
        chromosome_str = ''
        for i in range(self.size()):
            chromosome_str += str(self.__chromosome[i])
        return chromosome_str

    def __repr__(self):
        return self.get_chromosome()


class Population(object):
    """
    Population class represents the Population of Individuals under GA

    Private Properties
    -----------------
    1. individuals: list of individuals

    Public Methods
    -------------
    1. Population(): Constructor
    2. size(): returns number of Individuals in the population
    3. save_individual(): Saves individual at a given index
    4. get_individual(): gets Individual at a given index
    5. get_fittest(): gets the fittest Individual

    """

    def __init__(self, population_size, c_length, initialize=False):
        """
        Constructor of the Population class
        Parameters
        ----------
        population_size: integer.
            Determines the no of Individuals in the population

        c_length: integer
            Chromosome length

        initialize: boolean
            Determines whether to initialize all the individuals of the Population as well
        """
        self.__individuals = [Individual(
            chromosome_len=c_length) for x in range(population_size)]

        if initialize:
            for i in range(self.size()):
                indi = Individual(chromosome_len=c_length)
                indi.generate_individual()
                self.save_individual(i, indi)

    def size(self):
        """
        Returns the no of individuals in the Population
        """
        return len(self.__individuals)

    def save_individual(self, index, indiv):
        """
        Saves the individual at a specific index
        Parameters
        ----------
        index: integer.
            Determines the index to save the individual at
        indiv:  Individual
            the individual to save
        """
        self.__individuals[index] = indiv

    def get_individual(self, index):
        """
        Gets the individual at a specific index
        Parameter
        --------
        index: integer.
            Location of the individual to get

        Return
        ------
        Individual at the index
        """
        return self.__individuals[index]

    def get_fittest(self):
        """
        Determines and returns the fittest individual of the lot

        Return
        ------
        fittest: Individual
            Individual with the best Fitness score
        """
        fittest = self.__individuals[0]
        for i in range(self.size()):
            if fittest.get_fitness() <= self.get_individual(i).get_fitness():
                fittest = self.get_individual(i)
        return fittest


class Algorithm(object):
    """
    Implements the actual GA Algorithm
    Static Properties
    -----------------
    1. crossover_rate : Determines the rate of crossover
    2. mutation_rate : mutation rate
    3. tournament_size : determines the tournament size. tournament is the pool from which Individual are selected
    4. elitism : switches elitism on/off
    5. genome_len : length of the genome (intended solution)

    Static Methods
    --------------
    1. set_params() : set params for use
    2. evolve_population() : to evolve the current population
    3. tournament_selection() : Tournament Selection Algorithm
    4. crossover() : To make the crossover
    5. mutate() : To mutate the offsprings yielded by CrossOver
    """
    crossover_rate = 0.8
    mutation_rate = 0.01
    tournament_size = 10
    elitism = True
    genome_len = None

    @staticmethod
    def set_params(chromosome_len, crossover_rate=0.8, mutation_rate=0.01, tournament_size=10, elitism=True):
        """
        Used to set the parameters used internally by the Algorithm
        Parameters
        ----------
        chromosome_len: integer,
            determines the length of the chromosome (solution) of each of the Individuals of the Population

        crossover_rate: float. Optional (default=0.8)
            determines the crossover rate of the Crossover Procedure.
            optimum value range -> [0.8, 0.95]

        mutation_rate: float. Optional (default=0.01)
            governs the rate of mutation of the Mutation Procedure.
            optimum value range -> [1e-3, 1e-1]

        tournament_size: integer. Optional(default=10)
            determines the size of the Tournament

        elitism: boolean. Optional(default=True)
            determines whether we follow elitism or not
        """
        Algorithm.elitism = elitism
        Algorithm.crossover_rate = crossover_rate
        Algorithm.mutation_rate = mutation_rate
        Algorithm.tournament_size = tournament_size
        Algorithm.genome_len = chromosome_len

    @staticmethod
    def evolve_population(population):
        """
        Evolves the supplied Population and returns a new one.
        Parameters
        ----------
        population: Population data-type
            the population to evolve

        Returns
        -------
        new_population : Evolved Population
        """
        new_population = Population(
            population.size(), c_length=Algorithm.genome_len, initialize=False)

        if Algorithm.elitism:
            new_population.save_individual(0, population.get_fittest())
            elitism_offset = 1
        else:
            elitism_offset = 0

        for i in range(elitism_offset, population.size()):
            indiv = Algorithm.tournament_selection(population)
            indiv2 = Algorithm.tournament_selection(population)
            new_indiv = Algorithm.crossover(indiv, indiv2)
            new_population.save_individual(i, new_indiv)

        for i in range(elitism_offset, new_population.size()):
            Algorithm.mutate(new_population.get_individual(i))

        return new_population

    @staticmethod
    def tournament_selection(pop):
        """
        Applies the Fitness Proportionate Selection Procedure to return the fittest Individual
        Parameters
        ----------
        pop: of type Population.
            Population to select from

        Returns
        -------
        fittest: type- Individual
            the fittest individual of the lot
        """
        tournament = Population(Algorithm.tournament_size,
                                c_length=Algorithm.genome_len, initialize=False)
        for i in range(Algorithm.tournament_size):
            random_id = int(random.random() * pop.size())
            tournament.save_individual(i, pop.get_individual(random_id))

        fittest = tournament.get_fittest()
        return fittest

    @staticmethod
    def crossover(indiv1, indiv2):
        """
        Crosses over indiv1 and indiv2 to return new Individual
        Parameters
        ----------
        indiv1: Individual. Parent 1

        indiv2: Individual. Parent 2

        Returns
        -------
        new_sol: Individual
            child as result of the Crossover
        """
        new_sol = Individual(Algorithm.genome_len)
        for i in range(indiv1.size()):
            if random.random() <= Algorithm.crossover_rate:
                new_sol.set_gene(i, indiv1.get_gene(i))
            else:
                new_sol.set_gene(i, indiv2.get_gene(i))

        return new_sol

    @staticmethod
    def mutate(indiv):
        """
        Applies Mutation on Individual
        Parameter
        ---------
        indiv: Individual
            Individual to mutate (inplace)
        """
        for i in range(indiv.size()):
            rand = random.random()
            if rand <= Algorithm.mutation_rate:
                gene = rounder(rand)
                indiv.set_gene(i, gene)


def GA_FeatureSelection(data, target_feat, model=LogisticRegression(), population_size=10, chromosome_length=5,
                        generations=10, crossover_rate=0.80, mutation_rate=0.01, tournament_size=10, elitism=True, evaluation_method='cv', n_folds=5):
    """
    Feature Selection Algorithm implemented using GA that evaluates the best combination of features
    Incorporates cross validation. So you get valid and tested results

    Parameters
    ----------
    data: array like. Shape (MxChromosome_length)
        the data to work upon

    target_feat: string
        Target feature label

    model: Optional(default=LogisticRegression())
        Classifier to run check upon

    population_size: integer. Optional (default=20)
        Size of the population

    chromosome_length: integer. same as Number of features

    generations: integer. Optional (default=20)
        number of generations to run the heuristics for

    crossover_rate: float. Optional (default=0.80)
        Determines the rate of crossover. Optimum Range [0.80, 0.95]

    mutation_rate: float. Optional (default=0.01)
        Determines the rate of Mutation. Optimum Range [1e-3, 1e-1]

    tournament_size: integer. Optional (default=10)
        The size of tournament selection

    elitism: boolean. Optional(default=True)
        Switches Elitism

    evaluation_method: string. Optional(default='cv')
        'cv': yields StratifiedKFold Cross validation
        'train_test_split': train_test_split

    n_folds: integer. Optional(default=5)
        if evaluation_method is 'cv', it is used to set num_folds

    Return
    ------
    selected_features: list of string labels of selected features.
    """

    pop = Population(population_size=population_size,
                     c_length=chromosome_length, initialize=True)
    Algorithm.set_params(chromosome_len=chromosome_length, crossover_rate=crossover_rate,
                         mutation_rate=mutation_rate, elitism=elitism, tournament_size=tournament_size)

    FitnessCalc.set_params(chromosome_len=chromosome_length, data=data, target_feat=target_feat, model=model,
                           eval_method=evaluation_method, num_folds=n_folds)

    print("All features fitness score :", FitnessCalc.calculate_fitness(
        np.ones((chromosome_length, ), dtype=np.int8)))

    print("Running for %d generations...." % generations)
    for i in range(generations):
        print("Generation ", i)
        # we call the evolve method. Evolve method does crossover and mutation
        pop = Algorithm.evolve_population(pop)
        # The fitness score is Cross Validated already. So no need for Cross
        # Validation
        print("Fitness of current Generation :",
              pop.get_fittest().get_fitness())

    print("Finished")
    print("Final fitness : ", pop.get_fittest().get_fitness())
    print("Solution :")
    res = pop.get_fittest()
    print(res)
    selected_features = []
    features = data.columns
    features = features.difference([target_feat])
    i = 0
    for x in res.get_chromosome():
        if x == '1':
            selected_features.append(features[i])
        i += 1
    return selected_features
