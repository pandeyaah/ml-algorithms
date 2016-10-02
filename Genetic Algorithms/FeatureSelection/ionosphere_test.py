"""
We study on Ionosphere dataset (Iris only has 4 feats!)
Ionosphere has 34 features!
"""

from Algorithm import Algorithm
from Individual import Individual
from Fitness_Calc import FitnessCalc
from Population import Population
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


def trans_res(x):
    x = x.strip()
    if x == 'g':
        return 0
    else:
        return 1


if __name__ == '__main__':
    data = pd.read_csv("../../datasets/ionosphere/ionosphere.data")
    feats = data.columns[:-1]
    target_feat = 'result'
    data['result'] = data['result'].apply(trans_res)
    # shuffle data, since it is sequential in nature
    data = data.iloc[np.random.permutation(len(data))]

    # here we set the Initial population we will use further
    pop = Population(50, c_length=len(feats), initialize=True)

    # here we set the params of the algorithm
    Algorithm.set_params(chromosome_len=len(feats), crossover_rate=0.9, mutation_rate=0.001, elitism=True)
    print "len of feats :", len(feats)

    # here we specify the params of Fitness Calculator. You can set the model you wish to evaluate
    # features on
    FitnessCalc.set_params(len(feats), data, target_feat)

    print "All features Fitness :", FitnessCalc.calculate_fitness(np.ones((len(feats), ), dtype=np.int8))

    gen_count = 0
    print "running for 30 iterations......"

    for i in range(20):
        # we call the evolve method. Evolve method does crossover and mutation
        pop = Algorithm.evolve_population(pop)
        # The fitness score is Cross Validated already. So no need for Cross Validation
        print "Generation :", i
        print "Fitness of current Generation :", pop.get_fittest().get_fitness()

    print "Finished"
    # print "Num of Generations run : ", gen_count
    print "Final fitness : ", pop.get_fittest().get_fitness()
    print "Solution :"
    # print pop.get_fittest()
    print "the solution is printed in the Genome format itself. Will change it to return Actual feature names itself"
    res = pop.get_fittest()
    print res
    selected_features = []
    ind = 0
    for x in res.get_chromosome():
        if x == '1':
            selected_features.append(feats[ind])
        ind += 1

    print selected_features

    # 1 represents features used in the fittest, 0's are features that weren't used

