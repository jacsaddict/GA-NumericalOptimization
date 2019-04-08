import sys
import time
import numpy as np


def _generate_parent(get_fitness, N):
    pool = np.random.randint(1023, size=(100,N))
    papulation = []
    for gene in pool:
        papulation.append(Chromosomes(gene,get_fitness(gene)))
    return np.array(papulation)

def _bitflip(num, position):
    return num ^ (1 << position)

def _mutate(parent, get_fitness, N, mutateGate):
    mutatedMask = np.random.random([N,10]) < mutateGate
    mutatedIndexA, mutatedIndexB = np.where(mutatedMask==True)
    if(mutatedIndexA.size == 0):
        return None
    mutatedGenes = Chromosomes(np.copy(parent.Genes), -99999)
    for row in mutatedIndexA:
        for col in mutatedIndexB:
            mutatedGenes.Genes[row] = _bitflip(parent.Genes[row],col)
    mutatedGenes.Fitness = get_fitness(mutatedGenes.Genes)
    return mutatedGenes


def _crossover(parentGenesA, parentGenesB, get_fitness, N):
    # two point cross over
    arrayPoints = np.sort(np.random.choice(N, 2, replace=False))
    a,b,c = np.split(parentGenesA,arrayPoints)
    d,e,f = np.split(parentGenesB,arrayPoints)
    crossover1 = np.concatenate((a,e,c))
    crossover2 = np.concatenate((d,b,f))
    
    return np.array([Chromosomes(crossover1, get_fitness(crossover1)),
               Chromosomes(crossover2,get_fitness(crossover2))])


def _tournament(parents, get_fitness):
    winner = []
    for i in range(100):
        first, second = np.random.randint(100, size=(2))
        winner.append(first if parents[first].Fitness > parents[second].Fitness else second)
    return winner

def _selection(parent):
    Fitness = np.array([gene.Fitness for gene in parent])
    return np.argpartition(Fitness, -100)[-100:]

def get_best(get_fitness, N):
    # initialize
    parent = _generate_parent(get_fitness, N)
    mutateGate = np.multiply(np.ones([N,10]), 0.01)

    for generations in range(500):
        #tournament selection
        winner = _tournament(parent, get_fitness)
        #cross over
        crossOverRate = np.random.random(25)
        for rate in crossOverRate:
            if rate < 0.9:
                first, second = np.random.choice(100, 2, replace=False)
                parent = np.concatenate((parent,
                              _crossover(parent[winner[first]].Genes,
                                         parent[winner[second]].Genes,get_fitness,N)))
        #mutation
        mutationList = []
        for chromo in parent:
            mutated = _mutate(chromo, get_fitness, N, mutateGate)
            if mutated is None:
                continue
            mutationList.append(mutated)
        parent = np.concatenate((parent, mutationList))
        #Survivor Selection
        parent = parent[_selection(parent)]
        if(generations % 50 == 0):
            print("Generations %d th:  minimum is %d" % (generations, -parent[0].Fitness))
    return parent[0].Genes, parent[0].Fitness


class Chromosomes:
    def __init__(self, genes, fitness):
        self.Genes = genes
        self.Fitness = fitness