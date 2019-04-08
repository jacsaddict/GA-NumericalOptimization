import time
import numpy as np
import genetic

N = 10

def get_fitness(guess):
    x = np.subtract(guess, 512)
    return -(418.98291*N - np.sum(np.multiply(x, np.sin(np.sqrt(np.abs(x))))))


class NumericalOptimization():

    def Optimization(self):
        startTime = time.perf_counter()

        def fnGetFitness(genes):
            return get_fitness(genes)

        optimalFitness = 0
        best = genetic.get_best(fnGetFitness, N)
        endTime = time.perf_counter()
        print("Time Cost: ", endTime - startTime, "seconds")
        return best


if __name__ == '__main__':
    NO = NumericalOptimization()
    bestGene, bestFitness = NO.Optimization()
    print("The best X : ", np.subtract(bestGene, 512))
    print("With minimum : ", -bestFitness)