#!/usr/bin/env python
# coding: utf-8

# # Traveling Salesman
# # Genetic Algorithm

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Genes:

# In[2]:


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis**2) + (yDis**2))
        return distance

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"


# Erstellen von "zufälligen" Städten:

# In[3]:


def createCities(n, shape=None):
    cityList = []

    if shape == "circle":
        for i in range(n):
            theta = np.random.random() * 2 * np.pi
            x = 50 * np.cos(theta) + 50
            y = 50 * np.sin(theta) + 50
            cityList.append(City(x, y))
    else:
        for i in range(n):
            x = np.random.randint(1, 100)
            y = np.random.randint(1, 100)
            cityList.append(City(x, y))

    return np.array(cityList)


# In[6]:


cities = createCities(100, "circle")
# print(cities)

x = []
y = []
for city in cities:
    x.append(city.x)
    y.append(city.y)
plt.scatter(x, y)
plt.show()

cities = createCities(10)
print(cities)

x = []
y = []
for city in cities:
    x.append(city.x)
    y.append(city.y)
plt.scatter(x, y)
plt.show()


# Erstellen der zufälligen Route

# In[7]:


def createRoute(cityList):
    route = cityList.copy()
    np.random.shuffle(route)
    return route


# In[8]:


route = createRoute(cities)

print(route)
print(cities)


# # Erstellen der Population
# Anzahl an verschiedenen Routen

# In[9]:


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))

    return np.array(population)


# In[10]:


pop = initialPopulation(100, cities)
print(pop.shape)
print(pop[:3])


# # Bestimmen der Fitness
# für eine bestimmte Route

# In[11]:


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / self.routeDistance()
        return self.fitness

    def __str__(self):
        return f"Fitness: {self.routeFitness()}, Distanz: {self.routeDistance()}"


# In[12]:


print(Fitness(route))


# Routen nach der Fitness sortieren

# In[13]:


def rankRoutes(population):
    fitnessResults = {}
    for i in range(len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return np.array(sorted(fitnessResults.items(), key=lambda x: x[1], reverse=True))


# In[14]:


ranks = rankRoutes(pop)

print(ranks.shape)
print("Top five:")
print(ranks[:5])


# # Mating Pool
# Wer darf sich Fortpflanzen

# In[38]:


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.flip(popRanked, axis=0), columns=["Index", "Fitness"])
    df["cum_sum"] = df["Fitness"].cumsum()
    df["cum_perc"] = 100 * df["cum_sum"] / df["Fitness"].sum()

    #     print(df.tail())

    for i in range(eliteSize):
        selectionResults.append(popRanked[i][0])

    for i in range(len(popRanked) - eliteSize):
        pick = 100 * np.random.random()
        for i in range(len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return np.array(selectionResults, dtype="int")


# In[22]:


selects = selection(ranks, 20)
print(selects)
print(len(selects))
print(len(np.unique(selects)))


# In[23]:


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return np.array(matingpool)


# In[24]:


pop[3], selects[3]

# In[25]:


pool = matingPool(pop, selects)

print(pool.shape)
print(pool[:2])


# # Breed

# In[26]:


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(np.random.random() * len(parent1))
    geneB = int(np.random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return np.array(child)


# In[28]:


child = breed(pool[0], pool[1])

print(pool[0])
print(pool[1])
print(child)


# In[29]:


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = matingpool.copy()
    np.random.shuffle(pool)

    for i in range(eliteSize):
        children.append(matingpool[i])

    for i in range(length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return np.array(children)


# In[30]:


breedPop = breedPopulation(pool, 20)

print(breedPop.shape)
print(breedPop[:3])


# # Mutate

# In[31]:


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if np.random.random() < mutationRate:
            swapWith = np.random.randint(len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


# In[33]:


print(breedPop[0])
mutant = mutate(breedPop[0], 0.1)

print(mutant)


# In[34]:


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return np.array(mutatedPop)


# In[35]:


mutants = mutatePopulation(breedPop, 0.01)

print(mutants.shape)
print(mutants[:3])


# # Repeat with next Generation

# In[36]:


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


# In[39]:


nextGen = nextGeneration(mutants, 20, 0.01)

print(nextGen.shape)
print(nextGen[:3])


# # Evolve!

# In[40]:


def geneticAlgorithm(genes, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, genes)
    print(f"Initial distance: {1 / rankRoutes(pop)[0][1]}")

    progress = []

    for i in range(generations):
        progress.append(1 / rankRoutes(pop)[0][1])
        pop = nextGeneration(pop, eliteSize, mutationRate)

    progress.append(1 / rankRoutes(pop)[0][1])

    print(f"Final distance: {1 / rankRoutes(pop)[0][1]}")
    bestRouteIndex = int(rankRoutes(pop)[0][0])
    bestRoute = pop[bestRouteIndex]

    return bestRoute, progress


# # Laufen lassen

# In[51]:


cityList = createCities(10)

bestRoute, history = geneticAlgorithm(
    genes=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=100
)

# In[52]:


print(bestRoute)
# print(history)

plt.plot(np.arange(1, len(history) + 1), history)
plt.ylabel("Distance")
plt.xlabel("Generation")
plt.show()

# In[53]:


x = []
y = []
for city in cityList:
    x.append(city.x)
    y.append(city.y)
plt.scatter(x, y)

x = []
y = []
for city in bestRoute:
    x.append(city.x)
    y.append(city.y)
x.append(bestRoute[0].x)
y.append(bestRoute[0].y)

plt.plot(x, y, color="r")
plt.show()

# In[ ]:


# In[59]:


cityList = createCities(10, "circle")

x = []
y = []
for city in cityList:
    x.append(city.x)
    y.append(city.y)
plt.scatter(x, y)
plt.show()

bestRoute, history = geneticAlgorithm(
    genes=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=200
)

# In[60]:


# print(bestRoute)
# print(history)

plt.plot(np.arange(1, len(history) + 1), history)
plt.ylabel("Distance")
plt.xlabel("Generation")
plt.show()

# In[61]:


x = []
y = []
for city in cityList:
    x.append(city.x)
    y.append(city.y)
plt.scatter(x, y)

x = []
y = []
for city in bestRoute:
    x.append(city.x)
    y.append(city.y)
x.append(bestRoute[0].x)
y.append(bestRoute[0].y)

plt.plot(x, y, color="r")
plt.show()

# In[ ]:
