import numpy as np
import pyspiel
import random

class GeneticAlgorithm:
    def __init__(self, game, population_size, generations, mutation_rate, crossover_rate):
        self.game = game
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.initialize_population()
        self.current_individual = None

    def initialize_population(self):
        """Initializes the population with random strategies."""
        population = []
        for _ in range(self.population_size):
            individual = self.random_strategy()
            population.append(individual)
        return population

    def random_strategy(self):
        """Creates a random strategy for the game."""
        actions = []
        for _ in range(self.game.num_players()):
            actions.append(self.game.random_action())
        return actions

    def fitness(self, individual):
        """Evaluates the fitness of an individual."""
        state = self.game.new_initial_state()
        for action in individual:
            if state.is_terminal():
                break
            state.apply_action(action)
        return state.returns()

    def selection(self, population, fitness_scores):
        """Selects individuals based on their fitness."""
        selected = random.choices(population, weights=fitness_scores, k=2)
        return selected

    def crossover(self, parent1, parent2):
        """Performs crossover between two parents to produce offspring."""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            child1, child2 = parent1, parent2
        return child1, child2

    def mutate(self, individual):
        """Mutates an individual by randomly changing one of its actions."""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = self.game.random_action()
        return individual

    def evolve(self):
        """Runs the genetic algorithm to evolve the population."""
        for generation in range(self.generations):
            fitness_scores = [self.fitness(ind) for ind in self.population]
            new_population = []

            for _ in range(self.population_size // 2):
                parent1, parent2 = self.selection(self.population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population
            print(f"Generation {generation + 1}, Best Fitness: {max(fitness_scores)}")

    def step(self, state):
        """Selects an action based on the current state of the game."""
        # For simplicity, you might select the action from the best individual in the current population
        # Assuming a simple implementation where the best strategy from the population is chosen
        if self.current_individual is None:
            # Choose the best individual from the initial population
            fitness_scores = [self.fitness(ind) for ind in self.population]
            self.current_individual = self.population[fitness_scores.index(max(fitness_scores))]
        
        action = self.current_individual[state.current_player()]
        return action
