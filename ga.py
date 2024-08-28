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
        state = self.game.new_initial_state()
        actions = []
        while not state.is_terminal():
            legal_actions = state.legal_actions()  # Get all legal actions
            if not legal_actions:
                break  # Break if no legal actions are available

            random_action = random.choice(legal_actions)  # Choose a random action from the legal ones
            actions.append(random_action)
            state.apply_action(random_action)  # Apply the selected action

        return actions

    def fitness(self, individual):
        """Evaluates the fitness of an individual based on Go-specific metrics."""
        state = self.game.new_initial_state()
        for action in individual:
            if state.is_terminal():
                break
            state.apply_action(action)

        # Get the outcome of the game
        game_result = state.returns()
        territory_control = game_result[state.current_player()]  # Total score for the player

        # Fitness is higher for wins and based on territory control
        if game_result[state.current_player()] > 0:
            fitness_score = territory_control  # Positive score for winning
        else:
            fitness_score = -1 * (abs(game_result[state.current_player()]))  # Penalize for losing

        return fitness_score

    def heuristic_evaluation(self, state, action):
        """Heuristic to evaluate potential future actions."""
        future_state = state.clone()
        future_state.apply_action(action)
        future_returns = future_state.returns()[state.current_player()]  # Future reward estimation
        return future_returns

    def selection(self, population, fitness_scores):
        """Selects individuals based on their fitness using tournament selection."""
        tournament_size = 5
        selected = random.choices(population, weights=fitness_scores, k=tournament_size)
        return max(selected, key=self.fitness), max(selected, key=self.fitness)

    def crossover(self, parent1, parent2):
        """Performs crossover between two parents to produce offspring."""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            child1, child2 = parent1, parent2
        return child1, child2

    def mutate(self, individual):
        """Mutates an individual by randomly changing one of its actions."""
        state = self.game.new_initial_state()
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                legal_actions = state.legal_actions()
                if legal_actions:
                    individual[i] = random.choice(legal_actions)
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
        """Selects an action based on the current state of the game using long-term planning."""
        best_action = None
        best_fitness = -float('inf')

        for individual in self.population:
            # Apply the individual's strategy up to the current state
            current_state = self.game.new_initial_state()
            for action in individual:
                if current_state.is_terminal():
                    break
                current_state.apply_action(action)

            # Evaluate potential future actions based on heuristic
            for action in state.legal_actions():
                future_fitness = self.heuristic_evaluation(state, action)
                if future_fitness > best_fitness:
                    best_fitness = future_fitness
                    best_action = action

        if best_action:
            return best_action
        else:
            # If no best action found, return a random legal action
            return random.choice(state.legal_actions())

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions."""
        action_string = state.action_to_string(player_id, action)
        print(f"Player {player_id} took action: {action_string}")
        
    def restart(self):
        """Resets the board to the initial state."""
        self.state = self.game.new_initial_state()
        self.current_individual = None