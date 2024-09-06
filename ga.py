import random
import numpy as np
import pyspiel

class GeneticAlgorithm:
    def __init__(self, game: pyspiel.Game, population_size: int, generations: int, mutation_rate: float, crossover_rate: float, player: int):
        self.game = game
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.player = player
        self.population = self.initialize_population()

    def initialize_population(self):
        num_actions = self.game.num_distinct_actions()
        return [np.random.rand(num_actions) for _ in range(self.population_size)]

    def fitness(self, strategy):
        state = self.game.new_initial_state()
        return self.evaluate_strategy(state, strategy)

    def evaluate_strategy(self, state: pyspiel.State, strategy):
        while not state.is_terminal():
            legal_actions = state.legal_actions()
            action = self.select_action(strategy, legal_actions)
            state.apply_action(action)
        
        returns = state.returns()
        my_return = returns[self.player] if self.player < len(returns) else 0
        opponent_return = returns[1 - self.player] if len(returns) > 1 else 0
        fitness_value = (my_return - opponent_return) / (self.game.max_utility() - self.game.min_utility())
        return max(0.01, fitness_value)  # Ensure fitness is never zero

    def select_action(self, strategy, legal_actions):
        action_probabilities = [strategy[action] for action in legal_actions]
        total_prob = sum(action_probabilities)
        
        if total_prob <= 0:
            # Handle case where all probabilities are zero
            if not legal_actions:
                raise ValueError("No legal actions available for strategy")
            return random.choice(legal_actions)
        
        # Normalize action probabilities
        action_probabilities = [prob / total_prob for prob in action_probabilities]
        action = random.choices(legal_actions, weights=action_probabilities, k=1)[0]
        return action

    def selection(self):
        fitness_scores = [self.fitness(individual) for individual in self.population]
        selected_indices = np.argsort(fitness_scores)[-self.population_size//2:]
        return [self.population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] += random.gauss(0, 0.3)  # Gaussian mutation
        return individual

    def evolve(self):
        for _ in range(self.generations):
            new_population = []
            selected = self.selection()
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
            self.population = new_population

    def step(self, state):
        self.evolve()  # Evolve the population before choosing an action
        best_strategy = max(self.population, key=self.fitness)
        action = self.select_action(best_strategy, state.legal_actions())
        return action



    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions."""
        action_string = state.action_to_string(player_id, action)
        print(f"Player {player_id} took action: {action_string}")
        
    def restart(self):
        """Resets the board to the initial state."""
        self.state = self.game.new_initial_state()
        self.current_individual = None