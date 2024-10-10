import random
import numpy as np
import pyspiel
import pickle
from typing import List, Tuple

class GeneticAlgorithm:
    def __init__(self, game: pyspiel.Game, population_size: int, generations: int, mutation_rate: float, crossover_rate: float, player: int):
        self.game = game
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.player = player
        self.population = [self.generate_random_strategy() for _ in range(population_size)]
        self.best_strategy = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        self.top_strategies: List[Tuple[List[float], float]] = []
        self.max_top_strategies = 1000
        self.num_fitness_games = 10  # Number of games to play for fitness evaluation

    def generate_random_strategy(self):
        # Generate a random strategy (list of probabilities for each possible move)
        # Ensure that the probabilities are positive
        return [max(0.0001, random.random()) for _ in range(self.game.num_distinct_actions())]

    def fitness(self, strategy):
        # Play multiple games against a random opponent and return the win rate
        wins = 0
        for _ in range(self.num_fitness_games):
            if self.play_game(strategy) > 0:
                wins += 1
        return wins / self.num_fitness_games

    def play_game(self, strategy):
        state = self.game.new_initial_state()
        while not state.is_terminal():
            if state.current_player() == self.player:
                action = self.select_action(strategy, state.legal_actions())
            else:
                action = random.choice(state.legal_actions())
            state.apply_action(action)
        return state.returns()[self.player]

    def select_action(self, strategy, legal_actions):
        action_probs = [strategy[action] if action < len(strategy) else 0 for action in legal_actions]
        total_prob = sum(action_probs)
        if total_prob <= 0:
            # If all probabilities are zero or negative, choose randomly
            return random.choice(legal_actions)
        else:
            return random.choices(legal_actions, weights=action_probs)[0]

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
        for generation in range(self.generations):
            fitnesses = [self.fitness(individual) for individual in self.population]
            
            # Update best strategy
            max_fitness = max(fitnesses)
            best_index = fitnesses.index(max_fitness)
            if max_fitness > self.best_fitness:
                self.best_strategy = self.population[best_index]
                self.best_fitness = max_fitness
            
            # Selection
            selected = self.selection()
            
            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
            self.population = new_population
            
            # Update top strategies
            for individual, fitness in zip(self.population, fitnesses):
                self.update_top_strategies(individual, fitness)
            
            # Record average fitness for this generation
            avg_fitness = sum(fitnesses) / len(fitnesses)
            self.fitness_history.append((generation, avg_fitness))
            
            if generation % 10 == 0:  # Print progress every 10 generations
                print(f"Generation {generation}: Avg Fitness = {avg_fitness:.2f}, Max Fitness = {max_fitness:.2f}")

    def update_top_strategies(self, strategy: List[float], fitness: float):
        """Update the list of top strategies."""
        if len(self.top_strategies) < self.max_top_strategies:
            self.top_strategies.append((strategy, fitness))
            self.top_strategies.sort(key=lambda x: x[1], reverse=True)
        elif fitness > self.top_strategies[-1][1]:
            self.top_strategies.append((strategy, fitness))
            self.top_strategies.sort(key=lambda x: x[1], reverse=True)
            self.top_strategies = self.top_strategies[:self.max_top_strategies]

    def train(self):
        """Train the genetic algorithm and return the best strategy."""
        self.evolve()
        return self.best_strategy, self.fitness_history, self.top_strategies

    def save_model(self, filename):
        """Save the best strategy and top strategies to a file."""
        data = {
            'best_strategy': self.best_strategy,
            'best_fitness': self.best_fitness,
            'top_strategies': self.top_strategies,
            'fitness_history': self.fitness_history
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """Load a saved strategy from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.best_strategy = data['best_strategy']
        self.best_fitness = data['best_fitness']
        self.top_strategies = data['top_strategies']
        self.fitness_history = data['fitness_history']
        print(f"Model loaded from {filename}")

    def step(self, state):
        """Use one of the top strategies to select an action."""
        if not self.top_strategies:
            raise ValueError("No strategies available. Train or load a model first.")
        
        # Randomly select one of the top 10 strategies
        strategy, _ = random.choice(self.top_strategies[:10])
        action = self.select_action(strategy, state.legal_actions())
        return action

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions."""
        action_string = state.action_to_string(player_id, action)
        print(f"Player {player_id} took action: {action_string}")
        
    def restart(self):
        """Resets the board to the initial state."""
        self.state = self.game.new_initial_state()
        self.current_individual = None