import pyspiel
from ga import GeneticAlgorithm
import matplotlib.pyplot as plt

# Create a Go game (you can adjust the board size)
game = pyspiel.load_game("go", {"board_size": 9})

# Initialize the Genetic Algorithm
ga = GeneticAlgorithm(
    game=game,
    population_size=200,
    generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    player=0  # Assuming we're training for player 0
)

# Train the model
best_strategy, fitness_history, top_strategies = ga.train()

# Save the trained model
ga.save_model("go_ga_model.pkl")

print("Training complete. Model saved as go_ga_model.pkl")

# Plot average fitness history
generations, avg_fitnesses = zip(*fitness_history)
plt.figure(figsize=(10, 6))
plt.plot(generations, avg_fitnesses, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Average Fitness (Win Rate)')
plt.title('Average Fitness History')
plt.legend()
plt.grid(True)
plt.savefig('average_fitness_history.png')
plt.close()

print(f"Number of top strategies saved: {len(top_strategies)}")
print(f"Fitness range of top strategies: {top_strategies[-1][1]:.2f} - {top_strategies[0][1]:.2f}")
