import pyspiel
from ga import GeneticAlgorithm
import matplotlib.pyplot as plt
import csv

# Create a Go game (you can adjust the board size)
game = pyspiel.load_game("go", {"board_size": 19})

# Initialize the Genetic Algorithm
ga = GeneticAlgorithm(
    game=game,
    population_size=500,
    generations=200,
    mutation_rate=0.1,
    crossover_rate=0.8,
    player=0  # Assuming we're training for player 0
)

# Train the model
best_strategy, fitness_history, top_strategies = ga.train()

# Save the trained model
ga.save_model("19x19_ga_model.pkl")

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
plt.savefig('average_fitness_history_19x19.png')
plt.close()

# Save all fitness values to a CSV file
with open('fitness_values.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Generation', 'Average Fitness'])
    for generation, (_, avg_fitness) in enumerate(fitness_history):
        csv_writer.writerow([generation, avg_fitness])

print("Fitness values saved to fitness_values.csv")

print(f"Number of top strategies saved: {len(top_strategies)}")
print(f"Fitness range of top strategies: {top_strategies[-1][1]:.2f} - {top_strategies[0][1]:.2f}")