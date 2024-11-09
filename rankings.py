import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("./results/trained_mcts.csv")

# Initialize player ratings dictionary with initial ratings
# Replace with your actual initial ratings
initial_ratings = {
    "mcts_advanced": 2350,  # Example agent name and initial rating
    "mcts_intermediate": 1600,
    "mcts_casual": 600,
    "mcts_beginner": 100,
    "mcts": 1000
}

ratings = {**initial_ratings}  # Start with initial ratings

# Function to calculate expected score based on rating difference
def expected_score(diff, a):
    return 1 / (np.exp(diff / a) + 1)

# Function to update ratings with EGF-style Elo
def update_elo(player_a, player_b, result, K_a, K_b):
    # Retrieve or initialize ratings
    rating_a = ratings.get(player_a, 1000)
    rating_b = ratings.get(player_b, 1000)
    
    # Determine 'a' factor based on rating (higher fluctuation for lower-rated players)
    a_a = 350 if rating_a < 1000 else 300 if rating_a < 1800 else 250
    a_b = 350 if rating_b < 1000 else 300 if rating_b < 1800 else 250
    
    # Calculate expected scores
    S_E_a = expected_score(rating_b - rating_a, a_a)
    S_E_b = 1 - S_E_a  # For player B
    
    # Update ratings based on actual result and K factor
    ratings[player_a] = rating_a + K_a * (result - S_E_a)
    ratings[player_b] = rating_b + K_b * ((1 - result) - S_E_b)

# Iterate through each game and update EGF-style Elo ratings
for _, row in df.iterrows():
    player1 = row['Player 1 Name (Black)']
    player2 = row['Player 2 Name (White)']
    winner = row['Winner']
    
    # Ensure players have initial ratings if not already present
    if player1 not in ratings:
        ratings[player1] = 1000  # Or another default rating
    if player2 not in ratings:
        ratings[player2] = 1000  # Or another default rating

    # Determine K factors based on player ratings
    K_p1 = 116 if ratings[player1] < 1000 else 32 if ratings[player1] < 2700 else 10
    K_p2 = 116 if ratings[player2] < 1000 else 32 if ratings[player2] < 2700 else 10
    
    # Determine result
    result_p1 = 1 if winner == "Black" else 0  # 1 if player 1 wins, 0 if player 2 wins
    update_elo(player1, player2, result_p1, K_p1, K_p2)

# Convert Elo ratings to Go ranks using approximate EGF mappings
def elo_to_go_rank(elo_rating):
    if elo_rating < 1200:
        return f"{int((1200 - elo_rating) / 100) + 10}k"  # Double-digit kyu
    elif elo_rating < 2000:
        return f"{int((2000 - elo_rating) / 100)}k"  # Single-digit kyu
    else:
        return f"{int((elo_rating - 200) / 100) + 1}d"  # Dan ranks

# Display results
ranking = sorted(ratings.items(), key=lambda x: -x[1])
for player, rating in ranking:
    go_rank = elo_to_go_rank(rating)
    print(f"{player}: Elo Rating = {rating:.2f}, Go Rank = {go_rank}")
