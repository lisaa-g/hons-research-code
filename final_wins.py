import pandas as pd

# Load the CSV file
file_path = './results/trained_mcts.csv'
data = pd.read_csv(file_path)

# Get all unique players
players = pd.concat([data['Player 1 Name (Black)'], data['Player 2 Name (White)']]).unique()

# Initialize win counts and total games for each player
win_counts_black = {player: 0 for player in players}
win_counts_white = {player: 0 for player in players}
total_games = {player: 0 for player in players}

# Check the first few rows to understand the structure (for debugging)
print(data.head())

# Calculate wins and total games
for index, row in data.iterrows():
    player_1 = row['Player 1 Name (Black)']
    player_2 = row['Player 2 Name (White)']
    winner = row['Winner']

    # Update total games for each player
    total_games[player_1] += 1
    total_games[player_2] += 1

    # Update win counts based on who won the game
    if winner == 'Black':  # Player 1 (Black) won
        win_counts_black[player_1] += 1
    elif winner == 'White':  # Player 2 (White) won
        win_counts_white[player_2] += 1

# Print the results
print("Total Wins as Black and White:")
for player in players:
    print(f"{player}: Wins as Black = {win_counts_black[player]}, Wins as White = {win_counts_white[player]}, Total Games = {total_games[player]}")
