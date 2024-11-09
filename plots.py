import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV data for each board size
data_9x9 = pd.read_csv("./results/9x9_results.csv")
data_13x13 = pd.read_csv("./results/13x13_results.csv")
data_19x19 = pd.read_csv("./results/19x19_results.csv")
trained_19x19 = pd.read_csv("./results/trained_mcts.csv")

# Function to calculate win rates for a given dataset
def calculate_win_rates(data):
    players = pd.concat([data['Player 1 Name (Black)'], data['Player 2 Name (White)']]).unique()
    win_counts = {player: 0 for player in players}
    total_games = {player: 0 for player in players}

    # Calculate wins and total games
    for index, row in data.iterrows():
        player_1 = row['Player 1 Name (Black)']
        player_2 = row['Player 2 Name (White)']
        winner = row['Winner']

        # Update total games for each player
        total_games[player_1] += 1
        total_games[player_2] += 1

        # Update win count for the winner
        if winner == 'Black':
            win_counts[player_1] += 1
        elif winner == 'White':
            win_counts[player_2] += 1

    # Calculate win rate for each player
    win_rates = {player: (win_counts[player] / total_games[player] * 100) if total_games[player] > 0 else 0 for player in players}
    return win_rates

# Calculate win rates for each board size
win_rates_9x9 = calculate_win_rates(data_9x9)
win_rates_13x13 = calculate_win_rates(data_13x13)
win_rates_19x19 = calculate_win_rates(data_19x19)
win_rates_trained = calculate_win_rates(trained_19x19)

# Define agents common to all board sizes and agents specific to 19x19
common_agents = ['ga', 'dqn', 'mcts', 'random', 'mcts_trained']
mcts_variants_19x19 = ['mcts_advanced', 'mcts_beginner', 'mcts_casual', 'mcts_intermediate', 'mcts']

# Prepare win rate lists for the common agents
win_rates_9x9_common = [win_rates_9x9.get(agent, 0) for agent in common_agents]
win_rates_13x13_common = [win_rates_13x13.get(agent, 0) for agent in common_agents]
win_rates_19x19_common = [win_rates_19x19.get(agent, 0) for agent in common_agents]

# Prepare win rate list for MCTS variants (only 19x19)
win_rates_19x19_mcts_variants = [win_rates_trained.get(agent, 0) for agent in mcts_variants_19x19]

# Plot for common agents across all board sizes
x_common = np.arange(len(common_agents))
width = 0.3

plt.figure(figsize=(9, 6))
plt.bar(x_common - width, win_rates_9x9_common, width, label='9x9 Board', color='skyblue')
plt.bar(x_common, win_rates_13x13_common, width, label='13x13 Board', color='deeppink')
plt.bar(x_common + width, win_rates_19x19_common, width, label='19x19 Board', color='mediumseagreen')
plt.xlabel('Agent', fontsize=14)
plt.ylabel('Win Rate (%)', fontsize=14)
plt.title('Win Rates of Agents on 9x9, 13x13, and 19x19 Boards', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(x_common, common_agents, rotation=0, fontsize=13)
plt.legend(fontsize=13)
plt.tight_layout()
plt.show()

# Plot for MCTS variants on the 19x19 board only
x_mcts_variants = np.arange(len(mcts_variants_19x19))

width = 0.5
plt.figure(figsize=(9, 5))
plt.bar(x_mcts_variants, win_rates_19x19_mcts_variants, width, color='mediumseagreen')
plt.xlabel('MCTS Variant', fontsize=14)
plt.ylabel('Win Rate (%)', fontsize=14)
plt.title('Win Rates of MCTS Variants on 19x19 Board', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(x_mcts_variants, mcts_variants_19x19, rotation=0, fontsize=13)
plt.tight_layout()
plt.show()

# Function to calculate average scores for each player
def calculate_average_scores(data):
    players = pd.concat([data['Player 1 Name (Black)'], data['Player 2 Name (White)']]).unique()
    total_scores = {player: 0 for player in players}
    game_counts = {player: 0 for player in players}

    # Sum up scores and count games for each player
    for index, row in data.iterrows():
        player_1 = row['Player 1 Name (Black)']
        player_2 = row['Player 2 Name (White)']
        score_1 = row['Player 1 Score']
        score_2 = row['Player 2 Score']

        total_scores[player_1] += score_1
        total_scores[player_2] += score_2
        game_counts[player_1] += 1
        game_counts[player_2] += 1

    # Calculate average score for each player
    average_scores = {player: (total_scores[player] / game_counts[player]) if game_counts[player] > 0 else 0 for player in players}
    return average_scores

# Calculate average scores for each board size
average_scores_9x9 = calculate_average_scores(data_9x9)
average_scores_13x13 = calculate_average_scores(data_13x13)
average_scores_19x19 = calculate_average_scores(data_19x19)
average_scores_trained = calculate_average_scores(trained_19x19)

# Prepare lists of scores for common agents and MCTS variants for 19x19
average_scores_9x9_common = [average_scores_9x9.get(agent, 0) for agent in common_agents]
average_scores_13x13_common = [average_scores_13x13.get(agent, 0) for agent in common_agents]
average_scores_19x19_common = [average_scores_19x19.get(agent, 0) for agent in common_agents]
average_scores_19x19_mcts_variants = [average_scores_trained.get(agent, 0) for agent in mcts_variants_19x19]

# Plot for average scores of common agents across all board sizes
width = 0.3
plt.figure(figsize=(9, 6))
plt.bar(x_common - width, average_scores_9x9_common, width, label='9x9 Board', color='skyblue')
plt.bar(x_common, average_scores_13x13_common, width, label='13x13 Board', color='deeppink')
plt.bar(x_common + width, average_scores_19x19_common, width, label='19x19 Board', color='mediumseagreen')
plt.xlabel('Agent', fontsize=14)
plt.ylabel('Average Score', fontsize=14)
plt.title('Average Scores of Agents on 9x9, 13x13, and 19x19 Boards', fontsize=16)
plt.xticks(x_common, common_agents, rotation=0, fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=13)
plt.tight_layout()
plt.show()

# Plot for average scores of MCTS variants on the 19x19 board only
width = 0.5
plt.figure(figsize=(9, 5))
plt.bar(x_mcts_variants, average_scores_19x19_mcts_variants, width, color='mediumseagreen')
plt.xlabel('MCTS Variant', fontsize=14)
plt.ylabel('Average Score', fontsize=14)
plt.title('Average Scores of MCTS Variants on 19x19 Board', fontsize=16)
plt.xticks(x_mcts_variants, mcts_variants_19x19, rotation=0, fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Function to calculate average moves for each player
def calculate_average_moves(data):
    players = pd.concat([data['Player 1 Name (Black)'], data['Player 2 Name (White)']]).unique()
    total_moves = {player: 0 for player in players}
    game_counts = {player: 0 for player in players}

    # Sum up moves and count games for each player
    for index, row in data.iterrows():
        player_1 = row['Player 1 Name (Black)']
        player_2 = row['Player 2 Name (White)']
        moves = row['Number of Moves']

        total_moves[player_1] += moves
        total_moves[player_2] += moves
        game_counts[player_1] += 1
        game_counts[player_2] += 1

    # Calculate average moves for each player
    average_moves = {player: (total_moves[player] / game_counts[player]) if game_counts[player] > 0 else 0 for player in players}
    return average_moves

# Calculate average moves for each board size
average_moves_9x9 = calculate_average_moves(data_9x9)
average_moves_13x13 = calculate_average_moves(data_13x13)
average_moves_19x19 = calculate_average_moves(data_19x19)
average_moves_trained = calculate_average_moves(trained_19x19)

# Prepare lists of moves for common agents and MCTS variants for 19x19
average_moves_9x9_common = [average_moves_9x9.get(agent, 0) for agent in common_agents]
average_moves_13x13_common = [average_moves_13x13.get(agent, 0) for agent in common_agents]
average_moves_19x19_common = [average_moves_19x19.get(agent, 0) for agent in common_agents]
average_moves_19x19_mcts_variants = [average_moves_trained.get(agent, 0) for agent in mcts_variants_19x19]

# Plot for average moves of common agents across all board sizes
width = 0.3
plt.figure(figsize=(9, 6))
plt.bar(x_common - width, average_moves_9x9_common, width, label='9x9 Board', color='skyblue')
plt.bar(x_common, average_moves_13x13_common, width, label='13x13 Board', color='deeppink')
plt.bar(x_common + width, average_moves_19x19_common, width, label='19x19 Board', color='mediumseagreen')
plt.xlabel('Agent', fontsize=14)
plt.ylabel('Average Game Length', fontsize=14)
plt.title('Average Game Length of Agents on 9x9, 13x13, and 19x19 Boards', fontsize=16)
plt.xticks(x_common, common_agents, rotation=0, fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=13)
plt.tight_layout()
plt.show()

# Plot for average moves of MCTS variants on the 19x19 board only
width = 0.5
plt.figure(figsize=(9, 5))
plt.bar(x_mcts_variants, average_moves_19x19_mcts_variants, width, color='mediumseagreen')
plt.xlabel('MCTS Variant', fontsize=14)
plt.ylabel('Average Game Length', fontsize=14)
plt.title('Average Game Length of MCTS Variants on 19x19 Board', fontsize=16)
plt.xticks(x_mcts_variants, mcts_variants_19x19, rotation=0, fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()