import os
import csv
from sgfmill import sgf, boards

def area_score_with_komi(board, komi=6.5):
    """Calculate the area score of a position, including komi.
    
    Assumes all stones are alive and territory is uniquely owned.
    
    Returns the individual scores for Black and White.
    """
    scores = {'b': 0, 'w': 0}
    handled = set()

    for (row, col) in board.board_points:
        colour = board.board[row][col]
        if colour is not None:
            # Count stones on the board
            scores[colour] += 1
            continue
        point = (row, col)
        if point in handled:
            continue
        # Calculate territory for empty regions
        region = board._make_empty_region(row, col)
        region_size = len(region.points)
        if 'b' in region.neighbouring_colours and 'w' not in region.neighbouring_colours:
            scores['b'] += region_size
        elif 'w' in region.neighbouring_colours and 'b' not in region.neighbouring_colours:
            scores['w'] += region_size
        # Mark this region as handled
        handled.update(region.points)
    
    # Add komi to White's score
    white_score = scores['w'] + komi
    black_score = scores['b']
    
    return black_score, white_score

def calculate_individual_scores_with_komi(sgf_path, komi=6.5):
    with open(sgf_path, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    
    board = boards.Board(13)  # Set to 9x9 for your game
    
    for node in game.get_main_sequence():
        color, position = node.get_move()
        if color is not None and position is not None:
            row, col = position
            board.play(row, col, color)
    
    # Calculate the scores including komi
    black_score, white_score = area_score_with_komi(board, komi)
    
    # Get player names
    player_1 = game.get_player_name('b')  # Black player name
    player_2 = game.get_player_name('w')  # White player name
    number_of_moves = len(game.get_main_sequence())  # Number of moves played
    
    return player_1, player_2, black_score, white_score, number_of_moves

def process_multiple_sgf_files(directory, komi=6.5):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.sgf'):
            sgf_path = os.path.join(directory, filename)
            player_1, player_2, black_score, white_score, number_of_moves = calculate_individual_scores_with_komi(sgf_path, komi)
            
            # Determine the winner
            winner = "Black" if black_score > white_score else "White" if black_score < white_score else "Draw"
            score_difference = abs(black_score - white_score)
            
            results.append([player_1, player_2, black_score, white_score, winner, score_difference, number_of_moves])
    
    return results

def save_results_to_csv(results, output_file):
    headers = ['Player 1 Name (Black)', 'Player 2 Name (White)', 'Player 1 Score', 'Player 2 Score', 'Winner', 'Score Difference', 'Number of Moves']
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)

# Example usage
directory = "./13x13/"  # Update with your directory
output_file = "go_game_results.csv"
results = process_multiple_sgf_files(directory, komi=6.5)
save_results_to_csv(results, output_file)

print(f"Results saved to {output_file}.")
