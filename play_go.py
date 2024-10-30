"""
PLAY GO
player 1: black
player 2: white
komi = 6.5 and is added to white player (player 2) as compensation for going 2nd

Different agents:
1. MCTS
2. GA
3. DQN
4. Random

The games will also be played on different board sizes:
1. 9x9
2. 13x13
3. 19x19

For 19x19 MCTS will have different bots trained on previous data:
1. Beginner 30-20k
2. Casual 19-10k
3. Intermediate Amateur 9-1k
4. Advanced Amateur 1-7d
dont have any professional player ranked games

9x9 and 13x13 will just have games from tournaments as these sizes arent always played as much

Each bot will play against each other bot in a round-robin style tournament.
50 games as white and 50 games as black will be played for each pair of bots.
"""

import collections
import random
import sys
import os

from absl import app
from absl import flags
import numpy as np

import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel
import dqn
import tensorflow.compat.v1 as tf
from open_spiel.python import rl_environment
sys.path.append('/home/lisag/research')
import ga

_KNOWN_PLAYERS = [
    "mcts",
    "ga",
    "dqn",
    "mcts_beginner",
    "mcts_casual",
    "mcts_intermediate",
    "mcts_advanced",
    "mcts_9x9",
    "mcts_13x13"
]

flags.DEFINE_string("game", "go", "Name of the game.")
flags.DEFINE_enum("player1", "mcts_9x9", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "ga", _KNOWN_PLAYERS, "Who controls player 2.")
flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 1, "How many rollouts to do.")
flags.DEFINE_integer("max_simulations", 100, "How many simulations to run.")
flags.DEFINE_integer("num_games", 1, "How many games to play.")
flags.DEFINE_integer("seed", None, "Seed for the random number generator.")
flags.DEFINE_bool("random_first", False, "Play the first move randomly.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")
flags.DEFINE_integer("population_size", 100, "Size of the population.")
flags.DEFINE_float("mutation_rate", 0.3, "Mutation rate.")
flags.DEFINE_float("crossover_rate", 0.7, "Crossover rate.")
flags.DEFINE_integer("num_generations", 10, "Number of generations.")

FLAGS = flags.FLAGS


def _opt_print(*args, **kwargs):
  if not FLAGS.quiet:
    print(*args, **kwargs)


def _init_bot(bot_type, game, env, player_id):
  """Initializes a bot by type."""
  rng = np.random.RandomState(FLAGS.seed)
  #normal mcts
  if bot_type == "mcts":
    evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  #random
  if bot_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  #human
  if bot_type == "human":
    return human.HumanBot()
  #mcts beginner
  if bot_type == "mcts_beginner":
        sgf_file = "./training_data/beginner_training_data.sgf"
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return mcts.MCTSWithTraining(
            game,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            evaluator,
            sgf_file,
            random_state=rng,
            solve=FLAGS.solve,
            verbose=FLAGS.verbose)
  #mcts casual
  if bot_type == "mcts_casual":
        sgf_file = "./training_data/casual_training_data.sgf"
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return mcts.MCTSWithTraining(
            game,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            evaluator,
            sgf_file,
            random_state=rng,
            solve=FLAGS.solve,
            verbose=FLAGS.verbose)
  #mcts intermediate
  if bot_type == "mcts_intermediate":
        sgf_file = "./training_data/intermediate_training_data.sgf"
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return mcts.MCTSWithTraining(
            game,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            evaluator,
            sgf_file,
            random_state=rng,
            solve=FLAGS.solve,
            verbose=FLAGS.verbose)
  #mcts advanced
  if bot_type == "mcts_advanced":
        sgf_file = "./training_data/advanced_training_data.sgf"
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return mcts.MCTSWithTraining(
            game,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            evaluator,
            sgf_file,
            random_state=rng,
            solve=FLAGS.solve,
            verbose=FLAGS.verbose)
  #mcts 9x9
  if bot_type == "mcts_9x9":
        sgf_file = "./training_data/9x9_training_data.sgf"
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return mcts.MCTSWithTraining(
            game,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            evaluator,
            sgf_file,
            random_state=rng,
            solve=FLAGS.solve,
            verbose=FLAGS.verbose)
  #mcts 13x13
  if bot_type == "mcts_13x13":
        sgf_file = "./training_data/13x13_training_data.sgf"
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return mcts.MCTSWithTraining(
            game,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            evaluator,
            sgf_file,
            random_state=rng,
            solve=FLAGS.solve,
            verbose=FLAGS.verbose)
  #dqn
  if bot_type == "dqn":
        session = tf.Session()
        dqn_agent = dqn.DQN(
            game,
            session,
            player_id=player_id,
            state_representation_size=game.observation_tensor_size(),
            num_actions=game.num_distinct_actions()
        )
        dqn_agent.restore("./checkpoints/9x9/")  #change according to board size
        return dqn_agent
  #ga
  if bot_type == "ga":
        ga_bot = ga.GeneticAlgorithm(
            game=game,
            population_size=FLAGS.population_size,
            generations=FLAGS.num_generations,
            mutation_rate=FLAGS.mutation_rate,
            crossover_rate=FLAGS.crossover_rate,
            player=player_id
        )
        ga_bot.load_model("./trained_models_ga/13x13_ga_model.pkl") #change according to board size
        return ga_bot
  
  raise ValueError("Invalid bot type: %s" % bot_type)


def _get_action(state, action_str):
  for action in state.legal_actions():
    if action_str == state.action_to_string(state.current_player(), action):
      return action
  return None

def _play_game(game, bots, env, initial_actions):
    """Plays one game."""
    state = game.new_initial_state()
    _opt_print("Initial state:\n{}".format(state))

    history = []

    # Handling initial actions if required
    if FLAGS.random_first:
        assert not initial_actions
        initial_actions = [state.action_to_string(
            state.current_player(), random.choice(state.legal_actions()))]

    for action_str in initial_actions:
        action = _get_action(state, action_str)
        if action is None:
            sys.exit("Invalid action: {}".format(action_str))

        history.append(action_str)
        for bot in bots:
            bot.inform_action(state, state.current_player(), action)
        state.apply_action(action)
        _opt_print("Forced action", action_str)
        _opt_print("Next state:\n{}".format(state))

    while not state.is_terminal():
        current_player = state.current_player()
        bot = bots[current_player]

        if isinstance(bot, dqn.DQN):  # Check if bot is DQN
            # Wrap pyspiel.State into rl_environment.TimeStep
            time_step = convert_to_time_step(state, current_player)
            step_output = bot.step(time_step) # Call DQN agent's step function
            print(f"DQN Step Output: {step_output}")  # Print the full output for debugging
            action = step_output.action
            print(f"Extracted Action: {action}, Type: {type(action)}") 
        else:  # Assume it's MCTS or another type of bot
            action = bot.step(state)  # Call MCTS agent's step function

        action_str = state.action_to_string(current_player, action)
        _opt_print("Player {} sampled action: {}".format(current_player, action_str))

        # Inform other bots of the action
        for i, other_bot in enumerate(bots):
            if i != current_player:
                other_bot.inform_action(state, current_player, action)
        history.append(action_str)
        state.apply_action(action)

        _opt_print("Next state:\n{}".format(state))

    # Game is now done. Print return for each player
    returns = state.returns()
    print("Returns:", " ".join(map(str, returns)), ", Game actions:", " ".join(history))

    for bot in bots:
        bot.restart()

    return returns, history
  
def convert_to_time_step(state, player_id):
    """Convert pyspiel.State to rl_environment.TimeStep."""
    observations = {
        "info_state": {
            player_id: state.observation_tensor(player_id)  # Ensure this works as expected
        },
        "legal_actions": {
            player_id: state.legal_actions()  # Legal actions for the current player
        },
        "current_player": player_id  # Add the current player to observations
    }

    # Create a TimeStep instance
    return rl_environment.TimeStep(
        step_type=rl_environment.StepType.FIRST,  # Adjust as needed (FIRST, MID, LAST)
        rewards=0,  # Modify to set rewards according to the game state
        discounts=1.0,  # Set discounts as required
        observations=observations
    )

def convert_move_to_sgf(action_str, board_size):
    action_str = action_str.strip() 

    # Split the action into player and move (e.g., 'B h5' -> 'B', 'h5')
    parts = action_str.split()
    if len(parts) != 2:
        raise ValueError(f"Unexpected action format: {action_str}")
    
    move = parts[1]

    # Handle the case where the move is 'PASS'
    if move == 'PASS':
        return '' 

    col = move[0]
    row = move[1:] 

    # Make sure row is a valid number
    try:
        row = int(row)
    except ValueError:
        raise ValueError(f"Invalid row part in action: {action_str}")

    # Generate valid column letters excluding 'i' (Go boards skip the 'i' column)
    columns = [chr(i) for i in range(ord('a'), ord('a') + board_size + 1) if chr(i) != 'i']

    if col not in columns:
        raise ValueError(f"Invalid column: {col}. Valid columns are {columns}")

    sgf_col = columns.index(col)  # Convert column to SGF format
    sgf_row = board_size - row  # Convert row to SGF format

    return f"{chr(ord('a') + sgf_col)}{chr(ord('a') + sgf_row)}"

def save_sgf(history, filename="game.sgf", board_size=13, komi=6.5, player1_name="Player 1", player2_name="Player 2"):
    # SGF header with dynamic player names
    sgf = f"(;GM[1]FF[4]CA[UTF-8]SZ[{board_size}]KM[{komi}]\n"
    sgf += f"PB[{player1_name}]PW[{player2_name}]\n"

    # Convert moves to SGF format
    for i, action_str in enumerate(history):
        color = 'B' if i % 2 == 0 else 'W'  # Black or White move
        sgf_move = convert_move_to_sgf(action_str, board_size)
        sgf += f";{color}[{sgf_move}]\n"

    sgf += ")"

    # Write SGF file
    with open(filename, 'w') as sgf_file:
        sgf_file.write(sgf)
    print(f"SGF saved to {filename}")
  
def round_robin_tournament(game, agents, env, board_size=13):
    results = {}
    games_per_matchup = 100  # 50 as white, 50 as black

    # Prepare the results dictionary to store game outcomes for each pair of agents
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
             if i < j:
                matchup_key = f"{agent1}_{agent2}"
                results[matchup_key] = {"agent1_wins": 0, "agent2_wins": 0, "draws": 0}

                # Play 50 games with agent1 as black and agent2 as white, then switch
                for game_num in range(games_per_matchup//2):
                    bot1 = _init_bot(agent1, game, env, player_id=0)
                    bot2 = _init_bot(agent2, game, env, player_id=1)
                    bots = [bot1, bot2]
                    returns, history = _play_game(game, bots, env, initial_actions=[])

                    # Save SGF after each game
                    save_sgf(
                        history,
                        f"./13x13/{matchup_key}_game_{game_num + 1}.sgf",
                        board_size=board_size,
                        player1_name=agent1,
                        player2_name=agent2
                    )

                    if returns[0] > returns[1]:
                        results[matchup_key]["agent1_wins"] += 1
                    elif returns[1] > returns[0]:
                        results[matchup_key]["agent2_wins"] += 1
                    else:
                        results[matchup_key]["draws"] += 1

                # Switch roles for 50 more games
                for game_num in range(games_per_matchup // 2):
                    bot1 = _init_bot(agent1, game, env, player_id=1)
                    bot2 = _init_bot(agent2, game, env, player_id=0)
                    bots = [bot2, bot1]
                    returns, history = _play_game(game, bots, env, initial_actions=[])

                    # Save SGF after each game
                    save_sgf(
                        history,
                        f"./13x13/{matchup_key}_game_{game_num + 1 + (games_per_matchup // 2)}.sgf",
                        board_size=board_size,
                        player1_name=agent2,
                        player2_name=agent1
                    )

                    if returns[1] > returns[0]:
                        results[matchup_key]["agent1_wins"] += 1
                    elif returns[0] > returns[1]:
                        results[matchup_key]["agent2_wins"] += 1
                    else:
                        results[matchup_key]["draws"] += 1

    return results

def main(_):
  parameters = {'board_size': 13}
  game = pyspiel.load_game(FLAGS.game, parameters)
  env = rl_environment.Environment(game)
  agents = ["mcts",
    "ga",
    "random",
    "mcts_13x13",
    "dqn"]  # Define the agents in the tournament

  results = round_robin_tournament(game, agents, env, 13)
  print("Tournament Results:")
  for matchup, outcome in results.items():
      print(f"{matchup}: {outcome}")

if __name__ == "__main__":
    app.run(main)
