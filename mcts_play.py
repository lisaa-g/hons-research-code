# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCTS example."""

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
from open_spiel.python.bots import gtp
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

_KNOWN_PLAYERS = [
    # A generic Monte Carlo Tree Search agent.
    "mcts",

    # A generic random agent.
    "random",

    # You'll be asked to provide the moves.
    "human",

    # Run an external program that speaks the Go Text Protocol.
    # Requires the gtp_path flag.
    "gtp",

    # Run an alpha_zero checkpoint with MCTS. Uses the specified UCT/sims.
    # Requires the az_path flag.
    "az",
    
    "mcts_trained"
]

flags.DEFINE_string("game", "go", "Name of the game.")
flags.DEFINE_enum("player1", "mcts_trained", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "mcts", _KNOWN_PLAYERS, "Who controls player 2.")
flags.DEFINE_string("gtp_path", None, "Where to find a binary for gtp.")
flags.DEFINE_multi_string("gtp_cmd", [], "GTP commands to run at init.")
flags.DEFINE_string("az_path", None,
                    "Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 1, "How many rollouts to do.")
flags.DEFINE_integer("max_simulations", 100, "How many simulations to run.")
flags.DEFINE_integer("num_games", 1, "How many games to play.")
flags.DEFINE_integer("seed", None, "Seed for the random number generator.")
flags.DEFINE_bool("random_first", False, "Play the first move randomly.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS


def _opt_print(*args, **kwargs):
  if not FLAGS.quiet:
    print(*args, **kwargs)


def _init_bot(bot_type, game, player_id):
  """Initializes a bot by type."""
  rng = np.random.RandomState(FLAGS.seed)
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
  if bot_type == "az":
    model = az_model.Model.from_checkpoint(FLAGS.az_path)
    evaluator = az_evaluator.AlphaZeroEvaluator(game, model)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        child_selection_fn=mcts.SearchNode.puct_value,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  if bot_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  if bot_type == "human":
    return human.HumanBot()
  if bot_type == "gtp":
    bot = gtp.GTPBot(game, FLAGS.gtp_path)
    for cmd in FLAGS.gtp_cmd:
      bot.gtp_cmd(cmd)
    return bot
  if bot_type == "mcts_trained":
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
  raise ValueError("Invalid bot type: %s" % bot_type)


def _get_action(state, action_str):
  for action in state.legal_actions():
    if action_str == state.action_to_string(state.current_player(), action):
      return action
  return None


def _play_game(game, bots, initial_actions):
  """Plays one game."""
  state = game.new_initial_state()
  _opt_print("Initial state:\n{}".format(state))

  history = []

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
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      _opt_print("Chance node, got " + str(num_actions) + " outcomes")
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      action_str = state.action_to_string(current_player, action)
      _opt_print("Sampled action: ", action_str)
    elif state.is_simultaneous_node():
      raise ValueError("Game cannot have simultaneous nodes.")
    else:
      # Decision node: sample action for the single current player
      bot = bots[current_player]
      action = bot.step(state)
      action_str = state.action_to_string(current_player, action)
      _opt_print("Player {} sampled action: {}".format(current_player,
                                                       action_str))

    for i, bot in enumerate(bots):
      if i != current_player:
        bot.inform_action(state, current_player, action)
    history.append(action_str)
    state.apply_action(action)

    _opt_print("Next state:\n{}".format(state))

  # Game is now done. Print return for each player
  returns = state.returns()
  print("Returns:", " ".join(map(str, returns)), ", Game actions:",
        " ".join(history))

  for bot in bots:
    bot.restart()

  return returns, history

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

def save_sgf(history, filename="game.sgf", board_size=19, komi=6.5, player1_name="Player 1", player2_name="Player 2"):
    # SGF header with dynamic player names
    sgf = f"(;GM[1]FF[4]CA[UTF-8]SZ[{board_size}]KM[{komi}]\n"
    sgf += f"PW[{player1_name}]PB[{player2_name}]\n"

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
    
def main(argv):
    parameters = {'board_size': 19, 'komi': 6.5}
    game = pyspiel.load_game(FLAGS.game, parameters)
    if game.num_players() > 2:
        sys.exit("This game requires more players than the example can handle.")
    bots = [
        _init_bot(FLAGS.player1, game, 0),
        _init_bot(FLAGS.player2, game, 1),
    ]
    histories = collections.defaultdict(int)
    overall_returns = [0, 0]
    overall_wins = [0, 0]
    game_num = 0
    player1_name = FLAGS.player1
    player2_name = FLAGS.player2

    try:
        for game_num in range(FLAGS.num_games):
            returns, history = _play_game(game, bots, argv[1:])
            save_sgf(
                history,
                f"game_{game_num + 1}.sgf",
                board_size=parameters['board_size'],
                player1_name=player1_name,
                player2_name=player2_name
            )  # Save each game as SGF with specific player names
            histories[" ".join(history)] += 1
            for i, v in enumerate(returns):
                overall_returns[i] += v
                if v > 0:
                    overall_wins[i] += 1
    except (KeyboardInterrupt, EOFError):
        game_num -= 1
        print("Caught a KeyboardInterrupt, stopping early.")
    print("Number of games played:", game_num + 1)
    print("Number of distinct games played:", len(histories))
    print("Players:", player1_name, player2_name)
    print("Overall wins", overall_wins)
    print("Overall returns", overall_returns)


if __name__ == "__main__":
  app.run(main)
