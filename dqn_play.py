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
import dqn
import tensorflow.compat.v1 as tf
from open_spiel.python import rl_environment
from open_spiel.python import rl_agent

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
    
    "mcts_trained",
    
    "dqn"
]

flags.DEFINE_string("game", "go", "Name of the game.")
flags.DEFINE_enum("player1", "dqn", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "random", _KNOWN_PLAYERS, "Who controls player 2.")
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


def _init_bot(bot_type, game, env, player_id):
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
        sgf_file = "training_data.sgf"
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
  if bot_type == "dqn":
        session = tf.Session()
        dqn_agent = dqn.DQN(
            game,
            session,
            player_id=player_id,
            state_representation_size=game.observation_tensor_size(),
            num_actions=game.num_distinct_actions()
        )
        dqn_agent.restore("./checkpoints")  # Path to your saved DQN model
        return dqn_agent
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


def main(argv):
  game = pyspiel.load_game(FLAGS.game)
  env = rl_environment.Environment(game)
  if game.num_players() > 2:
    sys.exit("This game requires more players than the example can handle.")
  bots = [
      _init_bot(FLAGS.player1, game, env, 0),
      _init_bot(FLAGS.player2, game, env, 1),
  ]
  histories = collections.defaultdict(int)
  overall_returns = [0, 0]
  overall_wins = [0, 0]
  game_num = 0
  try:
    for game_num in range(FLAGS.num_games):
      returns, history = _play_game(game, bots, env, argv[1:])
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
  print("Players:", FLAGS.player1, FLAGS.player2)
  print("Overall wins", overall_wins)
  print("Overall returns", overall_returns)


if __name__ == "__main__":
  app.run(main)