import collections
import random
import sys

from absl import app
from absl import flags
import numpy as np
import pyspiel

from open_spiel.python.bots import uniform_random, human

sys.path.append('/home/lisag/research')

# Import GA module
import ga

_KNOWN_PLAYERS = [
    "ga",
    "random",
    "human"
]

# Define flags
flags.DEFINE_string("game", "go", "Name of the game.")
flags.DEFINE_enum("player1", "ga", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "random", _KNOWN_PLAYERS, "Who controls player 2.")
flags.DEFINE_integer("population_size", 100, "Size of the population.")
flags.DEFINE_float("mutation_rate", 0.3, "Mutation rate.")
flags.DEFINE_float("crossover_rate", 0.7, "Crossover rate.")
flags.DEFINE_integer("num_generations", 10, "Number of generations.")
flags.DEFINE_integer("num_games", 50, "How many games to play.")
flags.DEFINE_integer("seed", None, "Seed for the random number generator.")
flags.DEFINE_bool("random_first", False, "Whether to force a random move as the first action.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show detailed GA stats.")

FLAGS = flags.FLAGS

def _opt_print(*args, **kwargs):
    if not FLAGS.quiet:
        print(*args, **kwargs)

def _init_bot(bot_type, game, player_id):
    rng = np.random.RandomState(FLAGS.seed)
    if bot_type == "ga":
        ga_bot = ga.GeneticAlgorithm(
            game=game,
            population_size=FLAGS.population_size,
            generations=FLAGS.num_generations,
            mutation_rate=FLAGS.mutation_rate,
            crossover_rate=FLAGS.crossover_rate,
            player=player_id
        )
        # Load the trained model
        ga_bot.load_model("./trained_models_ga/19x19_ga_model.pkl")
        return ga_bot
    elif bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    elif bot_type == "human":
        return human.HumanBot()
    else:
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
        bot = bots[current_player]
        legal_actions = state.legal_actions()
        action = bot.step(state)
        action_str = state.action_to_string(current_player, action)
        
        if action not in legal_actions:
            raise ValueError(f"Illegal action attempted by bot: {action}")

        _opt_print("Player {} sampled action: {}".format(current_player, action_str))

        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
        history.append(action_str)
        state.apply_action(action)

        _opt_print("Next state:\n{}".format(state))

    returns = state.returns()
    print("Returns:", " ".join(map(str, returns)), ", Game actions:", " ".join(history))

    for bot in bots:
        bot.restart()

    return returns, history

def main(argv):
    parameters = {'board_size': 19}
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
    try:
        for game_num in range(FLAGS.num_games):
            returns, history = _play_game(game, bots, argv[1:])
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
