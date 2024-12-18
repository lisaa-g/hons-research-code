"""DQN agents trained by independent Q-learning."""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

import csv

def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    num_players = len(trained_agents)
    sum_episode_rewards = np.zeros(num_players)
    for player_pos in range(num_players):
        cur_agents = random_agents[:]
        cur_agents[player_pos] = trained_agents[player_pos]
        for _ in range(num_episodes):
            time_step = env.reset()
            episode_rewards = 0
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if env.is_turn_based:
                    agent_output = cur_agents[player_id].step(
                        time_step, is_evaluation=True)
                    action_list = [agent_output.action]
                else:
                    agents_output = [
                        agent.step(time_step, is_evaluation=True) for agent in cur_agents
                    ]
                    action_list = [agent_output.action for agent_output in agents_output]
                time_step = env.step(action_list)
                episode_rewards += time_step.rewards[player_pos]
            sum_episode_rewards[player_pos] += episode_rewards
    return sum_episode_rewards / num_episodes

def main(_):
    game = "go"
    num_players = 2
    env_configs = {"board_size": 9, "handicap": 1, "komi": 7.5 , "max_game_length": 100}
    env = rl_environment.Environment(game, **env_configs)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = 16
    replay_buffer_capacity = int(1000)
    train_episodes = 36000
    loss_report_interval = 1000

    # Initialize CSV file
    with open("training_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "DQN Loss", "Mean Episode Rewards"])

    with tf.Session() as sess:
        dqn_agent = dqn.DQN(
            sess,
            player_id=0,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity)
        tabular_q_agent = tabular_qlearner.QLearner(
            player_id=1, num_actions=num_actions)
        agents = [dqn_agent, tabular_q_agent]

        sess.run(tf.global_variables_initializer())

        for ep in range(train_episodes):
            if ep and ep % loss_report_interval == 0:
                dqn_loss = agents[0].loss
                logging.info("[%s/%s] DQN loss: %s", ep, train_episodes, dqn_loss)
          
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)

            # Save DQN loss to CSV at reporting interval
            if ep and ep % loss_report_interval == 0:
                # Evaluate against random agents and log results
                random_agents = [
                    random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
                    for idx in range(num_players)
                ]
                mean_rewards = eval_against_random_bots(env, agents, random_agents, 100)
                logging.info("Mean episode rewards: %s", mean_rewards)

                dqn_agent.save("./checkpoints/9x9")
                # Write to CSV
                with open("training_log.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ep, dqn_loss, mean_rewards[0]])  # Log for player 0

        dqn_agent.save("./checkpoints/9x9")
        dqn_agent.restore("./checkpoints/9x9")

if __name__ == "__main__":
    app.run(main)
