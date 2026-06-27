#!/usr/bin/env ruby
require "csv"
require "torch-rb"

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "rl_learning/grid_world/actor_critic_learning"

state_transitions = CSV.read(File.expand_path("../state_transitions.csv", __dir__), converters: :numeric)
Utility.table_print(state_transitions, title: "state_transitions data table")

rewards = Torch.zeros(16)
rewards[3] = 10
rewards[2] = -1
rewards[11] = -1
rewards[10] = -1
Utility.table_plot(rewards.type(:int).reshape(4, 4), title: "rewards", padding: [0, 1])

model = ActorCritic.new(state_transitions:, rewards:)
model.perform

Utility.table_plot(model.policy_logits, title: "Policy logits")
Utility.plot_episode_returns(model.score_log, title: "Actor-Critic: Episode Return")

_, state_log = model.test_agent
state_view = Torch.zeros(16)
state_view[state_log] = 1
state_view_grid = state_view.type(:int).reshape(4, 4)
Utility.table_plot(state_view_grid, title: "state log", padding: [0, 1])

# Utility.block_plot(state_view_grid, title: "state log")
