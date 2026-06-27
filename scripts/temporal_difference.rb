#!/usr/bin/env ruby
require "csv"
require "torch-rb"

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "rl_learning/grid_world/temporal_difference"

state_transitions = CSV.read(File.expand_path("../state_transitions.csv", __dir__), converters: :numeric)
Utility.table_print(state_transitions, title: "state_transitions data table")

rewards = Torch.zeros(16)
rewards[3] = 10
rewards[2] = -1
rewards[11] = -1
rewards[10] = -1
Utility.table_plot(rewards.type(:int).reshape(4, 4), title: "rewards", padding: [0, 1])

mc = TemporalDifference.new(state_transitions:, rewards:)
mc.perform

Utility.table_plot(mc.state_values.reshape(4, 4), title: "state values")
Utility.plot_episode_returns(mc.score_log, title: "Temporal Difference: Episode Return")

_, state_log = mc.test_agent
state_view = Torch.zeros(16)
state_view[state_log] = 1
state_view_grid = state_view.type(:int).reshape(4, 4)
Utility.table_plot(state_view_grid, title: "state log", padding: [0, 1])

# Utility.block_plot(state_view_grid, title: "state log")
