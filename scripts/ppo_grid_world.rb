#!/usr/bin/env ruby
require "csv"
require "torch-rb"

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "rl_learning/grid_world/ppo_grid_world"

# Seed both Torch sampling and Ruby's minibatch shuffling.
seed = Integer(ENV.fetch("SEED", "42"))
Torch.manual_seed(seed)
srand(seed)

state_transitions = CSV.read(File.expand_path("../state_transitions.csv", __dir__), converters: :numeric)
rewards = Torch.zeros(16)
rewards[3] = 10
[2, 11, 10].each { |state| rewards[state] = -1 }
Utility.table_plot(rewards.type(:int).reshape(4, 4), title: "rewards", padding: [0, 1])

model = PPOGridWorld.new(state_transitions:, rewards:)
model.perform(iterations: Integer(ENV.fetch("ITERATIONS", PPOGridWorld::TRAINING_ITERATIONS.to_s)))

Utility.table_plot(model.policy_logits.detach.softmax(1), title: "PPO: action probabilities (up, right, down, left)")
Utility.table_plot(model.state_values.detach.reshape(4, 4), title: "PPO: state values")
Utility.plot_episode_returns(model.score_log, title: "PPO: Episode Return") unless model.score_log.empty?

score, state_log = model.test_agent(greedy: true)
puts "Frames: #{model.frames_seen}, greedy return: #{score}, path: #{state_log.join(" -> ")}"
state_view = Torch.zeros(16)
state_view[state_log] = 1
Utility.table_plot(state_view.type(:int).reshape(4, 4), title: "PPO: greedy path", padding: [0, 1])
