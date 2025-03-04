require "csv"
require "torch-rb"
require_relative "common/global"
require_relative "common/utility"

state_transitions = CSV.read("./state_transitions.csv", converters: :numeric)
pp state_transitions

rewards = Torch.zeros(16)
rewards[3] = 1.0
# Utility.implot(rewards.reshape(4, 4).type(:int).to_a)

state_values = Torch.zeros(16)
terminal_state = 3

alpha = 0.005
score_log = []

