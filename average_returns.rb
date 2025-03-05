require "csv"
require "torch-rb"
require_relative "common/global"
require_relative "common/utility"

class AverageReturns
  ALPHA = 0.005

  state_values = Torch.zeros(16)
  terminal_state = 3

  score_log = []

  def initialize(state_transitions:, rewards:)
  end

  def perform
  end
end

state_transitions = CSV.read(File.expand_path("state_transitions.csv", __dir__), converters: :numeric)
Utility.table_print(state_transitions, title: "state_transitions data table")

rewards = Torch.zeros(16).reshape(4, 4)
rewards[0][3] = 1.0
Utility.ansi_plot(rewards, title: "rewards")
# Utility.implot(rewards.type(:int).to_a)

AverageReturns.new(state_transitions:, rewards:).perform
