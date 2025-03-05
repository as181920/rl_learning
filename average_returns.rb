require "csv"
require "torch-rb"
require_relative "common/global"
require_relative "common/utility"

class AverageReturns
  ALPHA = 0.005
  GAMMA = 0.9
  DELTA = 1e-10

  attr_reader :state_transitions, :rewards, :start_state, :terminal_state

  attr_accessor :state_values, :score_log

  def initialize(state_transitions:, rewards:)
    @state_transitions = state_transitions
    @rewards = rewards
    @start_state = 12
    @terminal_state = 3
    @state_values = Torch.zeros(16)
    @score_log = []
  end

  def calc_returns(_values, rewards:, states:)
    state_count = Torch.zeros(16)
    state_returns = Torch.zeros(16)
    reward = 0
    rewards.length.pred.downto(0) do |idx|
      reward = rewards[idx] + (GAMMA * reward)
      state_returns[states[idx]] += reward
      state_count[states[idx]] += 1
      # print(states[idx])
    end

    state_returns / (state_count + DELTA)
  end

  def run_agent # rubocop:disable Metrics/MethodLength
    state = start_state
    # done = false
    steps = 0
    total_rewards = 0
    states_log = []
    while (state != terminal_state) && (steps < 30)
      states_log.append(state)
      action = Torch.argmax(state_values[state_transitions[state]]).to_i
      state = transit_state(state, action:)
      total_rewards += rewards[state]
      steps += 1
    end
    states_log.append(state)
    [total_rewards, states_log]
  end

  def perform # rubocop:disable Metrics/MethodLength
    returns_log = []

    100.times do
      state = start_state
      state_log = [state]
      reward_log = [rewards[state]]
      steps = 0

      loop do
        state = transit_state(state)
        steps = steps.succ
        state_log.append(state)
        reward_log.append(rewards[state])
        break if (state == terminal_state) || (steps >= 30)
      end

      returns_log.append(calc_returns(state_values, rewards: reward_log, states: state_log))
      @state_values = Torch.mean(Torch.stack(returns_log), dim: 0)

      score_log.append(run_agent[0])
    end
  end

  private

    def transit_state(state, action: rand(4))
      state_transitions.dig(state, action)
    end
end

state_transitions = CSV.read(File.expand_path("state_transitions.csv", __dir__), converters: :numeric)
Utility.table_print(state_transitions, title: "state_transitions data table")

rewards = Torch.zeros(16)
rewards[3] = 1.0
Utility.ansi_plot(rewards.type(:int).reshape(4, 4), title: "rewards", padding: [0, 1])

ar = AverageReturns.new(state_transitions:, rewards:)
ar.perform
Utility.ansi_plot(ar.state_values.reshape(4, 4), title: "rewards")
