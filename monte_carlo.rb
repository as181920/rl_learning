require "csv"
require "torch-rb"
require_relative "common/global"
require_relative "common/utility"

class MonteCarlo
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

  def monte_carlo_update(rewards:, states:)
    returns = []
    reward = 0
    rewards.length.pred.downto(0) do |idx|
      reward = rewards[idx] + (GAMMA * reward)
      returns.prepend(reward)
    end
    state_values[states] = state_values[states] + (ALPHA * (Torch.stack(returns) - state_values[states]))
  end

  def test_agent # rubocop:disable Metrics/MethodLength
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
    1000.times do
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

      monte_carlo_update(rewards: reward_log, states: state_log)

      score_log.append(test_agent[0])
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
Utility.table_plot(rewards.type(:int).reshape(4, 4), title: "rewards", padding: [0, 1])

mc = MonteCarlo.new(state_transitions:, rewards:)
mc.perform

# plot data
Utility.table_plot(mc.state_values.reshape(4, 4), title: "state values")
Utility.pyplot(Array(0..mc.score_log.length.pred), mc.score_log.map(&:to_f))

# test agent
_, state_log = mc.test_agent
state_view = Torch.zeros(16)
state_view[state_log] = 1
Utility.block_plot(state_view.type(:int).reshape(4, 4), title: "state log")
