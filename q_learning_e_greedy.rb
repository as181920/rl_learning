require "csv"
require "torch-rb"
require_relative "common/global"
require_relative "common/utility"

class QLearningEGreedy
  ALPHA = 0.01
  EPSILON = 0.4
  GAMMA = 0.99
  DELTA = 1e-10

  attr_reader :state_transitions, :rewards, :start_state, :terminal_state

  attr_accessor :q_values, :score_log

  def initialize(state_transitions:, rewards:)
    @state_transitions = state_transitions
    @rewards = rewards
    @start_state = 12
    @terminal_state = 3
    @q_values = Torch.zeros(16, 4)
    @score_log = []
  end

  def q_update(next_q_values:, actions:, rewards:, states:)
    new_q_values = Torch.zeros(16, 4) + q_values

    rewards.length.pred.downto(0) do |idx|
      new_q_values[states[idx]][actions[idx]] = q_values[states[idx]][actions[idx]] +
        (ALPHA * (rewards[idx] + (GAMMA * next_q_values) - q_values[states[idx]][actions[idx]]))
      next_q_values = Torch.max(q_values[states[idx]])
    end

    new_q_values
  end

  def test_agent # rubocop:disable Metrics/MethodLength
    state = start_state
    # done = false
    steps = 0
    total_rewards = 0
    states_log = []
    while (state != terminal_state) && (steps < 100)
      states_log.append(state)
      action = Torch.argmax(q_values[state]).to_i
      state = transit_state(state, action:)
      total_rewards += rewards[state]
      steps += 1
    end
    states_log.append(state)
    [total_rewards, states_log]
  end

  def perform # rubocop:disable Metrics/MethodLength
    300.times do
      state = start_state
      state_log = []
      reward_log = []
      action_log = []
      steps = 0

      loop do
        break if (state == terminal_state) || (steps >= 30)

        state_log.append(state)
        action = rand >= EPSILON ? Torch.argmax(q_values[state]).to_i : rand(4)
        action_log.append(action)

        state = transit_state(state, action:)
        reward_log.append(rewards[state])

        steps = steps.succ
      end

      next_q_values = Torch.max(q_values[state])
      @q_values = q_update(next_q_values:, actions: action_log, rewards: reward_log, states: state_log)

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
rewards[3] = 10
rewards[2] = -1
rewards[11] = -1
rewards[10] = -1
Utility.table_plot(rewards.type(:int).reshape(4, 4), title: "rewards", padding: [0, 1])

ql = QLearningEGreedy.new(state_transitions:, rewards:)
ql.perform

# plot q_values
Utility.table_plot(ql.q_values, title: "Q values")

# plot data
Utility.pyplot(Array(0..ql.score_log.length.pred), ql.score_log.map(&:to_f))

# test agent
_, state_log = ql.test_agent
state_view = Torch.zeros(16)
state_view[state_log] = 1
Utility.block_plot(state_view.type(:int).reshape(4, 4), title: "state log")
