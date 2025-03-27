require "csv"
require "torch-rb"
require_relative "common/global"
require_relative "common/utility"

class ReinforceGridWorld
  ALPHA = 0.01
  EPSILON = 0.4
  GAMMA = 0.9
  DELTA = 1e-10

  attr_reader :state_transitions, :rewards, :start_state, :terminal_state

  attr_accessor :state_action_logprobs, :score_log

  def initialize(state_transitions:, rewards:)
    @state_transitions = state_transitions
    @rewards = rewards
    @start_state = 12
    @terminal_state = 3
    @state_action_logprobs = Torch.rand(16, 4, dtype: :float64)
    @score_log = []
  end

  def calc_returns(rewards:)
    returns = []
    r = 0
    rewards.length.pred.downto(0) do |idx|
      r = rewards[idx] + (GAMMA * r)
      returns.prepend(r)
    end
    returns
  end

  def test_agent # rubocop:disable Metrics/MethodLength
    state = start_state
    # done = false
    steps = 0
    total_rewards = 0
    states_log = []
    while (state != terminal_state) && (steps < 100)
      states_log.append(state)
      action = Torch.multinomial(Utility.softmax(state_action_logprobs[state]), 1).to_i
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
      state_log = []
      reward_log = []
      action_log = []
      steps = 0

      loop do
        break if (state == terminal_state) || (steps >= 30)

        state_log.append(state)
        action = Torch.multinomial(Utility.softmax(state_action_logprobs[state]), 1).to_i
        action_log.append(action)

        state = transit_state(state, action:)
        reward_log.append(rewards[state])

        steps = steps.succ
      end

      state_returns = calc_returns(rewards: reward_log)
      advantage = Torch.tensor(state_returns)
      state_action_logprobs[state_log, action_log] = state_action_logprobs[state_log, action_log] + (ALPHA * advantage)
      @state_action_logprobs = Torch.clip(state_action_logprobs, -5, 5)

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
rewards[2] = -5
rewards[11] = -5
rewards[10] = -5
Utility.table_plot(rewards.type(:int).reshape(4, 4), title: "rewards", padding: [0, 1])

model = ReinforceGridWorld.new(state_transitions:, rewards:)
model.perform

# plot q_values
Utility.table_plot(model.state_action_logprobs, title: "State Action Logprobs")

# plot data
Utility.pyplot(Array(0..model.score_log.length.pred), model.score_log.map(&:to_f))

# test agent
_, state_log = model.test_agent
state_view = Torch.zeros(16)
state_view[state_log] = 1
Utility.block_plot(state_view.type(:int).reshape(4, 4), title: "state log")
