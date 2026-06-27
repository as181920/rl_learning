require_relative "../common/global"
require_relative "../common/utility"

class AverageReturns
  GAMMA = 0.9
  DELTA = 1e-10
  MAX_STEPS_PER_EPISODE = 30
  TRAINING_EPISODES = 100
  ACTION_COUNT = 4

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

  def collect_episode
    state = start_state
    states = [state]
    rewards = [self.rewards[state]]
    steps = 0

    until state == terminal_state || steps >= MAX_STEPS_PER_EPISODE
      state = transit_state(state)
      steps += 1

      states << state
      rewards << self.rewards[state]
    end

    [states, rewards]
  end

  def compute_returns(rewards:, states:)
    state_count = Torch.zeros(16)
    state_returns = Torch.zeros(16)
    cumulative_return = 0
    rewards.length.pred.downto(0) do |idx|
      cumulative_return = rewards[idx] + (GAMMA * cumulative_return)
      state_returns[states[idx]] += cumulative_return
      state_count[states[idx]] += 1
    end

    state_returns / (state_count + DELTA)
  end

  def test_agent
    state = start_state
    steps = 0
    total_rewards = 0
    states_log = []
    while (state != terminal_state) && (steps < MAX_STEPS_PER_EPISODE)
      states_log.append(state)
      action = best_action_for(state)
      state = transit_state(state, action:)
      total_rewards += rewards[state]
      steps += 1
    end
    states_log.append(state)
    [total_rewards, states_log]
  end

  def perform
    returns_log = []

    TRAINING_EPISODES.times do
      states, rewards = collect_episode
      returns_log.append(compute_returns(rewards: rewards, states: states))
      @state_values = Torch.mean(Torch.stack(returns_log), dim: 0)

      score_log.append(test_agent[0])
    end
  end

  private

    def best_action_for(state)
      action_scores = state_values[state_transitions[state]]
      Torch.argmax(action_scores).to_i
    end

    def transit_state(state, action: rand(ACTION_COUNT))
      state_transitions.dig(state, action)
    end
end
