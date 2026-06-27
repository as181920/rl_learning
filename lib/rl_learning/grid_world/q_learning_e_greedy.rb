require_relative "../common/global"
require_relative "../common/utility"

class QLearningEGreedy
  ALPHA = 0.01
  EPSILON = 0.4
  GAMMA = 0.99
  MAX_STEPS_PER_EPISODE = 30
  TRAINING_EPISODES = 300
  ACTION_COUNT = 4

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

  def collect_episode
    state = start_state
    states = []
    actions = []
    rewards = []
    steps = 0

    until state == terminal_state || steps >= MAX_STEPS_PER_EPISODE
      states << state

      action = epsilon_greedy_action(state)
      actions << action

      state = transit_state(state, action:)
      rewards << self.rewards[state]
      steps += 1
    end

    [states, rewards, actions]
  end

  def q_update(next_value:, actions:, rewards:, states:)
    new_q_values = Torch.zeros(16, 4) + q_values

    rewards.length.pred.downto(0) do |idx|
      new_q_values[states[idx]][actions[idx]] = q_values[states[idx]][actions[idx]] +
        (ALPHA * (rewards[idx] + (GAMMA * next_value) - q_values[states[idx]][actions[idx]]))
      next_value = next_action_value(states[idx])
    end

    new_q_values
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
    TRAINING_EPISODES.times do
      states, rewards, actions = collect_episode
      next_q_value = next_action_value(terminal_state)
      @q_values = q_update(next_value: next_q_value, actions:, rewards:, states:)

      score_log.append(test_agent[0])
    end
  end

  private

    def best_action_for(state)
      q_values_for_state = q_values[state]
      Torch.argmax(q_values_for_state).to_i
    end

    def epsilon_greedy_action(state)
      rand >= EPSILON ? best_action_for(state) : rand(ACTION_COUNT)
    end

    def next_action_value(state)
      Torch.max(q_values[state]).to_f
    end

    def transit_state(state, action: rand(ACTION_COUNT))
      state_transitions.dig(state, action)
    end
end
