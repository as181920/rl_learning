require_relative "../common/global"
require_relative "../common/utility"

class ActorCritic
  ALPHA = 0.005
  GAMMA = 0.9
  ACTION_COUNT = 4
  MAX_STEPS_PER_EPISODE = 30
  TRAINING_EPISODES = 1000

  attr_reader :state_transitions, :rewards, :start_state, :terminal_state

  attr_accessor :state_values, :policy_logits, :score_log

  def initialize(state_transitions:, rewards:)
    @state_transitions = state_transitions
    @rewards = rewards
    @start_state = 12
    @terminal_state = 3
    @state_values = Torch.zeros(16)
    @policy_logits = Torch.rand(16, ACTION_COUNT, dtype: :float64)
    @score_log = []
  end

  def collect_episode
    state = start_state
    states = []
    rewards = []
    actions = []
    state_values_log = []
    steps = 0

    until state == terminal_state || steps >= MAX_STEPS_PER_EPISODE
      states << state
      state_values_log << state_values[state]

      action = sample_action(state)
      actions << action

      state = transit_state(state, action:)
      rewards << self.rewards[state]
      steps += 1
    end

    [states, rewards, actions, state_values_log]
  end

  def calc_returns(rewards:)
    returns = []
    return_sum = 0
    rewards.length.pred.downto(0) do |idx|
      return_sum = rewards[idx] + (GAMMA * return_sum)
      returns.prepend(return_sum)
    end

    returns
  end

  def temporal_difference_update(next_value:, rewards:, states:)
    rewards.length.pred.downto(0) do |idx|
      td_target = rewards[idx] + (GAMMA * next_value)
      state = states[idx]
      state_values[state] = state_values[state] + (ALPHA * (td_target - state_values[state]))
      next_value = state_values[state]
    end
  end

  def test_agent
    state = start_state
    steps = 0
    total_rewards = 0
    states_log = []
    while (state != terminal_state) && (steps < MAX_STEPS_PER_EPISODE)
      states_log << state
      action = sample_action(state)
      state = transit_state(state, action:)
      total_rewards += rewards[state]
      steps += 1
    end
    states_log << state
    [total_rewards, states_log]
  end

  def perform
    TRAINING_EPISODES.times do
      states, episode_rewards, actions, values_before = collect_episode
      next_value = state_values[states[-1]]
      temporal_difference_update(next_value:, rewards: episode_rewards, states:)

      returns = calc_returns(rewards: episode_rewards)
      advantages = Torch.tensor(returns) - Torch.tensor(values_before)
      update_policy(states:, actions:, advantages:)

      @policy_logits = Torch.clip(policy_logits, -5, 5)
      score_log << test_agent[0]
    end
  end

  private

    def policy_for(state)
      Utility.softmax(policy_logits[state])
    end

    def sample_action(state)
      Torch.multinomial(policy_for(state), 1).to_i
    end

    def update_policy(states:, actions:, advantages:)
      states.each_with_index do |state, idx|
        action = actions[idx]
        policy_probs = policy_for(state)
        advantage = advantages[idx].to_f

        ACTION_COUNT.times do |action_idx|
          target = action_idx == action ? 1.0 : 0.0
          @policy_logits[state, action_idx] += ALPHA * advantage * (target - policy_probs[action_idx].to_f)
        end
      end
    end

    def transit_state(state, action: rand(ACTION_COUNT))
      state_transitions.dig(state, action)
    end
end
