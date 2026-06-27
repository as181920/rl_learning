require_relative "../common/global"
require_relative "../common/utility"

class ReinforceGridWorld
  ALPHA = 0.01
  GAMMA = 0.9
  ACTION_COUNT = 4
  MAX_STEPS_PER_EPISODE = 30
  TRAINING_EPISODES = 1000

  attr_reader :state_transitions, :rewards, :start_state, :terminal_state

  attr_accessor :policy_logits, :score_log

  def initialize(state_transitions:, rewards:)
    @state_transitions = state_transitions
    @rewards = rewards
    @start_state = 12
    @terminal_state = 3
    @policy_logits = Torch.rand(16, ACTION_COUNT, dtype: :float64)
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

      action = sample_action(state)
      actions << action

      state = transit_state(state, action:)
      rewards << @rewards[state]
      steps += 1
    end

    [states, rewards, actions]
  end

  def compute_discounted_returns(rewards:)
    returns = []
    r = 0
    rewards.length.pred.downto(0) do |idx|
      r = rewards[idx] + (GAMMA * r)
      returns.prepend(r)
    end
    Torch.tensor(returns)
  end

  def policy_for(state)
    Utility.softmax(policy_logits[state])
  end

  def sample_action(state)
    Torch.multinomial(policy_for(state), 1).to_i
  end

  def test_agent
    state = start_state
    steps = 0
    total_rewards = 0
    states_log = []
    while (state != terminal_state) && (steps < MAX_STEPS_PER_EPISODE)
      states_log.append(state)
      action = sample_action(state)
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

      discounted_returns = compute_discounted_returns(rewards: rewards)
      baseline = discounted_returns.mean
      advantages = discounted_returns - baseline

      states.each_with_index do |state, idx|
        action = actions[idx]
        action_probs = policy_for(state)
        advantage = advantages[idx].to_f

        ACTION_COUNT.times do |action_idx|
          target = action_idx == action ? 1.0 : 0.0
          @policy_logits[state, action_idx] += ALPHA * advantage * (target - action_probs[action_idx].to_f)
        end
      end

      @policy_logits = Torch.clip(policy_logits, -5, 5)

      score_log.append(test_agent[0])
    end
  end

  private

    def transit_state(state, action: rand(ACTION_COUNT))
      state_transitions.dig(state, action)
    end
end
