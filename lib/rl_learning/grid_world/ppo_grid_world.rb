require_relative "../common/global"
require_relative "../common/utility"

# Tabular adaptation of section11_rl/solutions/Procgen_PPO.ipynb.
# Autograd and Adam replace the manual policy updates used in earlier examples.
# Keep the rollout and update stages together for comparison with the notebook.
# rubocop:disable-next Metrics/ClassLength
class PPOGridWorld
  GAMMA = 0.9
  TAU = 0.95
  CLIP_PARAM = 0.2
  ALPHA = 0.02
  ACTION_COUNT = 4
  MAX_STEPS_PER_EPISODE = 30
  TRAINING_ITERATIONS = 200
  ROLLOUT_STEPS = 256
  PPO_EPOCHS = 3
  MINI_BATCH_SIZE = 64
  ENTROPY_COEF = 0.01
  MAX_GRAD_NORM = 0.5

  attr_reader :state_transitions, :rewards, :start_state, :terminal_state,
              :policy_logits, :state_values, :score_log, :loss_log, :frames_seen

  # Expose independent training settings as named arguments for experiments.
  # rubocop:disable-next Metrics/ParameterLists
  def initialize(state_transitions:, rewards:, learning_rate: ALPHA,
                 rollout_steps: ROLLOUT_STEPS, ppo_epochs: PPO_EPOCHS, mini_batch_size: MINI_BATCH_SIZE)
    [rollout_steps, ppo_epochs, mini_batch_size].each do |size|
      raise ArgumentError, "Rollout, epoch and batch sizes must be positive integers" unless size.is_a?(Integer) && size.positive?
    end
    @state_transitions = state_transitions
    @rewards = rewards
    @start_state = 12
    @terminal_state = 3
    @rollout_steps = rollout_steps
    @ppo_epochs = ppo_epochs
    @mini_batch_size = mini_batch_size
    @policy_logits = Torch.zeros(16, ACTION_COUNT, dtype: :float64, requires_grad: true)
    @state_values = Torch.zeros(16, dtype: :float64, requires_grad: true)
    @optimizer = Torch::Optim::Adam.new([policy_logits, state_values], lr: learning_rate)
    @score_log = []
    @loss_log = []
    @frames_seen = 0
  end

  def policy_for(state)
    Utility.softmax(policy_logits[state])
  end

  def sample_action(state)
    Torch.multinomial(policy_for(state), 1).to_i
  end

  # Collect a fresh on-policy rollout. Old probabilities and values are numeric
  # snapshots, never views into parameters that the optimizer will mutate.
  def collect_rollout
    data = %i[states actions log_probs rewards values next_values masks trace_masks].to_h { |key| [key, []] }
    state = start_state
    steps = 0
    Torch.no_grad do
      @rollout_steps.times do
        action = sample_action(state)
        next_state = state_transitions.dig(state, action)
        steps += 1
        episode_end = log_transition(data, state, action, next_state, steps)
        state = episode_end ? start_state : next_state
        steps = 0 if episode_end
      end
    end
    returns, advantages = compute_gae(**data.slice(:rewards, :values, :next_values, :masks, :trace_masks))
    data.merge(returns:, advantages:)
  end

  # masks disable bootstrapping only at true terminals; trace_masks also stop
  # GAE at time limits so an advantage never leaks into the next episode.
  def compute_gae(rewards:, values:, next_values:, masks:, trace_masks: masks)
    advantages = Array.new(rewards.length)
    gae = 0.0
    rewards.length.pred.downto(0) do |idx|
      delta = rewards[idx] + (GAMMA * next_values[idx] * masks[idx]) - values[idx]
      gae = delta + (GAMMA * TAU * trace_masks[idx] * gae)
      advantages[idx] = gae
    end
    returns = advantages.each_with_index.map { |advantage, idx| advantage + values[idx] }
    [returns, advantages]
  end

  # Maximize this surrogate objective, hence the minus sign in the total loss.
  def ppo_loss(new_log_probs:, old_log_probs:, advantages:)
    ratio = (new_log_probs - old_log_probs).exp
    surrogate = ratio * advantages
    clipped_surrogate = ratio.clamp(1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * advantages
    Torch.minimum(surrogate, clipped_surrogate).mean
  end

  def clipped_critic_loss(new_values:, old_values:, returns:)
    clipped_values = old_values + (new_values - old_values).clamp(-CLIP_PARAM, CLIP_PARAM)
    Torch.maximum((new_values - returns).pow(2), (clipped_values - returns).pow(2)).mean
  end

  def ppo_update(rollout)
    data = rollout_tensors(rollout)
    @ppo_epochs.times do
      (0...rollout[:states].length).to_a.shuffle.each_slice(@mini_batch_size) do |indices|
        update_batch(data.transform_values { |tensor| tensor[indices] })
      end
    end
  end

  def perform(iterations: TRAINING_ITERATIONS)
    iterations.times do
      rollout = collect_rollout
      ppo_update(rollout)
      @frames_seen += rollout[:states].length
      score_log << test_agent[0]
    end
    self
  end

  def test_agent(greedy: false)
    state = start_state
    total_reward = 0.0
    states_log = [state]
    Torch.no_grad do
      MAX_STEPS_PER_EPISODE.times do
        break if state == terminal_state

        action = greedy ? policy_for(state).argmax.to_i : sample_action(state)
        state = state_transitions.dig(state, action)
        total_reward += rewards[state].to_f
        states_log << state
      end
    end
    [total_reward, states_log]
  end

  private

    def log_transition(data, state, action, next_state, steps)
      terminated = next_state == terminal_state
      episode_end = terminated || steps >= MAX_STEPS_PER_EPISODE
      transition = {
        states: state, actions: action,
        log_probs: policy_logits[state].log_softmax(0)[action].to_f,
        rewards: rewards[next_state].to_f, values: state_values[state].to_f,
        next_values: terminated ? 0.0 : state_values[next_state].to_f,
        masks: terminated ? 0.0 : 1.0, trace_masks: episode_end ? 0.0 : 1.0
      }
      transition.each { |key, value| data[key] << value }
      episode_end
    end

    def rollout_tensors(rollout)
      data = %i[states actions log_probs values returns advantages].to_h do |key|
        dtype = %i[states actions].include?(key) ? :int64 : :float64
        [key, Torch.tensor(rollout.fetch(key), dtype:)]
      end
      advantages = data[:advantages]
      # Population variance is finite even for a one-transition rollout.
      std = (advantages - advantages.mean).pow(2).mean.sqrt
      data[:advantages] = (advantages - advantages.mean) / (std + 1e-8)
      data
    end

    def update_batch(batch)
      log_probs = policy_logits[batch[:states]].log_softmax(1)
      action_log_probs = log_probs.gather(1, batch[:actions].unsqueeze(1)).squeeze(1)
      actor = ppo_loss(new_log_probs: action_log_probs, old_log_probs: batch[:log_probs], advantages: batch[:advantages])
      critic = clipped_critic_loss(new_values: state_values[batch[:states]], old_values: batch[:values], returns: batch[:returns])
      entropy = -(log_probs.exp * log_probs).sum(1).mean
      loss = critic - actor - (ENTROPY_COEF * entropy)
      apply_gradients(loss)
      loss_log << { actor: actor.to_f, critic: critic.to_f, entropy: entropy.to_f, total: loss.to_f }
    end

    def apply_gradients(loss)
      @optimizer.zero_grad
      loss.backward
      clip_gradients
      @optimizer.step
    end

    # torch-rb has no nn.utils.clip_grad_norm_; use the same global L2 scaling.
    def clip_gradients
      gradients = [policy_logits.grad, state_values.grad].compact
      norm = Math.sqrt(gradients.sum { |gradient| gradient.pow(2).sum.to_f })
      scale = [MAX_GRAD_NORM / (norm + 1e-6), 1.0].min
      Torch.no_grad { gradients.each { |gradient| gradient.mul!(scale) } }
    end
end
