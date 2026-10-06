# frozen_string_literal: true

require_relative "../../test_helper"

describe PPOGridWorld do
  include GridWorldTestHelpers

  def trainable_tensor(values)
    tensor = Torch.tensor(values, dtype: :float64)
    tensor.requires_grad = true
    tensor
  end

  before do
    state_transitions, rewards = build_simple_episode_data(terminal_reward: 10)
    @model = PPOGridWorld.new(state_transitions:, rewards:)
  end

  it "computes GAE backwards and stops at episode boundaries" do
    returns, advantages = @model.compute_gae(
      rewards: [1.0, 2.0, 3.0], values: [0.5, 1.0, 2.0],
      next_values: [1.0, 0.0, 4.0], masks: [1.0, 0.0, 1.0]
    )

    # The third sample belongs to another episode; it must not affect the first two.
    assert_in_delta 2.255, advantages[0]
    assert_in_delta 1.0, advantages[1]
    assert_in_delta 4.6, advantages[2]
    assert_in_delta 2.755, returns[0]
    assert_in_delta 2.0, returns[1]
    assert_in_delta 6.6, returns[2]
  end

  it "clips beneficial probability changes for both advantage signs" do
    new_log_probs = trainable_tensor([Math.log(1.5), Math.log(0.5)])
    objective = @model.ppo_loss(
      new_log_probs:, old_log_probs: Torch.zeros(2, dtype: :float64),
      advantages: Torch.tensor([2.0, -2.0], dtype: :float64)
    )
    objective.backward

    assert_in_delta 0.4, objective.to_f
    new_log_probs.grad.to_a.each { |gradient| assert_in_delta 0.0, gradient }
  end

  it "retains corrective gradients outside the clip interval" do
    new_log_probs = trainable_tensor([Math.log(0.5), Math.log(1.5)])
    objective = @model.ppo_loss(
      new_log_probs:, old_log_probs: Torch.zeros(2, dtype: :float64),
      advantages: Torch.tensor([2.0, -2.0], dtype: :float64)
    )
    objective.backward

    assert_in_delta(-1.0, objective.to_f)
    assert_in_delta 0.5, new_log_probs.grad[0].to_f
    assert_in_delta(-1.5, new_log_probs.grad[1].to_f)
  end

  it "uses the larger critic error and blocks the clipped branch gradient" do
    new_values = trainable_tensor([0.0, 0.5, -0.5])
    loss = @model.clipped_critic_loss(
      new_values:, old_values: Torch.zeros(3, dtype: :float64),
      returns: Torch.tensor([1.0, 1.0, -1.0], dtype: :float64)
    )
    loss.backward

    assert_in_delta 0.76, loss.to_f
    assert_in_delta(-2.0 / 3, new_values.grad[0].to_f)
    assert_in_delta 0.0, new_values.grad[1].to_f
    assert_in_delta 0.0, new_values.grad[2].to_f
  end

  it "takes numeric snapshots and never bootstraps a terminal state" do
    Torch.no_grad do
      @model.state_values[12] = 7.0
      @model.state_values[3] = 999.0
    end
    rollout = @model.collect_rollout
    old_log_prob = rollout[:log_probs].first
    Torch.no_grad do
      @model.policy_logits[12, 0] = 3.0
      @model.state_values[12] = 9.0
    end

    assert_equal PPOGridWorld::ROLLOUT_STEPS, rollout[:states].length
    assert_equal [12], rollout[:states].uniq
    assert_equal [0.0], rollout[:masks].uniq
    assert_equal [0.0], rollout[:next_values].uniq
    assert_equal [10.0], rollout[:returns].uniq
    assert_equal [7.0], rollout[:values].uniq
    assert_in_delta(-Math.log(4), old_log_prob)
    assert_equal old_log_prob, rollout[:log_probs].first
  end

  it "bootstraps time limits without propagating GAE across resets" do
    transitions = Array.new(16) { |state| Array.new(4, state) }
    model = PPOGridWorld.new(state_transitions: transitions, rewards: Torch.zeros(16), rollout_steps: 31)
    Torch.no_grad { model.state_values[12] = 2.0 }
    rollout = model.collect_rollout

    assert_in_delta 1.0, rollout[:masks][29]
    assert_in_delta 0.0, rollout[:trace_masks][29]
    assert_in_delta(-0.2, rollout[:advantages][29])
    assert_in_delta 1.8, rollout[:returns][29]
    assert_in_delta(-0.371, rollout[:advantages][28])
    # The end of the rollout also bootstraps the actual next state.
    assert_in_delta 1.8, rollout[:returns][30]
  end

  it "updates actor and critic across epochs including a partial minibatch" do
    Torch.manual_seed(123)
    transitions, rewards = build_simple_episode_data(terminal_reward: 10, penalty_states: { 2 => -1 })
    transitions[12] = [3, 2, 2, 2]
    model = PPOGridWorld.new(state_transitions: transitions, rewards:, rollout_steps: 17, mini_batch_size: 8, ppo_epochs: 2)
    old_logits = model.policy_logits.to_a
    old_values = model.state_values.to_a
    rollout = model.collect_rollout
    snapshot = Marshal.dump(rollout)
    model.ppo_update(rollout)

    refute_equal old_logits, model.policy_logits.to_a
    refute_equal old_values, model.state_values.to_a
    assert_equal 6, model.loss_log.length
    assert_equal snapshot, Marshal.dump(rollout)
    assert(model.loss_log.all? { |loss| loss.values.all?(&:finite?) })
  end

  it "handles a one-transition rollout with zero advantage variance" do
    model = PPOGridWorld.new(state_transitions: @model.state_transitions, rewards: @model.rewards, rollout_steps: 1)
    model.perform(iterations: 1)

    assert_equal 1, model.frames_seen
    assert_equal [10.0], model.score_log
    assert(model.loss_log.all? { |loss| loss.values.all?(&:finite?) })
    assert_operator model.state_values[12].to_f, :>, 0.0
    assert model.policy_logits.requires_grad
    assert model.state_values.requires_grad
  end

  it "caps the combined actor and critic gradient norm" do
    (100 * (@model.policy_logits.sum + @model.state_values.sum)).backward
    @model.send(:clip_gradients)
    norm = Math.sqrt([@model.policy_logits.grad, @model.state_values.grad].sum { |gradient| gradient.pow(2).sum.to_f })

    assert_in_delta PPOGridWorld::MAX_GRAD_NORM, norm
  end

  it "evaluates a greedy policy without recording gradients" do
    score, path = @model.test_agent(greedy: true)

    assert_in_delta 10.0, score
    assert_equal [12, 3], path
    assert_nil @model.policy_logits.grad
    assert_nil @model.state_values.grad
  end

  it "rejects invalid rollout and minibatch sizes" do
    [0, -1, 1.5].each do |size|
      assert_raises(ArgumentError) do
        PPOGridWorld.new(state_transitions: @model.state_transitions, rewards: @model.rewards, rollout_steps: size)
      end
    end
    assert_raises(ArgumentError) do
      PPOGridWorld.new(state_transitions: @model.state_transitions, rewards: @model.rewards, mini_batch_size: 0)
    end
    assert_raises(ArgumentError) do
      PPOGridWorld.new(state_transitions: @model.state_transitions, rewards: @model.rewards, ppo_epochs: 0)
    end
  end
end
