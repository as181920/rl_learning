# frozen_string_literal: true

require_relative "../../test_helper"

describe ReinforceGridWorld do
  include GridWorldTestHelpers

  it "runs one training episode and updates policy logits" do
    state_transitions = Array.new(16) { Array.new(4, 3) }
    state_transitions[12] = [0, 0, 0, 0]
    rewards = Torch.zeros(16)
    rewards[3] = 10
    rewards[0] = 0

    with_temp_constant(ReinforceGridWorld, :TRAINING_EPISODES, 1) do
      model = ReinforceGridWorld.new(state_transitions:, rewards:)
      old_logits = model.policy_logits.to_a
      model.perform

      assert_equal 1, model.score_log.length
      refute_equal old_logits, model.policy_logits.to_a
      states, episode_rewards = model.collect_episode

      assert_operator states.length, :<=, 2
      assert_operator episode_rewards.length, :<=, 2
    end
  end
end
