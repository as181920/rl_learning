# frozen_string_literal: true

require_relative "../../test_helper"

describe QLearningEGreedy do
  include GridWorldTestHelpers

  it "learns first-step Q values" do
    state_transitions, rewards = build_simple_episode_data

    with_temp_constant(QLearningEGreedy, :TRAINING_EPISODES, 1) do
      model = QLearningEGreedy.new(state_transitions:, rewards:)
      model.perform

      q_values = model.q_values.to_a

      assert_equal [1], model.score_log
      assert(q_values.any? { |values| values.any?(&:positive?) })
    end
  end

  it "returns max action value for known q row" do
    state_transitions, rewards = build_simple_episode_data
    model = QLearningEGreedy.new(state_transitions:, rewards:)
    model.q_values[3] = Torch.tensor([0.1, 0.3, -0.2, 0.0])

    assert_in_delta 0.3, model.send(:next_action_value, 3), 1e-6
    assert_equal 1, model.send(:best_action_for, 3)
  end
end
