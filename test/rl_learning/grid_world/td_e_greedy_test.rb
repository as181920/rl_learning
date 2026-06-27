# frozen_string_literal: true

require_relative "../../test_helper"

describe TemporalDifferenceEGreedy do
  include GridWorldTestHelpers

  it "uses epsilon-greedy action choice and updates after one episode" do
    state_transitions, rewards = build_simple_episode_data(terminal_reward: 10)

    with_temp_constant(TemporalDifferenceEGreedy, :TRAINING_EPISODES, 1) do
      model = TemporalDifferenceEGreedy.new(state_transitions:, rewards:)
      model.perform

      values = model.state_values.to_a

      assert_in_delta 0.2, values[3], 1e-6
      assert_in_delta 0.0036, values[12], 1e-6
      refute_empty model.score_log
      assert model.send(:greedy_action_for, 12).between?(0, 3)
    end
  end
end
