# frozen_string_literal: true

require_relative "../../test_helper"

describe TemporalDifference do
  include GridWorldTestHelpers

  it "updates values with TD(0) style targets" do
    state_transitions, rewards = build_simple_episode_data
    with_temp_constant(TemporalDifference, :TRAINING_EPISODES, 1) do
      model = TemporalDifference.new(state_transitions:, rewards:)
      model.perform

      values = model.state_values.to_a

      assert_in_delta 0.00036, values[12], 1e-6
      assert_in_delta 0.02, values[3], 1e-6
      assert_equal [1], model.score_log
    end
  end
end
