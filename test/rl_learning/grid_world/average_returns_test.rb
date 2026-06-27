# frozen_string_literal: true

require_relative "../../test_helper"

describe AverageReturns do
  include GridWorldTestHelpers

  it "collects an episode that ends at terminal state" do
    state_transitions, rewards = build_simple_episode_data
    model = AverageReturns.new(state_transitions:, rewards:)
    states, episode_rewards = model.collect_episode

    assert_equal [12, 3], states
    assert_equal [0, 1], episode_rewards
  end

  it "updates averages across episodes and keeps score log" do
    state_transitions, rewards = build_simple_episode_data

    with_temp_constant(AverageReturns, :TRAINING_EPISODES, 2) do
      model = AverageReturns.new(state_transitions:, rewards:)
      model.perform

      assert_equal 2, model.score_log.length
      values = model.state_values.to_a

      assert_in_delta 0.9, values[12], 1e-6
      assert_in_delta 1.0, values[3], 1e-6
    end
  end
end
