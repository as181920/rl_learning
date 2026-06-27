# frozen_string_literal: true

require_relative "../../test_helper"

describe MonteCarlo do
  include GridWorldTestHelpers

  it "collects one-step episodes for immediate terminal transition" do
    state_transitions, rewards = build_simple_episode_data
    model = MonteCarlo.new(state_transitions:, rewards:)
    states, episode_rewards = model.collect_episode

    assert_equal [12, 3], states
    assert_equal [0, 1], episode_rewards
  end

  it "updates state values by Monte Carlo return" do
    state_transitions, rewards = build_simple_episode_data
    with_temp_constant(MonteCarlo, :TRAINING_EPISODES, 1) do
      model = MonteCarlo.new(state_transitions:, rewards:)
      model.perform

      values = model.state_values.to_a

      assert_in_delta 0.0045, values[12], 1e-6
      assert_in_delta 0.005, values[3], 1e-6
      assert_equal [1], model.score_log
    end
  end
end
