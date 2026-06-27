# frozen_string_literal: true

require_relative "../../test_helper"

describe ActorCritic do
  include GridWorldTestHelpers

  it "runs one episode, updates values and policy" do
    state_transitions, rewards = build_simple_episode_data(terminal_reward: 10, penalty_states: { 2 => -1, 11 => -1, 10 => -1 })
    with_temp_constant(ActorCritic, :TRAINING_EPISODES, 1) do
      model = ActorCritic.new(state_transitions:, rewards:)
      old_logits = model.policy_logits.to_a
      old_values = model.state_values.to_a
      model.perform

      refute_equal old_values, model.state_values.to_a
      assert_equal 1, model.score_log.length
      refute_equal old_logits, model.policy_logits.to_a
    end
  end
end
