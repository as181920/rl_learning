# frozen_string_literal: true

module GridWorldTestHelpers
  module_function

  def build_simple_episode_data(terminal_reward: 1.0, penalty_states: {})
    state_transitions = Array.new(16) { Array.new(4, 3) }
    state_transitions[12] = [3, 3, 3, 3]
    state_transitions[3] = [3, 3, 3, 3]

    rewards = Torch.zeros(16)
    rewards[3] = terminal_reward
    penalty_states.each do |state, reward|
      rewards[state] = reward
    end

    [state_transitions, rewards]
  end

  def with_temp_constant(klass, const_name, value)
    original_value = klass.const_get(const_name)
    klass.send(:remove_const, const_name)
    klass.const_set(const_name, value)

    yield
  ensure
    klass.send(:remove_const, const_name)
    klass.const_set(const_name, original_value)
  end
end
