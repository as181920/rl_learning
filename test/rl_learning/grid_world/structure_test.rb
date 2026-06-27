# frozen_string_literal: true

require_relative "../../test_helper"

LEARNERS = [
  {
    class_name: :AverageReturns,
    script: "average_returns.rb",
    implementation: "average_returns.rb",
    test_file: "average_returns_test.rb"
  },
  {
    class_name: :MonteCarlo,
    script: "monte_carlo.rb",
    implementation: "monte_carlo.rb",
    test_file: "monte_carlo_test.rb"
  },
  {
    class_name: :TemporalDifference,
    script: "temporal_difference.rb",
    implementation: "temporal_difference.rb",
    test_file: "temporal_difference_test.rb"
  },
  {
    class_name: :TemporalDifferenceEGreedy,
    script: "td_e_greedy.rb",
    implementation: "td_e_greedy.rb",
    test_file: "td_e_greedy_test.rb"
  },
  {
    class_name: :QLearningEGreedy,
    script: "q_learning_e_greedy.rb",
    implementation: "q_learning_e_greedy.rb",
    test_file: "q_learning_e_greedy_test.rb"
  },
  {
    class_name: :ReinforceGridWorld,
    script: "reinforce_grid_world.rb",
    implementation: "reinforce_grid_world.rb",
    test_file: "reinforce_grid_world_test.rb"
  },
  {
    class_name: :ActorCritic,
    script: "actor_critic_learning.rb",
    implementation: "actor_critic_learning.rb",
    test_file: "actor_critic_learning_test.rb"
  }
].freeze

ROOT_DIR = File.join(__dir__, "../../../").freeze
SCRIPTS_DIR = File.join(__dir__, "../../../scripts").freeze
LIB_DIR = File.join(__dir__, "../../../lib/rl_learning/grid_world").freeze
TEST_DIR = File.join(__dir__, "../../../test/rl_learning/grid_world").freeze

describe "Grid World project structure" do
  it "loads all grid world learners through rl_learning entrypoint" do
    LEARNERS.each do |learner|
      const_name = learner.fetch(:class_name)

      assert Object.const_defined?(const_name), "Expected #{const_name} to be loaded"
    end
  end

  it "keeps executable entry scripts in scripts/ directory" do
    LEARNERS.each do |learner|
      path = File.join(SCRIPTS_DIR, learner.fetch(:script))

      assert File.file?(path), "Missing script: #{learner.fetch(:script)}"
      assert File.executable?(path), "Script is not executable: #{learner.fetch(:script)}"
    end
  end

  it "maps each learner implementation to a grid_world test file" do
    LEARNERS.each do |learner|
      path = File.join(TEST_DIR, learner.fetch(:test_file))

      assert File.file?(path), "Missing test file: #{learner.fetch(:test_file)}"
    end
  end

  it "keeps core Ruby files under lib/ and test/ directories" do
    LEARNERS.each do |learner|
      path = File.join(LIB_DIR, learner.fetch(:implementation))

      assert File.file?(path), "Missing implementation file: #{learner.fetch(:implementation)}"
    end

    root_files = Dir[File.join(ROOT_DIR, "*.rb")]

    assert_empty root_files, "Top-level Ruby files should be moved to lib/ or scripts/: #{root_files.join(", ")}"
  end
end
