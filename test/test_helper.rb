# frozen_string_literal: true

require "debug"
require "minitest/autorun"
begin
  require "minitest/mock"
rescue LoadError
  # Optional in minimal environments.
end
begin
  require "mocha/minitest"
rescue LoadError
  # Optional in minimal environments.
end

begin
  require "minitest/reporters"
  Minitest::Reporters.use!
rescue LoadError
  # Optional in minimal environments.
end

$LOAD_PATH.unshift File.expand_path("../support", __dir__)
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)

require "torch-rb"

require "rl_learning"

require_relative "support/grid_world_test_helper"
