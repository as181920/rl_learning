# frozen_string_literal: true

require_relative "../../test_helper"

describe Utility do
  it "implements softmax from scratch" do
    x = Torch.tensor([1.0, 2.0, 99.123])

    assert Torch.equal(Torch.softmax(x, dim: 0), Utility.softmax_from_scratch(x))
  end

  it "implements softmax" do
    x = Torch.tensor([1.0, 2.0, 99.123])

    assert Torch.equal(Torch.softmax(x, dim: 0), Utility.softmax(x, dim: 0))
  end

  it "allows empty episode log in trend plot call" do
    assert_nil Utility.plot_episode_returns([], title: "empty")
  end
end
