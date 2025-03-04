require_relative "../test/test_helper"
require_relative "utility"

describe Utility do
  it "should implement softmax from scratch" do
    x = Torch.tensor([1.0, 2.0, 99.123])

    assert Torch.equal(Torch.softmax(x, dim: 0), Utility.softmax_from_scratch(x))
  end

  it "should implement softmax" do
    x = Torch.tensor([1.0, 2.0, 99.123])

    assert Torch.equal(Torch.softmax(x, dim: 0), Utility.softmax(x))
  end
end
