require "active_support/all"
require "matplotlib/pyplot"
require "torch-rb"
require "vips"

module Utility
  module_function

  def softmax_from_scratch(x)
    x -= x.max(0)[0]
    x.exp / x.exp.sum
  end

  def softmax(x, dim: 0)
    x.softmax(dim:)
  end

  def plot(x, y, ylim: [-0.1, 1.1], title: nil)
    plt = Matplotlib::Pyplot
    plt.plot(x.to_a, y.to_a)
    plt.title(title) if title.present?
    plt.ylim(*ylim) if ylim.present?
    plt.show
  end

  def implot(image_array, ylim: [], title: nil)
    plt = Matplotlib::Pyplot
    plt.imshow(image_array)
    plt.title(title) if title.present?
    plt.ylim(*ylim) if ylim.present?
    plt.show
  end
end
