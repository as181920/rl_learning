require "active_support/all"
require "matplotlib/pyplot"
require "rainbow"
require "torch-rb"
require "tty-table"
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

  def table_print(table_data, title: nil)
    puts Rainbow(title).bright.aqua if title.present?
    puts TTY::Table.new(table_data).render(:unicode, border: { separator: :each_row })
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

  def ansi_plot(image_array, title: nil)
    puts Rainbow(title).bright.aqua if title.present?

    filler = "⬛" # "  "
    image_array.each do |row|
      row.each do |cell|
        # print cell.nonzero? ? colorize(r: 255, g: 170, b: 51) : colorize
        color = cell.nonzero? ? [(cell.to_f * 255).clamp(0, 255), 0, 0] : [245, 245, 245]
        print Rainbow(filler).color(*color)
      end
      print "\n"
    end
  end

  def colorize(text = "  ", r: 245, g: 245, b: 245)
    "\e[48;2;#{r};#{g};#{b}m#{text}\e[0m"
  end
end
