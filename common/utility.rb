require "active_support/all"
require "json"
require "matplotlib/pyplot"
require "rainbow"
require "torch-rb"
require "tty-table"
require "vips"

module Utility
  module_function

  DELTA = 1e-10
  VIRIDIS_COLORS = JSON.load_file(File.expand_path("cmap_viridis.json", __dir__))

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

  def pyplot(x, y, ylim: [-0.1, 1.1], title: nil)
    plt = Matplotlib::Pyplot
    plt.plot(x.to_a, y.to_a)
    plt.title(title) if title.present?
    plt.ylim(*ylim) if ylim.present?
    plt.show
  end

  def imshow(image_array, ylim: [], title: nil)
    plt = Matplotlib::Pyplot
    plt.imshow(image_array)
    plt.title(title) if title.present?
    plt.ylim(*ylim) if ylim.present?
    plt.show
  end

  def block_plot(image_array, title: nil)
    puts Rainbow(title).bright.aqua if title.present?

    filler = "⬛" # "  "
    image_array.each do |row|
      row.each do |value|
        # print value.nonzero? ? colorize(r: 255, g: 170, b: 51) : colorize
        # color = value.nonzero? ? [(value.to_f * 255).clamp(0, 255), 0, 0] : [245, 245, 245]
        color = color_map((value.to_f - image_array.min) / (image_array.max - image_array.min + DELTA))
        print Rainbow(filler).color(*color)
      end
      print "\n"
    end
  end

  def ansi_plot(image_array, title: nil, padding: 0)
    puts Rainbow(title).bright.aqua if title.present?

    filter = proc do |value, _row_index, _col_index|
      color = color_map((value.to_f - image_array.min) / (image_array.max - image_array.min + DELTA))
      Rainbow(value.to_s).bg(*color).color(:black)
    end

    puts TTY::Table.new(image_array.to_a).render(:unicode, border: { separator: :each_row }, padding:, filter:)
  end

  def colorize(text = "  ", r: 245, g: 245, b: 245)
    "\e[48;2;#{r};#{g};#{b}m#{text}\e[0m"
  end

  def color_map(value, colormap: VIRIDIS_COLORS)
    idx = (value.to_f.clamp(0, 1) * (colormap.size - 1)).to_i
    colormap[idx] # returns [r, g, b]
  end
end
