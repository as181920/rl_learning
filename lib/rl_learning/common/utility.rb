require "active_support/all"
require "json"
require "rainbow"
require "torch-rb"
require "unicode_plot"
require "tty-logger"
require "tty-pie"
require "tty-spinner"
require "tty-table"
require "vips"

module Utility
  module_function

  DELTA = 1e-10
  VIRIDIS_COLORS = JSON.load_file(File.expand_path("cmap_viridis.json", __dir__))

  Logger = TTY::Logger.new

  def softmax_from_scratch(x)
    x -= x.max(0)[0]
    x.exp / x.exp.sum
  end

  def softmax(x, dim: 0)
    return x.softmax(dim:) if x.respond_to?(:softmax)

    Torch.softmax(x, dim:)
  end

  def table_print(table_data, title: nil)
    puts Rainbow(title).bright.aqua if title.present?
    puts TTY::Table.new(table_data).render(:unicode, border: { separator: :each_row })
  end

  def pyplot(x, y, ylim: [-0.1, 1.1], title: nil)
    require "matplotlib/pyplot"

    plt = Matplotlib::Pyplot
    plt.plot(x.to_a, y.to_a)
    plt.title(title) if title.present?
    plt.ylim(*ylim) if ylim.present?
    plt.show
  end

  def imshow(image_array, ylim: [], title: nil)
    require "matplotlib/pyplot"

    plt = Matplotlib::Pyplot
    plt.imshow(image_array)
    plt.title(title) if title.present?
    plt.ylim(*ylim) if ylim.present?
    plt.show
  end

  def block_plot(image_array, title: nil)
    puts Rainbow(title).bright.aqua if title.present?

    filler = "  "
    image_array.each do |row|
      row.each do |value|
        color = color_map((value.to_f - image_array.min) / (image_array.max - image_array.min + DELTA))
        print Rainbow(filler).bg(*color)
      end
      print "\n"
    end
  end

  def table_plot(image_array, title: nil, padding: 0)
    puts Rainbow(title).bright.aqua if title.present?

    filter = proc do |value, _row_index, _col_index|
      color = color_map((value.to_f - image_array.min) / (image_array.max - image_array.min + DELTA))
      Rainbow(value.to_s).bg(*color).color(:black)
    end

    puts TTY::Table.new(image_array.to_a).render(:unicode, border: { separator: :each_row }, padding:, filter:)
  end

  def pie_plot(
    data = [
      { name: "filler1", value: 309, color: :bright_yellow, fill: "¥" },
      { name: "filler2", value: 382, color: :bright_green, fill: "$" },
      { name: "filler2", value: 309, color: :bright_magenta, fill: "€" }
    ],
    title: nil
  )
    puts Rainbow(title).bright.aqua if title.present?

    puts TTY::Pie.new(data:)
  end

  def colorize(text = "  ", r: 245, g: 245, b: 245)
    "\e[48;2;#{r};#{g};#{b}m#{text}\e[0m"
  end

  def color_map(value, colormap: VIRIDIS_COLORS)
    idx = (value.to_f.clamp(0, 1) * (colormap.size - 1)).to_i
    colormap[idx] # returns [r, g, b]
  end

  def plot_episode_returns(score_log, title:, width: 70, height: 12)
    return if score_log.empty?

    scores = score_log.map(&:to_f)
    score_min = scores.min.to_f
    score_max = scores.max.to_f
    score_range = score_max - score_min
    score_pad = [score_range * 0.05, 0.2].max
    y_min = score_min.negative? ? score_min - score_pad : 0

    puts UnicodePlot.lineplot(
      Array(0...scores.length),
      scores,
      title: title,
      xlabel: "Episode",
      ylabel: "Return",
      ylim: [y_min, score_max],
      width: width,
      height: height
    )
  end
end
