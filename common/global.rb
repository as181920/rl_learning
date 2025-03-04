require "torch-rb"
require "debug"

module Global
  DEVICE = Torch::CUDA.available? ? ENV.fetch("DEVICE", "cuda") : "cpu"
  PRECISION = Torch::CUDA.available? ? ENV.fetch("PRECISION", "float64").to_sym : :float64
end
