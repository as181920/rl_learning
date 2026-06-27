# README

## python code

[pytorch-tutorials/rl](https://github.com/LukeDitria/pytorch_tutorials/tree/main/section11_rl/notebooks/Gridworlds)

## install torch-rb

[torch-rb](https://github.com/ankane/torch.rb)

detech your cuda version and install with libtorch, for example:


```bash
aria2c https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.12.1%2Bcu126.zip

gem install torch-rb -v 0.24.0 --verbose -- \
  --with-torch-dir=/home/andersen/Installed/libtorch \
  --with-cuda-dir=/usr/local/cuda-12.6
```

## run examples

Run scripts from repo root:

```bash
ruby scripts/average_returns.rb
ruby scripts/monte_carlo.rb
ruby scripts/temporal_difference.rb
ruby scripts/td_e_greedy.rb
ruby scripts/q_learning_e_greedy.rb
ruby scripts/reinforce_grid_world.rb
ruby scripts/actor_critic_learning.rb
```

All learning implementations live in `lib/rl_learning/...` and all tests live in
`test/rl_learning/...` for consistent structure.

## tests

```bash
bundle exec ruby -Itest -e 'Dir["test/rl_learning/**/*_test.rb"].sort.each { |f| require_relative f }'
# or
rake test
```

Tips:
- `test/rl_learning/grid_world/structure_test.rb` is the "structure guardrail": it verifies the class entrypoint and that all scripts stay under `scripts/`.
- If tests fail before you run training examples, check `torch-rb` environment first (missing shared library or gem version in this repo setup can block test bootstrapping).
