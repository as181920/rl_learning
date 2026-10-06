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
ruby scripts/ppo_grid_world.rb
```

All learning implementations live in `lib/rl_learning/...` and all tests live in
`test/rl_learning/...` for consistent structure.

## PPO（接续 Gridworld 学习）

目前已实现 Q-learning、REINFORCE 和 Actor-Critic，新增的 `PPOGridWorld` 接在
Actor-Critic 后。课程原始 [Procgen PPO notebook](https://github.com/LukeDitria/pytorch_tutorials/blob/main/section11_rl/solutions/Procgen_PPO.ipynb)
使用 CoinRun 图像、并行环境和 IMPALA CNN；这里将其 PPO 训练算法适配到已有的
16 状态、4 动作 Gridworld，策略和值函数用可求导的表表示，不需要 Python/Gym/Procgen。
这不是 Procgen 环境或 CNN 的逐行移植。

实现与课程代码对应的步骤：

- `collect_rollout`：固定长度采样，保存旧策略 log probability 和旧值的数值快照。
- `compute_gae`：反向计算 GAE 和值函数目标；真正终点不 bootstrap，30 步超限时
  使用实际下一状态的值 bootstrap，并切断不同回合间的 GAE。
- `ppo_loss`：计算 `mean(min(ratio * advantage, clip(ratio, 1−ε, 1+ε) * advantage))`，
  其中 `ratio = exp(new_log_prob − old_log_prob)`，`ε = 0.2`。
- `clipped_critic_loss`：取原始值预测误差和裁剪后的预测误差的较大者。
- `ppo_update`：优势标准化，每轮重新打乱小批量，多轮使用同一份 rollout；
  总损失为 `critic_loss − actor_objective − 0.01 * entropy`，使用 autograd、Adam 和梯度范数裁剪。
  完成更新后重新采样，不保留跨更新的旧经验。

为适应小型 Gridworld，默认 `gamma = 0.9`、学习率 `0.02`，每次采样 256 步，
小批量 64，更新 3 轮，共 200 次采样更新。其余参数见类中的常量。
每次更新后的随机策略评估回报记录在 `score_log`，每个小批量的损失在 `loss_log`，
脚本最后展示贪心策略路径。训练与评估均在 CPU 上进行，与前面的表格示例一致。

```bash
ruby scripts/ppo_grid_world.rb
SEED=42 ITERATIONS=20 ruby scripts/ppo_grid_world.rb # 较短的试运行
ruby -Itest test/rl_learning/grid_world/ppo_grid_world_test.rb
```

也可通过 `PPOGridWorld.new(..., learning_rate:, rollout_steps:, ppo_epochs:, mini_batch_size:)`
调整训练参数，通过 `model.perform(iterations: 200)` 启动训练。

## tests

```bash
bundle exec ruby -Itest -e 'Dir["test/rl_learning/**/*_test.rb"].sort.each { |f| require_relative f }'
# or
rake test
```

Tips:
- `test/rl_learning/grid_world/structure_test.rb` is the "structure guardrail": it verifies the class entrypoint and that all scripts stay under `scripts/`.
- If tests fail before you run training examples, check `torch-rb` environment first (missing shared library or gem version in this repo setup can block test bootstrapping).
