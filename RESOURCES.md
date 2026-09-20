# RL Duck 仿真与强化学习实验资源

## Knowledge

- [本仓库讲义：强化学习入门（小鸭子版）](docs/强化学习入门-小鸭子版.md)
  与本仓代码一一对应的中文主教材。用于：先形成 agent—environment、观测—动作—奖励和 REINFORCE→A2C→PPO 的整体地图。
- [本仓库实际训练代码](code/train_v3_ppo.py)
  当前实验的第一事实来源。用于：核对这个自写 PPO 真正使用的超参数、日志字段和 checkpoint 内容。
- [Microduck RL：本仓锁定的上游提交](https://github.com/pollen-robotics/microduck_rl/tree/5946fd9cdbc58956424420153e51975af3b30d77)
  机器人、任务、奖励、课程表与域随机化的来源。用于：设计或解释环境侧改动；必须优先看锁定提交，不把 develop 分支当成本地版本。
- [Microduck RL 的工程经验清单](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md)
  上游团队记录的奖励设计与训练诊断经验。用于：形成实验假设；内容会随 develop 变化，不能代替锁定版本源码。
- [Mjlab：任务注册与 RSL-RL 训练文档](https://github.com/mujocolab/mjlab/blob/main/docs/source/training/rsl_rl.rst)
  解释 task ID 如何绑定环境配置和训练配置。用于：理解 `load_env_cfg`、`load_rl_cfg` 与上游 runner；版本变化时以本地安装源码为准。
- [MuJoCo MJX 文档](https://mujoco.readthedocs.io/en/latest/mjx.html)
  MuJoCo 的加速器与批量仿真说明。用于：理解为什么并行环境数影响吞吐，以及批量物理世界不是“同一个场景里的很多机器人”。
- [论文：Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  PPO 原始论文。用于：理解 clipped surrogate 与多轮 minibatch 数据复用，不用于猜测本仓具体实现细节。
- [论文：High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
  GAE 原始论文。用于：理解 `gamma`、`lambda` 与偏差—方差权衡。
- [PyTorch Lightning DataHooks 文档](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.DataHooks.html)
  说明 dataloader 何时重建。用于：理解本仓为何用 `reload_dataloaders_every_n_epochs=1` 保持 on-policy 数据新鲜。
- [uv：锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)
  官方依赖复现说明。用于：服务器上用 `uv sync --frozen`，避免运行时悄悄改锁文件。
- [W&B：离线运行与稍后同步](https://docs.wandb.ai/fr/models/ref/cli/wandb-offline)
  官方离线日志说明。用于：理解本仓 `wandb_mode="offline"` 产生什么，以及如何 `wandb sync`。
- [Git 官方工作流说明](https://git-scm.com/docs/gitworkflows)
  区分 fetch、push 与 pull 的官方资料。用于：建立本地改代码、服务器只拉取和运行的安全习惯。

## Wisdom (Communities)

- [pollen-robotics/microduck_rl Issues](https://github.com/pollen-robotics/microduck_rl/issues)
  适合检索具体任务、sim2real、奖励或上游版本问题；提问前附 task、commit、命令和最小证据。
- [mujocolab/mjlab Issues](https://github.com/mujocolab/mjlab/issues)
  适合确认管理器、MuJoCo Warp、渲染、域随机化等平台行为；先用本地锁定版本复现。

## Gaps

- 用户希望迁移到的第一个目标领域尚未明确；确定目标后需要补充该领域的权威环境建模与评估资料。
- 当前自写训练器未把每个 reward term、episode length、success rate 写进 `metrics.jsonl`；在补齐诊断日志前，曲线只能回答有限问题。
