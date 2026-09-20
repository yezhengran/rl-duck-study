"""无需创建 GPU 环境，展开一个 mjlab 任务的静态配置。

这个工具回答“参数实际声明在哪里、当前值是什么”。它读取任务注册表中的环境配置和
上游 runner 配置，但不会构建 ``ManagerBasedRlEnv``，因此适合在本地开发机上先检查。

注意：本仓训练使用自写的 ``train_v*_*.py``，不是上游 RSL-RL runner。``algorithm``
一节用于对照任务随附的上游配方；真正生效的自写 PPO 超参数仍以
``train_v3_ppo.py`` 为准。
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pprint import pformat
from typing import Any

import mjlab_microduck.tasks  # noqa: F401  # 导入即注册 MicroDuck 任务
from env import TASK
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

SECTIONS = (
    "overview",
    "observations",
    "actions",
    "commands",
    "rewards",
    "terminations",
    "curriculum",
    "events",
    "algorithm",
)


def callable_name(value: Any) -> str:
    """返回可调用对象的稳定点分名称。"""
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", getattr(value, "__name__", None))
    if module and name:
        return f"{module}.{name}"
    return repr(value)


def public_fields(value: Any, names: tuple[str, ...] | None = None) -> dict[str, Any]:
    """把配置对象压成适合终端阅读的公开字段字典。"""
    raw = value if isinstance(value, Mapping) else vars(value)
    keys = names if names is not None else tuple(k for k in raw if not k.startswith("_"))
    return {key: raw[key] for key in keys if key in raw}


def print_heading(title: str) -> None:
    print(f"\n[{title}]")


def print_terms(terms: Mapping[str, Any], *, extra: tuple[str, ...] = ()) -> None:
    for name, term in terms.items():
        function = callable_name(term.func) if hasattr(term, "func") else type(term).__name__
        bits = [f"{name}: {function}"]
        for field in extra:
            if hasattr(term, field):
                bits.append(f"{field}={getattr(term, field)!r}")
        print("  " + " | ".join(bits))
        params = getattr(term, "params", None)
        if params:
            print("    params=" + pformat(params, width=96, compact=True))


def print_overview(cfg: Any, rl_cfg: Any, task: str) -> None:
    sim_dt = float(cfg.sim.mujoco.timestep)
    control_dt = sim_dt * int(cfg.decimation)
    print_heading("overview")
    print(f"  task={task}")
    print(f"  physics_dt={sim_dt:g}s | decimation={cfg.decimation}")
    print(f"  control_dt={control_dt:g}s | control_hz={1.0 / control_dt:g}")
    print(f"  episode_length={cfg.episode_length_s:g}s")
    print(f"  observation_groups={list(cfg.observations)}")
    print(f"  action_groups={list(cfg.actions)}")
    print(f"  commands={list(cfg.commands)}")
    print(f"  rewards={len(cfg.rewards)} | terminations={len(cfg.terminations)}")
    print(f"  curriculum_terms={len(cfg.curriculum)} | events={len(cfg.events)}")
    print(f"  upstream_runner_steps={rl_cfg.num_steps_per_env}")


def print_observations(cfg: Any) -> None:
    print_heading("observations")
    for group_name, group in cfg.observations.items():
        print(
            f"  <{group_name}> concatenate={group.concatenate_terms} "
            f"corruption={group.enable_corruption}"
        )
        print_terms(group.terms, extra=("scale", "noise"))


def print_actions(cfg: Any) -> None:
    print_heading("actions")
    selected = ("entity_name", "actuator_names", "joint_names", "scale", "use_default_offset")
    for name, action in cfg.actions.items():
        print(f"  {name}: {type(action).__module__}.{type(action).__qualname__}")
        print("    " + pformat(public_fields(action, selected), width=96, compact=True))


def print_commands(cfg: Any) -> None:
    print_heading("commands")
    selected = (
        "resampling_time_range",
        "rel_standing_envs",
        "rel_heading_envs",
        "rel_turn_in_place_envs",
        "heading_command",
        "ranges",
    )
    for name, command in cfg.commands.items():
        fields = public_fields(command, selected)
        if "ranges" in fields and hasattr(fields["ranges"], "__dict__"):
            fields["ranges"] = public_fields(fields["ranges"])
        print(f"  {name}: {type(command).__module__}.{type(command).__qualname__}")
        print("    " + pformat(fields, width=96, compact=True))


def print_algorithm(rl_cfg: Any) -> None:
    print_heading("algorithm (上游任务配方；自写训练器以 train_v3_ppo.py 为准)")
    runner_fields = (
        "num_steps_per_env",
        "max_iterations",
        "save_interval",
        "experiment_name",
        "run_name",
    )
    print("  runner=" + pformat(public_fields(rl_cfg, runner_fields), width=96, compact=True))
    print("  actor=" + pformat(public_fields(rl_cfg.actor), width=96, compact=True))
    print("  critic=" + pformat(public_fields(rl_cfg.critic), width=96, compact=True))
    print("  ppo=" + pformat(public_fields(rl_cfg.algorithm), width=96, compact=True))


def inspect(task: str, sections: list[str]) -> None:
    """打印所选任务的指定配置节。"""
    available = list_tasks()
    if task not in available:
        raise SystemExit(f"未注册任务 {task!r}；先运行 --list")
    cfg = load_env_cfg(task)
    rl_cfg = load_rl_cfg(task)

    for section in sections:
        if section == "overview":
            print_overview(cfg, rl_cfg, task)
        elif section == "observations":
            print_observations(cfg)
        elif section == "actions":
            print_actions(cfg)
        elif section == "commands":
            print_commands(cfg)
        elif section == "algorithm":
            print_algorithm(rl_cfg)
        else:
            print_heading(section)
            extra = ("weight",) if section == "rewards" else ("time_out",)
            print_terms(getattr(cfg, section), extra=extra)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="只列出 MicroDuck 任务")
    parser.add_argument("--task", default=TASK, help=f"任务 ID；默认 {TASK}")
    parser.add_argument(
        "--section",
        nargs="+",
        choices=(*SECTIONS, "all"),
        default=["overview"],
        help="要展开的配置节；可一次给多个，all 表示全部",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        microduck_tasks = [name for name in list_tasks() if "MicroDuck" in name]
        print(f"MicroDuck tasks: {len(microduck_tasks)}")
        print("\n".join(f"  {name}" for name in microduck_tasks))
        return
    sections = list(SECTIONS) if "all" in args.section else args.section
    inspect(args.task, sections)


if __name__ == "__main__":
    main()
