import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent_system import AgentSystemConfig, ArmCodeAgentSystem


HOME_POSE_MMRAD = [345.0, 0.0, 450.0, 3.14, 0.0, -1.57]


def check_code(arm, code, label: str = "") -> bool:
    if code != 0:
        print(f"[xArm][ERROR] {label} failed, code={code}")
        print(f"[xArm] state={arm.state}, error={arm.error_code}, warn={arm.warn_code}")
        return False
    return True


def init_arm(ip: str):
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=True)
    time.sleep(0.5)

    print("[xArm] connected:", arm.connected)
    print("[xArm] state:", arm.state)
    print("[xArm] error:", arm.error_code)
    print("[xArm] warn:", arm.warn_code)

    arm.clean_warn()
    arm.clean_error()
    check_code(arm, arm.motion_enable(enable=True), "motion_enable")
    check_code(arm, arm.set_mode(0), "set_mode")
    check_code(arm, arm.set_state(0), "set_state")
    time.sleep(0.5)
    return arm


def move_xarm_pose(arm, pose, speed: float, acc: float) -> bool:
    x, y, z, rx, ry, rz = [float(v) for v in pose]
    code = arm.set_position(
        x=x,
        y=y,
        z=z,
        roll=rx,
        pitch=ry,
        yaw=rz,
        speed=speed,
        mvacc=acc,
        wait=True,
    )
    return check_code(arm, code, "set_position")


def call_grip_service(
    value: int,
    ros_ws: str,
    service_name: str,
    timeout: float,
    ignore_error: bool,
) -> bool:
    cmd = f"source {ros_ws}/devel/setup.bash && rosservice call {service_name} {int(value)}"
    print(f"[Gripper] {cmd}")

    try:
        result = subprocess.run(
            ["bash", "-lc", cmd],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        if result.stdout.strip():
            print("[Gripper] output:")
            print(result.stdout)
        if result.returncode != 0:
            print(f"[Gripper][WARN] rosservice returned code={result.returncode}")
            return bool(ignore_error)
        return True
    except subprocess.TimeoutExpired:
        print(f"[Gripper][WARN] rosservice timeout: {service_name}")
        return bool(ignore_error)
    except Exception as e:
        print(f"[Gripper][WARN] rosservice failed: {e}")
        return bool(ignore_error)


def print_help():
    print(
        """
可输入自然语言任务，例如：
  将桌面上所有的药品盒子整理到粉色托盘中
  把布洛芬放到托盘里

内置命令：
  /help, help, 帮助              显示帮助
  /quit, quit, exit, 退出        退出系统
  /home, home, 回到初始位置       移动到初始位姿 [345,0,450,3.14,0,-1.57]
  /open, 打开夹爪                rosservice call /grip 1
  /close, 关闭夹爪               rosservice call /grip 0

安全提示：
  未加 --execute_robot 时，内置运动/夹爪命令只打印，不会真的执行。
  普通任务会走 planning -> coding -> executing；若未加 --execute_robot，生成脚本也只干跑。
"""
    )


def normalize_command(text: str) -> str:
    return " ".join(str(text or "").strip().split()).lower()


def is_quit_command(command: str) -> bool:
    return command in {"/quit", "quit", "exit", "q", "退出", "结束"}


def is_help_command(command: str) -> bool:
    return command in {"/help", "help", "?", "帮助"}


def is_home_command(command: str) -> bool:
    return command in {"/home", "home", "init", "initial", "回到初始位置", "初始位置", "回初始位"}


def is_open_gripper_command(command: str) -> bool:
    return command in {"/open", "open", "打开夹爪", "夹爪打开"}


def is_close_gripper_command(command: str) -> bool:
    return command in {"/close", "close", "关闭夹爪", "夹爪关闭"}


def run_builtin_command(command: str, args) -> bool:
    if is_help_command(command):
        print_help()
        return True

    if is_home_command(command):
        print(f"[builtin] home pose: {HOME_POSE_MMRAD}")
        if not args.execute_robot:
            print("[builtin] dry-run: 未加 --execute_robot，不执行机械臂移动。")
            return True
        arm = init_arm(args.xarm_ip)
        try:
            ok = move_xarm_pose(arm, HOME_POSE_MMRAD, args.move_speed, args.move_acc)
            print("[builtin] home result:", ok)
        finally:
            try:
                arm.disconnect()
            except Exception:
                pass
        return True

    if is_open_gripper_command(command):
        print(f"[builtin] open gripper value={args.grip_open_value}")
        if not args.execute_robot:
            print("[builtin] dry-run: 未加 --execute_robot，不执行夹爪服务。")
            return True
        call_grip_service(
            value=args.grip_open_value,
            ros_ws=args.ros_ws,
            service_name=args.grip_service,
            timeout=args.grip_timeout,
            ignore_error=args.ignore_grip_error,
        )
        return True

    if is_close_gripper_command(command):
        print(f"[builtin] close gripper value={args.grip_close_value}")
        if not args.execute_robot:
            print("[builtin] dry-run: 未加 --execute_robot，不执行夹爪服务。")
            return True
        call_grip_service(
            value=args.grip_close_value,
            ros_ws=args.ros_ws,
            service_name=args.grip_service,
            timeout=args.grip_timeout,
            ignore_error=args.ignore_grip_error,
        )
        return True

    return False


def build_config(args, goal: str, executor_args):
    return AgentSystemConfig(
        goal=goal,
        project_root=str(PROJECT_ROOT),
        use_vdm=args.use_vdm,
        use_llm_planner=args.use_llm_planner,
        scene_text=args.scene_text,
        plan_json=args.plan_json,
        plan_md=args.plan_md,
        generated_code=args.generated_code,
        execute_robot=args.execute_robot,
        run_executor=not args.no_run_executor,
        executor_args=executor_args,
        verbose_events=args.verbose_events,
        vlm_base_url=args.vlm_base_url,
        vlm_api_key=args.vlm_api_key,
        vlm_model=args.vlm_model,
        vlm_timeout=args.vlm_timeout,
        fallback_vlm_base_url=args.fallback_vlm_base_url,
        fallback_vlm_api_key=args.fallback_vlm_api_key,
        fallback_vlm_model=args.fallback_vlm_model,
        planner_llm_base_url=args.planner_llm_base_url,
        planner_llm_api_key=args.planner_llm_api_key,
        planner_llm_model=args.planner_llm_model,
        global_serial=args.global_serial,
        wrist_serial=args.wrist_serial,
    )


def run_goal(args, goal: str, executor_args) -> bool:
    config = build_config(args, goal=goal, executor_args=executor_args)
    result = ArmCodeAgentSystem(config).run()

    print("\n=== Agent System Summary ===")
    print("success:", result.success)
    print("plan_json:", result.plan_json)
    print("plan_md:", result.plan_md)
    print("generated_code:", result.generated_code)
    print("executor_returncode:", result.executor_returncode)
    print("error:", result.error)
    return bool(result.success)


def interactive_loop(args, executor_args):
    print("\nArmCode Agent CLI")
    print("输入 /help 查看命令；输入 /quit 退出。")
    print("当前执行模式:", "robot" if args.execute_robot else "dry-run")

    while True:
        try:
            raw = input("\narmcode> ")
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            return 0

        command = normalize_command(raw)
        if not command:
            continue
        if is_quit_command(command):
            print("退出。")
            return 0
        if run_builtin_command(command, args):
            continue

        ok = run_goal(args, goal=raw.strip(), executor_args=executor_args)
        if not ok:
            print("[agent] 任务失败。你可以调整场景或输入更明确的任务后重试。")


def main():
    parser = argparse.ArgumentParser(
        description="Simple CLI frontend: user command -> planning -> coding -> executing."
    )
    parser.add_argument(
        "--goal",
        default="",
        help="用户任务，例如：将桌面上所有的药品盒子整理到粉色托盘中",
    )
    parser.add_argument("--scene_text", default="")
    parser.add_argument("--use_vdm", action="store_true")
    parser.add_argument("--use_llm_planner", action="store_true")
    parser.add_argument("--plan_json", default="runs/current_plan/plan_graph.json")
    parser.add_argument("--plan_md", default="runs/current_plan/plan_graph.md")
    parser.add_argument("--generated_code", default="runs/generated_code/current_plan_executor.py")
    parser.add_argument("--no_run_executor", action="store_true")
    parser.add_argument("--execute_robot", action="store_true")
    parser.add_argument("--verbose_events", action="store_true")

    parser.add_argument(
        "--vlm_base_url",
        default=os.getenv("PLANNER_LLM_BASE_URL", "http://10.201.102.163:8001/v1"),
    )
    parser.add_argument(
        "--vlm_api_key",
        default=os.getenv("PLANNER_LLM_API_KEY", "wpywpywpy"),
    )
    parser.add_argument(
        "--vlm_model",
        default=os.getenv("PLANNER_LLM_MODEL", "qwen-vl-8b"),
    )
    parser.add_argument("--vlm_timeout", type=int, default=120)
    parser.add_argument(
        "--fallback_vlm_base_url",
        default=os.getenv("PLANNER_VDM_FALLBACK_BASE_URL", os.getenv("PLANNER_API_BASE_URL", "")),
    )
    parser.add_argument(
        "--fallback_vlm_api_key",
        default=os.getenv("PLANNER_VDM_FALLBACK_API_KEY", os.getenv("PLANNER_API_KEY", "")),
    )
    parser.add_argument(
        "--fallback_vlm_model",
        default=os.getenv("PLANNER_VDM_FALLBACK_MODEL", os.getenv("PLANNER_API_MODEL", "")),
    )
    parser.add_argument(
        "--planner_llm_base_url",
        default=os.getenv("PLANNER_LLM_FALLBACK_BASE_URL", os.getenv("PLANNER_API_BASE_URL", "")),
    )
    parser.add_argument(
        "--planner_llm_api_key",
        default=os.getenv("PLANNER_LLM_FALLBACK_API_KEY", os.getenv("PLANNER_API_KEY", "")),
    )
    parser.add_argument(
        "--planner_llm_model",
        default=os.getenv("PLANNER_LLM_FALLBACK_MODEL", os.getenv("PLANNER_API_MODEL", "")),
    )

    parser.add_argument("--global_serial", default="213522251050")
    parser.add_argument("--wrist_serial", default="243222070053")
    parser.add_argument("--xarm_ip", default="192.168.1.237")
    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)
    parser.add_argument("--ros_ws", default="/home/wpy/gt200_ws")
    parser.add_argument("--grip_service", default="/grip")
    parser.add_argument("--grip_timeout", type=float, default=5.0)
    parser.add_argument("--grip_open_value", type=int, default=1)
    parser.add_argument("--grip_close_value", type=int, default=0)
    parser.add_argument("--ignore_grip_error", action="store_true", default=True)

    parser.add_argument(
        "executor_args",
        nargs=argparse.REMAINDER,
        help="传给生成脚本的额外参数。用 -- 分隔，例如：-- --visualize_selected_grasp",
    )
    args = parser.parse_args()

    executor_args = list(args.executor_args or [])
    if executor_args and executor_args[0] == "--":
        executor_args = executor_args[1:]

    if not args.goal:
        raise SystemExit(interactive_loop(args, executor_args))

    ok = run_goal(args, goal=args.goal, executor_args=executor_args)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
