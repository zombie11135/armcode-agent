import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.plan_graph import PlanGraph, PlanNode


@dataclass
class GeneratedSkillCall:
    node_id: str
    label: str
    skill: str
    args: Dict[str, Any]


class CodeGenerator:
    """
    Generate a sequential Python executor from a PlanGraph.

    The generated script is intentionally conservative for a single xArm:
    graph branches are linearized by topological order, and only supported
    executable skill nodes are emitted.
    """

    SUPPORTED_SKILLS = {
        "pick_medicine_and_place_to_container",
    }

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path.cwd()).resolve()

    def generate(
        self,
        graph: PlanGraph,
        output_path: Optional[str] = None,
    ) -> str:
        calls = self.build_skill_calls(graph)
        if not calls:
            raise ValueError("PlanGraph 中没有可生成执行代码的技能节点。")

        script = self._render_script(graph=graph, calls=calls)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script, encoding="utf-8")

        return script

    def build_skill_calls(self, graph: PlanGraph) -> List[GeneratedSkillCall]:
        return self._build_skill_calls(graph)

    def _build_skill_calls(self, graph: PlanGraph) -> List[GeneratedSkillCall]:
        nodes = self._topological_nodes(graph)

        calls = []
        invalid_nodes = []
        for node in nodes:
            if node.skill not in self.SUPPORTED_SKILLS:
                continue

            args = self._normalize_pick_place_args(node)
            validation_error = self._validate_pick_place_call(node, args)
            if validation_error:
                invalid_nodes.append(validation_error)
                continue

            calls.append(
                GeneratedSkillCall(
                    node_id=node.node_id,
                    label=node.label,
                    skill=node.skill,
                    args=args,
                )
            )

        if invalid_nodes:
            details = "\n".join([f"- {item}" for item in invalid_nodes])
            raise ValueError(
                "PlanGraph 中存在无法安全生成抓取代码的节点：\n"
                f"{details}\n"
                "请重新规划：pick_medicine_and_place_to_container 的 medicine_query "
                "必须是药盒上的可读药名/文字，不能是颜色、形状或方位描述。"
            )

        # If a macro root expands into granular children, prefer the explicit
        # pick/place macro calls only once. Current LLM plans already produce
        # one macro node per medicine target, so this mainly protects heuristic
        # or hand-authored plans from duplicate execution.
        seen = set()
        unique_calls = []
        for call in calls:
            key = (
                call.skill,
                call.args.get("medicine_query"),
                call.args.get("container_name"),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_calls.append(call)
        return unique_calls

    def _topological_nodes(self, graph: PlanGraph) -> List[PlanNode]:
        node_by_id = {node.node_id: node for node in graph.nodes}
        order_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
        indegree = {node.node_id: 0 for node in graph.nodes}
        outgoing = {node.node_id: [] for node in graph.nodes}

        for edge in graph.edges:
            if edge.source not in node_by_id or edge.target not in node_by_id:
                continue
            outgoing[edge.source].append(edge.target)
            indegree[edge.target] += 1

        ready = sorted(
            [node_id for node_id, degree in indegree.items() if degree == 0],
            key=lambda node_id: order_index[node_id],
        )
        result = []

        while ready:
            node_id = ready.pop(0)
            result.append(node_by_id[node_id])
            for target in sorted(outgoing[node_id], key=lambda item: order_index[item]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort(key=lambda item: order_index[item])

        if len(result) != len(graph.nodes):
            return list(graph.nodes)
        return result

    def _normalize_pick_place_args(self, node: PlanNode) -> Dict[str, Any]:
        args = dict(node.args or {})
        text = "\n".join([node.label or "", node.instruction or ""])

        medicine_query = (
            args.get("medicine_query")
            or args.get("query")
            or args.get("medicine_name")
            or args.get("medicine")
            or args.get("target")
            or self._infer_medicine_query(text)
        )
        container_name = (
            args.get("container_name")
            or args.get("container")
            or args.get("target_container")
            or self._infer_container_name(text)
        )

        return {
            "medicine_query": str(medicine_query or "目标药盒").strip(),
            "container_name": self._canonical_container_name(container_name),
            "source_args": args,
        }

    @classmethod
    def _validate_pick_place_call(cls, node: PlanNode, args: Dict[str, Any]) -> Optional[str]:
        medicine_query = str(args.get("medicine_query") or "").strip()
        if not medicine_query:
            return f"{node.node_id} {node.label}: 缺少 medicine_query。"

        if cls._looks_like_non_box_object(medicine_query, node):
            return (
                f"{node.node_id} {node.label}: “{medicine_query}” 看起来不是药盒名称，"
                "当前抓取技能只支持药盒/药品包装盒。"
            )

        if cls._looks_like_visual_description(medicine_query):
            return (
                f"{node.node_id} {node.label}: medicine_query=“{medicine_query}” 是颜色/形状/方位描述，"
                "OCR 匹配不稳定。请让 planner 使用药盒上的文字，例如 布洛芬/抗病毒口服液。"
            )

        return None

    @staticmethod
    def _looks_like_visual_description(text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return True

        known_text_tokens = [
            "布洛芬",
            "抗病毒",
            "口服液",
            "蒙脱石",
            "西瓜霜",
            "胶囊",
            "颗粒",
            "片",
            "OTC",
            "IBUPROFEN",
            "999",
        ]
        if any(token.lower() in text.lower() for token in known_text_tokens):
            return False

        visual_tokens = [
            "白色",
            "蓝色",
            "绿色",
            "红色",
            "黄色",
            "黑色",
            "粉色",
            "左边",
            "右边",
            "中间",
            "前方",
            "后方",
            "上方",
            "下方",
            "较大",
            "较小",
            "长方形",
            "矩形",
            "盒子",
            "药盒",
            "包装",
        ]
        visual_hit_count = sum(1 for token in visual_tokens if token in text)
        return visual_hit_count > 0 and len(text) <= 12

    @staticmethod
    def _looks_like_non_box_object(text: str, node: PlanNode) -> bool:
        joined = "\n".join([text, node.label or "", node.instruction or ""])
        non_box_tokens = [
            "圆柱",
            "管状",
            "瓶",
            "饮料",
            "线材",
            "电缆",
            "急停",
        ]
        return any(token in joined for token in non_box_tokens)

    @staticmethod
    def _infer_medicine_query(text: str) -> str:
        known = [
            "布洛芬缓释胶囊",
            "布洛芬",
            "抗病毒口服液",
            "抗病毒",
            "西瓜霜",
            "蒙脱石散",
            "999",
        ]
        for name in known:
            if name in text:
                return name

        patterns = [
            r"将(.+?)(?:药盒|药品)?(?:放入|放到|放置到)",
            r"抓取(.+?)(?:药盒|药品)?(?:并|然后|放)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(" “”。")

        return "目标药盒"

    @staticmethod
    def _infer_container_name(text: str) -> str:
        if any(token in text for token in ["粉色圆盘", "圆盘", "托盘", "盘子"]):
            return "tray"
        if any(token in text for token in ["收纳盒", "容器", "盒子"]):
            return "container"
        if "碗" in text:
            return "bowl"
        return "tray"

    @staticmethod
    def _canonical_container_name(value: Any) -> str:
        text = str(value or "").strip()
        if any(token in text for token in ["粉色圆盘", "圆盘", "托盘", "盘子", "tray", "plate"]):
            return "tray"
        if any(token in text for token in ["收纳盒", "容器", "盒子", "container", "box"]):
            return "container"
        if any(token in text for token in ["碗", "bowl"]):
            return "bowl"
        return text or "tray"

    def _render_script(self, graph: PlanGraph, calls: List[GeneratedSkillCall]) -> str:
        calls_json = json.dumps([call.__dict__ for call in calls], ensure_ascii=False, indent=2)
        goal_json = json.dumps(graph.goal, ensure_ascii=False)
        project_root_json = json.dumps(str(self.project_root), ensure_ascii=False)

        return f'''#!/usr/bin/env python3
# Auto-generated by agents.code_generator.CodeGenerator.
# Goal: {graph.goal}

import argparse
import subprocess
import sys
import time
from pathlib import Path
from pprint import pprint


PROJECT_ROOT = Path({project_root_json})
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.context import RuntimeContext
from services.grounding_dino_detector_service import GroundingDINODetectorService
from services.ocr_service import PaddleOCRService
from services.realsense_dual_camera_service import CameraConfig, RealSenseDualCameraService
from services.anygrasp_service import AnyGraspService
from skill_lib.combound.find_and_grasp_medicine import FindAndGraspMedicineSkill
from skill_lib.combound.place_to_container import PlaceToContainerSkill


GOAL = {goal_json}
PLAN_STEPS = {calls_json}


def print_event(event: dict, verbose: bool = False):
    event_type = event.get("type")
    if not verbose and event_type not in ["warning", "error"]:
        return

    if event_type == "image":
        print(f"[Image] {{event.get('content')}}: {{event.get('image_path')}}")
    elif event_type == "text":
        print(f"[Text] {{event.get('content')}}")
    elif event_type == "warning":
        print(f"[Warning] {{event.get('content')}}")
    elif event_type == "error":
        print(f"[Error] {{event.get('content')}}")
    else:
        print(event)


def check_code(arm, code, label: str = "") -> bool:
    if code != 0:
        print(f"[xArm][ERROR] {{label}} failed, code={{code}}")
        print(f"[xArm] state={{arm.state}}, error={{arm.error_code}}, warn={{arm.warn_code}}")
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


def call_grip_service(
    value: int,
    ros_ws: str,
    service_name: str,
    timeout: float,
    ignore_error: bool,
) -> bool:
    cmd = f"source {{ros_ws}}/devel/setup.bash && rosservice call {{service_name}} {{int(value)}}"
    print(f"[Gripper] {{cmd}}")
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
            print(f"[Gripper][WARN] rosservice returned code={{result.returncode}}")
            return bool(ignore_error)
        return True
    except subprocess.TimeoutExpired:
        print(f"[Gripper][WARN] rosservice timeout: {{service_name}}")
        return bool(ignore_error)
    except Exception as e:
        print(f"[Gripper][WARN] rosservice failed: {{e}}")
        return bool(ignore_error)


def build_context(args):
    camera_service = RealSenseDualCameraService(
        global_config=CameraConfig(
            name="global",
            serial=args.global_serial,
            color_width=1280,
            color_height=720,
            color_fps=30,
            enable_depth=False,
        ),
        wrist_config=CameraConfig(
            name="wrist",
            serial=args.wrist_serial,
            color_width=1280,
            color_height=720,
            color_fps=30,
            enable_depth=True,
            depth_width=1280,
            depth_height=720,
            depth_fps=30,
        ),
        save_dir=args.capture_dir,
        show_window=True,
    )

    detector_service = GroundingDINODetectorService(
        model_name_or_path=args.grounding_dino_model,
        device=args.detector_device,
        threshold=min(args.container_box_threshold, args.grasp_box_threshold),
        text_threshold=min(args.container_text_threshold, args.grasp_text_threshold),
        nms_iou_threshold=0.50,
        local_files_only=True,
        use_fp16=False,
    )

    ocr_service = PaddleOCRService(
        lang="ch",
        device="cpu",
        use_textline_orientation=True,
    )

    anygrasp_service = AnyGraspService(
        grasp_detection_dir=args.anygrasp_dir,
        checkpoint_path=args.anygrasp_ckpt,
        max_gripper_width=0.10,
        gripper_height=0.03,
        top_down_grasp=False,
        debug=False,
    )

    context = RuntimeContext(
        camera=camera_service,
        detector=detector_service,
        ocr=ocr_service,
        anygrasp=anygrasp_service,
        event_callback=lambda event: print_event(event, verbose=args.verbose_events),
    )
    return context, camera_service


def group_steps_by_container(steps):
    groups = []
    group_by_container = {{}}
    for step in steps:
        container_name = step["args"]["container_name"]
        if container_name not in group_by_container:
            group = {{"container_name": container_name, "steps": []}}
            group_by_container[container_name] = group
            groups.append(group)
        group_by_container[container_name]["steps"].append(step)
    return groups


def move_xarm(arm, pose, speed, acc) -> bool:
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
    return code == 0


def execute_cached_place(context, place_plan, args, gripper_open):
    arm = context.require("xarm")
    approach_pose = place_plan["approach_pose_mmrad"]
    place_pose = place_plan["place_pose_mmrad"]
    lift_pose = place_plan["lift_pose_mmrad"]

    if not move_xarm(arm, approach_pose, args.move_speed, args.move_acc):
        return {{"success": False, "error": "移动到缓存 approach pose 失败。"}}

    if not move_xarm(arm, place_pose, args.move_speed, args.move_acc):
        return {{"success": False, "error": "移动到缓存 place pose 失败。"}}

    if gripper_open and not gripper_open():
        return {{"success": False, "error": "放置时打开夹爪失败。"}}

    if not move_xarm(arm, lift_pose, args.move_speed, args.move_acc):
        return {{"success": False, "error": "放置后抬升失败。"}}

    return {{"success": True, "message": "缓存放置位姿执行完成。"}}


def run_plan(args):
    context, camera_service = build_context(args)
    arm = None

    try:
        camera_service.start()
        time.sleep(args.camera_warmup_s)

        if args.execute:
            arm = init_arm(args.xarm_ip)
            context.xarm = arm

        gripper_open = None
        gripper_close = None
        if args.execute and not args.disable_grip:
            gripper_open = lambda: call_grip_service(
                value=args.grip_open_value,
                ros_ws=args.ros_ws,
                service_name=args.grip_service,
                timeout=args.grip_timeout,
                ignore_error=args.ignore_grip_error,
            )
            gripper_close = lambda: call_grip_service(
                value=args.grip_close_value,
                ros_ws=args.ros_ws,
                service_name=args.grip_service,
                timeout=args.grip_timeout,
                ignore_error=args.ignore_grip_error,
            )

        place_skill = PlaceToContainerSkill(context)
        grasp_skill = FindAndGraspMedicineSkill(context)
        results = []
        step_groups = group_steps_by_container(PLAN_STEPS)

        print("\\n=== Generated Plan Executor ===")
        print("goal:", GOAL)
        print("steps:", len(PLAN_STEPS))
        print("container observations:", len(step_groups))

        finished_steps = 0
        for group_index, group in enumerate(step_groups, start=1):
            container_name = group["container_name"]
            group_steps = group["steps"]

            print(f"\\n=== Container Observation {{group_index}}/{{len(step_groups)}} ===")
            print("container_name:", container_name)
            print("reuse_for_steps:", [step["node_id"] for step in group_steps])

            place_result = place_skill.run(
                container_name=container_name,
                camera_name="wrist",
                detection_threshold=args.container_box_threshold,
                text_threshold=args.container_text_threshold,
                detection_index=args.container_detection_index,
                depth_window_px=args.container_depth_window_px,
                xarm_ip=args.xarm_ip,
                handeye_config_path=args.handeye_config,
                place_clearance_mm=args.place_clearance_mm,
                approach_clearance_mm=args.place_approach_clearance_mm,
                lift_after_place_mm=args.place_lift_after_mm,
                refine_observation=not args.no_container_refine,
                refine_only_when_execute=not args.execute,
                refine_observe_z_mm=args.container_refine_observe_z_mm,
                refine_wait_s=args.container_refine_wait_s,
                execute=False,
                move_speed=args.move_speed,
                move_acc=args.move_acc,
                gripper_open=None,
                fallback_intrinsics={{
                    "fx": 909.59,
                    "fy": 909.80,
                    "cx": 658.97,
                    "cy": 355.42,
                }},
            )

            if not place_result.success:
                print("\\n=== Container Observation Failed ===")
                print("container_name:", container_name)
                print("error:", place_result.error)
                results.append({{
                    "container_name": container_name,
                    "success": False,
                    "error": place_result.error,
                }})
                return False, results

            place_plan = place_result.data
            print("cached_place_pose_mmrad:")
            pprint(place_plan.get("place_pose_mmrad"))

            for step in group_steps:
                finished_steps += 1
                call_args = step["args"]
                medicine_query = call_args["medicine_query"]

                print(f"\\n=== Step {{finished_steps}}/{{len(PLAN_STEPS)}}: {{step['node_id']}} {{step['label']}} ===")
                print("medicine_query:", medicine_query)
                print("container_name:", container_name)

                if args.execute and not args.no_pre_grasp_pose:
                    print("move_pre_grasp_pose:")
                    pprint(args.pre_grasp_pose)
                    if not move_xarm(arm, args.pre_grasp_pose, args.move_speed, args.move_acc):
                        results.append({{
                            "node_id": step["node_id"],
                            "success": False,
                            "error": "移动到抓取前观察位姿失败。",
                        }})
                        return False, results

                grasp_result = grasp_skill.run(
                    query=medicine_query,
                    camera_name="wrist",
                    min_score=args.grasp_min_score,
                    detection_threshold=args.grasp_box_threshold,
                    text_threshold=args.grasp_text_threshold,
                    use_sam_for_selection=False,
                    use_mask_for_grasp=False,
                    plan_grasp=True,
                    execute=args.execute,
                    top_k=args.grasp_top_k,
                    grasp_selection_mode=args.grasp_selection_mode,
                    random_select_top_k=args.grasp_random_select_top_k,
                    bbox_filter_inner_margin_ratio=args.bbox_filter_inner_margin_ratio,
                    infer_grasps_on_full_cloud=True,
                    filter_grasps_by_bbox=True,
                    visualize_selected_grasp=args.visualize_selected_grasp,
                    visualize_bbox_filtered_grasps=args.visualize_bbox_filtered_grasps,
                    move_speed=args.move_speed,
                    move_acc=args.move_acc,
                    gripper_open=gripper_open,
                    gripper_close=gripper_close,
                    gripper_close_wait_s=args.gripper_close_wait_s,
                    fallback_intrinsics={{
                        "fx": 909.59,
                        "fy": 909.80,
                        "cx": 658.97,
                        "cy": 355.42,
                    }},
                    xarm_ip=args.xarm_ip,
                    handeye_config_path=args.handeye_config,
                )

                if not grasp_result.success:
                    print("\\n=== Step Failed ===")
                    print("node_id:", step["node_id"])
                    print("error:", grasp_result.error)
                    results.append({{
                        "node_id": step["node_id"],
                        "success": False,
                        "error": grasp_result.error,
                    }})
                    return False, results

                place_execution = None
                if args.execute:
                    place_execution = execute_cached_place(
                        context=context,
                        place_plan=place_plan,
                        args=args,
                        gripper_open=gripper_open,
                    )
                    if not place_execution.get("success", False):
                        print("\\n=== Cached Place Failed ===")
                        print("node_id:", step["node_id"])
                        print("error:", place_execution.get("error"))
                        results.append({{
                            "node_id": step["node_id"],
                            "success": False,
                            "error": place_execution.get("error"),
                        }})
                        return False, results

                grasp_plan = (grasp_result.data or {{}}).get("grasp_plan", {{}})

                print("\\n=== Step Result ===")
                print("success:", grasp_result.success)
                print("grasp_pose_mmrad:")
                pprint(grasp_plan.get("best_grasp_xarm_mmrad"))
                print("grasp_choose_reason:", grasp_plan.get("choose_reason"))
                print("place_execution:", place_execution)
                results.append({{
                    "node_id": step["node_id"],
                    "success": True,
                    "container_reused": True,
                }})

        return True, results

    finally:
        camera_service.stop()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Generated executor for PlanGraph.")
    parser.add_argument("--execute", action="store_true")

    parser.add_argument("--grounding_dino_model", default="/home/wpy/models/grounding-dino-tiny")
    parser.add_argument("--detector_device", default="cuda")
    parser.add_argument(
        "--anygrasp_dir",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection",
    )
    parser.add_argument(
        "--anygrasp_ckpt",
        default="/home/wpy/pythonproject/thirdparty/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar",
    )

    parser.add_argument("--global_serial", default="213522251050")
    parser.add_argument("--wrist_serial", default="243222070053")
    parser.add_argument("--capture_dir", default="runs/current_capture")
    parser.add_argument("--camera_warmup_s", type=float, default=2.0)

    parser.add_argument("--xarm_ip", default="192.168.1.237")
    parser.add_argument(
        "--handeye_config",
        default="/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
    )
    parser.add_argument(
        "--pre_grasp_pose",
        type=float,
        nargs=6,
        default=[350, 0, 450, 3.14, 0.0, -1.57],
        metavar=("x", "y", "z", "rx", "ry", "rz"),
    )
    parser.add_argument("--no_pre_grasp_pose", action="store_true")
    parser.add_argument("--move_speed", type=float, default=50)
    parser.add_argument("--move_acc", type=float, default=500)

    parser.add_argument("--container_box_threshold", type=float, default=0.25)
    parser.add_argument("--container_text_threshold", type=float, default=0.20)
    parser.add_argument("--container_detection_index", type=int, default=0)
    parser.add_argument("--container_depth_window_px", type=int, default=25)
    parser.add_argument("--no_container_refine", action="store_true")
    parser.add_argument("--container_refine_observe_z_mm", type=float, default=400.0)
    parser.add_argument("--container_refine_wait_s", type=float, default=0.8)
    parser.add_argument("--place_clearance_mm", type=float, default=250.0)
    parser.add_argument("--place_approach_clearance_mm", type=float, default=120.0)
    parser.add_argument("--place_lift_after_mm", type=float, default=120.0)

    parser.add_argument("--grasp_box_threshold", type=float, default=0.22)
    parser.add_argument("--grasp_text_threshold", type=float, default=0.18)
    parser.add_argument("--grasp_min_score", type=float, default=0.50)
    parser.add_argument("--grasp_top_k", type=int, default=50)
    parser.add_argument(
        "--grasp_selection_mode",
        default="random_top_k_good",
        choices=["best_good", "random_good", "random_top_k_good"],
    )
    parser.add_argument("--grasp_random_select_top_k", type=int, default=5)
    parser.add_argument("--bbox_filter_inner_margin_ratio", type=float, default=0.15)
    parser.add_argument("--visualize_selected_grasp", action="store_true")
    parser.add_argument("--visualize_bbox_filtered_grasps", action="store_true")

    parser.add_argument("--disable_grip", action="store_true")
    parser.add_argument("--ros_ws", default="/home/wpy/gt200_ws")
    parser.add_argument("--grip_service", default="/grip")
    parser.add_argument("--grip_timeout", type=float, default=5.0)
    parser.add_argument("--grip_open_value", type=int, default=1)
    parser.add_argument("--grip_close_value", type=int, default=0)
    parser.add_argument("--gripper_close_wait_s", type=float, default=1.5)
    parser.add_argument("--ignore_grip_error", action="store_true", default=True)
    parser.add_argument("--verbose_events", action="store_true")
    args = parser.parse_args()

    success, results = run_plan(args)
    print("\\n=== Plan Execution Summary ===")
    pprint(results)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''
