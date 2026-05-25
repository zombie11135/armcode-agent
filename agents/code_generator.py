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
        for node in nodes:
            if node.skill not in self.SUPPORTED_SKILLS:
                continue

            calls.append(
                GeneratedSkillCall(
                    node_id=node.node_id,
                    label=node.label,
                    skill=node.skill,
                    args=self._normalize_pick_place_args(node),
                )
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
from skill_lib.combound.pick_medicine_and_place_to_container import (
    PickMedicineAndPlaceToContainerSkill,
)


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

        skill = PickMedicineAndPlaceToContainerSkill(context)
        results = []

        print("\\n=== Generated Plan Executor ===")
        print("goal:", GOAL)
        print("steps:", len(PLAN_STEPS))

        for index, step in enumerate(PLAN_STEPS, start=1):
            call_args = step["args"]
            medicine_query = call_args["medicine_query"]
            container_name = call_args["container_name"]

            print(f"\\n=== Step {{index}}/{{len(PLAN_STEPS)}}: {{step['node_id']}} {{step['label']}} ===")
            print("medicine_query:", medicine_query)
            print("container_name:", container_name)

            result = skill.run(
                medicine_query=medicine_query,
                container_name=container_name,
                execute=args.execute,
                xarm_ip=args.xarm_ip,
                handeye_config_path=args.handeye_config,
                pre_grasp_pose_mmrad=None if args.no_pre_grasp_pose else args.pre_grasp_pose,
                move_speed=args.move_speed,
                move_acc=args.move_acc,
                gripper_open=gripper_open,
                gripper_close=gripper_close,
                container_detection_threshold=args.container_box_threshold,
                container_text_threshold=args.container_text_threshold,
                container_detection_index=args.container_detection_index,
                container_depth_window_px=args.container_depth_window_px,
                container_refine_observation=not args.no_container_refine,
                container_refine_observe_z_mm=args.container_refine_observe_z_mm,
                container_refine_wait_s=args.container_refine_wait_s,
                place_clearance_mm=args.place_clearance_mm,
                grasp_detection_threshold=args.grasp_box_threshold,
                grasp_text_threshold=args.grasp_text_threshold,
                grasp_min_score=args.grasp_min_score,
                grasp_top_k=args.grasp_top_k,
                grasp_selection_mode=args.grasp_selection_mode,
                grasp_random_select_top_k=args.grasp_random_select_top_k,
                bbox_filter_inner_margin_ratio=args.bbox_filter_inner_margin_ratio,
                visualize_selected_grasp=args.visualize_selected_grasp,
                visualize_bbox_filtered_grasps=args.visualize_bbox_filtered_grasps,
                fallback_intrinsics={{
                    "fx": 909.59,
                    "fy": 909.80,
                    "cx": 658.97,
                    "cy": 355.42,
                }},
            )

            if not result.success:
                print("\\n=== Step Failed ===")
                print("node_id:", step["node_id"])
                print("error:", result.error)
                print("stage:", result.data.get("stage") if isinstance(result.data, dict) else None)
                results.append({{"node_id": step["node_id"], "success": False, "error": result.error}})
                return False, results

            data = result.data
            grasp_plan = (data.get("grasp_result") or {{}}).get("grasp_plan", {{}})
            place_plan = data.get("place_plan") or {{}}

            print("\\n=== Step Result ===")
            print("success:", result.success)
            print("cached_place_pose_mmrad:")
            pprint(place_plan.get("place_pose_mmrad"))
            print("grasp_pose_mmrad:")
            pprint(grasp_plan.get("best_grasp_xarm_mmrad"))
            print("grasp_choose_reason:", grasp_plan.get("choose_reason"))
            print("place_execution:", data.get("place_execution"))
            results.append({{"node_id": step["node_id"], "success": True}})

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
