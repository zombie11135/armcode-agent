import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.code_generator import CodeGenerator
from agents.planning_agent import PlanningAgent
from core.context import RuntimeContext


@dataclass
class AgentSystemConfig:
    goal: str
    project_root: str
    use_vdm: bool = False
    use_llm_planner: bool = False
    scene_text: str = ""
    plan_json: str = "runs/current_plan/plan_graph.json"
    plan_md: str = "runs/current_plan/plan_graph.md"
    generated_code: str = "runs/generated_code/current_plan_executor.py"
    execute_robot: bool = False
    run_executor: bool = True
    executor_args: List[str] = field(default_factory=list)
    verbose_events: bool = False

    vlm_base_url: str = "http://10.201.102.163:8001/v1"
    vlm_api_key: str = "wpywpywpy"
    vlm_model: str = "qwen-vl-8b"
    vlm_timeout: int = 120
    fallback_vlm_base_url: str = ""
    fallback_vlm_api_key: str = ""
    fallback_vlm_model: str = ""
    planner_llm_base_url: str = ""
    planner_llm_api_key: str = ""
    planner_llm_model: str = ""

    global_serial: str = "213522251050"
    wrist_serial: str = "243222070053"


@dataclass
class AgentSystemResult:
    success: bool
    plan_json: str
    plan_md: str
    generated_code: str
    executor_returncode: Optional[int] = None
    error: Optional[str] = None


class ArmCodeAgentSystem:
    def __init__(self, config: AgentSystemConfig):
        self.config = config
        self.project_root = Path(config.project_root).resolve()

    def run(self) -> AgentSystemResult:
        try:
            graph = self._planning_stage()
            self._coding_stage(graph)
            returncode = None

            if self.config.run_executor:
                returncode = self._executing_stage()
                if returncode != 0:
                    return AgentSystemResult(
                        success=False,
                        plan_json=self.config.plan_json,
                        plan_md=self.config.plan_md,
                        generated_code=self.config.generated_code,
                        executor_returncode=returncode,
                        error=f"executor failed with return code {returncode}",
                    )

            return AgentSystemResult(
                success=True,
                plan_json=self.config.plan_json,
                plan_md=self.config.plan_md,
                generated_code=self.config.generated_code,
                executor_returncode=returncode,
            )

        except Exception as e:
            return AgentSystemResult(
                success=False,
                plan_json=self.config.plan_json,
                plan_md=self.config.plan_md,
                generated_code=self.config.generated_code,
                error=str(e),
            )

    def _planning_stage(self):
        self._stage("planning", f"goal: {self.config.goal}")

        camera_service = None
        context = RuntimeContext(
            event_callback=lambda event: self._print_event(
                event,
                verbose=self.config.verbose_events,
            )
        )

        try:
            if self.config.use_vdm:
                from services.realsense_dual_camera_service import (
                    CameraConfig,
                    RealSenseDualCameraService,
                )

                camera_service = RealSenseDualCameraService(
                    global_config=CameraConfig(
                        name="global",
                        serial=self.config.global_serial,
                        color_width=1280,
                        color_height=720,
                        color_fps=30,
                        enable_depth=False,
                    ),
                    wrist_config=CameraConfig(
                        name="wrist",
                        serial=self.config.wrist_serial,
                        color_width=1280,
                        color_height=720,
                        color_fps=30,
                        enable_depth=True,
                        depth_width=1280,
                        depth_height=720,
                        depth_fps=30,
                    ),
                    save_dir="runs/current_capture",
                    show_window=True,
                )
                camera_service.start()
                time.sleep(2.0)
                context.camera = camera_service

            if self.config.use_vdm or self.config.use_llm_planner:
                from services.vlm_client import QwenVLClient

                context.vlm = QwenVLClient(
                    base_url=self.config.vlm_base_url,
                    api_key=self.config.vlm_api_key,
                    model_name=self.config.vlm_model,
                    timeout=self.config.vlm_timeout,
                )

                if (
                    self.config.fallback_vlm_base_url
                    and self.config.fallback_vlm_api_key
                    and self.config.fallback_vlm_model
                ):
                    context.fallback_vlm = QwenVLClient(
                        base_url=self.config.fallback_vlm_base_url,
                        api_key=self.config.fallback_vlm_api_key,
                        model_name=self.config.fallback_vlm_model,
                        timeout=self.config.vlm_timeout,
                    )

                if (
                    self.config.planner_llm_base_url
                    and self.config.planner_llm_api_key
                    and self.config.planner_llm_model
                ):
                    context.planner_llm = QwenVLClient(
                        base_url=self.config.planner_llm_base_url,
                        api_key=self.config.planner_llm_api_key,
                        model_name=self.config.planner_llm_model,
                        timeout=self.config.vlm_timeout,
                    )

            planner = PlanningAgent(
                context=context,
                use_llm=self.config.use_llm_planner,
            )
            graph = planner.plan(
                task_goal=self.config.goal,
                scene_text=self.config.scene_text,
                use_vdm=self.config.use_vdm,
                output_path=self.config.plan_json,
            )

            md = graph.to_markdown()
            plan_md = Path(self.config.plan_md)
            plan_md.parent.mkdir(parents=True, exist_ok=True)
            plan_md.write_text(md, encoding="utf-8")

            print("\n=== Plan Graph ===")
            print(md)
            self._done("planning", f"json={self.config.plan_json}, md={self.config.plan_md}")
            return graph

        finally:
            if camera_service is not None:
                camera_service.stop()

    def _coding_stage(self, graph):
        self._stage("coding", "generate executable python from PlanGraph")
        generator = CodeGenerator(project_root=str(self.project_root))
        generator.generate(graph, output_path=self.config.generated_code)
        calls = generator.build_skill_calls(graph)
        print("\n=== Executable Skill Calls ===")
        for index, call in enumerate(calls, start=1):
            print(
                f"- {index}. {call.node_id} {call.skill}: "
                f"{call.args['medicine_query']} -> {call.args['container_name']}"
            )
        self._done("coding", f"script={self.config.generated_code}")

    def _executing_stage(self) -> int:
        mode = "robot" if self.config.execute_robot else "dry-run"
        self._stage("executing", mode)

        cmd = [
            sys.executable,
            self.config.generated_code,
        ]
        if self.config.execute_robot:
            cmd.append("--execute")
        cmd.extend(self.config.executor_args)

        print("\n=== Executor Command ===")
        print(" ".join(cmd))
        returncode = subprocess.run(cmd, cwd=str(self.project_root)).returncode
        self._done("executing", f"returncode={returncode}")
        return int(returncode)

    @staticmethod
    def _print_event(event: Dict[str, Any], verbose: bool = False):
        event_type = event.get("type")
        if not verbose and event_type not in ["warning", "error"]:
            return

        if event_type == "image":
            print(f"[Image] {event.get('content')}: {event.get('image_path')}")
        elif event_type == "text":
            print(f"[Text] {event.get('content')}")
        elif event_type == "warning":
            print(f"[Warning] {event.get('content')}")
        elif event_type == "error":
            print(f"[Error] {event.get('content')}")
        else:
            print(event)

    @staticmethod
    def _stage(name: str, message: str):
        print(f"\n[{name}] {message}")

    @staticmethod
    def _done(name: str, message: str):
        print(f"[{name}] done: {message}")
