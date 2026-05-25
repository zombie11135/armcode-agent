import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.plan_graph import PlanGraph, PlanNode


PLANNER_SYSTEM_PROMPT = """
你是机械臂任务规划 agent。你需要参考 InstructFlow 的思想：
- 将长程任务分解成层级 instruction graph；
- 每个节点必须是语义明确的子目标；
- 可执行节点必须绑定已有技能；
- 失败约束会被注入到图中，用于后续重新规划或局部修复。

请只输出 JSON，不要输出 Markdown。JSON schema:
{
  "goal": "...",
  "root_id": "n0",
  "constraints": ["..."],
  "nodes": [
    {
      "node_id": "n0",
      "label": "...",
      "node_type": "macro|observe|locate|pick|place|verify",
      "instruction": "...",
      "skill": null 或技能名,
      "args": {},
      "preconditions": [],
      "effects": [],
      "constraints": [],
      "parent": null 或父节点 id,
      "metadata": {}
    }
  ],
  "edges": [
    {"source": "n1", "target": "n2", "edge_type": "control", "condition": null}
  ]
}

当前可用的关键技能：
- describe_scene: 使用全局和腕部相机生成场景描述。
- place_to_container: 空手定位容器并生成放置位姿，可选执行放置。
- find_and_grasp_medicine: 根据药名识别药盒并抓取。
- inspect_medicine_candidates: 粗检测药盒类似物，必要时移动到候选上方近距离 OCR，输出 confirmed_targets。
- pick_medicine_and_place_to_container: 宏技能，空手定位容器、抓药盒、使用缓存容器位姿放置。

技能绑定规则：
- pick_medicine_and_place_to_container 只能用于“药盒/药品包装盒”，不要用于圆柱形管状物、瓶子、饮料、线材等非药盒物体。
- pick_medicine_and_place_to_container 的 args 必须包含 medicine_query 和 container_name。
- medicine_query 必须优先使用包装上可读文字或药品名称，例如“布洛芬”“抗病毒口服液”“蒙脱石散”。
- 禁止把“白色蓝色药盒”“绿色白色药盒”“左边药盒”“圆柱形管状物”这类颜色、形状或方位描述当作 medicine_query。
- 如果只能看到颜色/形状，无法读到药名或关键文字，则该物体不要绑定抓取技能；应生成 observe/verify 节点并在 constraints 中写明“需要更近距离 OCR 或人工确认药名后再抓取”。
- 当任务要求整理所有药盒，但 VDM 只能给出“白色蓝色药盒/绿色白色药盒”等视觉描述时，应先使用 inspect_medicine_candidates 进行近距离 OCR 确认，再让后续执行只抓取 confirmed_targets。
- 对“整理所有药品盒子”任务，只规划已确认是药盒且有可检索文字的目标；不确定的物体应保留为待确认目标，不要直接执行抓取。
"""


class PlanningAgent:
    """
    Planner that creates an InstructFlow-style instruction graph.

    The default planner is intentionally deterministic and conservative, so it
    can be used before an LLM planner is wired in. If context.vlm is available,
    use_llm=True can ask the VLM/LLM service to produce the same graph schema.
    """

    def __init__(self, context=None, use_llm: bool = False):
        self.context = context
        self.use_llm = use_llm

    def plan(
        self,
        task_goal: str,
        scene_text: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        use_vdm: bool = False,
        output_path: Optional[str] = None,
    ) -> PlanGraph:
        scene_text = scene_text or ""
        constraints = constraints or []

        if use_vdm and not scene_text:
            scene_text = self._run_vdm()

        graph = None

        if self.use_llm and self.context is not None and getattr(self.context, "vlm", None):
            try:
                graph = self._plan_with_llm(
                    task_goal=task_goal,
                    scene_text=scene_text,
                    constraints=constraints,
                )
            except Exception as e:
                graph = self._plan_heuristic(
                    task_goal=task_goal,
                    scene_text=scene_text,
                    constraints=[
                        *constraints,
                        f"LLM planner failed, fallback to heuristic planner: {e}",
                    ],
                )
        else:
            graph = self._plan_heuristic(
                task_goal=task_goal,
                scene_text=scene_text,
                constraints=constraints,
            )

        if output_path:
            graph.save_json(output_path)

        return graph

    def replan_with_constraints(
        self,
        previous_graph: PlanGraph,
        new_constraints: List[str],
        failed_node_id: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> PlanGraph:
        graph = PlanGraph.from_dict(previous_graph.to_dict())
        graph.apply_constraints(new_constraints, target_node_id=failed_node_id)
        graph.metadata["replanned_at"] = datetime.now().isoformat(timespec="seconds")
        graph.metadata["failed_node_id"] = failed_node_id

        # For now constraints are injected directly. This gives the code
        # generator and future LLM planner a stable symbolic repair surface.
        if output_path:
            graph.save_json(output_path)

        return graph

    def _run_vdm(self) -> str:
        if self.context is None:
            raise RuntimeError("PlanningAgent missing RuntimeContext for VDM.")

        from skill_lib.perception.vdm import VDMSkill

        result = VDMSkill(self.context).run(use_wrist=True)
        if not result.success:
            raise RuntimeError(result.error)

        return result.data.get("scene_text", "")

    def _plan_with_llm(
        self,
        task_goal: str,
        scene_text: str,
        constraints: List[str],
    ) -> PlanGraph:
        vlm = (
            getattr(self.context, "planner_llm", None)
            or getattr(self.context, "fallback_vlm", None)
            or self.context.require("vlm")
        )
        prompt = self._build_llm_prompt(task_goal, scene_text, constraints)
        text = vlm.chat(text=prompt, max_tokens=2048, temperature=0.1)
        data = self._extract_json(text)
        graph = PlanGraph.from_dict(data)
        graph.scene_text = scene_text
        graph.metadata.setdefault("planner", "llm")
        graph.metadata.setdefault("source", "PlanningAgent._plan_with_llm")
        return graph

    def _plan_heuristic(
        self,
        task_goal: str,
        scene_text: str,
        constraints: List[str],
    ) -> PlanGraph:
        medicine_query = self._extract_medicine_query(task_goal, scene_text)
        container_name = self._extract_container_name(task_goal, scene_text)

        graph = PlanGraph(
            goal=task_goal,
            scene_text=scene_text,
            constraints=list(constraints),
            metadata={
                "planner": "heuristic",
                "style": "instructflow_instruction_graph",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "medicine_query": medicine_query,
                "container_name": container_name,
            },
        )

        root = graph.add_node(
            PlanNode(
                node_id="n0",
                label="完成药盒抓取并放置到容器",
                node_type="macro",
                instruction=(
                    f"将目标药品“{medicine_query}”放置到“{container_name}”中。"
                    "采用空手先定位容器、再抓取、最后复用缓存容器位姿放置的流程。"
                ),
                skill="pick_medicine_and_place_to_container",
                args={
                    "medicine_query": medicine_query,
                    "container_name": container_name,
                    "execute": True,
                },
                preconditions=["机器人处于可运动状态", "夹爪可控", "相机服务可用"],
                effects=[f"{medicine_query} 位于 {container_name} 中"],
                constraints=[
                    "先在空手状态下定位容器，避免抓取后遮挡腕部相机视野。",
                    "放置阶段必须使用空手阶段缓存的容器位姿，不在抓取后重新依赖视觉定位容器。",
                ],
            )
        )
        graph.root_id = root.node_id

        graph.add_node(
            PlanNode(
                node_id="n1",
                label="多视角观察场景",
                node_type="observe",
                instruction="使用全局相机和腕部相机生成整体场景描述，确认桌面物体和可用容器。",
                skill="describe_scene",
                args={"use_wrist": True},
                parent=root.node_id,
                effects=["获得 scene_text"],
            )
        )

        graph.add_node(
            PlanNode(
                node_id="n2",
                label="空手定位放置容器",
                node_type="locate",
                instruction=(
                    f"在空手状态下检测 {container_name}，进行必要的二次观察，"
                    "并缓存 approach/place/lift 放置位姿。"
                ),
                skill="place_to_container",
                args={
                    "container_name": container_name,
                    "execute": False,
                    "refine_observation": True,
                    "refine_only_when_execute": False,
                    "place_clearance_mm": 250.0,
                },
                parent=root.node_id,
                preconditions=["夹爪没有遮挡容器视野"],
                effects=["获得 cached_place_plan"],
                constraints=[
                    "容器可能只被局部看到，允许移动到 z=400mm 的二次观察位重新定位。",
                ],
            )
        )

        graph.add_node(
            PlanNode(
                node_id="n3",
                label="抓取目标药盒",
                node_type="pick",
                instruction=f"识别写有“{medicine_query}”的药盒并执行抓取，抓取后抬升。",
                skill="find_and_grasp_medicine",
                args={
                    "query": medicine_query,
                    "plan_grasp": True,
                    "execute": True,
                    "use_mask_for_grasp": False,
                    "infer_grasps_on_full_cloud": True,
                    "filter_grasps_by_bbox": True,
                    "grasp_selection_mode": "random_top_k_good",
                    "random_select_top_k": 5,
                },
                parent=root.node_id,
                preconditions=["cached_place_plan 已生成"],
                effects=[f"夹爪抓住 {medicine_query}"],
                constraints=["抓取完成后不要再依赖腕部相机重新定位容器。"],
            )
        )

        graph.add_node(
            PlanNode(
                node_id="n4",
                label="使用缓存容器位姿放置",
                node_type="place",
                instruction="移动到缓存的容器 approach/place/lift 位姿，打开夹爪释放物体。",
                skill="use_cached_place_plan",
                args={
                    "place_plan_ref": "n2.cached_place_plan",
                    "gripper_open_value": 1,
                },
                parent=root.node_id,
                preconditions=[f"夹爪抓住 {medicine_query}", "cached_place_plan 可用"],
                effects=[f"{medicine_query} 已释放到 {container_name}"],
            )
        )

        graph.add_node(
            PlanNode(
                node_id="n5",
                label="验证任务结果",
                node_type="verify",
                instruction="用可用视觉或执行反馈判断药盒是否已放入容器。",
                skill=None,
                args={},
                parent=root.node_id,
                preconditions=[f"{medicine_query} 已释放到 {container_name}"],
                effects=["任务完成或生成失败约束"],
            )
        )

        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")
        graph.add_edge("n3", "n4")
        graph.add_edge("n4", "n5")

        return graph

    def _build_llm_prompt(
        self,
        task_goal: str,
        scene_text: str,
        constraints: List[str],
    ) -> str:
        constraints_text = "\n".join([f"- {c}" for c in constraints]) or "- 无"
        return f"""{PLANNER_SYSTEM_PROMPT}

任务目标：
{task_goal}

VDM 场景描述：
{scene_text or "无"}

已有约束：
{constraints_text}

请生成 instruction graph JSON。
"""

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _extract_medicine_query(task_goal: str, scene_text: str) -> str:
        text = f"{task_goal}\n{scene_text}"
        known = [
            "布洛芬",
            "西瓜霜",
            "抗病毒口服液",
            "抗病毒",
            "蒙脱石散",
            "999",
        ]
        for name in known:
            if name in text:
                return name

        patterns = [
            r"把(.+?)(?:放|放置|移动|拿|抓)",
            r"抓取(.+?)(?:并|然后|放|到)",
            r"目标(?:药品|药盒|物体)[：: ]*([^，。,.\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, task_goal)
            if match:
                return match.group(1).strip()

        return "目标药盒"

    @staticmethod
    def _extract_container_name(task_goal: str, scene_text: str) -> str:
        text = f"{task_goal}\n{scene_text}"
        container_aliases = [
            ("tray", ["托盘", "盘子", "tray", "plate"]),
            ("container", ["容器", "盒子", "收纳盒", "container", "box"]),
            ("bowl", ["碗", "bowl"]),
        ]
        for canonical, aliases in container_aliases:
            if any(alias in text for alias in aliases):
                return canonical

        patterns = [
            r"(?:放到|放入|放置到|放进)([^，。,.\n]+)",
            r"容器[：: ]*([^，。,.\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, task_goal)
            if match:
                value = match.group(1).strip()
                if value:
                    return value

        return "tray"
