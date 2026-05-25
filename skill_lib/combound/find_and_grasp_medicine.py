from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import math

from core.skill_base import BaseSkill, SkillResult
from skill_lib.combound.select_obj_by_text import SelectObjectByTextSkill
from skill_lib.manipulation.plan_grasp_with_anygrasp import PlanGraspWithAnyGraspSkill
from skill_lib.perception.seg_obj import SegmentObjectSkill


class FindAndGraspMedicineSkill(BaseSkill):
    name = "find_and_grasp_medicine"
    description = "高级复合技能：稳定识别指定药盒，生成 AnyGrasp 抓取输入，并可选规划/执行抓取。"

    input_schema = {
        "query": "药品名称或包装文字，例如 布洛芬 / 西瓜霜 / 蒙脱石散。",
        "camera_name": "默认 wrist。",
        "target_prompts": "可选。GroundingDINO 多提示词列表。",
        "min_score": "目标文本/视觉融合最低分数。",
        "use_sam_for_selection": "是否用 SAM 生成候选 mask/crop，默认 False。",
        "use_mask_for_grasp": "是否把 SAM mask 传给 AnyGrasp，默认 False，推荐 bbox ROI。",
        "plan_grasp": "是否调用 AnyGrasp 规划抓取。",
        "execute": "是否执行 approach -> grasp -> close -> lift。",
        "gripper_open": "可选夹爪打开函数。",
        "gripper_close": "可选夹爪闭合函数。",
    }

    output_schema = {
        "selected_instance": "最终选中的药盒实例。",
        "grasp_input": "可直接传给 plan_grasp_with_anygrasp 的输入。",
        "grasp_plan": "可选 AnyGrasp 规划结果。",
        "ranked_candidates": "候选药盒排序明细。",
    }

    DEFAULT_TARGET_PROMPTS = [
        "medicine_box",
        "rectangular medicine carton",
        "pharmaceutical paper box",
        "medicine package with printed text",
        "drug package box",
    ]

    def run(
        self,
        query: str,
        camera_name: str = "wrist",
        target_prompts: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        min_score: float = 0.50,
        max_detections_per_prompt: int = 12,
        max_instances: int = 12,
        detection_threshold: Optional[float] = 0.22,
        text_threshold: Optional[float] = 0.18,
        merge_iou: float = 0.62,
        expand_bbox_px: int = 12,
        ocr_padding_px: int = 24,
        use_sam_for_selection: bool = False,
        use_mask_for_grasp: bool = False,
        bbox_margin_px: int = 16,
        infer_grasps_on_full_cloud: bool = True,
        filter_grasps_by_bbox: bool = True,
        bbox_filter_margin_px: int = 0,
        bbox_filter_inner_margin_ratio: float = 0.15,
        fallback_to_roi_points_when_bbox_empty: bool = True,
        fallback_intrinsics: Optional[Dict[str, float]] = None,
        plan_grasp: bool = True,
        execute: bool = False,
        xarm_ip: str = "192.168.1.237",
        handeye_config_path: str = "/home/wpy/pythonproject/armcode-agent/config/calibration/wrist_handeye.yaml",
        top_k: int = 50,
        lims_margin: float = 0.03,
        min_mask_points: int = 200,
        grasp_selection_mode: str = "random_top_k_good",
        random_select_top_k: int = 5,
        random_seed: Optional[int] = None,
        fallback_to_raw_when_no_good: bool = False,
        visualize_grasps: bool = False,
        visualize_top_k: int = 20,
        visualize_best_only: bool = False,
        visualize_selected_grasp: bool = False,
        visualize_bbox_filtered_grasps: bool = False,
        visualize_bbox_filtered_top_k: int = 100,
        approach_clearance_mm: float = 80.0,
        lift_after_grasp_mm: float = 120.0,
        move_speed: float = 50,
        move_acc: float = 500,
        gripper_open: Optional[Callable[[], bool]] = None,
        gripper_close: Optional[Callable[[], bool]] = None,
        **kwargs,
    ) -> SkillResult:
        try:
            self.context.emit_text(f"开始高级药盒识别抓取：query={query}")

            capture = self._capture(camera_name)
            image_path = capture["color_path"]
            self.context.emit_image(image_path=image_path, caption="高级药盒识别输入图像")

            prompts = self._build_prompts(target_prompts)
            detections = self._detect_with_prompt_fusion(
                image_path=image_path,
                prompts=prompts,
                threshold=detection_threshold,
                text_threshold=text_threshold,
                max_detections_per_prompt=max_detections_per_prompt,
                merge_iou=merge_iou,
            )

            if not detections:
                return SkillResult(
                    success=False,
                    data={"stage": "detect", "image_path": image_path, "prompts": prompts},
                    error=f"没有检测到候选药盒：{query}",
                )

            detections = detections[:max_instances]
            self.context.emit_text(f"多提示词融合后得到 {len(detections)} 个候选 bbox。")

            if use_sam_for_selection:
                instances = self._segment_candidates(
                    image_path=image_path,
                    detections=detections,
                    expand_bbox_px=expand_bbox_px,
                    **kwargs,
                )
                instances = [inst for inst in instances if inst.get("success")]
            else:
                instances = self._instances_from_detections(
                    image_path=image_path,
                    detections=detections,
                )

            if not instances:
                return SkillResult(
                    success=False,
                    data={"stage": "segment", "image_path": image_path, "detections": detections},
                    error="没有可用于 OCR/抓取的候选药盒。",
                )

            instances = self._augment_ocr(
                image_path=image_path,
                instances=instances,
                query=query,
                padding=ocr_padding_px,
            )

            select_result = self._select_best_instance(
                instances=instances,
                query=query,
                aliases=aliases,
                min_score=min_score,
            )

            if not select_result.success:
                return SkillResult(
                    success=False,
                    data={
                        "stage": "select",
                        "image_path": image_path,
                        "detections": detections,
                        "instances": instances,
                        "selection_result": select_result.data,
                    },
                    error=select_result.error,
                )

            selected = select_result.data["selected_instance"]
            grasp_input = self._build_grasp_input(
                capture=capture,
                image_path=image_path,
                selected=selected,
                query=query,
                use_mask_for_grasp=use_mask_for_grasp,
                fallback_intrinsics=fallback_intrinsics,
            )

            result_data = {
                "image_path": image_path,
                "prompts": prompts,
                "detections": detections,
                "instances": instances,
                "selected_instance": selected,
                "selection_result": select_result.data,
                "ranked_candidates": select_result.data["ranked_candidates"],
                "grasp_input": grasp_input,
            }

            if plan_grasp:
                plan_result = PlanGraspWithAnyGraspSkill(self.context).run(
                    grasp_input=grasp_input,
                    xarm_ip=xarm_ip,
                    handeye_config_path=handeye_config_path,
                    approach_clearance_mm=approach_clearance_mm,
                    lift_after_grasp_mm=lift_after_grasp_mm,
                    auto_lims_from_mask=use_mask_for_grasp,
                    auto_lims_from_bbox=not infer_grasps_on_full_cloud,
                    lims_margin=lims_margin,
                    bbox_margin_px=bbox_margin_px,
                    use_mask_points=True,
                    use_roi_points_for_inference=not infer_grasps_on_full_cloud,
                    filter_grasps_by_bbox=filter_grasps_by_bbox,
                    bbox_filter_margin_px=bbox_filter_margin_px,
                    bbox_filter_inner_margin_ratio=bbox_filter_inner_margin_ratio,
                    fallback_to_roi_points_when_bbox_empty=fallback_to_roi_points_when_bbox_empty,
                    min_mask_points=min_mask_points,
                    top_k=top_k,
                    grasp_selection_mode=grasp_selection_mode,
                    random_select_top_k=random_select_top_k,
                    random_seed=random_seed,
                    fallback_to_raw_when_no_good=fallback_to_raw_when_no_good,
                    apply_object_mask=True,
                    visualize_grasps=visualize_grasps,
                    visualize_top_k=visualize_top_k,
                    visualize_best_only=visualize_best_only,
                    visualize_selected_grasp=visualize_selected_grasp,
                    visualize_bbox_filtered_grasps=visualize_bbox_filtered_grasps,
                    visualize_bbox_filtered_top_k=visualize_bbox_filtered_top_k,
                )
                result_data["grasp_plan"] = plan_result.data

                if not plan_result.success:
                    return SkillResult(
                        success=False,
                        data=result_data,
                        error=f"药盒已识别，但抓取规划失败: {plan_result.error}",
                    )

                if execute:
                    exec_result = self._execute_grasp(
                        plan=plan_result.data,
                        move_speed=move_speed,
                        move_acc=move_acc,
                        gripper_open=gripper_open,
                        gripper_close=gripper_close,
                    )
                    result_data["execution"] = exec_result

                    if not exec_result.get("success", False):
                        return SkillResult(
                            success=False,
                            data=result_data,
                            error=exec_result.get("error", "抓取执行失败。"),
                        )

            self.context.emit_text(
                "高级药盒识别完成："
                f"object_id={selected.get('object_id')}, "
                f"score={select_result.data.get('score'):.3f}, "
                f"text={selected.get('ocr_text')}"
            )

            if selected.get("overlay_path"):
                self.context.emit_image(
                    image_path=selected["overlay_path"],
                    caption=f"最终选择药盒：{query}",
                )

            return SkillResult(
                success=True,
                data=result_data,
                message=f"已稳定识别目标药盒：{query}",
            )

        except Exception as e:
            error = f"find_and_grasp_medicine 执行失败: {e}"
            try:
                self.context.emit_error(error)
            except Exception:
                pass
            return SkillResult(success=False, data={}, error=error)

    def _capture(self, camera_name: str) -> Dict[str, Any]:
        camera = self.context.require("camera")
        if camera_name == "wrist":
            return camera.capture_wrist()
        if camera_name == "global":
            return camera.capture_global()
        raise ValueError(f"Unknown camera_name: {camera_name}")

    @classmethod
    def _build_prompts(cls, target_prompts: Optional[List[str]]) -> List[str]:
        prompts = list(target_prompts or cls.DEFAULT_TARGET_PROMPTS)
        clean = []
        for prompt in prompts:
            prompt = str(prompt).strip()
            if prompt and prompt not in clean:
                clean.append(prompt)
        return clean

    def _detect_with_prompt_fusion(
        self,
        image_path: str,
        prompts: List[str],
        threshold: Optional[float],
        text_threshold: Optional[float],
        max_detections_per_prompt: int,
        merge_iou: float,
    ) -> List[Dict[str, Any]]:
        detector = self.context.require("detector")
        fused: List[Dict[str, Any]] = []

        for prompt in prompts:
            self.context.emit_text(f"GroundingDINO 检测候选药盒 prompt={prompt}")
            result = detector.detect(
                image_path=image_path,
                target_name=prompt,
                threshold=threshold,
                text_threshold=text_threshold,
                max_objects=max_detections_per_prompt,
                enable_filter=True,
            )

            for det in result.get("detections", []):
                det = dict(det)
                det["source_prompt"] = prompt
                self._merge_detection(fused, det, merge_iou=merge_iou)

        fused.sort(
            key=lambda d: (
                len(d.get("prompt_hits", [])),
                float(d.get("confidence", 0.0) or 0.0),
                self._bbox_area(d.get("bbox")),
            ),
            reverse=True,
        )

        for idx, det in enumerate(fused):
            det["object_id"] = f"obj_{idx + 1}"

        return fused

    @classmethod
    def _merge_detection(
        cls,
        fused: List[Dict[str, Any]],
        det: Dict[str, Any],
        merge_iou: float,
    ) -> None:
        for old in fused:
            if cls._bbox_iou(old["bbox"], det["bbox"]) < merge_iou:
                continue

            old_conf = float(old.get("confidence", 0.0) or 0.0)
            new_conf = float(det.get("confidence", 0.0) or 0.0)
            if new_conf > old_conf:
                old["bbox"] = det["bbox"]
                old["confidence"] = new_conf
                old["phrase"] = det.get("phrase")
                old["source_prompt"] = det.get("source_prompt")

            hits = old.setdefault("prompt_hits", [])
            prompt = det.get("source_prompt")
            if prompt and prompt not in hits:
                hits.append(prompt)
            old["fused_confidence"] = max(old_conf, new_conf)
            return

        new_det = dict(det)
        new_det["prompt_hits"] = [det.get("source_prompt")] if det.get("source_prompt") else []
        new_det["fused_confidence"] = float(det.get("confidence", 0.0) or 0.0)
        fused.append(new_det)

    def _segment_candidates(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        expand_bbox_px: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        segment_skill = SegmentObjectSkill(self.context)
        instances = []

        for idx, det in enumerate(detections):
            obj_id = det.get("object_id", f"obj_{idx + 1}")
            prefix = f"runs/current_capture/advanced_medicine_{idx + 1:02d}_{obj_id}"
            seg_result = segment_skill.run(
                image_path=image_path,
                bbox=det["bbox"],
                target_name="medicine_box",
                segment_all=False,
                output_prefix=prefix,
                expand_bbox_px=expand_bbox_px,
                select_by="iou",
                **kwargs,
            )

            if not seg_result.success:
                self.context.emit_warning(f"{obj_id} 分割失败: {seg_result.error}")
                continue

            seg_instances = seg_result.data.get("instances", [])
            if not seg_instances:
                continue

            inst = dict(seg_instances[0])
            inst["object_id"] = obj_id
            inst["detection"] = det
            inst["confidence"] = det.get("confidence")
            inst["prompt_hits"] = det.get("prompt_hits", [])
            inst["source_prompt"] = det.get("source_prompt")
            instances.append(inst)

        return instances

    @staticmethod
    def _instances_from_detections(
        image_path: str,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        instances = []
        for idx, det in enumerate(detections):
            instances.append(
                {
                    "object_id": det.get("object_id", f"obj_{idx + 1}"),
                    "name": det.get("name", "medicine_box"),
                    "target_name": "medicine_box",
                    "detection": det,
                    "bbox": det.get("bbox"),
                    "confidence": det.get("confidence"),
                    "phrase": det.get("phrase"),
                    "image_path": image_path,
                    "prompt_hits": det.get("prompt_hits", []),
                    "source_prompt": det.get("source_prompt"),
                    "success": True,
                    "segmenter": "bbox_only",
                    "mask_path": None,
                    "mask_npy_path": None,
                    "mask_area": None,
                }
            )
        return instances

    def _augment_ocr(
        self,
        image_path: str,
        instances: List[Dict[str, Any]],
        query: str,
        padding: int,
    ) -> List[Dict[str, Any]]:
        ocr = self.context.require("ocr")

        for idx, inst in enumerate(instances):
            texts = []
            confidences = []
            items = []
            vis_paths = []

            sources = []
            if inst.get("crop_path"):
                sources.append(("crop", inst["crop_path"]))
            if inst.get("masked_image_path"):
                sources.append(("masked", inst["masked_image_path"]))

            for source_name, source_path in sources:
                try:
                    result = ocr.read_image(
                        image_path=source_path,
                        output_prefix=f"runs/current_capture/advanced_ocr_{idx + 1:02d}_{source_name}",
                        min_confidence=0.0,
                    )
                    texts.extend(result.get("texts", []))
                    confidences.extend(result.get("confidences", []))
                    items.extend(result.get("items", []))
                    if result.get("vis_path"):
                        vis_paths.append(result["vis_path"])
                except Exception as e:
                    inst.setdefault("ocr_errors", []).append(f"{source_name}: {e}")

            if inst.get("bbox"):
                try:
                    result = ocr.read_region(
                        image_path=image_path,
                        bbox=inst["bbox"],
                        output_prefix=f"runs/current_capture/advanced_ocr_{idx + 1:02d}_region",
                        padding=padding,
                        min_confidence=0.0,
                    )
                    texts.extend(result.get("texts", []))
                    confidences.extend(result.get("confidences", []))
                    items.extend(result.get("items", []))
                    inst["ocr_region_crop_path"] = result.get("crop_path")
                    if result.get("vis_path"):
                        vis_paths.append(result["vis_path"])
                except Exception as e:
                    inst.setdefault("ocr_errors", []).append(f"region: {e}")

            inst["ocr_success"] = bool(texts)
            inst["ocr_texts"] = self._dedupe_texts(texts)
            inst["ocr_text"] = " ".join(inst["ocr_texts"])
            inst["ocr_confidences"] = confidences
            inst["ocr_avg_confidence"] = (
                sum(float(x) for x in confidences) / len(confidences)
                if confidences else 0.0
            )
            inst["ocr_items"] = items
            inst["ocr_vis_paths"] = vis_paths

            self.context.emit_text(
                f"候选 {inst.get('object_id')} OCR: {inst['ocr_text'] or '<empty>'}"
            )
            if vis_paths:
                self.context.emit_image(
                    image_path=vis_paths[0],
                    caption=f"候选 {inst.get('object_id')} OCR 增强结果：{query}",
                )

        return instances

    def _select_best_instance(
        self,
        instances: List[Dict[str, Any]],
        query: str,
        aliases: Optional[List[str]],
        min_score: float,
    ) -> SkillResult:
        keywords = SelectObjectByTextSkill._build_keywords(query, aliases)
        ranked = []

        for idx, inst in enumerate(instances):
            text = SelectObjectByTextSkill._collect_instance_text(inst)
            text_score, matched_keyword, match_type = SelectObjectByTextSkill._score_text(
                text=text,
                keywords=keywords,
            )
            ocr_conf = self._clamp(float(inst.get("ocr_avg_confidence", 0.0) or 0.0))
            det_conf = self._clamp(float(inst.get("confidence", 0.0) or 0.0))
            prompt_score = min(len(inst.get("prompt_hits", [])) / 3.0, 1.0)
            mask_score = self._mask_quality_score(inst)

            final_score = (
                0.68 * text_score
                + 0.10 * ocr_conf
                + 0.10 * det_conf
                + 0.08 * prompt_score
                + 0.04 * mask_score
            )

            if not text.strip():
                final_score *= 0.62

            ranked.append(
                {
                    "index": idx,
                    "object_id": inst.get("object_id"),
                    "bbox": inst.get("bbox"),
                    "crop_path": inst.get("crop_path"),
                    "overlay_path": inst.get("overlay_path"),
                    "ocr_text": text,
                    "ocr_avg_confidence": ocr_conf,
                    "detection_confidence": det_conf,
                    "prompt_hits": inst.get("prompt_hits", []),
                    "score": final_score,
                    "raw_text_score": text_score,
                    "matched_keyword": matched_keyword,
                    "match_type": match_type,
                    "mask_score": mask_score,
                    "instance": inst,
                }
            )

        ranked.sort(key=lambda x: x["score"], reverse=True)
        best = ranked[0] if ranked else None

        if best is None or best["score"] < min_score:
            highest = 0.0 if best is None else best["score"]
            return SkillResult(
                success=False,
                data={"query": query, "keywords": keywords, "ranked_candidates": ranked},
                error=f"未找到稳定匹配的药盒。最高分={highest:.3f}, min_score={min_score:.3f}",
            )

        return SkillResult(
            success=True,
            data={
                "query": query,
                "keywords": keywords,
                "selected_instance": best["instance"],
                "selected_index": best["index"],
                "score": best["score"],
                "matched_keyword": best["matched_keyword"],
                "match_type": best["match_type"],
                "ranked_candidates": ranked,
            },
            message=f"成功选择药盒：{query}",
        )

    def _build_grasp_input(
        self,
        capture: Dict[str, Any],
        image_path: str,
        selected: Dict[str, Any],
        query: str,
        use_mask_for_grasp: bool,
        fallback_intrinsics: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        depth_path = self._existing_or_none(
            capture.get("depth_path")
            or capture.get("wrist_depth_path")
            or "runs/current_capture/wrist_depth.png"
        )
        depth_npy_path = self._existing_or_none(
            capture.get("depth_npy_path")
            or capture.get("depth_meter_path")
        )

        if depth_path is None and depth_npy_path is None:
            raise FileNotFoundError("缺少同一帧 depth_path/depth_npy_path，无法抓取。")

        intrinsics = (
            fallback_intrinsics
            or capture.get("depth_intrinsics")
            or capture.get("color_intrinsics")
            or capture.get("camera_intrinsics")
        )
        intrinsics = self._normalize_intrinsics(intrinsics)

        if intrinsics is None:
            raise RuntimeError("缺少相机内参，无法构造 AnyGrasp 输入。")

        mask_path = None
        mask_npy_path = None
        if use_mask_for_grasp:
            mask_path = self._existing_or_none(selected.get("mask_path"))
            mask_npy_path = self._existing_or_none(selected.get("mask_npy_path"))
            if mask_path is None and mask_npy_path is None:
                raise FileNotFoundError("use_mask_for_grasp=True，但缺少目标 mask。")

        bbox = selected.get("bbox")
        if bbox is None and mask_path is None and mask_npy_path is None:
            raise FileNotFoundError("缺少目标 bbox/mask，无法构造 AnyGrasp 输入。")

        return {
            "object_id": selected.get("object_id"),
            "target_name": "medicine_box",
            "query": query,
            "color_path": image_path,
            "depth_path": depth_path,
            "depth_npy_path": depth_npy_path,
            "mask_path": mask_path,
            "mask_npy_path": mask_npy_path,
            "bbox": bbox,
            "crop_path": selected.get("crop_path"),
            "overlay_path": selected.get("overlay_path"),
            "depth_intrinsics": intrinsics,
            "color_intrinsics": intrinsics,
            "camera_intrinsics": intrinsics,
            "depth_scale": capture.get("depth_scale", 1000.0),
            "ocr_text": selected.get("ocr_text"),
            "ocr_texts": selected.get("ocr_texts"),
        }

    def _execute_grasp(
        self,
        plan: Dict[str, Any],
        move_speed: float,
        move_acc: float,
        gripper_open: Optional[Callable[[], bool]],
        gripper_close: Optional[Callable[[], bool]],
    ) -> Dict[str, Any]:
        arm = self.context.require("xarm")
        approach = plan.get("approach_pose_mmrad")
        grasp = plan.get("best_grasp_xarm_mmrad")
        lift = plan.get("lift_pose_mmrad")

        for name, pose in [("approach", approach), ("grasp", grasp), ("lift", lift)]:
            if pose is None or len(pose) != 6:
                return {"success": False, "error": f"抓取规划缺少 {name} pose。"}

        if gripper_open and not gripper_open():
            return {"success": False, "error": "夹爪打开失败。"}

        for label, pose in [("approach", approach), ("grasp", grasp)]:
            ok = self._move_xarm(arm, pose, move_speed, move_acc)
            if not ok:
                return {"success": False, "error": f"xArm 移动到 {label} 失败。"}

        if gripper_close and not gripper_close():
            return {"success": False, "error": "夹爪闭合失败。"}

        if not self._move_xarm(arm, lift, move_speed, move_acc):
            return {"success": False, "error": "xArm 抬升失败。"}

        return {"success": True, "message": "抓取执行完成。"}

    @staticmethod
    def _move_xarm(arm, pose: List[float], speed: float, acc: float) -> bool:
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

    @staticmethod
    def _dedupe_texts(texts: List[str]) -> List[str]:
        clean = []
        for text in texts:
            text = str(text).strip()
            if text and text not in clean:
                clean.append(text)
        return clean

    @staticmethod
    def _existing_or_none(path) -> Optional[str]:
        if not path:
            return None
        p = Path(path)
        return str(p) if p.exists() else None

    @staticmethod
    def _normalize_intrinsics(intrinsics):
        if intrinsics is None:
            return None
        if isinstance(intrinsics, dict):
            fx = intrinsics.get("fx")
            fy = intrinsics.get("fy")
            cx = intrinsics.get("cx", intrinsics.get("ppx"))
            cy = intrinsics.get("cy", intrinsics.get("ppy"))
            if fx is not None and fy is not None and cx is not None and cy is not None:
                return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}
        return intrinsics

    @staticmethod
    def _bbox_area(bbox) -> float:
        if not bbox:
            return 0.0
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

    @classmethod
    def _bbox_iou(cls, box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union = cls._bbox_area(box_a) + cls._bbox_area(box_b) - inter
        return 0.0 if union <= 0 else float(inter / union)

    @staticmethod
    def _clamp(value: float) -> float:
        if math.isnan(value):
            return 0.0
        return max(0.0, min(1.0, value))

    @classmethod
    def _mask_quality_score(cls, inst: Dict[str, Any]) -> float:
        bbox_area = cls._bbox_area(inst.get("bbox"))
        mask_area = float(inst.get("mask_area", 0.0) or 0.0)
        if bbox_area <= 0 or mask_area <= 0:
            return 0.0
        fill = mask_area / bbox_area
        if fill < 0.05 or fill > 1.20:
            return 0.15
        if 0.20 <= fill <= 0.95:
            return 1.0
        return 0.55
