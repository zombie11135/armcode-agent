# services/anygrasp_service.py

import os
import sys
import json
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
from PIL import Image
import open3d as o3d

@contextmanager
def pushd(path: str):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class AnyGraspService:
    """
    简化版 AnyGrasp service。

    重点：
    - AnyGrasp SDK 保持在原始 grasp_detection 目录
    - 初始化时只加载一次权重
    - 推理时直接复用 self.anygrasp
    """

    def __init__(
        self,
        grasp_detection_dir: str,
        checkpoint_path: str,
        max_gripper_width: float = 0.10,
        gripper_height: float = 0.03,
        top_down_grasp: bool = False,
        debug: bool = False,
    ):
        self.grasp_detection_dir = str(Path(grasp_detection_dir).resolve())
        self.checkpoint_path = checkpoint_path

        if self.grasp_detection_dir not in sys.path:
            sys.path.insert(0, self.grasp_detection_dir)

        cfgs = SimpleNamespace(
            checkpoint_path=checkpoint_path,
            max_gripper_width=max(0.0, min(0.1, max_gripper_width)),
            gripper_height=gripper_height,
            top_down_grasp=top_down_grasp,
            debug=debug,
        )

        print("[AnyGrasp] loading in original demo directory:")
        print("  grasp_detection_dir:", self.grasp_detection_dir)
        print("  checkpoint_path:", checkpoint_path)

        with pushd(self.grasp_detection_dir):
            from gsnet import AnyGrasp
            self.anygrasp = AnyGrasp(cfgs)
            self.anygrasp.load_net()

        print("[AnyGrasp] loaded successfully.")

    def plan_grasp(
        self,
        color_path: str,
        depth_path: Optional[str] = None,
        depth_npy_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        mask_npy_path: Optional[str] = None,
        intrinsics: Optional[Dict[str, float]] = None,
        depth_scale: float = 1000.0,
        lims: Optional[List[float]] = None,
        auto_lims_from_mask: bool = True,
        lims_margin: float = 0.03,
        top_k: int = 20,
        apply_object_mask: bool = True,
        dense_grasp: bool = False,
        collision_detection: bool = True,
        output_json_path: str = "runs/current_capture/anygrasp_result.json",
        use_mask_points: bool = True,
        min_mask_points: int = 200,
        
    ) -> Dict[str, Any]:
        """
        color_path:
            wrist_color.png

        depth_path:
            wrist_depth.png，单位通常是 uint16 mm

        depth_npy_path:
            可选，推荐。如果保存的是米制 depth，直接读取。

        mask_path / mask_npy_path:
            可选。用于自动计算 tight lims。
        """
        if intrinsics is None:
            raise ValueError("AnyGrasp 需要相机内参 intrinsics: fx, fy, cx, cy")

        colors = np.array(Image.open(color_path).convert("RGB"), dtype=np.float32) / 255.0
        depths = self._load_depth(depth_path, depth_npy_path, depth_scale)

        if colors.shape[:2] != depths.shape[:2]:
            colors = cv2.resize(
                colors,
                (depths.shape[1], depths.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # 先根据深度反投影得到全图 points_map
        points_map = self._depth_to_points(
            depths=depths,
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
        )

        valid_mask = np.isfinite(depths) & (depths > 0.02) & (depths < 1.5)

        target_mask = self._load_mask(mask_path, mask_npy_path)

        if target_mask is not None and target_mask.shape[:2] != depths.shape[:2]:
            target_mask = cv2.resize(
                target_mask.astype(np.uint8),
                (depths.shape[1], depths.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # 1. 根据 mask 自动算 tight lims
        if lims is None:
            if auto_lims_from_mask and target_mask is not None:
                lims = self._compute_lims_from_mask(
                    points_map=points_map,
                    target_mask=target_mask,
                    valid_mask=valid_mask,
                    margin=lims_margin,
                )
            else:
                lims = [-1, 1, -1, 1, 0.0, 1.0]

        # 2. 是否直接只使用 mask 内点云
        if use_mask_points and target_mask is not None:
            final_mask = valid_mask & target_mask
        else:
            final_mask = valid_mask

        points = points_map[final_mask].astype(np.float32)
        point_colors = colors[final_mask].astype(np.float32)

        if len(points) < min_mask_points:
            raise RuntimeError(
                f"mask 内有效点云太少: {len(points)}，"
                f"请检查 depth 是否 align 到 color，或增大 mask / bbox。"
            )

        print("[AnyGrasp] points:", points.shape)
        print("[AnyGrasp] lims:", lims)

        with pushd(self.grasp_detection_dir):
            gg, cloud = self.anygrasp.get_grasp(
                points,
                point_colors,
                lims=lims,
                apply_object_mask=apply_object_mask,
                dense_grasp=dense_grasp,
                collision_detection=collision_detection,
            )

        if len(gg) == 0:
            return {
                "success": False,
                "error": "No grasp detected.",
                "lims": lims,
            }

        if hasattr(gg, "grasp_group_array"):
            print("[AnyGrasp] grasp_group_array dtype before:", gg.grasp_group_array.dtype)
            gg.grasp_group_array = gg.grasp_group_array.astype(np.float32, copy=False)
            print("[AnyGrasp] grasp_group_array dtype after:", gg.grasp_group_array.dtype)

        try:
            gg = gg.nms().sort_by_score()
        except RuntimeError as e:
            print(f"[AnyGrasp][WARN] nms failed: {e}")
            print("[AnyGrasp][WARN] fallback to sort_by_score only")
            gg = gg.sort_by_score()

        candidates = self._extract_candidates(
            gg,
            top_k=top_k,
            intrinsics=intrinsics,
        )
        result = {
            "success": True,
            "best_grasp_camera": candidates[0],
            "candidate_grasps_camera": candidates,
            "num_candidates": len(candidates),
            "lims": lims,
            "color_path": color_path,
            "depth_path": depth_path,
            "depth_npy_path": depth_npy_path,
            "mask_path": mask_path,
            "mask_npy_path": mask_npy_path,
        }

        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        result["result_json_path"] = output_json_path
        return result

    @staticmethod
    def _load_depth(
        depth_path: Optional[str],
        depth_npy_path: Optional[str],
        depth_scale: float,
    ) -> np.ndarray:
        if depth_npy_path:
            depth = np.load(depth_npy_path).astype(np.float32)

            # 如果看起来是 mm，自动转米
            if np.nanmax(depth) > 10:
                depth = depth / 1000.0

            return depth

        if depth_path is None:
            raise ValueError("需要 depth_path 或 depth_npy_path")

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(depth_path)

        # demo 中 scale=1000.0，depths / scale 得到米
        return depth_raw.astype(np.float32) / float(depth_scale)

    @staticmethod
    def _depth_to_points(depths, fx, fy, cx, cy):
        h, w = depths.shape[:2]
        xmap, ymap = np.arange(w), np.arange(h)
        xmap, ymap = np.meshgrid(xmap, ymap)

        z = depths
        x = (xmap - cx) / fx * z
        y = (ymap - cy) / fy * z

        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def _load_mask(mask_path, mask_npy_path):
        if mask_npy_path:
            mask = np.load(mask_npy_path)
            if mask.ndim == 3:
                mask = mask[..., 0]
            return mask > 0

        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            return mask > 0

        return None

    @staticmethod
    def _compute_lims_from_mask(points_map, target_mask, valid_mask, margin=0.03):
        mask = target_mask & valid_mask

        pts = points_map[mask]

        if len(pts) < 50:
            raise RuntimeError("mask 内有效点太少，无法计算 AnyGrasp lims")

        xmin, ymin, zmin = np.percentile(pts, 2, axis=0)
        xmax, ymax, zmax = np.percentile(pts, 98, axis=0)

        lims = [
            float(xmin - margin),
            float(xmax + margin),
            float(ymin - margin),
            float(ymax + margin),
            float(max(0.0, zmin - margin)),
            float(zmax + margin),
        ]

        return lims







    @staticmethod
    def _extract_candidates(gg, top_k, intrinsics):
        candidates = []
        n = min(len(gg), top_k)

        translations = getattr(gg, "translations", None)
        rotations = getattr(gg, "rotation_matrices", None)
        scores = getattr(gg, "scores", None)
        widths = getattr(gg, "widths", None)
        heights = getattr(gg, "heights", None)
        depths = getattr(gg, "depths", None)

        for i in range(n):
            if translations is not None:
                t = np.asarray(translations[i], dtype=float)
                r = np.asarray(rotations[i], dtype=float)
                score = float(scores[i]) if scores is not None else None
                width = float(widths[i]) if widths is not None else None
                height = float(heights[i]) if heights is not None else None
                depth = float(depths[i]) if depths is not None else None
            else:
                g = gg[i]
                t = np.asarray(g.translation, dtype=float)
                r = np.asarray(g.rotation_matrix, dtype=float)
                score = float(getattr(g, "score", 0.0))
                width = float(getattr(g, "width", 0.0))
                height = float(getattr(g, "height", 0.0))
                depth = float(getattr(g, "depth", 0.0))

            u = None
            v = None
            if t[2] > 1e-6:
                u = int(round(intrinsics["fx"] * t[0] / t[2] + intrinsics["cx"]))
                v = int(round(intrinsics["fy"] * t[1] / t[2] + intrinsics["cy"]))

            candidates.append({
                "index": i,
                "translation": t.tolist(),
                "rotation_matrix": r.tolist(),
                "score": score,
                "width": width,
                "height": height,
                "depth": depth,
                "projected_pixel": [u, v] if u is not None else None,
            })

        return candidates