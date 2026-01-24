import queue
import random
import re
import threading
import time
from threading import Thread

from makcu import MouseButton
from pyclick import HumanCurve

from src.infer_function import nms, nms_v8, read_img, sunone_postprocess


class AimingMixin:
    """Mixin class for aiming, targeting, mouse control and trigger system."""

    def _clear_target_lock(self):
        self._target_lock_active = False
        self._target_lock_id = None
        self._target_lock_center = None
        self._target_lock_lost_time = None

    def _get_lock_distance(self, cfg):
        dyn_cfg = cfg.get("dynamic_scope", {}) or {}
        if not bool(dyn_cfg.get("enabled", False)):
            return 0.0
        try:
            return max(1.0, float(cfg.get("target_lock_distance", 100) or 100))
        except Exception:
            return 0.0

    def _get_lock_reacquire_time(self, cfg):
        try:
            return max(0.0, float(cfg.get("target_lock_reacquire_time", 0.3) or 0.3))
        except Exception:
            return 0.0

    def _make_lock_id(self, target, grid=10):
        box = target.get("box")
        if not box:
            return None
        left, top, width, height = box
        try:
            x1 = int(left // grid) * grid
            y1 = int(top // grid) * grid
            x2 = int((left + width) // grid) * grid
            y2 = int((top + height) // grid) * grid
        except Exception:
            return None
        return (x1, y1, x2, y2)

    def _get_box_center(self, target):
        box = target.get("box")
        if not box:
            return None
        left, top, width, height = box
        try:
            return (left + width * 0.5, top + height * 0.5)
        except Exception:
            return None

    def _get_lock_center(self, target):
        pos = target.get("pos")
        if not pos:
            return None
        try:
            offset_x = getattr(self, "identify_rect_left", 0) or 0
            offset_y = getattr(self, "identify_rect_top", 0) or 0
            return (pos[0] - offset_x, pos[1] - offset_y)
        except Exception:
            return None

    def _update_target_lock(self, target, use_box_center=False):
        if target is None:
            self._clear_target_lock()
            return
        lock_id = self._make_lock_id(target)
        if lock_id is None:
            self._clear_target_lock()
            return
        if use_box_center:
            center = self._get_box_center(target)
        else:
            center = self._get_lock_center(target)
        if center is None:
            self._clear_target_lock()
            return
        self._target_lock_active = True
        self._target_lock_id = lock_id
        self._target_lock_center = center
        self._target_lock_lost_time = None

    def _apply_target_lock(self, targets, lock_distance, reacquire_time):
        if not getattr(self, "_target_lock_active", False):
            return targets, False
        now = time.time()
        if not targets:
            if self._target_lock_lost_time is None:
                self._target_lock_lost_time = now
            lost_for = now - self._target_lock_lost_time
            if lost_for >= reacquire_time:
                self._clear_target_lock()
                return targets, False
            return [], True
        locked_idx = -1
        locked_id = getattr(self, "_target_lock_id", None)
        if locked_id is not None:
            for idx, target in enumerate(targets):
                if self._make_lock_id(target) == locked_id:
                    locked_idx = idx
                    break
        if locked_idx == -1:
            center = self._target_lock_center
            if center:
                best_dist = float("inf")
                for idx, target in enumerate(targets):
                    candidate_center = self._get_box_center(target)
                    if candidate_center is None:
                        continue
                    dx = candidate_center[0] - center[0]
                    dy = candidate_center[1] - center[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        locked_idx = idx
                if locked_idx != -1 and best_dist > lock_distance:
                    locked_idx = -1
        if locked_idx != -1:
            candidate = targets[locked_idx]
            candidate_center = self._get_box_center(candidate)
            if candidate_center is not None:
                self._target_lock_center = candidate_center
            self._target_lock_id = self._make_lock_id(candidate)
            self._target_lock_lost_time = None
            return [candidate], False
        if self._target_lock_lost_time is None:
            self._target_lock_lost_time = now
        lost_for = now - self._target_lock_lost_time
        if lost_for >= reacquire_time:
            self._clear_target_lock()
            return targets, False
        return [], True

    def _update_target_counts(self, targets, reference_class):
        current_total_count = len(targets)
        current_target_count = len(
            [t for t in targets if t.get("class_id") == reference_class]
        )
        self.last_target_count_by_class[reference_class] = current_target_count
        self.last_target_count = current_total_count

    def smooth_small_targets(self, targets):
        """
        Apply historical smoothing to small targets to improve detection stability
        """
        current_frame = time.time()
        for target_id in list(self.target_history.keys()):
            history = self.target_history[target_id]
            max_frames = self.config["small_target_enhancement"]["smooth_frames"]
            history["frames"] = [
                frame
                for frame in history["frames"]
                if not len(history["frames"]) - history["frames"].index(frame)
                <= max_frames
                or current_frame - frame["time"] < 1.0
            ]
            if not history["frames"]:
                del self.target_history[target_id]
        smoothed_targets = []
        for target in targets:
            target_id = target.get("id")
            relative_size = target.get("relative_size", 0)
            if (
                self.config["small_target_enhancement"]["enabled"]
                and self.config["small_target_enhancement"]["smooth_enabled"]
                and (
                    relative_size < self.config["small_target_enhancement"]["threshold"]
                )
                and target_id
            ):
                if target_id not in self.target_history:
                    self.target_history[target_id] = {"frames": []}
                frame_info = {
                    "time": current_frame,
                    "pos": target["pos"],
                    "size": target["size"],
                    "confidence": target.get("confidence", 0.5),
                }
                self.target_history[target_id]["frames"].append(frame_info)
                max_frames = self.config["small_target_enhancement"]["smooth_frames"]
                if len(self.target_history[target_id]["frames"]) > max_frames:
                    self.target_history[target_id]["frames"] = self.target_history[
                        target_id
                    ]["frames"][-max_frames:]
                frames = self.target_history[target_id]["frames"]
                if len(frames) >= 2:
                    avg_x = sum((f["pos"][0] for f in frames)) / len(frames)
                    avg_y = sum((f["pos"][1] for f in frames)) / len(frames)
                    avg_size = sum((f["size"] for f in frames)) / len(frames)
                    smoothed_target = target.copy()
                    smoothed_target["pos"] = (avg_x, avg_y)
                    smoothed_target["size"] = avg_size
                    smoothed_target["smoothed"] = True
                    smoothed_targets.append(smoothed_target)
                else:
                    smoothed_targets.append(target)
            else:
                smoothed_targets.append(target)
        return smoothed_targets

    def select_target_by_priority(self, targets):
        """Intelligent target selection: priority algorithm based on distance to center, with target count monitoring and integral reset"""
        lock_distance = self._get_lock_distance(self.pressed_key_config)
        lock_enabled = lock_distance > 0
        lock_reacquire_time = self._get_lock_reacquire_time(self.pressed_key_config)
        priority_order = self.pressed_key_config.get("class_priority_order", [])
        if isinstance(priority_order, str):
            parts = re.split(r"[-,\s]+", priority_order.strip())
            parsed = []
            for part in parts:
                if not part:
                    continue
                try:
                    parsed.append(int(part))
                except ValueError:
                    continue
            priority_order = parsed

        reference_class = self.pressed_key_config.get("target_reference_class", 0)
        aim_scope = self.get_dynamic_aim_scope()
        base_scope = float(self.pressed_key_config.get("aim_bot_scope", 0) or 0)
        if base_scope <= 0:
            base_scope = aim_scope

        def _get_tiebreak_ratio():
            try:
                return max(0.0, float(self.config.get("aim_weight_tiebreak_ratio", 0.1)))
            except Exception:
                return 0.1

        def _choose_with_weights(candidates):
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0]
            ratio = _get_tiebreak_ratio()
            if ratio <= 0:
                return min(candidates, key=lambda t: t["distance_to_center"])
            min_dist = min(t["distance_to_center"] for t in candidates)
            scope = aim_scope if aim_scope > 0 else max(1.0, min_dist)
            threshold = scope * ratio
            near = [t for t in candidates if t["distance_to_center"] <= min_dist + threshold]
            if len(near) == 1:
                return near[0]
            try:
                w_dist = float(self.config.get("distance_scoring_weight", 1.0) or 1.0)
                w_center = float(self.config.get("center_scoring_weight", 1.0) or 1.0)
                w_size = float(self.config.get("size_scoring_weight", 1.0) or 1.0)
            except Exception:
                w_dist, w_center, w_size = (1.0, 1.0, 1.0)
            if w_dist == 0 and w_center == 0 and w_size == 0:
                return min(near, key=lambda t: t["distance_to_center"])
            max_dist = max(1.0, max(t["distance_to_center"] for t in near))
            max_center = max(1.0, max(abs(t.get("dx", 0.0)) + abs(t.get("dy", 0.0)) for t in near))
            max_size = max(1e-6, max(float(t.get("size", 0.0)) for t in near))

            def _score(t):
                dist_norm = t["distance_to_center"] / max_dist
                center_norm = (abs(t.get("dx", 0.0)) + abs(t.get("dy", 0.0))) / max_center
                size_norm = float(t.get("size", 0.0)) / max_size
                return (w_dist * dist_norm) + (w_center * center_norm) + (w_size * (1.0 - size_norm))

            return min(near, key=_score)
        if lock_enabled:
            targets, lock_blocking = self._apply_target_lock(
                targets, lock_distance, lock_reacquire_time
            )
            if lock_blocking:
                return None
        if not targets:
            self._clear_target_lock()
            if self.last_target_count > 0:
                self.last_target_count = 0
                self.last_target_count_by_class.clear()
                if hasattr(self, "dual_pid"):
                    self.dual_pid._i_term["x"] = 0
                    self.dual_pid._i_term["y"] = 0
            return None
        if lock_enabled and getattr(self, "_target_lock_active", False) and len(targets) == 1:
            self._update_target_counts(targets, reference_class)
            self._update_target_lock(targets[0], use_box_center=True)
            return targets[0]
        valid_targets = []
        for target in targets:
            dx = target["pos"][0] - self.screen_center_x
            dy = target["pos"][1] - self.screen_center_y
            distance = (dx * dx + dy * dy) ** 0.5
            target["distance_to_center"] = distance
            target["dx"] = dx
            target["dy"] = dy
            if distance <= aim_scope:
                valid_targets.append(target)
        if not lock_enabled:
            self._clear_target_lock()
        if priority_order:
            for class_id in priority_order:
                candidates = []
                for target in targets:
                    try:
                        target_class = int(target.get("class_id"))
                    except (TypeError, ValueError):
                        continue
                    if target_class != class_id:
                        continue
                    if base_scope > 0 and target["distance_to_center"] > base_scope:
                        continue
                    candidates.append(target)
                if candidates:
                    selected = _choose_with_weights(candidates)
                    if lock_distance > 0:
                        self._update_target_lock(selected, use_box_center=True)
                    else:
                        self._clear_target_lock()
                    return selected
        if not valid_targets:
            if lock_enabled and getattr(self, "_target_lock_active", False):
                return None
            self._clear_target_lock()
            if self.last_target_count > 0:
                self.last_target_count = 0
                self.last_target_count_by_class.clear()
                if hasattr(self, "dual_pid"):
                    self.dual_pid._i_term["x"] = 0
                    self.dual_pid._i_term["y"] = 0
                    print("Target moved out of range, reset PID integral term")
            return None
        target_switch_delay = self.pressed_key_config.get("target_switch_delay", 0)
        current_total_count = len(valid_targets)
        prev_total_count = self.last_target_count
        current_target_count = len(
            [t for t in valid_targets if t.get("class_id") == reference_class]
        )
        last_count = self.last_target_count_by_class.get(reference_class, 0)
        if (
            target_switch_delay > 0
            and (not self.is_waiting_for_switch)
            and (prev_total_count > 1)
            and (current_total_count < prev_total_count)
        ):
            self.is_waiting_for_switch = True
            self.target_switch_time = time.time() * 1000
            if hasattr(self, "dual_pid"):
                self.dual_pid._i_term["x"] = 0
                self.dual_pid._i_term["y"] = 0
            return None
        if self.is_waiting_for_switch and current_total_count > prev_total_count:
            self.is_waiting_for_switch = False
        if (
            target_switch_delay == 0
            and current_total_count < prev_total_count
            and (prev_total_count > 0)
            and hasattr(self, "dual_pid")
        ):
            self.dual_pid._i_term["x"] = 0
            self.dual_pid._i_term["y"] = 0
        if self.is_waiting_for_switch:
            current_time = time.time() * 1000
            if current_time - self.target_switch_time >= target_switch_delay:
                self.is_waiting_for_switch = False
            else:
                return None
        self.last_target_count_by_class[reference_class] = current_target_count
        self.last_target_count = current_total_count
        if priority_order:
            priority_map = {cid: idx for idx, cid in enumerate(priority_order)}

            def priority_key(target):
                class_id = target.get("class_id")
                try:
                    class_id = int(class_id)
                except (TypeError, ValueError):
                    class_id = None
                return (
                    priority_map.get(class_id, len(priority_map)),
                    target["distance_to_center"],
                )

            valid_targets.sort(key=priority_key)
        else:
            valid_targets.sort(key=lambda x: x["distance_to_center"])
        selected = _choose_with_weights(valid_targets)
        if lock_distance > 0:
            self._update_target_lock(selected, use_box_center=True)
        else:
            self._clear_target_lock()
        return selected

    def aim_bot_func(self, uTimerID, uMsg, dwUser, dw1, dw2):
        if self.aim_key_status:
            try:
                aim_data = self.que_aim.get_nowait()
                if isinstance(aim_data, tuple):
                    aim_targets, class_ids = aim_data
                else:
                    aim_targets = aim_data
                    class_ids = []
            except queue.Empty:
                aim_targets = []
                class_ids = []
            nearest = None
            if len(aim_targets):
                target_objects = []
                for i in range(len(aim_targets)):
                    item = aim_targets[i]
                    result_center_x, result_center_y, width, height = item
                    class_id = class_ids[i] if i < len(class_ids) else 0
                    aim_position = self.get_aim_position_for_class(class_id)
                    if hasattr(self, "engine") and self.engine:
                        model_width = self.engine.get_input_shape()[3]
                        model_height = self.engine.get_input_shape()[2]
                        model_area = model_width * model_height
                    else:
                        model_area = 102400
                    absolute_area = width * height
                    relative_size = absolute_area / model_area
                    if self.config["small_target_enhancement"]["enabled"]:
                        small_threshold = self.config["small_target_enhancement"][
                            "threshold"
                        ]
                        medium_threshold = self.config["small_target_enhancement"][
                            "medium_threshold"
                        ]
                        small_boost = self.config["small_target_enhancement"][
                            "boost_factor"
                        ]
                        medium_boost = self.config["small_target_enhancement"][
                            "medium_boost"
                        ]
                        if relative_size < small_threshold:
                            size_boost = small_boost
                        else:
                            if relative_size < medium_threshold:
                                size_boost = medium_boost
                            else:
                                size_boost = 1.0
                    else:
                        size_boost = 1.0
                    final_size_score = relative_size * size_boost
                    target_obj = {
                        "pos": (
                            self.identify_rect_left + result_center_x,
                            self.identify_rect_top
                            + (result_center_y - height / 2)
                            + max(
                                height * aim_position,
                                self.pressed_key_config["min_position_offset"],
                            ),
                        ),
                        "box": (
                            result_center_x - width / 2,
                            result_center_y - height / 2,
                            width,
                            height,
                        ),
                        "size": final_size_score,
                        "absolute_size": absolute_area,
                        "relative_size": relative_size,
                        "class_id": class_id,
                        "aim_position": aim_position,
                        "id": f"{int(self.identify_rect_left + result_center_x)}_{int(self.identify_rect_top + (result_center_y - height / 2) + max(height * aim_position, self.pressed_key_config['min_position_offset']))}",
                    }
                    target_objects.append(target_obj)
                if self.config["auto_flashbang"]["enabled"]:
                    original_count = len(target_objects)
                    target_objects = [t for t in target_objects if t["class_id"] != 4]
                    filtered_count = original_count - len(target_objects)
                try:
                    aim_bot_scope = float(self.get_dynamic_aim_scope())
                except Exception:
                    aim_bot_scope = 0
                if (
                    self.config["small_target_enhancement"]["enabled"]
                    and self.config["small_target_enhancement"]["smooth_enabled"]
                ):
                    smoothed_targets = self.smooth_small_targets(target_objects)
                else:
                    smoothed_targets = target_objects
                nearest = self.select_target_by_priority(smoothed_targets)
                if nearest is not None:
                    result_center_x = nearest["pos"][0] - self.screen_center_x
                    result_center_y = nearest["pos"][1] - self.screen_center_y
                    if self.aim_key_status:
                        controller = self.pressed_key_config.get(
                            "aim_controller",
                            self.config.get("aim_controller", "pid"),
                        )
                        if controller == "sunone" and hasattr(self, "sunone_aim"):
                            settings = self.config.get("sunone", {})
                            if hasattr(self, "engine") and self.engine:
                                region_w = self.engine.get_input_shape()[3]
                                region_h = self.engine.get_input_shape()[2]
                            else:
                                region_w = self.screen_width
                                region_h = self.screen_height
                            dx, dy = self.sunone_aim.compute_move(
                                nearest["pos"][0],
                                nearest["pos"][1],
                                self.screen_center_x,
                                self.screen_center_y,
                                region_w,
                                region_h,
                                max(self.fps, 1.0),
                                self.last_infer_time_ms,
                                settings,
                            )
                            relative_move_x, relative_move_y = (dx, dy)
                        elif controller == "sunone":
                            return
                        else:
                            relative_move_x, relative_move_y = self.dual_pid.compute(
                                result_center_x, result_center_y
                            )
                        current_key = self.old_pressed_aim_key
                        if (
                            current_key in self.aim_keys_dist
                            and self.aim_keys_dist[current_key].get("auto_y", False)
                            and self.left_pressed_long
                        ):
                            relative_move_y = 0
                        move_threshold = self.pressed_key_config.get(
                            "move_deadzone", 1.0
                        )
                        if (
                            abs(relative_move_x) > move_threshold
                            or abs(relative_move_y) > move_threshold
                        ):
                            self.execute_move(relative_move_x, relative_move_y)
                else:
                    if hasattr(self, "sunone_aim"):
                        self.sunone_aim.reset_if_stale()
            else:
                if hasattr(self, "sunone_aim"):
                    self.sunone_aim.reset_if_stale()

    def execute_move(self, relative_move_x, relative_move_y):
        if self.config.get("use_async_move", False):
            move_thread = Thread(
                target=self._execute_move_async, args=(relative_move_x, relative_move_y)
            )
            move_thread.daemon = True
            move_thread.start()
        else:
            self._execute_move_async(relative_move_x, relative_move_y)

    def _execute_move_async(self, relative_move_x, relative_move_y):
        if self.config["is_curve"]:
            curve = HumanCurve(
                (0, 0),
                (round(relative_move_x), round(relative_move_y)),
                offsetBoundaryX=self.config["offset_boundary_x"],
                offsetBoundaryY=self.config["offset_boundary_y"],
                knotsCount=self.config["knots_count"],
                distortionMean=self.config["distortion_mean"],
                distortionStdev=self.config["distortion_st_dev"],
                distortionFrequency=self.config["distortion_frequency"],
                targetPoints=self.config["target_points"],
            )
            curve = curve.points
            if isinstance(curve, tuple):
                self.move_r(round(relative_move_x), round(relative_move_y))
            else:
                if self.config["is_show_curve"]:
                    print(f"Curve points: {len(curve)}")
                for i in range(1, len(curve)):
                    x = round(curve[i][0] - curve[i - 1][0])
                    y = round(curve[i][1] - curve[i - 1][1])
                    if x == 0 and y == 0:
                        continue
                    self.move_r(round(x), round(y))
        else:
            if self.config["is_curve_uniform"] and self.AimController.is_uniform_motion(
                self.config["show_motion_speed"]
            ):
                curve = HumanCurve(
                    (0, 0),
                    (round(relative_move_x), round(relative_move_y)),
                    offsetBoundaryX=self.config["offset_boundary_x"],
                    offsetBoundaryY=self.config["offset_boundary_y"],
                    knotsCount=self.config["knots_count"],
                    distortionMean=self.config["distortion_mean"],
                    distortionStdev=self.config["distortion_st_dev"],
                    distortionFrequency=self.config["distortion_frequency"],
                    targetPoints=self.config["target_points"],
                )
                curve = curve.points
                if isinstance(curve, tuple):
                    self.move_r(round(relative_move_x), round(relative_move_y))
                else:
                    if self.config["is_show_curve"]:
                        print(f"Curve points: {len(curve)}")
                    for i in range(1, len(curve)):
                        x = round(curve[i][0] - curve[i - 1][0])
                        y = round(curve[i][1] - curve[i - 1][1])
                        self.move_r(round(x), round(y))
            else:
                self.move_r(round(relative_move_x), round(relative_move_y))

    def infer(self):
        import numpy as np

        self.time_begin_period(1)
        if self.engine is None:
            model_path = self.config["groups"][self.group]["infer_model"]
            while self.engine is None and (not self.end):
                time.sleep(0.1)
                if self.end:
                    return
            if self.engine is None:
                return
        group_cfg = self.config["groups"][self.group]
        use_sunone_processing = group_cfg.get("use_sunone_processing", False)
        yolo_format = group_cfg.get("yolo_format", "auto")
        yolo_version = str(
            group_cfg.get("yolo_version", group_cfg.get("sunone_model_variant", "yolo11"))
        ).strip().lower()
        if yolo_version and not yolo_version.startswith("yolo"):
            yolo_version = f"yolo{yolo_version}"
        v8_like_versions = {"yolo8", "yolo9", "yolo10", "yolo11", "yolo12"}
        if not use_sunone_processing and yolo_version in v8_like_versions:
            yolo_format = "v8"
        v8_count = self.engine.get_class_num_v8()
        v5_count = self.engine.get_class_num()
        if yolo_format == "v8":
            class_num = v8_count
            is_v8_like = True
        elif yolo_format == "v5":
            class_num = v5_count
            is_v8_like = False
        else:
            if v8_count > 1000 and v5_count > 0:
                class_num = v5_count
                is_v8_like = False
            elif v5_count > 1000 and v8_count > 0:
                class_num = v8_count
                is_v8_like = True
            elif 1 <= v8_count <= 256:
                class_num = v8_count
                is_v8_like = True
            elif 1 <= v5_count <= 256:
                class_num = v5_count
                is_v8_like = False
            else:
                class_num = v8_count or v5_count
                is_v8_like = self.config["groups"][self.group].get("is_v8", False)
        if not isinstance(class_num, int):
            class_num = int(class_num) if class_num else 0
        input_shape_weight = self.engine.get_input_shape()[3]
        input_shape_height = self.engine.get_input_shape()[2]
        print("Model input size:", input_shape_weight, input_shape_height)
        frame_count = 0
        start_time = time.perf_counter()
        last_fps_update_time = time.perf_counter()
        fps_text = "FPS: 0.00"
        self.fps = 0
        last_latency_text = "latency: 0.00ms"
        latency_values = []
        last_latency_update_time = time.perf_counter()
        display_latency_ms = 0.0
        offset_x = int(self.config.get("capture_offset_x", 0))
        offset_y = int(self.config.get("capture_offset_y", 0))
        left = (self.screen_width - input_shape_weight) // 2 + offset_x
        top = (self.screen_height - input_shape_height) // 2 + offset_y
        right = left + input_shape_weight
        bottom = top + input_shape_height
        left = max(0, min(left, self.screen_width - input_shape_weight))
        top = max(0, min(top, self.screen_height - input_shape_height))
        right = min(self.screen_width, left + input_shape_weight)
        bottom = min(self.screen_height, top + input_shape_height)
        screenshot_region = (left, top, right, bottom)
        print_fps = self.config["print_fps"]
        infer_debug = self.config["infer_debug"]
        frame_skip_ratio = self.config.get("frame_skip_ratio", 0)
        frame_skip_counter = 0
        while self.running:
            if frame_skip_ratio > 0:
                frame_skip_counter += 1
                if frame_skip_counter % (frame_skip_ratio + 1) != 0:
                    continue
            t0 = time.perf_counter()
            screenshot = self.screenshot_manager.get_screenshot(screenshot_region)
            cap_ms = (time.perf_counter() - t0) * 1000
            if screenshot is None:
                continue
            frame_count += 1
            current_fps_time = time.perf_counter()
            if current_fps_time - last_fps_update_time >= 1.0:
                time_elapsed = current_fps_time - start_time
                if time_elapsed > 0:
                    self.fps = frame_count / time_elapsed
                    fps_text = f"FPS: {self.fps:.2f}"
                    if print_fps:
                        print(fps_text)
                frame_count = 0
                start_time = current_fps_time
                last_fps_update_time = current_fps_time
            t1 = time.perf_counter()

            # Use GPU preprocessing for TensorRT engine, CPU fallback for ONNX Runtime
            img_input = None
            if hasattr(self.engine, 'infer_with_preprocess'):
                # GPU preprocessing path (TensorRT) - preprocessing included in inference
                outputs = self.engine.infer_with_preprocess(
                    screenshot, (input_shape_weight, input_shape_height)
                )
                pre_ms = 0.0  # Preprocessing is now part of inference timing
                current_infer_time_ms = (time.perf_counter() - t1) * 1000
            else:
                # CPU preprocessing fallback (ONNX Runtime)
                img_input = read_img(screenshot, (input_shape_weight, input_shape_height))
                pre_ms = (time.perf_counter() - t1) * 1000
                infer_start_time = time.perf_counter()
                outputs = self.engine.infer(img_input)
                current_infer_time_ms = (time.perf_counter() - infer_start_time) * 1000
            self.last_infer_time_ms = current_infer_time_ms
            latency_values.append(current_infer_time_ms)
            current_latency_time = time.perf_counter()
            if (
                current_latency_time - last_latency_update_time >= 1.0
                and latency_values
            ):
                avg_latency = sum(latency_values) / len(latency_values)
                display_latency_ms = avg_latency
                last_latency_text = f"latency: {avg_latency:.2f}ms"
                latency_values = []
                last_latency_update_time = current_latency_time
            infer_time_ms = display_latency_ms
            pred = outputs[0]
            if pred.ndim == 1:
                if yolo_version == "yolo10":
                    C = 6
                elif is_v8_like:
                    C = self.engine.get_class_num_v8() + 4
                else:
                    C = self.engine.get_class_num() + 5
                if pred.size % C != 0:
                    raise ValueError(
                        f"Inference output length {pred.size} cannot be divided by feature count per row {C}, please check model!"
                    )
                pred = pred.reshape((-1), C)
            class_aim_positions = self.pressed_key_config.get("class_aim_positions", {})
            if not isinstance(class_aim_positions, dict):
                class_aim_positions = {}
            min_confidence_threshold = 0.05
            class_confidence_thresholds = {}
            class_iou_thresholds = {}
            for class_str, config in class_aim_positions.items():
                if isinstance(config, dict):
                    conf_thresh = config.get("confidence_threshold", 0.5)
                    iou_thresh = config.get("iou_t", 1.0)
                    class_confidence_thresholds[int(class_str)] = conf_thresh
                    class_iou_thresholds[int(class_str)] = iou_thresh
                    min_confidence_threshold = min(
                        min_confidence_threshold, conf_thresh
                    )
            if not class_confidence_thresholds:
                confidence_threshold = self.pressed_key_config.get(
                    "confidence_threshold", 0.5
                )
                iou_t = self.pressed_key_config.get("iou_t", 1.0)
            else:
                confidence_threshold = min_confidence_threshold
                iou_t = (
                    min(class_iou_thresholds.values()) if class_iou_thresholds else 1.0
                )
            t3 = time.perf_counter()
            group_config = self.config["groups"][self.group]
            use_sunone_processing = group_config.get("use_sunone_processing", False)
            if use_sunone_processing:
                sunone_variant = group_config.get(
                    "yolo_version", group_config.get("sunone_model_variant", "yolo11")
                )
                boxes, scores, classes = sunone_postprocess(
                    pred,
                    confidence_threshold,
                    iou_t,
                    1.0,
                    self.engine.get_class_num(),
                    self.config["small_target_enhancement"]["adaptive_nms"],
                    self.config.get("sunone_max_detections", 0),
                    variant=sunone_variant,
                )
                is_v8_like = True
                nms_algo = "v8"
            else:
                engine_nms_handled = False
                if yolo_version == "yolo10":
                    boxes, scores, classes = sunone_postprocess(
                        pred,
                        confidence_threshold,
                        iou_t,
                        1.0,
                        self.engine.get_class_num(),
                        self.config["small_target_enhancement"]["adaptive_nms"],
                        self.config.get("sunone_max_detections", 0),
                        variant="yolo10",
                    )
                    is_v8_like = True
                    nms_algo = "v8"
                    engine_nms_handled = True
                else:
                    nms_algo = yolo_format
                    if nms_algo == "auto":
                        if pred.ndim == 3 and pred.shape[1] != pred.shape[2]:
                            nms_algo = "v8" if pred.shape[1] < pred.shape[2] else "v5"
                        else:
                            nms_algo = "v8" if is_v8_like else "v5"
                    use_engine_nms = nms_algo in {"v5", "standard"}
                    if (
                        use_engine_nms
                        and img_input is not None
                        and hasattr(self.engine, "infer_with_nms")
                    ):
                        try:
                            boxes, scores, classes = self.engine.infer_with_nms(
                                img_input, confidence_threshold, iou_t, nms_algo
                            )
                            engine_nms_handled = True
                        except Exception as e:
                            print(
                                f"[NMS] Engine NMS failed: {e}, falling back to manual NMS"
                            )
                    if not engine_nms_handled and nms_algo == "v8":
                        adaptive_nms_enabled = (
                            self.config["small_target_enhancement"]["enabled"]
                            and self.config["small_target_enhancement"]["adaptive_nms"]
                        )
                        boxes, scores, classes = nms_v8(
                            pred, confidence_threshold, iou_t, adaptive_nms_enabled
                        )
                        is_v8_like = True
                    elif not engine_nms_handled:
                        boxes, scores, classes = nms(
                            pred, confidence_threshold, iou_t, class_num
                        )
                        is_v8_like = False
            post_ms = (time.perf_counter() - t3) * 1000
            total_ms = cap_ms + pre_ms + current_infer_time_ms + post_ms

            # Debug profiling output every 30 frames
            if infer_debug and frame_count % 30 == 0:
                print(f"[PROFILE] cap:{cap_ms:.2f}ms pre:{pre_ms:.2f}ms infer:{current_infer_time_ms:.2f}ms post:{post_ms:.2f}ms total:{total_ms:.2f}ms")

            current_selected_classes = self.pressed_key_config.get("classes", [])
            selected_classes_set = (
                set(current_selected_classes) if current_selected_classes else set()
            )
            class_ids = []
            if len(boxes) > 0:
                if nms_algo == "v8":
                    all_class_ids = np.array(classes).astype(int)
                else:
                    all_class_ids = np.argmax(classes, axis=1).astype(int)
                if selected_classes_set:
                    mask = np.array(
                        [cls_id in selected_classes_set for cls_id in all_class_ids],
                        dtype=bool,
                    )
                    boxes = boxes[mask]
                    scores = scores[mask]
                    classes = classes[mask]
                    class_ids = all_class_ids[mask].tolist()
                    if class_confidence_thresholds and len(boxes) > 0:
                        confidence_mask = []
                        for i, cls_id in enumerate(class_ids):
                            cls_conf_thresh = class_confidence_thresholds.get(
                                cls_id, 0.5
                            )
                            confidence_mask.append(scores[i] >= cls_conf_thresh)
                        if confidence_mask:
                            confidence_mask = np.array(confidence_mask, dtype=bool)
                            boxes = boxes[confidence_mask]
                            scores = scores[confidence_mask]
                            classes = classes[confidence_mask]
                            class_ids = [
                                class_ids[i]
                                for i, keep in enumerate(confidence_mask)
                                if keep
                            ]
                else:
                    class_ids = all_class_ids.tolist()
            if (
                self.config["auto_flashbang"]["enabled"]
                and len(boxes) > 0
                and (len(class_ids) > 0)
            ):
                if self.is_using_dopa_model():
                    self.detect_and_handle_flashbang(
                        boxes, class_ids, input_shape_weight, input_shape_height, scores
                    )
                else:
                    if 4 in class_ids and (not hasattr(self, "_dopa_warning_shown")):
                        print(
                            "Auto flashbang feature only supports ZTX models, current model not supported"
                        )
                        self._dopa_warning_shown = True
            else:
                if self.config["auto_flashbang"]["enabled"]:
                    if len(class_ids) > 0 and 4 in class_ids:
                        if self.is_using_dopa_model():
                            print(
                                "Detected class 4, but boxes is empty or filtered empty"
                            )
                        else:
                            if not hasattr(self, "_dopa_warning_shown"):
                                print("Auto flashbang feature only supports ZTX models")
                                self._dopa_warning_shown = True
            if len(boxes) > 0:
                if self.aim_key_status:
                    try:
                        self.que_aim.put_nowait((boxes, class_ids))
                    except queue.Full:
                        try:
                            self.que_aim.get_nowait()
                        except queue.Empty:
                            pass
                        self.que_aim.put_nowait((boxes, class_ids))
                trigger_payload = None
                trigger_key = group_config.get("triggerbot_button_key", "")
                targeting_key = group_config.get("targeting_button_key", "")
                triggerbot_active = getattr(self, "triggerbot_key_status", False)
                if triggerbot_active:
                    trigger_cfg = group_config.get("aim_keys", {}).get(
                        targeting_key, self.pressed_key_config
                    )
                    trigger_payload = ("triggerbot", boxes, trigger_cfg)
                elif self.aim_key_status:
                    trigger_cfg = self.pressed_key_config
                    if trigger_cfg.get("trigger", {}).get("status", False):
                        trigger_payload = ("aim", boxes, trigger_cfg)
                if trigger_payload:
                    try:
                        self.que_trigger.put_nowait(trigger_payload)
                    except queue.Full:
                        try:
                            self.que_trigger.get_nowait()
                        except queue.Empty:
                            pass
                        self.que_trigger.put_nowait(trigger_payload)
            if infer_debug and self.screenshot_manager and (frame_count % 3 == 0):
                if self.aim_key_status:
                    current_key = self.old_pressed_aim_key
                else:
                    current_key = (
                        self.select_key
                        if hasattr(self, "select_key") and self.select_key
                        else self.old_pressed_aim_key
                    )
                current_scope = 0
                if (
                    current_key
                    and current_key in self.config["groups"][self.group]["aim_keys"]
                ):
                    current_scope = self.get_dynamic_aim_scope()
                display_boxes, display_scores, display_classes = (
                    boxes,
                    scores,
                    classes,
                )
                is_v8 = is_v8_like
                current_move_deadzone = self.pressed_key_config.get(
                    "move_deadzone", 1.0
                )
                current_smooth_deadzone = self.pressed_key_config.get(
                    "smooth_deadzone", 0.0
                )
                sunone_debug = None
                try:
                    controller = self.pressed_key_config.get(
                        "aim_controller",
                        self.config.get("aim_controller", "pid"),
                    )
                    debug_cfg = self.config.get("sunone", {}).get("debug", {})
                    if (
                        controller == "sunone"
                        and (
                            debug_cfg.get("show_prediction")
                            or debug_cfg.get("show_step")
                            or debug_cfg.get("show_future")
                        )
                        and hasattr(self, "sunone_aim")
                    ):
                        pred, step, future = self.sunone_aim.get_debug_snapshot()

                        def _to_local(pt):
                            if not pt:
                                return None
                            return (
                                pt[0] - self.identify_rect_left,
                                pt[1] - self.identify_rect_top,
                            )

                        pred_local = _to_local(pred) if debug_cfg.get("show_prediction") else None
                        step_local = _to_local(step) if debug_cfg.get("show_step") else None
                        future_local = []
                        if debug_cfg.get("show_future") and future:
                            for item in future:
                                future_local.append(
                                    (
                                        item[0] - self.identify_rect_left,
                                        item[1] - self.identify_rect_top,
                                    )
                                )
                        sunone_debug = {
                            "prediction": pred_local,
                            "step": step_local,
                            "future": future_local,
                        }
                except Exception:
                    sunone_debug = None
                self.screenshot_manager.put_screenshot_result(
                    screenshot,
                    display_boxes,
                    display_scores,
                    display_classes,
                    fps_text,
                    infer_time_ms,
                    current_key,
                    current_scope,
                    self.aim_key_status,
                    is_v8,
                    current_move_deadzone,
                    current_smooth_deadzone,
                    sunone_debug,
                )
                self.showed = True
        self.time_end_period(1)

    def _update_dynamic_aim_scope(self):
        # Based on lock status and switch delay, linearly adjust aim scope.
        now_ms = time.time() * 1000
        cfg = self.pressed_key_config
        base_scope = float(cfg.get("aim_bot_scope", 0) or 0)
        if base_scope <= 0:
            self._dynamic_scope["value"] = 0
            self._dynamic_scope["phase"] = "idle"
            self._dynamic_scope["last_ms"] = now_ms
            return 0
        dyn_cfg = cfg.get("dynamic_scope", {}) or {}
        enabled = bool(dyn_cfg.get("enabled", False))
        min_ratio = dyn_cfg.get("min_ratio", None)
        if min_ratio is not None:
            try:
                min_ratio = float(min_ratio)
            except Exception:
                min_ratio = 0.5
            min_scope = base_scope * max(0, min(1, float(min_ratio)))
        else:
            try:
                min_scope = float(dyn_cfg.get("min_scope", base_scope))
            except Exception:
                min_scope = base_scope
        shrink_ms = int(dyn_cfg.get("shrink_duration_ms", 300))
        recover_ms = int(dyn_cfg.get("recover_duration_ms", 300))
        lock_active = self.last_target_count > 0 and not self.is_waiting_for_switch
        if self._dynamic_scope_lock_active_prev and not lock_active:
            if not (cfg.get("target_switch_delay", 0) and self.is_waiting_for_switch):
                self._dynamic_scope["phase"] = "recover"
                self._dynamic_scope["last_ms"] = now_ms
        if not self._dynamic_scope_lock_active_prev and lock_active:
            self._dynamic_scope["phase"] = "shrink"
            self._dynamic_scope["last_ms"] = now_ms
        self._dynamic_scope_lock_active_prev = lock_active
        if not enabled:
            self._dynamic_scope["value"] = base_scope
            self._dynamic_scope["phase"] = "idle"
            self._dynamic_scope["last_ms"] = now_ms
            return base_scope
        phase = self._dynamic_scope["phase"]
        elapsed = max(0, now_ms - self._dynamic_scope["last_ms"])
        if phase == "shrink":
            if shrink_ms <= 0:
                val = min_scope
            else:
                t = max(0, min(1, elapsed / float(shrink_ms)))
                val = base_scope + (min_scope - base_scope) * t
            self._dynamic_scope["value"] = val
            if elapsed >= shrink_ms:
                self._dynamic_scope["phase"] = "hold"
                self._dynamic_scope["last_ms"] = now_ms
        elif phase == "hold":
            self._dynamic_scope["value"] = min_scope
        elif phase == "recover":
            if recover_ms <= 0:
                val = base_scope
            else:
                t = max(0, min(1, elapsed / float(recover_ms)))
                val = min_scope + (base_scope - min_scope) * t
            self._dynamic_scope["value"] = val
            if elapsed >= recover_ms:
                self._dynamic_scope["phase"] = "idle"
                self._dynamic_scope["last_ms"] = now_ms
        else:
            self._dynamic_scope["value"] = base_scope
            self._dynamic_scope["phase"] = "idle"
            self._dynamic_scope["last_ms"] = now_ms
        return float(self._dynamic_scope["value"])

    def get_dynamic_aim_scope(self):
        """Get the aim scope (in pixels) to use for the current frame."""
        try:
            return float(self.pressed_key_config.get("aim_bot_scope", 0) or 0)
        except Exception:
            return 0.0

    def reset_dynamic_aim_scope(self, for_key=None):
        """When dynamic aim scope is enabled, reset current scope to base scope for specified key.

        Args:
            for_key: Specified key name; if omitted, uses currently active key.
        """
        try:
            key = for_key or (
                self.old_pressed_aim_key
                if self.old_pressed_aim_key
                else self.select_key
            )
            key_cfg = self.config["groups"][self.group]["aim_keys"].get(
                key, self.pressed_key_config
            )
        except Exception:
            key_cfg = self.pressed_key_config
        dyn_cfg = key_cfg.get("dynamic_scope") or {}
        if not bool(dyn_cfg.get("enabled", False)):
            return
        try:
            base_scope = float(key_cfg.get("aim_bot_scope", 0) or 0)
        except Exception:
            base_scope = 0.0
        self._dynamic_scope["value"] = base_scope
        self._dynamic_scope["phase"] = "idle"
        self._dynamic_scope["last_ms"] = time.time() * 1000.0

    def mouse_left_down(self):
        if self.config["move_method"] == "makcu" and self.makcu is not None:
            self.makcu.press(MouseButton.LEFT)

    def mouse_left_up(self):
        if self.config["move_method"] == "makcu" and self.makcu is not None:
            self.makcu.release(MouseButton.LEFT)

    def _force_reset_input_states(self, reason=""):
        try:
            print(f"[Safety] Resetting input states: {reason}")
        except Exception:
            pass
        try:
            self.trigger_status = False
        except Exception:
            pass
        try:
            self.continuous_trigger_active = False
        except Exception:
            pass
        try:
            self.triggerbot_key_status = False
            self.triggerbot_key = ""
            self.triggerbot_key_config = None
        except Exception:
            pass
        try:
            self.aim_key_status = False
            self.old_pressed_aim_key = ""
        except Exception:
            pass
        if hasattr(self, "stop_continuous_trigger"):
            try:
                self.stop_continuous_trigger()
            except Exception:
                pass
        if hasattr(self, "stop_trigger_recoil"):
            try:
                self.stop_trigger_recoil()
            except Exception:
                pass
        if hasattr(self, "reset_target_lock"):
            try:
                self.reset_target_lock(self.old_pressed_aim_key)
            except Exception:
                pass
        if hasattr(self, "reset_pid"):
            try:
                self.reset_pid()
            except Exception:
                pass

    def _safe_mouse_left_down(self):
        try:
            self.mouse_left_down()
            return True
        except Exception as e:
            print(f"[Trigger] mouse_left_down failed: {e}")
            self._force_reset_input_states("mouse_left_down error")
            return False

    def _safe_mouse_left_up(self):
        try:
            self.mouse_left_up()
            return True
        except Exception as e:
            print(f"[Trigger] mouse_left_up failed: {e}")
            try:
                if self.config.get("move_method") == "makcu" and self.makcu is not None:
                    self.makcu.unlock(MouseButton.LEFT)
            except Exception as unlock_e:
                print(f"[Trigger] mouse_left_up unlock failed: {unlock_e}")
            self._force_reset_input_states("mouse_left_up error")
            return False

    def _is_trigger_source_active(self, source):
        if source == "triggerbot":
            return bool(getattr(self, "triggerbot_key_status", False))
        return bool(self.aim_key_status)

    def trigger_process(
        self,
        start_delay=0,
        press_delay=1,
        end_delay=0,
        random_delay=0,
        recoil_enabled=False,
    ):
        self.time_begin_period(1)
        pressed = False
        try:
            if start_delay > 0:
                if random_delay > 0:
                    start_delay = random.randint(
                        max(0, start_delay - random_delay), start_delay + random_delay
                    )
                time.sleep(start_delay / 1000)
            pressed = self._safe_mouse_left_down()
            if press_delay > 0:
                if random_delay > 0:
                    press_delay = random.randint(
                        max(0, press_delay - random_delay), press_delay + random_delay
                    )
                time.sleep(press_delay / 1000)
        finally:
            if pressed:
                self._safe_mouse_left_up()
            else:
                self._safe_mouse_left_up()
            self.trigger_status = False
        if end_delay > 0:
            if random_delay > 0:
                end_delay = random.randint(
                    max(0, end_delay - random_delay), end_delay + random_delay
                )
            time.sleep(end_delay / 1000)

    def continuous_trigger_process(self, trigger_cfg, recoil_enabled=False, source="aim"):
        """Continuous trigger process - keep firing until key released"""
        self.time_begin_period(1)
        start_delay = trigger_cfg["trigger"]["start_delay"]
        random_delay = trigger_cfg["trigger"]["random_delay"]
        if start_delay > 0:
            if random_delay > 0:
                actual_start_delay = random.randint(
                    max(0, start_delay - random_delay), start_delay + random_delay
                )
            else:
                actual_start_delay = start_delay
            time.sleep(actual_start_delay / 1000)
        pressed = self._safe_mouse_left_down()
        if not pressed:
            self.continuous_trigger_active = False
            return
        try:
            while self._is_trigger_source_active(source) and self.continuous_trigger_active:
                time.sleep(0.01)
        finally:
            self._safe_mouse_left_up()
            self.continuous_trigger_active = False

    def stop_continuous_trigger(self):
        """Stop continuous trigger"""
        if self.continuous_trigger_active:
            self.continuous_trigger_active = False

    def start_trigger_recoil(self):
        """Trigger recoil removed."""
        self.trigger_recoil_active = False
        self.trigger_recoil_pressed = False

    def stop_trigger_recoil(self):
        """Trigger recoil removed."""
        self.trigger_recoil_active = False
        self.trigger_recoil_pressed = False
        if hasattr(self, "timer_id2") and self.timer_id2:
            self.time_kill_event(self.timer_id2)
            self.timer_id2 = 0

    def trigger(self):
        self.time_begin_period(1)
        input_shape_weight = self.engine.get_input_shape()[3]
        input_shape_height = self.engine.get_input_shape()[2]
        identify_rect_left = self.screen_center_x - input_shape_weight / 2
        identify_rect_top = self.screen_center_y - input_shape_height / 2
        last_check_time = time.perf_counter()
        check_interval = 0.002
        while self.running:
            current_time = time.perf_counter()
            if current_time - last_check_time < check_interval:
                time.sleep(0.001)
                continue
            last_check_time = current_time
            try:
                payload = self.que_trigger.get_nowait()
            except queue.Empty:
                payload = None
            if payload:
                if (
                    isinstance(payload, tuple)
                    and len(payload) == 3
                    and isinstance(payload[2], dict)
                ):
                    trigger_mode, aim_targets, trigger_cfg = payload
                else:
                    trigger_mode = "aim"
                    aim_targets = payload
                    trigger_cfg = self.pressed_key_config
                if not self._is_trigger_source_active(trigger_mode):
                    continue
                for item in aim_targets:
                    result_center_x, result_center_y, width, height = item
                    x_trigger_offset = trigger_cfg["trigger"][
                        "x_trigger_offset"
                    ]
                    y_trigger_offset = trigger_cfg["trigger"][
                        "y_trigger_offset"
                    ]
                    x_trigger_scope = trigger_cfg["trigger"][
                        "x_trigger_scope"
                    ]
                    y_trigger_scope = trigger_cfg["trigger"][
                        "y_trigger_scope"
                    ]
                    left = result_center_x - width / 2
                    top = result_center_y - height / 2
                    left = left + width * x_trigger_offset
                    top = top + height * y_trigger_offset
                    width = width * x_trigger_scope
                    height = height * y_trigger_scope
                    right = left + width
                    bottom = top + height
                    relative_screen_top = identify_rect_top + round(top, 2)
                    relative_screen_left = identify_rect_left + round(left, 2)
                    relative_screen_bottom = identify_rect_top + round(bottom, 2)
                    relative_screen_right = identify_rect_left + round(right, 2)
                    if (
                        relative_screen_left
                        < self.screen_center_x
                        < relative_screen_right
                        and relative_screen_top
                        < self.screen_center_y
                        < relative_screen_bottom
                    ):
                        continuous_enabled = trigger_cfg["trigger"].get("continuous", False)
                        key_active = self._is_trigger_source_active(trigger_mode)
                        if continuous_enabled:
                            if (
                                not self.continuous_trigger_active
                                and key_active
                            ):
                                self.continuous_trigger_active = True
                                self.continuous_trigger_thread = Thread(
                                    target=self.continuous_trigger_process,
                                    args=(trigger_cfg, False, trigger_mode),
                                )
                                self.continuous_trigger_thread.daemon = True
                                self.continuous_trigger_thread.start()
                        else:
                            if not self.trigger_status and key_active:
                                self.trigger_status = True
                                Thread(
                                    target=self.trigger_process,
                                    args=(
                                        trigger_cfg["trigger"]["start_delay"],
                                        trigger_cfg["trigger"]["press_delay"],
                                        trigger_cfg["trigger"]["end_delay"],
                                        trigger_cfg["trigger"]["random_delay"],
                                        False,
                                    ),
                                ).start()
                        break

    def reset_pid(self):
        self.dual_pid.reset()
        self.last_target_count = 0
        self.last_target_count_by_class.clear()
        self.is_waiting_for_switch = False
        self.target_switch_time = 0
        if not getattr(self, "triggerbot_key_status", False):
            self.stop_continuous_trigger()
            self.stop_trigger_recoil()
        if hasattr(self, "sunone_aim"):
            self.sunone_aim.reset()
