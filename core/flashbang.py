import math
import random
import threading
import time

import dearpygui.dearpygui as dpg
from pyclick import HumanCurve


class FlashbangMixin:
    """Mixin class for auto-flashbang feature methods."""

    def on_auto_flashbang_enabled_change(self, sender, app_data):
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            self.config["auto_flashbang"]["enabled"] = False
            try:
                import dearpygui.dearpygui as dpg

                if dpg.does_item_exist("auto_flashbang_enabled_checkbox"):
                    dpg.set_value("auto_flashbang_enabled_checkbox", False)
            except:
                return None
        else:
            self.config["auto_flashbang"]["enabled"] = app_data
            print(f"Auto flashbang {('enabled' if app_data else 'disabled')}")

    def on_auto_flashbang_delay_change(self, sender, app_data):
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            return
        self.config["auto_flashbang"]["delay_ms"] = app_data
        print(f"Flashbang delay set to: {app_data}ms")

    def on_auto_flashbang_angle_change(self, sender, app_data):
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            return
        self.config["auto_flashbang"]["turn_angle"] = app_data
        print(f"Flashbang turn angle set to: {app_data} degrees")

    def on_auto_flashbang_sensitivity_change(self, sender, app_data):
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            return
        self.config["auto_flashbang"]["sensitivity_multiplier"] = app_data
        print(f"Flashbang sensitivity multiplier set to: {app_data}")

    def on_auto_flashbang_return_delay_change(self, sender, app_data):
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            return
        self.config["auto_flashbang"]["return_delay"] = app_data
        print(f"Flashbang return delay set to: {app_data}ms")

    def on_test_flashbang_left(self, sender, app_data):
        """Test left turn flashbang"""
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            return
        print("Testing left turn flashbang...")
        self.execute_flashbang_turn((-1))

    def on_test_flashbang_right(self, sender, app_data):
        """Test right turn flashbang"""
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            return
        print("Testing right turn flashbang...")
        self.execute_flashbang_turn(1)

    def on_auto_flashbang_curve_change(self, sender, app_data):
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            return
        self.config["auto_flashbang"]["use_curve"] = app_data
        print(f"Flashbang curve movement {('enabled' if app_data else 'disabled')}")

    def on_auto_flashbang_curve_speed_change(self, sender, app_data):
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            return
        self.config["auto_flashbang"]["curve_speed"] = app_data
        print(f"Flashbang curve speed set to: {app_data}")

    def on_auto_flashbang_curve_knots_change(self, sender, app_data):
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            return
        self.config["auto_flashbang"]["curve_knots"] = app_data
        print(f"Flashbang curve control points set to: {app_data}")

    def on_auto_flashbang_min_confidence_change(self, sender, app_data):
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            return
        self.config["auto_flashbang"]["min_confidence"] = app_data
        print(f"Flashbang minimum confidence set to: {app_data}")

    def on_auto_flashbang_min_size_change(self, sender, app_data):
        if not self.is_using_dopa_model():
            print(
                "Auto flashbang feature only supports ZTX models, current model not supported"
            )
            return
        self.config["auto_flashbang"]["min_size"] = app_data
        print(f"Flashbang minimum size set to: {app_data} pixels")

    def on_flashbang_debug_info(self, sender, app_data):
        """Display auto-flashbang debug information"""
        import time

        print("=== Auto Flashbang Debug Info ===")
        print(
            f"Feature status: {('enabled' if self.config['auto_flashbang']['enabled'] else 'disabled')}"
        )
        dopa_status = self.is_using_dopa_model()
        print(
            f"ZTX model status: {('loaded' if dopa_status else 'not used/not loaded')}"
        )
        if hasattr(self, "group") and self.group:
            current_model = self.config["groups"][self.group].get("infer_model", "")
        print(
            f"Feature availability: {('available' if self.config['auto_flashbang']['enabled'] and dopa_status else 'unavailable')}"
        )
        print("Configuration parameters:")
        for key, value in self.config["auto_flashbang"].items():
            print(f"  {key}: {value}")
        print(
            f"Last trigger time: {time.time() - self.last_flashbang_time:.1f} seconds ago"
        )
        print(f"Cooldown time: {self.flashbang_cooldown} seconds")
        print(f"Is turning back: {self.is_turning_back}")
        if hasattr(self, "group") and self.group:
            current_model = self.config["groups"][self.group].get("infer_model", "")
            print(f"Current model: {current_model}")
        print("=====================")

    def update_auto_flashbang_ui_state(self):
        """Update enable/disable state of auto-flashbang UI controls"""
        try:
            import dearpygui.dearpygui as dpg

            is_dopa = self.is_using_dopa_model()
            auto_flashbang_controls = [
                "auto_flashbang_enabled_checkbox",
                "auto_flashbang_delay_input",
                "auto_flashbang_angle_input",
                "auto_flashbang_sensitivity_input",
                "auto_flashbang_return_delay_input",
                "auto_flashbang_curve_checkbox",
                "auto_flashbang_curve_speed_input",
                "auto_flashbang_curve_knots_input",
                "auto_flashbang_min_confidence_input",
                "auto_flashbang_min_size_input",
                "auto_flashbang_test_left_button",
                "auto_flashbang_test_right_button",
                "auto_flashbang_debug_button",
            ]
            for control_tag in auto_flashbang_controls:
                if dpg.does_item_exist(control_tag):
                    dpg.configure_item(control_tag, enabled=is_dopa)
            if not is_dopa:
                if self.config["auto_flashbang"]["enabled"]:
                    self.config["auto_flashbang"]["enabled"] = False
                    if dpg.does_item_exist("auto_flashbang_enabled_checkbox"):
                        dpg.set_value("auto_flashbang_enabled_checkbox", False)
        except Exception as e:
            print(f"Error updating auto flashbang UI state: {e}")

    def detect_and_handle_flashbang(
        self, boxes, class_ids, model_width, model_height, scores=None
    ):
        """
        Detect flashbang and execute evasion action

        Args:
            boxes: Detection box array
            class_ids: Class ID array
            model_width: Model input width
            model_height: Model input height
            scores: Confidence score array (optional)
        """
        import threading
        import time

        current_time = time.time()
        if current_time - self.last_flashbang_time < self.flashbang_cooldown:
            return
        min_confidence = self.config["auto_flashbang"]["min_confidence"]
        min_size = self.config["auto_flashbang"]["min_size"]
        flashbang_indices = []
        class4_detected = False
        for i, class_id in enumerate(class_ids):
            if class_id == 4:
                class4_detected = True
                current_confidence = scores[i] if scores is not None else 1.0
                box = boxes[i]
                print(f"Original box data: {box}")
                try:
                    x1, y1, x2, y2 = (box[0], box[1], box[2], box[3])
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    min_box_size = min(width, height)
                    if width == 0 or height == 0:
                        print(
                            f"  Warning: Abnormal detection box size width={width}, height={height}"
                        )
                        continue
                except (IndexError, TypeError) as e:
                    print(f"  Error: Abnormal box data format {e}")
                    continue
                print(
                    f"Found class 4: confidence={current_confidence:.3f}, size={width:.1f}x{height:.1f}, min_edge={min_box_size:.1f}"
                )
                if scores is not None and scores[i] < min_confidence:
                    print(
                        f"  Skip: confidence {current_confidence:.3f} < {min_confidence}"
                    )
                else:
                    if min_box_size < min_size:
                        print(
                            f"  Skip: size {min_box_size:.1f} < {min_size} (distant flashbang may need to lower this threshold)"
                        )
                    else:
                        print("  Passed check, adding to flashbang list")
                        flashbang_indices.append(i)
        if class4_detected and (not flashbang_indices):
            print(
                "Detected class 4 but all filtered - suggest lowering minimum confidence or size threshold"
            )
        if not flashbang_indices:
            return
        print(
            f"Detected {len(flashbang_indices)} valid flashbangs, preparing to execute evasion"
        )
        left_count = 0
        right_count = 0
        center_x = model_width / 2
        for idx in flashbang_indices:
            box = boxes[idx]
            try:
                x1, y1, x2, y2 = (box[0], box[1], box[2], box[3])
                flashbang_center_x = (x1 + x2) / 2
                relative_pos = flashbang_center_x / model_width
                print(
                    f"Flashbang position: x={flashbang_center_x:.1f} (relative position: {relative_pos:.2f})"
                )
                if flashbang_center_x < center_x:
                    left_count += 1
                else:
                    right_count += 1
            except (IndexError, TypeError) as e:
                print(f"Error calculating flashbang position: {e}")
        if left_count > right_count:
            turn_direction = -1
            direction_text = "left"
        else:
            if right_count > left_count:
                turn_direction = 1
                direction_text = "right"
            else:
                import random

                turn_direction = random.choice([(-1), 1])
                direction_text = "left" if turn_direction == (-1) else "right"
        print(
            f"Flashbang distribution: {left_count} on left, {right_count} on right, evading to {direction_text}"
        )
        self.last_flashbang_time = current_time
        delay_ms = self.config["auto_flashbang"]["delay_ms"]
        print(f"Will execute evasion in {delay_ms}ms")
        threading.Timer(
            delay_ms / 1000.0, self.execute_flashbang_turn, args=(turn_direction,)
        ).start()

    def execute_flashbang_turn(self, turn_direction):
        """
        Execute flashbang evasion turn action

        Args:
            turn_direction: Turn direction, -1 for left, 1 for right
        """
        import threading
        import time

        if self.is_turning_back:
            return
        turn_angle = self.config["auto_flashbang"]["turn_angle"]
        actual_turn_angle = turn_angle * turn_direction
        print(f"Executing evasion: turn {actual_turn_angle} degrees")
        try:
            sensitivity = self.config["auto_flashbang"]["sensitivity_multiplier"]
            mouse_move_x = int(actual_turn_angle * sensitivity)
            print(f"Planned mouse movement distance: {mouse_move_x} pixels")
            self.flashbang_actual_move_x = mouse_move_x
            self.flashbang_actual_move_y = 0
            if self.config["auto_flashbang"]["use_curve"]:
                actual_move = self.execute_flashbang_ultra_fast_move(mouse_move_x, 0)
                if actual_move:
                    self.flashbang_actual_move_x, self.flashbang_actual_move_y = (
                        actual_move
                    )
            else:
                self.execute_move(mouse_move_x, 0)
            print(
                f"Actual movement distance: X={self.flashbang_actual_move_x}, Y={self.flashbang_actual_move_y}"
            )
            return_delay = self.config["auto_flashbang"]["return_delay"] / 1000.0
            threading.Timer(
                return_delay, self.execute_flashbang_return, args=(turn_direction,)
            ).start()
        except Exception as e:
            print(f"Error executing flashbang turn: {e}")

    def execute_flashbang_return(self, original_turn_direction):
        #
        #         Execute return action after flashbang - precise return version, ensure return to original position
        #
        #         Args:
        #             original_turn_direction: Original turn direction
        #
        import time

        if self.is_turning_back:
            return
        self.is_turning_back = True
        self.turn_back_start_time = time.time()
        try:
            try:
                return_move_x = -self.flashbang_actual_move_x
                return_move_y = -self.flashbang_actual_move_y
                print(
                    "Executing precise return: X="
                    + f"{return_move_x}"
                    + ", Y="
                    + f"{return_move_y}"
                    + " pixels (based on actual movement distance)"
                )
                if self.config["auto_flashbang"]["use_curve"]:
                    self.execute_flashbang_ultra_fast_move(return_move_x, return_move_y)
                else:
                    self.execute_move(return_move_x, return_move_y)
                self.flashbang_actual_move_x = 0
                self.flashbang_actual_move_y = 0
            except Exception as e:
                print("Error executing return: " + f"{e}")
        finally:
            time.sleep(0.05)
            self.is_turning_back = False

    def execute_flashbang_curve_move(self, relative_move_x, relative_move_y):
        """
        Execute flashbang curve movement - optimized version, more like natural human reaction
        Fast start, fast end, smooth in the middle

        Args:
            relative_move_x: X-axis relative movement distance
            relative_move_y: Y-axis relative movement distance
        """
        import math
        import random
        import time

        try:
            speed_multiplier = self.config["auto_flashbang"]["curve_speed"]
            knots_count = self.config["auto_flashbang"]["curve_knots"]
            curve = HumanCurve(
                (0, 0),
                (round(relative_move_x), round(relative_move_y)),
                offsetBoundaryX=self.config["offset_boundary_x"],
                offsetBoundaryY=self.config["offset_boundary_y"],
                knotsCount=knots_count,
                distortionMean=self.config["distortion_mean"],
                distortionStdev=self.config["distortion_st_dev"],
                distortionFrequency=self.config["distortion_frequency"],
                targetPoints=self.config["target_points"],
            )
            curve_points = curve.points
            if isinstance(curve_points, tuple):
                print("Curve generation failed, using linear movement")
                self.move_r(round(relative_move_x), round(relative_move_y))
            else:
                print(
                    f"Flashbang curve control points: {knots_count}, generated trajectory points: {len(curve_points)}"
                )
                print(f"Target movement: X={relative_move_x}, Y={relative_move_y}")
                total_distance = math.sqrt(relative_move_x**2 + relative_move_y**2)
                base_duration = 0.001
                frame_count = max(1, min(3, int(total_distance / 500)))
                if len(curve_points) < frame_count:
                    interpolated_points = []
                    for i in range(frame_count):
                        t = i / (frame_count - 1)
                        idx = t * (len(curve_points) - 1)
                        idx_floor = int(idx)
                        idx_ceil = min(idx_floor + 1, len(curve_points) - 1)
                        frac = idx - idx_floor
                        if idx_floor == idx_ceil:
                            point = curve_points[idx_floor]
                        else:
                            x = (
                                curve_points[idx_floor][0] * (1 - frac)
                                + curve_points[idx_ceil][0] * frac
                            )
                            y = (
                                curve_points[idx_floor][1] * (1 - frac)
                                + curve_points[idx_ceil][1] * frac
                            )
                            point = (x, y)
                        interpolated_points.append(point)
                    curve_points = interpolated_points

                def human_like_easing(t):
                    """
                    Human-like easing function: simulates human emergency reaction speed curve
                    - Fast acceleration at start (emergency reaction)
                    - Constant speed in middle (control phase)
                    - Fast deceleration at end (precise positioning)
                    """
                    if t < 0.15:
                        normalized_t = t / 0.15
                        return 0.4 * normalized_t**2
                    if t > 0.85:
                        normalized_t = (t - 0.85) / 0.15
                        return 0.6 + 0.4 * (1 - (1 - normalized_t) ** 2)
                    normalized_t = (t - 0.15) / 0.7
                    return 0.4 + 0.2 * normalized_t

                frame_moves = []
                total_x = 0
                total_y = 0
                for i in range(1, len(curve_points)):
                    x = curve_points[i][0] - curve_points[i - 1][0]
                    y = curve_points[i][1] - curve_points[i - 1][1]
                    if abs(x) < 0.1 and abs(y) < 0.1:
                        continue
                    frame_moves.append((x, y))
                    total_x += x
                    total_y += y
                target_x = relative_move_x
                target_y = relative_move_y
                if abs(total_x - target_x) > 1 or abs(total_y - target_y) > 1:
                    print(
                        f"Warning: Curve movement total distance mismatch, target:({target_x},{target_y}), actual:({total_x:.2f},{total_y:.2f})"
                    )
                    if len(frame_moves) > 0:
                        correction_x = target_x / total_x if total_x != 0 else 1
                        correction_y = target_y / total_y if total_y != 0 else 1
                        corrected_moves = []
                        for dx, dy in frame_moves:
                            corrected_moves.append(
                                (dx * correction_x, dy * correction_y)
                            )
                        frame_moves = corrected_moves
                total_frames = len(frame_moves)
                if total_frames == 0:
                    self.move_r(round(relative_move_x), round(relative_move_y))
                    return
                frame_time = base_duration / total_frames if total_frames > 0 else 0
                for i, (dx, dy) in enumerate(frame_moves):
                    move_x = round(dx)
                    move_y = round(dy)
                    if move_x == 0 and move_y == 0:
                        continue
                    self.move_r(move_x, move_y)
                    if i < total_frames - 1:
                        base_delay = frame_time
                        pass
                print(
                    f"Flashbang curve movement complete, total frames: {total_frames}, total time: {frame_time * total_frames:.3f}s"
                )
        except Exception as e:
            print(f"Error executing flashbang curve movement: {e}")
            self.move_r(round(relative_move_x), round(relative_move_y))

    def execute_flashbang_curve_move_fast(self, relative_move_x, relative_move_y):
        """
        Execute flashbang return fast curve movement - specifically for return, faster and more direct

        Args:
            relative_move_x: X-axis relative movement distance
            relative_move_y: Y-axis relative movement distance
        """
        import math
        import random
        import time

        try:
            speed_multiplier = self.config["auto_flashbang"]["curve_speed"] * 10
            knots_count = 1
            curve = HumanCurve(
                (0, 0),
                (round(relative_move_x), round(relative_move_y)),
                offsetBoundaryX=self.config["offset_boundary_x"],
                offsetBoundaryY=self.config["offset_boundary_y"],
                knotsCount=knots_count,
                distortionMean=self.config["distortion_mean"] * 0.7,
                distortionStdev=self.config["distortion_st_dev"] * 0.7,
                distortionFrequency=self.config["distortion_frequency"] * 0.8,
                targetPoints=self.config["target_points"],
            )
            curve_points = curve.points
            if isinstance(curve_points, tuple):
                print("Fast curve generation failed, using linear movement")
                self.move_r(round(relative_move_x), round(relative_move_y))
            else:
                print(
                    f"Return fast curve: {knots_count} control points, trajectory points: {len(curve_points)}"
                )
                total_distance = math.sqrt(relative_move_x**2 + relative_move_y**2)
                base_duration = 0
                frame_count = 1
                if len(curve_points) < frame_count:
                    interpolated_points = []
                    for i in range(frame_count):
                        t = i / (frame_count - 1)
                        idx = t * (len(curve_points) - 1)
                        idx_floor = int(idx)
                        idx_ceil = min(idx_floor + 1, len(curve_points) - 1)
                        frac = idx - idx_floor
                        if idx_floor == idx_ceil:
                            point = curve_points[idx_floor]
                        else:
                            x = (
                                curve_points[idx_floor][0] * (1 - frac)
                                + curve_points[idx_ceil][0] * frac
                            )
                            y = (
                                curve_points[idx_floor][1] * (1 - frac)
                                + curve_points[idx_ceil][1] * frac
                            )
                            point = (x, y)
                        interpolated_points.append(point)
                    curve_points = interpolated_points

                def fast_return_easing(t):
                    """Return-specific easing: fast in fast out, constant in middle"""
                    if t < 0.1:
                        return 0.5 * (t / 0.1) ** 1.5
                    if t > 0.9:
                        normalized_t = (t - 0.9) / 0.1
                        return 0.8 + 0.2 * (1 - (1 - normalized_t) ** 1.5)
                    return 0.5 + 0.3 * ((t - 0.1) / 0.8)

                frame_moves = []
                total_x = 0
                total_y = 0
                for i in range(1, len(curve_points)):
                    x = curve_points[i][0] - curve_points[i - 1][0]
                    y = curve_points[i][1] - curve_points[i - 1][1]
                    if abs(x) < 0.1 and abs(y) < 0.1:
                        continue
                    frame_moves.append((x, y))
                    total_x += x
                    total_y += y
                target_x = relative_move_x
                target_y = relative_move_y
                if (abs(total_x - target_x) > 1 or abs(total_y - target_y) > 1) and len(
                    frame_moves
                ) > 0:
                    correction_x = target_x / total_x if total_x != 0 else 1
                    correction_y = target_y / total_y if total_y != 0 else 1
                    corrected_moves = []
                    for dx, dy in frame_moves:
                        corrected_moves.append((dx * correction_x, dy * correction_y))
                    frame_moves = corrected_moves
                total_frames = len(frame_moves)
                if total_frames == 0:
                    self.move_r(round(relative_move_x), round(relative_move_y))
                    return
                frame_time = base_duration / total_frames
                for i, (dx, dy) in enumerate(frame_moves):
                    move_x = round(dx)
                    move_y = round(dy)
                    if move_x == 0 and move_y == 0:
                        continue
                    self.move_r(move_x, move_y)
                print(
                    f"Fast return complete, total frames: {total_frames}, total time: {frame_time * total_frames:.3f}s"
                )
        except Exception as e:
            print(f"Error executing fast return curve movement: {e}")
            self.move_r(round(relative_move_x), round(relative_move_y))

    def execute_flashbang_curve_move_with_tracking(
        self, relative_move_x, relative_move_y
    ):
        """
        Execute flashbang curve movement and track actual movement distance

        Args:
            relative_move_x: X-axis relative movement distance
            relative_move_y: Y-axis relative movement distance

        Returns:
            tuple: (actual_move_x, actual_move_y) actual movement distance
        """
        import math
        import random
        import time

        actual_total_x = 0
        actual_total_y = 0
        try:
            speed_multiplier = self.config["auto_flashbang"]["curve_speed"]
            knots_count = self.config["auto_flashbang"]["curve_knots"]
            curve = HumanCurve(
                (0, 0),
                (round(relative_move_x), round(relative_move_y)),
                offsetBoundaryX=self.config["offset_boundary_x"],
                offsetBoundaryY=self.config["offset_boundary_y"],
                knotsCount=knots_count,
                distortionMean=self.config["distortion_mean"],
                distortionStdev=self.config["distortion_st_dev"],
                distortionFrequency=self.config["distortion_frequency"],
                targetPoints=self.config["target_points"],
            )
            curve_points = curve.points
            if isinstance(curve_points, tuple):
                print("Curve generation failed, using linear movement")
                self.move_r(round(relative_move_x), round(relative_move_y))
                return (relative_move_x, relative_move_y)
            print(
                f"Flashbang curve control points: {knots_count}, generated trajectory points: {len(curve_points)}"
            )
            print(f"Target movement: X={relative_move_x}, Y={relative_move_y}")
            total_distance = math.sqrt(relative_move_x**2 + relative_move_y**2)
            base_duration = 0.001
            frame_count = max(1, min(3, int(total_distance / 500)))
            if len(curve_points) < frame_count:
                interpolated_points = []
                for i in range(frame_count):
                    t = i / (frame_count - 1)
                    idx = t * (len(curve_points) - 1)
                    idx_floor = int(idx)
                    idx_ceil = min(idx_floor + 1, len(curve_points) - 1)
                    frac = idx - idx_floor
                    if idx_floor == idx_ceil:
                        point = curve_points[idx_floor]
                    else:
                        x = (
                            curve_points[idx_floor][0] * (1 - frac)
                            + curve_points[idx_ceil][0] * frac
                        )
                        y = (
                            curve_points[idx_floor][1] * (1 - frac)
                            + curve_points[idx_ceil][1] * frac
                        )
                        point = (x, y)
                    interpolated_points.append(point)
                curve_points = interpolated_points

            def human_like_easing(t):
                """
                Human-like easing function: simulates human emergency reaction speed curve
                - Fast acceleration at start (emergency reaction)
                - Constant speed in middle (control phase)
                - Fast deceleration at end (precise positioning)
                """
                if t < 0.15:
                    normalized_t = t / 0.15
                    return 0.4 * normalized_t**2
                if t > 0.85:
                    normalized_t = (t - 0.85) / 0.15
                    return 0.6 + 0.4 * (1 - (1 - normalized_t) ** 2)
                normalized_t = (t - 0.15) / 0.7
                return 0.4 + 0.2 * normalized_t

            frame_moves = []
            total_x = 0
            total_y = 0
            for i in range(1, len(curve_points)):
                x = curve_points[i][0] - curve_points[i - 1][0]
                y = curve_points[i][1] - curve_points[i - 1][1]
                if abs(x) < 0.1 and abs(y) < 0.1:
                    continue
                frame_moves.append((x, y))
                total_x += x
                total_y += y
            target_x = relative_move_x
            target_y = relative_move_y
            if abs(total_x - target_x) > 1 or abs(total_y - target_y) > 1:
                print(
                    f"Warning: Curve movement total distance mismatch, target:({target_x},{target_y}), actual:({total_x:.2f},{total_y:.2f})"
                )
                if len(frame_moves) > 0:
                    correction_x = target_x / total_x if total_x != 0 else 1
                    correction_y = target_y / total_y if total_y != 0 else 1
                    corrected_moves = []
                    for dx, dy in frame_moves:
                        corrected_moves.append((dx * correction_x, dy * correction_y))
                    frame_moves = corrected_moves
            total_frames = len(frame_moves)
            if total_frames == 0:
                self.move_r(round(relative_move_x), round(relative_move_y))
                return (relative_move_x, relative_move_y)
            frame_time = base_duration / total_frames if total_frames > 0 else 0
            for i, (dx, dy) in enumerate(frame_moves):
                move_x = round(dx)
                move_y = round(dy)
                if move_x == 0 and move_y == 0:
                    continue
                self.move_r(move_x, move_y)
                actual_total_x += move_x
                actual_total_y += move_y
                if i < total_frames - 1:
                    base_delay = frame_time
                    pass
            print(
                f"Flashbang curve movement complete, total frames: {total_frames}, total time: {frame_time * total_frames:.3f}s"
            )
            print(f"Actual movement: X={actual_total_x}, Y={actual_total_y}")
            return (actual_total_x, actual_total_y)
        except Exception as e:
            print(f"Error executing flashbang curve movement: {e}")
            self.move_r(round(relative_move_x), round(relative_move_y))
            return (relative_move_x, relative_move_y)

    def execute_flashbang_ultra_fast_move(self, relative_move_x, relative_move_y):
        """
        Ultra-fast flashbang movement: large steps, no delay, most direct path

        Args:
            relative_move_x: X-axis relative movement distance
            relative_move_y: Y-axis relative movement distance

        Returns:
            tuple: (actual_move_x, actual_move_y) actual movement distance
        """
        try:
            print(
                f"Ultra-fast flashbang movement: X={relative_move_x}, Y={relative_move_y}"
            )
            total_distance = math.sqrt(relative_move_x**2 + relative_move_y**2)
            if total_distance <= 100:
                self.move_r(round(relative_move_x), round(relative_move_y))
                actual_total_x = round(relative_move_x)
                actual_total_y = round(relative_move_y)
                print("Small distance single movement complete")
            else:
                if total_distance <= 500:
                    step1_x = round(relative_move_x * 0.6)
                    step1_y = round(relative_move_y * 0.6)
                    step2_x = round(relative_move_x - step1_x)
                    step2_y = round(relative_move_y - step1_y)
                    self.move_r(step1_x, step1_y)
                    self.move_r(step2_x, step2_y)
                    actual_total_x = step1_x + step2_x
                    actual_total_y = step1_y + step2_y
                    print(
                        f"Medium distance 2-step movement complete: ({step1_x},{step1_y}) -> ({step2_x},{step2_y})"
                    )
                else:
                    step1_x = round(relative_move_x * 0.5)
                    step1_y = round(relative_move_y * 0.5)
                    step2_x = round(relative_move_x * 0.3)
                    step2_y = round(relative_move_y * 0.3)
                    step3_x = round(relative_move_x - step1_x - step2_x)
                    step3_y = round(relative_move_y - step1_y - step2_y)
                    self.move_r(step1_x, step1_y)
                    self.move_r(step2_x, step2_y)
                    self.move_r(step3_x, step3_y)
                    actual_total_x = step1_x + step2_x + step3_x
                    actual_total_y = step1_y + step2_y + step3_y
                    print(
                        f"Large distance 3-step movement complete: ({step1_x},{step1_y}) -> ({step2_x},{step2_y}) -> ({step3_x},{step3_y})"
                    )
            print(
                f"Ultra-fast movement complete, actual movement: X={actual_total_x}, Y={actual_total_y}"
            )
            return (actual_total_x, actual_total_y)
        except Exception as e:
            print(f"Ultra-fast movement error: {e}")
            self.move_r(round(relative_move_x), round(relative_move_y))
            return (relative_move_x, relative_move_y)
