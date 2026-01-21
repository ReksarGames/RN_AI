import json
import os
import random
import threading

from .utils import TENSORRT_AVAILABLE


class ConfigMixin:
    """Mixin class for configuration management."""

    def save_config_callback(self):
        """Async save configuration callback - saves to local cfg.json"""

        def _async_save_callback():
            try:
                with open("cfg.json", "w", encoding="utf-8") as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                print("Configuration saved successfully")
                return True
            except Exception as e:
                print(f"Configuration save callback exception: {e}")
                return False

        save_thread = threading.Thread(target=_async_save_callback, daemon=True)
        save_thread.start()
        return True

    def build_config(self):
        """
        Build configuration and return related parameters

        Process TRT related path settings and get current group's key configuration

        Returns:
            tuple: (config, aim_keys_dist, aim_keys, group) config dict, key config dict, key list and current group name
        """
        if hasattr(self, "config") and self.config:
            config = self.config
        else:
            # Load from local cfg.json
            config = None
            if os.path.exists("cfg.json"):
                try:
                    with open("cfg.json", "r", encoding="utf-8") as f:
                        config = json.load(f)
                except Exception as e:
                    print(f"Failed to load cfg.json: {e}")
            if config is None:
                print("cfg.json not found or invalid, generating default configuration.")
                config = self._create_default_config()
                self.config = config
                self.save_config_callback()
        if config.get("move_method") != "makcu":
            print("Only makcu move method is supported, forcing move_method to 'makcu'")
            config["move_method"] = "makcu"
        if "enable_parallel_processing" not in config:
            config["enable_parallel_processing"] = True
        if "turbo_mode" not in config:
            config["turbo_mode"] = True
        if "skip_frame_processing" not in config:
            config["skip_frame_processing"] = True
        if "performance_mode" not in config:
            config["performance_mode"] = "balanced"
        if "use_async_move" not in config:
            config["use_async_move"] = False
        if "frame_skip_ratio" not in config:
            config["frame_skip_ratio"] = 0
        if "cpu_optimization" not in config:
            config["cpu_optimization"] = True
        if "memory_optimization" not in config:
            config["memory_optimization"] = True
        if "capture_offset_x" not in config:
            config["capture_offset_x"] = 0
        if "capture_offset_y" not in config:
            config["capture_offset_y"] = 0
        if "aim_controller" not in config:
            config["aim_controller"] = "pid"
        if "ui_language" not in config:
            config["ui_language"] = "en"
        if "gui_width_scale" not in config:
            config["gui_width_scale"] = 1.0
        if "gui_font_scale" not in config:
            config["gui_font_scale"] = 1.0
        if "small_target_enhancement" not in config:
            config["small_target_enhancement"] = {
                "enabled": True,
                "boost_factor": 0.1,
                "threshold": 0.02,
                "medium_threshold": 0.05,
                "medium_boost": 1.5,
                "smooth_enabled": True,
                "smooth_frames": 2,
                "adaptive_nms": True,
            }
        if "class_names" not in config:
            config["class_names"] = []
        if "class_names_file" not in config:
            config["class_names_file"] = ""
        if "sunone" not in config:
            config["sunone"] = {}
        sunone = config["sunone"]
        if "use_smoothing" not in sunone:
            sunone["use_smoothing"] = True
        if "smoothness" not in sunone:
            sunone["smoothness"] = 6
        if "tracking_smoothing" not in sunone:
            sunone["tracking_smoothing"] = False
        if "use_kalman" not in sunone:
            sunone["use_kalman"] = False
        if "kalman_process_noise" not in sunone:
            sunone["kalman_process_noise"] = 0.01
        if "kalman_measurement_noise" not in sunone:
            sunone["kalman_measurement_noise"] = 0.1
        if "kalman_speed_multiplier_x" not in sunone:
            sunone["kalman_speed_multiplier_x"] = 1.0
        if "kalman_speed_multiplier_y" not in sunone:
            sunone["kalman_speed_multiplier_y"] = 1.0
        if "reset_threshold" not in sunone:
            sunone["reset_threshold"] = 4.0
        if "speed" not in sunone:
            sunone["speed"] = {}
        speed = sunone["speed"]
        if "min_multiplier" not in speed:
            speed["min_multiplier"] = 0.5
        if "max_multiplier" not in speed:
            speed["max_multiplier"] = 0.7
        if "snap_radius" not in speed:
            speed["snap_radius"] = 3.2
        if "near_radius" not in speed:
            speed["near_radius"] = 40.0
        if "speed_curve_exponent" not in speed:
            speed["speed_curve_exponent"] = 10.0
        if "snap_boost_factor" not in speed:
            speed["snap_boost_factor"] = 4.0
        if "prediction" not in sunone:
            sunone["prediction"] = {}
        prediction = sunone["prediction"]
        if "mode" not in prediction:
            prediction["mode"] = 0
        if "interval" not in prediction:
            prediction["interval"] = 0.01
        if "kalman_lead_ms" not in prediction:
            prediction["kalman_lead_ms"] = 0.0
        if "kalman_max_lead_ms" not in prediction:
            prediction["kalman_max_lead_ms"] = 0.0
        if "velocity_smoothing" not in prediction:
            prediction["velocity_smoothing"] = 0.4
        if "velocity_scale" not in prediction:
            prediction["velocity_scale"] = 1.0
        if "kalman_process_noise" not in prediction:
            prediction["kalman_process_noise"] = 0.01
        if "kalman_measurement_noise" not in prediction:
            prediction["kalman_measurement_noise"] = 0.1
        if "future_positions" not in prediction:
            prediction["future_positions"] = 6
        if "draw_future_positions" not in prediction:
            prediction["draw_future_positions"] = False
        if "debug" not in sunone:
            sunone["debug"] = {}
        debug = sunone["debug"]
        if "show_prediction" not in debug:
            debug["show_prediction"] = False
        if "show_step" not in debug:
            debug["show_step"] = False
        if "show_future" not in debug:
            debug["show_future"] = False
        if "sunone_max_detections" not in config:
            config["sunone_max_detections"] = 0
        if "auto_flashbang" not in config:
            config["auto_flashbang"] = {
                "enabled": False,
                "delay_ms": 150,
                "turn_angle": 90,
                "sensitivity_multiplier": 2.5,
                "return_delay": 80,
                "min_confidence": 0.3,
                "min_size": 5,
                "use_curve": True,
                "curve_speed": 8.0,
                "curve_knots": 3,
            }
        for group_key, group_val in config.get("groups", {}).items():
            if "disable_headshot" not in group_val:
                group_val["disable_headshot"] = False
            if "disable_headshot_keys" not in group_val:
                group_val["disable_headshot_keys"] = ["m"]
            if "disable_headshot_class_id" not in group_val:
                group_val["disable_headshot_class_id"] = -1
            if "targeting_button_key" not in group_val:
                default_target = ""
                for key_name, key_cfg in group_val.get("aim_keys", {}).items():
                    if not key_cfg.get("trigger_only", False):
                        default_target = key_name
                        break
                if not default_target and group_val.get("aim_keys"):
                    default_target = next(iter(group_val["aim_keys"].keys()))
                group_val["targeting_button_key"] = default_target
            if "triggerbot_button_key" not in group_val:
                default_trigger = ""
                for key_name, key_cfg in group_val.get("aim_keys", {}).items():
                    if key_cfg.get("trigger_only", False):
                        default_trigger = key_name
                        break
                group_val["triggerbot_button_key"] = default_trigger
            if "disable_headshot_button_key" not in group_val:
                keys = group_val.get("disable_headshot_keys", [])
                group_val["disable_headshot_button_key"] = keys[0] if keys else "m"
            disable_key = group_val.get("disable_headshot_button_key", "")
            group_val["disable_headshot_keys"] = [disable_key] if disable_key else []
            if "is_trt" not in group_val:
                group_val["is_trt"] = False
            if "yolo_format" not in group_val:
                group_val["yolo_format"] = (
                    "v8" if group_val.get("is_v8", False) else "auto"
                )
            if "use_sunone_processing" not in group_val:
                group_val["use_sunone_processing"] = False
            if "sunone_model_variant" not in group_val:
                group_val["sunone_model_variant"] = "yolo11"
            if "infer_model" not in group_val:
                continue
            current_model = group_val["infer_model"]
            if "original_infer_model" not in group_val:
                if current_model.endswith(".engine"):
                    onnx_path = os.path.splitext(current_model)[0] + ".onnx"
                    if os.path.exists(onnx_path):
                        group_val["original_infer_model"] = onnx_path
                elif current_model.endswith(".onnx"):
                    group_val["original_infer_model"] = current_model
            if group_val.get("is_trt", False):
                if not TENSORRT_AVAILABLE:
                    print(
                        f"Group {group_key} is set to use TRT, but TensorRT environment is unavailable, automatically switching to original mode"
                    )
                    group_val["is_trt"] = False
                    original_path = group_val.get(
                        "original_infer_model", group_val["infer_model"]
                    )
                    if original_path != group_val["infer_model"] and os.path.exists(
                        original_path
                    ):
                        group_val["infer_model"] = original_path
                        print(
                            f"Automatically switched back to ONNX mode: {original_path}"
                        )
                else:
                    original_path = group_val.get(
                        "original_infer_model", group_val["infer_model"]
                    )
                    engine_path = os.path.splitext(original_path)[0] + ".engine"
                    if os.path.exists(engine_path):
                        group_val["infer_model"] = engine_path
                    else:
                        print(f"Warning: TRT engine file does not exist: {engine_path}")
                        if original_path != group_val["infer_model"] and os.path.exists(
                            original_path
                        ):
                            group_val["infer_model"] = original_path
            else:
                original_path = group_val.get(
                    "original_infer_model", group_val["infer_model"]
                )
                if os.path.exists(original_path):
                    group_val["infer_model"] = original_path
        group = config["group"]
        if group and group in config["groups"]:
            aim_keys_dist = config["groups"][group]["aim_keys"]
            aim_keys = list(aim_keys_dist.keys())
            self.migrate_config_to_class_based(config)
            self.init_all_keys_class_aim_positions(group, config)
        else:
            aim_keys_dist = {}
            aim_keys = []
        return (config, aim_keys_dist, aim_keys, group)

    def _create_default_config(self):
        default_trigger = {
            "status": False,
            "start_delay": 150,
            "press_delay": 1,
            "end_delay": 200,
            "random_delay": 20,
            "x_trigger_scope": 0.5,
            "y_trigger_scope": 0.5,
            "x_trigger_offset": 0.0,
            "y_trigger_offset": 0.0,
            "continuous": False,
            "recoil": False,
        }
        default_key = {
            "confidence_threshold": 0.5,
            "iou_t": 0.8,
            "aim_bot_position": 0.5,
            "aim_bot_position2": 0.5,
            "aim_bot_scope": 160,
            "smoothing_factor": 0.5,
            "base_step": 0.17,
            "distance_weight": 1.0,
            "fov_angle": 90,
            "history_size": 12,
            "output_scale_x": 1.0,
            "output_scale_y": 1.0,
            "deadzone": 2.0,
            "uniform_threshold": 1.0,
            "compensation_factor": 1.0,
            "trigger": default_trigger,
            "classes": [0],
            "class_priority_order": [0],
            "class_aim_positions": {
                "0": {
                    "aim_bot_position": 0.5,
                    "aim_bot_position2": 0.5,
                    "confidence_threshold": 0.5,
                    "iou_t": 1.0,
                }
            },
            "overshoot_threshold": 3.0,
            "overshoot_x_factor": 0.5,
            "overshoot_y_factor": 0.3,
            "move_deadzone": 1.0,
            "pid_kp_x": 0.4,
            "pid_kp_y": 0.4,
            "pid_ki_x": 0.001,
            "pid_ki_y": 0.001,
            "pid_kd_x": 0.05,
            "pid_kd_y": 0.05,
            "smooth_x": 0,
            "smooth_y": 0,
            "smooth_deadzone": 0.0,
            "smooth_algorithm": 1.0,
            "target_switch_delay": 0,
            "dynamic_scope": {"enabled": False},
            "trigger_only": False,
            "disable_headshot_removed": False,
        }
        return {
            "group": "Default",
            "groups": {
                "Default": {
                    "infer_model": "",
                    "original_infer_model": "",
                    "is_trt": False,
                    "yolo_format": "auto",
                    "sunone_model_variant": "yolo11",
                    "use_sunone_processing": False,
                    "right_down": False,
                    "aim_keys": {"mouse_side2": default_key},
                    "disable_headshot": False,
                    "disable_headshot_keys": ["m"],
                    "disable_headshot_class_id": -1,
                    "targeting_button_key": "mouse_side2",
                    "triggerbot_button_key": "",
                    "disable_headshot_button_key": "m",
                }
            },
            "class_names": [
                "player",
                "bot",
                "weapon",
                "outline",
                "dead_body",
                "hideout_target_human",
                "hideout_target_balls",
                "head",
                "smoke",
                "fire",
                "third_person",
            ],
            "class_names_file": "",
            "infer_debug": False,
            "print_fps": False,
            "show_motion_speed": False,
            "move_method": "makcu",
            "screen_width": 1920,
            "screen_height": 1080,
            "capture_offset_x": 0,
            "capture_offset_y": 0,
            "ui_language": "en",
            "gui_width_scale": 1.0,
            "gui_font_scale": 1.0,
            "small_target_enhancement": {
                "enabled": True,
                "boost_factor": 0.1,
                "threshold": 0.02,
                "medium_threshold": 0.05,
                "medium_boost": 1.5,
                "smooth_enabled": True,
                "smooth_frames": 2,
                "adaptive_nms": True,
            },
        }

    def init_all_keys_class_aim_positions(self, group, config):
        """Initialize class aim position configuration for all keys"""
        try:
            old_group = getattr(self, "group", None)
            self.group = group
            self.config = config
            class_num = self.get_current_class_num()
            for key_name in config["groups"][group]["aim_keys"]:
                key_config = config["groups"][group]["aim_keys"][key_name]
                if "class_aim_positions" not in key_config:
                    key_config["class_aim_positions"] = {}
                cap = key_config["class_aim_positions"]
                if isinstance(cap, list):
                    converted = {}
                    for idx, item in enumerate(cap):
                        if isinstance(item, dict):
                            converted[str(idx)] = {
                                "aim_bot_position": float(
                                    item.get("aim_bot_position", 0.0)
                                ),
                                "aim_bot_position2": float(
                                    item.get("aim_bot_position2", 0.0)
                                ),
                                "confidence_threshold": float(
                                    item.get("confidence_threshold", 0.5)
                                ),
                                "iou_t": float(item.get("iou_t", 1.0)),
                            }
                        else:
                            converted[str(idx)] = {
                                "aim_bot_position": 0.0,
                                "aim_bot_position2": 0.0,
                                "confidence_threshold": 0.5,
                                "iou_t": 1.0,
                            }
                    key_config["class_aim_positions"] = converted
                else:
                    if not isinstance(cap, dict):
                        key_config["class_aim_positions"] = {}
                if "class_priority_order" not in key_config:
                    key_config["class_priority_order"] = list(range(class_num))
                if "overshoot_threshold" not in key_config:
                    key_config["overshoot_threshold"] = 3.0
                if "overshoot_x_factor" not in key_config:
                    key_config["overshoot_x_factor"] = 0.5
                if "overshoot_y_factor" not in key_config:
                    key_config["overshoot_y_factor"] = 0.3
                if "trigger_only" not in key_config:
                    key_config["trigger_only"] = False
                if "disable_headshot_removed" not in key_config:
                    key_config["disable_headshot_removed"] = False
                for i in range(class_num):
                    class_str = str(i)
                    if class_str not in key_config["class_aim_positions"]:
                        key_config["class_aim_positions"][class_str] = {
                            "aim_bot_position": 0.0,
                            "aim_bot_position2": 0.0,
                            "confidence_threshold": 0.5,
                            "iou_t": 1.0,
                        }
            if old_group is not None:
                self.group = old_group
        except Exception as e:
            print(f"Failed to initialize class aim configuration for all keys: {e}")
            import traceback

            traceback.print_exc()

    def migrate_config_to_class_based(self, config):
        """Migrate configuration: migrate global confidence threshold and IOU config to class-based config"""
        try:
            for group_name, group_config in config.get("groups", {}).items():
                for key_name, key_config in group_config.get("aim_keys", {}).items():
                    old_conf_thresh = key_config.get("confidence_threshold")
                    old_iou_t = key_config.get("iou_t")
                    if old_conf_thresh is not None or old_iou_t is not None:
                        if "class_aim_positions" not in key_config:
                            key_config["class_aim_positions"] = {}
                        class_aim_positions = key_config["class_aim_positions"]
                        if isinstance(class_aim_positions, list):
                            key_config["class_aim_positions"] = {}
                            class_aim_positions = {}
                        else:
                            if not isinstance(class_aim_positions, dict):
                                key_config["class_aim_positions"] = {}
                                class_aim_positions = {}
                        for class_str, class_config in class_aim_positions.items():
                            if isinstance(class_config, dict):
                                if (
                                    old_conf_thresh is not None
                                    and "confidence_threshold" not in class_config
                                ):
                                    class_config["confidence_threshold"] = (
                                        old_conf_thresh
                                    )
                                if (
                                    old_iou_t is not None
                                    and "iou_t" not in class_config
                                ):
                                    class_config["iou_t"] = old_iou_t
        except Exception as e:
            print(f"Configuration migration failed: {e}")
            import traceback

            traceback.print_exc()

    def calculate_max_pixel_distance(self, screen_width, screen_height, fov_angle):
        diagonal_distance = (screen_width**2 + screen_height**2) ** 0.5
        max_pixel_distance = diagonal_distance / 2 * (fov_angle / 180)
        return max_pixel_distance

    def refresh_controller_params(self):
        self.dual_pid.set_pid_params(
            kp=[
                self.pressed_key_config.get("pid_kp_x", 0.4),
                self.pressed_key_config.get("pid_kp_y", 0.4),
            ],
            ki=[
                self.pressed_key_config.get("pid_ki_x", 0.02),
                self.pressed_key_config.get("pid_ki_y", 0.02),
            ],
            kd=[
                self.pressed_key_config.get("pid_kd_x", 0.002),
                self.pressed_key_config.get("pid_kd_y", 0),
            ],
        )
        integral_limit_x = self.pressed_key_config.get("pid_integral_limit_x", 0.0)
        integral_limit_y = self.pressed_key_config.get("pid_integral_limit_y", 0.0)
        self.dual_pid.set_windup_guard([integral_limit_x, integral_limit_y])
        smooth_x = self.pressed_key_config.get("smooth_x", 0)
        smooth_y = self.pressed_key_config.get("smooth_y", 0)
        smooth_deadzone = self.pressed_key_config.get("smooth_deadzone", 0.0)
        smooth_algorithm = self.pressed_key_config.get("smooth_algorithm", 1.0)
        self.dual_pid.set_smooth_params(
            smooth_x, smooth_y, smooth_deadzone, smooth_algorithm
        )

    def refresh_pressed_key_config(self, key):
        """
        Refresh currently pressed key's configuration

        Args:
            key: Key name
        """
        if key != self.old_refreshed_aim_key:
            self.old_refreshed_aim_key = key
            self.pressed_key_config = self.aim_keys_dist[key]
            if hasattr(self, "dual_pid"):
                self.dual_pid.reset()
            if hasattr(self, "sunone_aim"):
                self.sunone_aim.reset()
            self.refresh_controller_params()

    def get_aim_position_for_class(self, class_id):
        """Get aim position based on class ID"""
        disable_headshot = False
        try:
            disable_headshot = bool(
                self.config["groups"][self.group].get("disable_headshot", False)
            )
        except Exception:
            disable_headshot = False

        def pick_position(a, b):
            if disable_headshot:
                return max(a, b)
            return random.uniform(a, b)

        if "class_aim_positions" not in self.pressed_key_config:
            return pick_position(
                self.pressed_key_config.get("aim_bot_position", 0.5),
                self.pressed_key_config.get("aim_bot_position2", 0.5),
            )
        class_str = str(class_id)
        if class_str in self.pressed_key_config["class_aim_positions"]:
            config = self.pressed_key_config["class_aim_positions"][class_str]
            return pick_position(config["aim_bot_position"], config["aim_bot_position2"])
        return pick_position(
            self.pressed_key_config.get("aim_bot_position", 0.5),
            self.pressed_key_config.get("aim_bot_position2", 0.5),
        )
