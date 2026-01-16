import json
import os
import random
import string
import time
import threading
from ctypes import *
from queue import Queue
from threading import Thread

import pydirectinput

pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False

from core.aiming import AimingMixin
from core.config import ConfigMixin
from core.flashbang import FlashbangMixin
from core.gui import GuiMixin
from core.input import InputMixin
from core.model import ModelMixin
from core.recoil import RecoilMixin
from core.sunone_controller import SunoneAimController
from core.utils import TENSORRT_AVAILABLE
from src.function import *
from src.gui_handlers import ConfigChangeHandler
from src.infer_class import *
from src.pid import DualAxisPID
from src.profiler import FrameProfiler

TensorRTInferenceEngine = None
ensure_engine_from_memory = None
if TENSORRT_AVAILABLE:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dll_path = os.path.join(current_dir, "dll")
        if os.path.exists(dll_path):
            os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(dll_path)
        from src.inference_engine import (
            TensorRTInferenceEngine,
            ensure_engine_from_memory,
        )

        print("TensorRT inference engine module loaded successfully")
    except Exception as e:
        print(f"TensorRT inference engine module load failed: {e}")
        print(f"Error details: {str(e)}")
        TENSORRT_AVAILABLE = False
        TensorRTInferenceEngine = None
        ensure_engine_from_memory = None
if not TENSORRT_AVAILABLE:
    print("Skipping TensorRT module import, using pure ONNX mode")

from src.screenshot_manager import ScreenshotManager

try:
    from web.server import start_web_server
except ImportError:
    start_web_server = None

# Generate random functions for obfuscation
random_function_num = random.randint(1, 5)
for i in range(random_function_num):
    random_function_name = "a"
    random_function_name += "".join(
        random.sample(string.ascii_letters + string.digits, 8)
    )
    random_function_content_num = random.randint(1, 5)
    random_function_content = ""
    for ii in range(random_function_content_num):
        content = "".join(random.sample(string.ascii_letters + string.digits, 8))
        content = f"_{content} = True"
        random_function_content += content + "\n"
    exec(f"def {random_function_name}(): {random_function_content}")
    exec(f"{random_function_name}()")

print("")


class Aassist(
    InputMixin,
    ConfigMixin,
    AimingMixin,
    GuiMixin,
    RecoilMixin,
    FlashbangMixin,
    ModelMixin,
):
    """Main application class combining all mixin functionality."""

    def __init__(self):
        self.is_v8_checkbox = None
        self.is_trt_checkbox = None
        self.press_timer = None
        self.auto_y_checkbox = None
        self.right_down_checkbox = None
        self.down_switch = False
        self.decimal_x = 0
        self.decimal_y = 0
        self.end = False
        self.left_pressed = False
        self.left_pressed_long = False
        self.right_pressed = False
        self.number_input = None
        self.x_input = None
        self.y_input = None
        self.picked_stage = 0
        self.stages_combo = None
        self.add_gun_name = ""
        self.picked_gun = ""
        self.guns_combo = None
        self.add_game_name = ""
        self.games_combo = None
        self.timer_id2 = 0
        self.delay = 10
        self.now_num = 0
        self.now_stage = 0
        self.target_priority = {
            "distance_scoring_weight": 0.6,
            "center_scoring_weight": 0.4,
            "size_scoring_weight": 0.3,
            "small_target_boost": 2.0,
            "small_target_threshold": 0.01,
            "medium_target_threshold": 0.05,
            "medium_target_boost": 1.5,
        }
        self.last_flashbang_time = 0
        self.flashbang_cooldown = 1.0
        self.current_yaw = 0
        self.is_turning_back = False
        self.turn_back_start_time = 0
        self.flashbang_actual_move_x = 0
        self.flashbang_actual_move_y = 0
        self._dopa_warning_shown = False
        self.fps = 0
        self.last_infer_time_ms = 0.0
        self.kalman_filter = None
        self.tracker = None
        self.showed = False
        self.km_listen_switch = False
        self.pnmh_listen_switch = False
        self.makcu_listen_switch = False
        self.trigger_only_active = False
        self.move_r = None
        self.move_dll = None
        self.add_key_name = ""
        self.add_group_name = ""
        self.selected_items = []
        self.checkboxes = []
        self.checkbox_group_tag = None
        self.identify_rect_top = None
        self.identify_rect_left = None
        self.engine = None
        self.running = False
        self.start_button_tag = None
        self.select_key = ""
        self.end_delay_input = None
        self.press_delay_input = None
        self.start_delay_input = None
        self.random_delay_input = None
        self.x_trigger_scope_input = None
        self.y_trigger_scope_input = None
        self.x_trigger_offset_input = None
        self.y_trigger_offset_input = None
        self.status_input = None
        self.continuous_trigger_input = None
        self.trigger_recoil_input = None
        self.deadzone_input = None
        self.history_size_input = None
        self.output_scale_x_input = None
        self.output_scale_y_input = None
        self.uniform_threshold_input = None
        self.min_velocity_threshold_input = None
        self.max_velocity_threshold_input = None
        self.compensation_factor_input = None
        self.fov_angle_input = None
        self.distance_weight_input = None
        self.base_step_input = None
        self.smoothing_factor_input = None
        self.aim_bot_scope_input = None
        self.min_position_offset_input = None
        self.aim_bot_position_input = None
        self.aim_bot_position2_input = None
        self.class_aim_combo = None
        self.dynamic_scope_enabled_input = None
        self.dynamic_scope_min_scope_input = None
        self.dynamic_scope_shrink_ms_input = None
        self.dynamic_scope_recover_ms_input = None
        self.current_selected_class = "0"
        self.class_priority_input = None
        self.infer_model_input = None
        self.confidence_threshold_input = None
        self.iou_t_input = None
        self.key_tag = None
        self.move_group_tag = None
        self.window_tag = None
        self.old_refreshed_aim_key = ""
        self.config, self.aim_keys_dist, self.aim_key, self.group = self.build_config()
        self.refresh_class_names()
        if "small_target_enhancement" not in self.config:
            self.config["small_target_enhancement"] = {
                "enabled": True,
                "boost_factor": 2.0,
                "threshold": 0.01,
                "medium_threshold": 0.05,
                "medium_boost": 1.5,
                "smooth_enabled": True,
                "smooth_frames": 5,
                "adaptive_nms": True,
            }
        auto_dpi_scale = self.get_system_dpi_scale()
        config_dpi_scale = self.config.get("gui_dpi_scale", 0.0)
        self.dpi_scale = config_dpi_scale if config_dpi_scale > 0 else auto_dpi_scale
        font_scale = float(self.config.get("gui_font_scale", 1.0))
        font_scale = max(0.6, min(font_scale, 3.0))
        self.font_scale = font_scale
        base_width, base_height = (840, 750)
        width_scale = self.config.get("gui_width_scale", 1.0)
        self.gui_window_width = int(base_width * self.dpi_scale * max(width_scale, 0.5))
        self.gui_window_height = int(base_height * self.dpi_scale)
        self.scaled_bar_height = int(2 * self.dpi_scale)
        self.scaled_sidebar_width = int(60 * self.dpi_scale)
        self.scaled_font_size_main = int(12 * self.dpi_scale * self.font_scale)
        self.scaled_font_size_custom = int(14 * self.dpi_scale * self.font_scale)
        self.scaled_width_small = int(50 * self.dpi_scale)
        self.scaled_width_60 = int(60 * self.dpi_scale)
        self.scaled_width_medium = int(80 * self.dpi_scale)
        self.scaled_width_normal = int(100 * self.dpi_scale)
        self.scaled_width_large = int(120 * self.dpi_scale)
        self.scaled_width_xlarge = int(200 * self.dpi_scale)
        self.scaled_height_normal = int(100 * self.dpi_scale)
        if not TENSORRT_AVAILABLE:
            trt_used = False
            for group_name, group_data in self.config["groups"].items():
                if group_data.get("is_trt", False):
                    trt_used = True
                    group_data["is_trt"] = False
                    if group_data["infer_model"].endswith(".engine"):
                        onnx_path = group_data.get("original_infer_model")
                        if onnx_path and os.path.exists(onnx_path):
                            group_data["infer_model"] = onnx_path
                        else:
                            possible_onnx = (
                                os.path.splitext(group_data["infer_model"])[0] + ".onnx"
                            )
                            if os.path.exists(possible_onnx):
                                group_data["infer_model"] = possible_onnx
        self.config_handler = ConfigChangeHandler(self.config, None)
        self.config_handler.register_context_provider("group", lambda: self.group)
        self.config_handler.register_context_provider("key", lambda: self.select_key)
        self._init_config_handlers()
        self.init_target_priority()
        self._sensitivity_display_initialized = False
        self.mouse_listener = None
        self.keyboard_listener = None
        self.old_pressed_aim_key = ""
        self.que_aim = Queue(maxsize=1)
        self.que_trigger = Queue(maxsize=1)
        self.aim_key_status = False
        self.aim_bot = CFUNCTYPE(
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p
        )(self.aim_bot_func)
        self.down = CFUNCTYPE(
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p
        )(self.down_func)
        self.timer_id = 0
        self.pressed_key = []
        self.time_set_event = windll.winmm.timeSetEvent
        self.time_kill_event = windll.winmm.timeKillEvent
        self.time_begin_period = windll.winmm.timeBeginPeriod
        self.time_end_period = windll.winmm.timeEndPeriod
        self.screen_width, self.screen_height = self.get_dpi_aware_screen_size()
        self.screen_center_x = int(self.screen_width / 2)
        self.screen_center_y = int(self.screen_height / 2)
        self.screenshot_manager = None
        if len(self.aim_key) > 0:
            self.select_key = self.aim_key[0]
            self.pressed_key_config = self.aim_keys_dist[self.aim_key[0]]
        self.dual_pid = DualAxisPID(
            kp=[0.4, 0.4], ki=[0.02, 0.02], kd=[0.12, 0.12], windup_guard=[0.0, 0.0]
        )
        self.sunone_aim = SunoneAimController()
        self.last_target_count = 0
        self.last_target_count_by_class = {}
        self.target_switch_time = 0
        self.is_waiting_for_switch = False
        self._dynamic_scope = {
            "value": 0,
            "phase": "idle",
            "last_ms": time.time() * 1000.0,
        }
        self._dynamic_scope_lock_active_prev = False
        self.refresh_controller_params()
        self.trigger_status = False
        self.continuous_trigger_active = False
        self.continuous_trigger_thread = None
        self.trigger_recoil_active = False
        self.trigger_recoil_thread = None
        self.trigger_recoil_pressed = False
        self.picked_game = self.config["picked_game"]
        self.games = list(self.config["games"].keys())
        self.pnmh = None
        self.makcu = None
        self.temp_aim_bot_position = 0.0
        self.target_history = {}
        self.target_history_max_frames = 5
        if start_web_server is not None:
            start_web_server(self)
        else:
            print(
                "[Web control panel] web/server.py not found, Web service not started."
            )
        self.change = self._change_callback
        self._save_timer = None
        self.frame_profiler = FrameProfiler()
        if "recoil" not in self.config:
            self.config["recoil"] = {
                "use_mouse_re_trajectory": False,
                "replay_speed": 1.0,
                "pixel_enhancement_ratio": 1.0,
                "mapping": {},
            }
        else:
            self.config["recoil"].setdefault("use_mouse_re_trajectory", False)
            self.config["recoil"].setdefault("replay_speed", 1.0)
            self.config["recoil"].setdefault("pixel_enhancement_ratio", 1.0)
            self.config["recoil"].setdefault("mapping", {})
            if not isinstance(self.config["recoil"]["mapping"], dict):
                print("[Fix] recoil.mapping type error, reset to empty dict")
                self.config["recoil"]["mapping"] = {}
        self._current_mouse_re_points = None
        self._recoil_replay_thread = None
        self._recoil_is_replaying = False
        self.mouse_re_picked_game = self.config.get("picked_game", "")
        self.mouse_re_picked_gun = ""
        self.mouse_re_games_combo = None
        self.mouse_re_guns_combo = None

    def _change_callback(self, path, value):
        if path == "inference":
            if value == "start" and (not self.running):
                self.running = True
                self.go()
                return
            if value == "stop" and self.running:
                self.running = False
                if self.timer_id != 0:
                    self.time_kill_event(self.timer_id)
                    self.timer_id = 0
                if self.timer_id2 != 0:
                    self.time_kill_event(self.timer_id2)
                    self.timer_id2 = 0
                self.close_screenshot()
                self.disconnect_device()
            return None
        parts = path.split(".")
        value_changed = False
        if len(parts) == 1:
            method = getattr(self, f"on_{parts[0]}_change", None)
            if callable(method):
                method(None, value)
            else:
                if parts[0] not in self.config or self.config[parts[0]] != value:
                    self.config[parts[0]] = value
                    value_changed = True
        else:
            if len(parts) == 3 and parts[0] == "groups":
                group, key = (parts[1], parts[2])
                method = getattr(self, f"on_{key}_change", None)
                if callable(method):
                    method(None, value)
                else:
                    if group not in self.config["groups"]:
                        self.config["groups"][group] = {}
                    if (
                        key not in self.config["groups"][group]
                        or self.config["groups"][group][key] != value
                    ):
                        self.config["groups"][group][key] = value
                        value_changed = True
            else:
                if (
                    len(parts) >= 5
                    and parts[0] == "groups"
                    and (parts[2] == "aim_keys")
                ):
                    group, aim_key, param = (parts[1], parts[3], parts[4])
                    if group not in self.config["groups"]:
                        self.config["groups"][group] = {}
                    if "aim_keys" not in self.config["groups"][group]:
                        self.config["groups"][group]["aim_keys"] = {}
                    if aim_key not in self.config["groups"][group]["aim_keys"]:
                        self.config["groups"][group]["aim_keys"][aim_key] = {}
                    if param == "trigger" and len(parts) == 6:
                        trigger_param = parts[5]
                        if (
                            "trigger"
                            not in self.config["groups"][group]["aim_keys"][aim_key]
                        ):
                            self.config["groups"][group]["aim_keys"][aim_key][
                                "trigger"
                            ] = {
                                "status": False,
                                "continuous": False,
                                "recoil": False,
                                "start_delay": 0,
                                "press_delay": 1,
                                "end_delay": 0,
                                "random_delay": 0,
                                "x_trigger_scope": 1.0,
                                "y_trigger_scope": 1.0,
                                "x_trigger_offset": 0.0,
                                "y_trigger_offset": 0.0,
                            }
                        if (
                            trigger_param
                            not in self.config["groups"][group]["aim_keys"][aim_key][
                                "trigger"
                            ]
                            or self.config["groups"][group]["aim_keys"][aim_key][
                                "trigger"
                            ][trigger_param]
                            != value
                        ):
                            self.config["groups"][group]["aim_keys"][aim_key][
                                "trigger"
                            ][trigger_param] = value
                            value_changed = True
                    else:
                        if (
                            param
                            not in self.config["groups"][group]["aim_keys"][aim_key]
                            or self.config["groups"][group]["aim_keys"][aim_key][param]
                            != value
                        ):
                            self.config["groups"][group]["aim_keys"][aim_key][param] = (
                                value
                            )
                            value_changed = True
                    self.refresh_pressed_key_config(aim_key)
                else:
                    obj = self.config
                    for i, p in enumerate(parts[:-1]):
                        if p not in obj:
                            obj[p] = {}
                        obj = obj[p]
                    last_part = parts[-1]
                    if last_part not in obj or obj[last_part] != value:
                        obj[last_part] = value
                        value_changed = True
        if value_changed:
            print(f"Configuration changed: {path} = {value}")

    def start(self):
        self.gui()

    def go(self):
        model_path = self.config["groups"][self.group]["infer_model"]
        if not os.path.exists(model_path):
            print("Model file does not exist")
            return False
        self.config["screen_width"] = self.screen_width
        self.config["screen_height"] = self.screen_height
        if self.screenshot_manager is None:
            self.screenshot_manager = ScreenshotManager(self.config, self.engine)
        if not self.screenshot_manager.init_sources():
            print("Failed to initialize screenshot source")
            return False
        self.init_mouse()
        if self.timer_id != 0:
            self.time_kill_event(self.timer_id)
            self.timer_id = 0
        self.timer_id = self.time_set_event(1, 1, self.aim_bot, 0, 1)
        infer_thread = Thread(target=self.infer)
        infer_thread.setDaemon(True)
        infer_thread.start()
        trigger_thread = Thread(target=self.trigger)
        trigger_thread.setDaemon(True)
        trigger_thread.start()
        return True

    def save_config(self):
        """Save configuration to local file"""
        import json

        def _async_save():
            try:
                with open("cfg.json", "w", encoding="utf-8") as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                print("Configuration saved successfully")
                return True
            except Exception as e:
                print(f"Configuration save exception: {e}")
                return False

        save_thread = threading.Thread(target=_async_save, daemon=True)
        save_thread.start()
        return True

    def _secure_cleanup(self):
        """Clean up resources on exit"""
        try:
            if (
                hasattr(self, "screenshot_manager")
                and self.screenshot_manager is not None
            ):
                try:
                    self.screenshot_manager.close()
                except Exception:
                    pass
        except Exception:
            pass
