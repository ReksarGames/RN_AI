import copy
import ctypes
import os
import math
import random
import time
import string
import tkinter as tk
from tkinter import filedialog

import dearpygui.dearpygui as dpg
from makcu import MouseButton

from src.gui_handlers import ConfigItemGroup

from .utils import TENSORRT_AVAILABLE, UPDATE_TIME, VERSION, create_gradient_image

class DemoKalman1D:
    def __init__(self, process_noise, measurement_noise):
        self.x = 0.0
        self.v = 0.0
        self.P = 1.0
        self.Q = float(process_noise)
        self.R = float(measurement_noise)

    def update(self, z, dt):
        self.x += self.v * dt
        self.P += self.Q * dt
        k = self.P / (self.P + self.R) if (self.P + self.R) != 0 else 0.0
        self.x += k * (z - self.x)
        self.P *= (1.0 - k)
        denom = dt if dt > 1e-8 else 1e-8
        self.v = (1.0 - k) * self.v + k * ((z - self.x) / denom)
        return self.x

TRANSLATIONS = {
    "en": {
        "tab_system": "System",
        "tab_driver": "Driver",
        "tab_bypass": "Bypass",
        "tab_strafe": "Strafe",
        "tab_config": "Config",
        "tab_aim": "Aim",
        "tab_classes": "Classes",
        "label_ui_language": "UI Language",
        "label_ui_width_scale": "UI Width Scale",
        "label_ui_width_hint": "Adjust UI width (effective after restart)",
        "label_ui_font_scale": "Font Scale",
        "label_aim_controller": "Aim Controller",
        "label_controller_pid": "PID",
        "label_controller_sunone": "Sunone",
        "label_sunone_settings": "Sunone Settings",
        "label_sunone_smoothing": "Enable Smoothing",
        "label_sunone_tracking": "Tracking Smooth",
        "label_sunone_smoothness": "Smoothness",
        "label_sunone_kalman": "Enable Kalman",
        "label_sunone_kalman_q": "Kalman Process Noise",
        "label_sunone_kalman_r": "Kalman Measurement Noise",
        "label_sunone_kalman_mul_x": "Kalman Speed X",
        "label_sunone_kalman_mul_y": "Kalman Speed Y",
        "label_sunone_reset_threshold": "Reset Threshold",
        "label_sunone_prediction": "Prediction",
        "label_sunone_prediction_mode": "Prediction Mode",
        "label_sunone_prediction_interval": "Prediction Interval",
        "label_sunone_prediction_lead": "Kalman Lead (ms)",
        "label_sunone_prediction_max_lead": "Kalman Max Lead (ms)",
        "label_sunone_velocity_smoothing": "Velocity Smoothing",
        "label_sunone_velocity_scale": "Velocity Scale",
        "label_sunone_prediction_q": "Prediction Kalman Q",
        "label_sunone_prediction_r": "Prediction Kalman R",
        "label_sunone_future_positions": "Future Positions",
        "label_sunone_draw_future": "Draw Future Positions",
        "label_sunone_debug": "Prediction Debug",
        "label_sunone_debug_pred": "Show Predicted Target",
        "label_sunone_debug_step": "Show Step",
        "label_sunone_debug_future": "Show Future Points",
        "label_long_press_section": "Long Press",
        "label_sunone_speed": "Speed Curve",
        "label_sunone_min_speed": "Min Speed",
        "label_sunone_max_speed": "Max Speed",
        "label_sunone_snap_radius": "Snap Radius",
        "label_sunone_near_radius": "Near Radius",
        "label_sunone_curve_exp": "Curve Exponent",
        "label_sunone_snap_boost": "Snap Boost",
        "label_prediction_preview": "Prediction Preview",
        "label_speed_curve_preview": "Speed Curve Preview",
        "label_class_settings": "Class Settings",
        "label_class_priority": "Class Priority",
        "label_class_priority_hint": "Format: 0-1-2-3",
        "label_infer_classes": "Inference Classes",
        "label_class_aim_config": "Class Aim Config",
        "label_select_class": "Select Class",
        "label_class_names_file": "Class Names File",
        "label_load_class_names": "Load Names",
        "label_class_aim_preview": "Aim Preview",
        "label_class_names_manual": "Class Names (Manual)",
        "label_apply_class_names": "Apply Names",
        "label_class_names_list": "All Classes",
        "help_aim_controller": "Select aiming controller. PID is classic; Sunone adds smoothing/kalman/prediction.",
        "help_sunone_settings": "Sunone smoothing and Kalman options for stable tracking.",
        "help_prediction": "Predicts target movement. Standard uses velocity; Kalman filters velocity; combined uses both.",
        "help_trigger": "Auto-fire when target is inside the trigger zone. Continuous keeps LMB pressed while the key is held.",
        "help_class_settings": "Manage model classes: enable/disable targets and priorities.",
        "help_class_aim_config": "Choose a class and set aim positions. 0 = top of body, 1 = bottom.",
        "help_aim_position": "Aim position inside the box: 0.0 top, 1.0 bottom.",
        "help_aim_position2": "Second aim point for two-stage aim/smoothing.",
        "help_small_target_settings": "Improves stability for small/far targets; may cost performance.",
        "help_mouse_re": "Recoil compensation using mouse_re trajectory files.",
        "help_capture_source": "Select capture input: Standard (screen), OBS, or Capture Card.",
        "help_capture_offsets": "Offsets shift the capture region from screen center.",
        "help_capture_size": "Override standard capture size (e.g., 320x320 or 640x640). Auto uses model size.",
        "label_buttons": "Buttons",
        "label_targeting_buttons": "Targeting Buttons",
        "label_triggerbot_buttons": "Triggerbot Buttons",
        "label_disable_headshot_buttons": "Disable Headshot Buttons",
        "label_disable_headshot_class": "Disable Headshot Class ID",
        "label_add_button": "Add Button",
        "label_remove_button": "Remove",
        "label_button_none": "None",
        "label_disable_headshot_status": "Disable headshot",
        "help_smart_target": "Locks current target for stability; improves tracking on moving targets.",
        "help_target_lock_distance": "Max distance (px) to keep the locked target.",
        "help_target_lock_reacquire_time": "How long (s) to keep lock after target disappears; 0 = instant switch.",
        "help_aim_weights": "Weights for target selection: distance, center, and size influence priority. Tie-break applies weights only when targets are within the chosen % of the closest target.",
        "help_aim_weight_tiebreak": "Apply aim weights only when targets are within this % of the closest target distance. 0 disables.",
        "help_speed_curve": "Speed curve controls how fast the aim moves based on distance to target.",
        "help_kalman": "Kalman filter smooths noisy target positions for steadier tracking.",
        "help_prediction_lead": "Lead time (ms) adds forward prediction to aim position.",
        "help_velocity_smoothing": "Smooths target velocity to reduce jitter in prediction.",
        "help_long_press_no_lock_y": "Long press keeps Y axis unlocked while aiming; useful for vertical control.",
        "help_long_press_threshold": "Duration in ms to treat a press as long press.",
        "help_trigger_only": "Trigger only: no aim movement, only auto-fire for this key.",
        "help_trigger_recoil": "Apply recoil compensation when triggerbot fires.",
        "help_small_target_enhancement": "Boosts detection stability for small targets.",
        "help_small_target_smoothing": "Smooths small target positions across frames.",
        "help_small_target_boost": "Extra weight for small targets in selection.",
        "help_small_target_history": "Number of frames used for small target smoothing.",
        "help_small_target_threshold": "Size threshold treated as small target.",
        "help_medium_target_threshold": "Threshold for medium targets in small-target smoothing.",
        "help_adaptive_nms": "Adaptive NMS adjusts suppression for small targets.",
        "label_help_sunone": "Sunone Info",
        "label_help_prediction": "Prediction Info",
        "label_help_speed_curve": "Speed Curve Info",
        "help_driver": "Driver controls how mouse movement is sent (Makcu).",
        "help_bypass": "Bypass masks physical inputs while the driver is active.",
        "help_strafe": "Strafe tab stores recoil/weapon profiles and movement offsets.",
        "label_key_bindings": "Key Bindings",
        "label_bind_key": "Bind Key",
        "label_key_preset": "Key Preset",
        "label_key_name": "Key Name",
        "label_key_add": "Add Key",
        "label_key_delete": "Delete Key",
        "label_model_params": "Model Settings",
        "label_group_name": "Group Name",
        "label_add_group": "Add Group",
        "label_delete_group": "Delete Group",
        "label_infer_model": "Inference Model",
        "label_select_model": "Select Model",
        "label_yolo_format": "YOLO Format",
        "label_sunone_variant": "YOLO Version",
        "label_sunone_variant_yolo8": "YOLOv8",
        "label_sunone_variant_yolo9": "YOLOv9",
        "label_sunone_variant_yolo10": "YOLOv10",
        "label_sunone_variant_yolo11": "YOLOv11",
        "label_sunone_variant_yolo12": "YOLOv12",
        "label_use_sunone_processing": "Use Sunone Processing",
        "help_sunone_variant": "Select YOLO version (v8-v12) for decoding; used by both standard and Sunone paths.",
        "label_yolo_auto": "Auto",
        "label_yolo_auto": "Auto",
        "label_yolo_v5": "YOLOv5/v7",
        "label_yolo_v8": "YOLOv8/v10/v11",
        "help_use_sunone_processing": "Switch detection/NMS to the Sunone processing pipeline.",
        "label_capture_status": "Current Capture",
        "label_capture_source": "Capture Source",
        "label_capture_standard": "Standard",
        "label_capture_obs": "OBS",
        "label_capture_card": "Capture Card",
        "label_capture_bettercam": "BetterCam (Desktop)",
        "label_obs_ip": "OBS IP",
        "label_obs_port": "OBS Port",
        "label_obs_fps": "OBS FPS",
        "label_capture_device": "Capture Device",
        "label_capture_fps": "Capture FPS",
        "label_capture_resolution": "Capture Resolution",
        "label_capture_crop": "Capture Crop Size",
        "label_capture_size": "Capture Size",
        "label_video_codec": "Video Codec",
        "label_capture_offset_x": "Capture Offset X",
        "label_capture_offset_y": "Capture Offset Y",
        "label_gui_dpi_scale": "GUI DPI Scale",
        "label_auto_detect": "Auto Detect",
        "label_infer_window": "Inference Window",
        "label_print_fps": "Print FPS",
        "label_show_motion_speed": "Show Motion Speed",
        "label_show_curve": "Show Curve",
        "label_show_infer_time": "Show Infer Time",
        "label_screenshot_separation": "Screenshot Separation (Multi-thread)",
        "label_small_target_enhancement": "Enable Small Target Enhancement",
        "label_small_target_smoothing": "Enable Small Target Smoothing",
        "label_adaptive_nms": "Adaptive NMS",
        "label_small_target_boost": "Small Target Boost",
        "label_small_target_history": "Smooth History Frames",
        "label_small_target_threshold": "Small Target Threshold",
        "label_medium_target_threshold": "Medium Target Threshold",
        "label_turbo_mode": "Turbo Mode",
        "label_skip_frame_processing": "Skip Frame Processing",
        "label_recoil_debug": "Recoil Debug",
        "label_class_priority_debug": "Class Priority Debug",
        "label_show_aim_scope": "Show Aim Scope",
        "label_aim_weights": "Aim Weights",
        "label_aim_weight_tiebreak": "Tie-break (%)",
        "label_distance_weight": "Distance Weight",
        "label_center_weight": "Center Weight",
        "label_size_weight": "Size Weight",
        "label_move_curve": "Move Curve",
        "label_compensation_curve": "Compensation Curve",
        "label_horizontal_boundary": "Horizontal Boundary",
        "label_vertical_boundary": "Vertical Boundary",
        "label_control_points": "Control Points",
        "label_distortion_mean": "Distortion Mean",
        "label_distortion_stddev": "Distortion StdDev",
        "label_distortion_frequency": "Distortion Frequency",
        "label_path_points_total": "Path Points Total",
        "label_com_port": "COM",
        "label_move_method": "Move Method",
        "label_mask_left": "Mask Left",
        "label_mask_right": "Mask Right",
        "label_mask_middle": "Mask Middle",
        "label_mask_side1": "Mask Side1",
        "label_mask_side2": "Mask Side2",
        "label_mask_x_axis": "Mask X Axis",
        "label_mask_y_axis": "Mask Y Axis",
        "label_aim_mask_x": "Aim Mask X",
        "label_aim_mask_y": "Aim Mask Y",
        "label_mask_wheel": "Mask Wheel",
        "label_check_right_button": "Check Right Button",
        "label_delete_game": "Delete Game",
        "label_game_name": "Game Name",
        "label_add_game": "Add Game",
        "label_delete_gun": "Delete Gun",
        "label_gun_name": "Gun Name",
        "label_add_gun": "Add Gun",
        "label_count": "Count",
        "label_axis_x": "X",
        "label_axis_y": "Y",
        "label_delete_index": "Delete Index",
        "label_add_index": "Add Index",
        "label_mouse_re_header": "mouse_re Trajectory Recoil",
        "label_mouse_re_enable": "Enable mouse_re Trajectory Recoil",
        "label_replay_speed": "Replay Speed",
        "label_pixel_enhance_ratio": "Pixel Enhancement Ratio",
        "label_import_trajectory_file": "Import Trajectory File",
        "label_clear_mapping": "Clear Mapping",
        "label_trt": "TRT",
        "label_long_press_no_lock_y": "Long Press No Lock Y",
        "label_long_press_threshold": "Long Press Threshold",
        "label_target_switch_delay": "Target Switch Delay (ms)",
        "label_target_reference_class": "Target Reference Class",
        "label_min_offset": "Min Offset",
        "label_aim_scope": "Aim Scope",
        "label_dynamic_scope": "Smart Target Lock",
        "label_min_scope": "Min Scope",
        "label_shrink_duration": "Shrink Duration",
        "label_recover_duration": "Recover Duration",
        "label_target_lock_distance": "Lock Distance",
        "label_target_lock_reacquire_time": "Reacquire Time (s)",
        "label_pid_params": "PID Controller Parameters",
        "label_pid_x_p": "X Proportional",
        "label_pid_x_i": "X Integral",
        "label_pid_x_d": "X Derivative",
        "label_pid_y_p": "Y Proportional",
        "label_pid_y_i": "Y Integral",
        "label_pid_y_d": "Y Derivative",
        "label_pid_x_limit": "X Limit",
        "label_pid_x_smooth": "X Smooth",
        "label_smooth_algorithm": "Smooth Algorithm",
        "label_pid_y_limit": "Y Limit",
        "label_pid_y_smooth": "Y Smooth",
        "label_smooth_deadzone": "Smooth Deadzone",
        "label_move_deadzone": "Move Deadzone",
        "label_trigger_config": "Trigger Config",
        "label_auto_trigger": "Auto Trigger",
        "label_continuous_trigger": "Continuous Trigger",
        "label_trigger_recoil": "Trigger Recoil",
        "label_trigger_only": "Trigger Only",
        "label_trigger_delay": "Trigger Delay",
        "label_press_duration": "Press Duration",
        "label_trigger_cooldown": "Trigger Cooldown",
        "label_random_delay": "Random Delay",
        "label_x_trigger_scope": "X Trigger Scope",
        "label_y_trigger_scope": "Y Trigger Scope",
        "label_x_trigger_offset": "X Trigger Offset",
        "label_y_trigger_offset": "Y Trigger Offset",
        "label_confidence_threshold": "Confidence Threshold",
        "label_iou": "IOU",
        "label_aim_position": "Aim Position",
        "label_aim_position2": "Aim Position2",
        "label_game": "Game",
        "label_gun": "Gun",
        "label_index": "Index",
        "label_ui_scale_detail": "Adjust GUI interface scale (Current system detection: {scale:.2f}, effective after restart)",
        "label_small_target_settings": "Small Target Enhancement Settings",
        "label_small_target_note": "Note: Small target enhancement can improve detection stability for distant or small-sized targets",
        "label_mouse_re_config": "mouse_re Recoil Config:",
        "label_mouse_re_note": "Note: Supports loading JSON files generated by mouse_re.py, hold left button to replay trajectory for recoil",
        "label_current_status": "Current Status:",
        "label_switch_prefix": "Switch",
        "label_switch_on": "ON",
        "label_switch_off": "OFF",
        "label_mapping_file_prefix": "Mapping File",
        "label_trajectory_points_prefix": "Trajectory Points",
        "label_none": "None",
        "label_start": "Start",
        "label_stop": "Stop",
        "label_save_config": "Save Config",
        "label_do_not_click": "Do Not Click!!!",
    },
    "ru": {
        "tab_system": "Система",
        "tab_driver": "Драйвер",
        "tab_bypass": "Обход",
        "tab_strafe": "Стрейф",
        "tab_config": "Конфиг",
        "tab_aim": "Прицел",
        "tab_classes": "Классы",
        "label_ui_language": "Язык интерфейса",
        "label_ui_width_scale": "Масштаб ширины интерфейса",
        "label_ui_width_hint": "Настройка ширины интерфейса (вступит в силу после перезапуска)",
        "label_ui_font_scale": "Масштаб шрифта",
        "label_aim_controller": "Контроллер прицеливания",
        "label_controller_pid": "PID",
        "label_controller_sunone": "Sunone",
        "label_sunone_settings": "Настройки Sunone",
        "label_sunone_smoothing": "Включить сглаживание",
        "label_sunone_tracking": "Плавность трекинга",
        "label_sunone_smoothness": "Сглаживание",
        "label_sunone_kalman": "Включить Калман",
        "label_sunone_kalman_q": "Шум процесса Калмана",
        "label_sunone_kalman_r": "Шум измерения Калмана",
        "label_sunone_kalman_mul_x": "Скорость Калмана X",
        "label_sunone_kalman_mul_y": "Скорость Калмана Y",
        "label_sunone_reset_threshold": "Порог сброса",
        "label_sunone_prediction": "Предсказание",
        "label_sunone_prediction_mode": "Режим предсказания",
        "label_sunone_prediction_interval": "Интервал предсказания",
        "label_sunone_prediction_lead": "Опережение Калмана (мс)",
        "label_sunone_prediction_max_lead": "Макс. опережение Калмана (мс)",
        "label_sunone_velocity_smoothing": "Сглаживание скорости",
        "label_sunone_velocity_scale": "Масштаб скорости",
        "label_sunone_prediction_q": "Калман Q предсказания",
        "label_sunone_prediction_r": "Калман R предсказания",
        "label_sunone_future_positions": "Будущие позиции",
        "label_sunone_draw_future": "Рисовать будущие позиции",
        "label_sunone_debug": "Отладка предсказания",
        "label_sunone_debug_pred": "Показывать предсказанную цель",
        "label_sunone_debug_step": "Показывать шаг",
        "label_sunone_debug_future": "Показывать будущие точки",
        "label_long_press_section": "Долгое нажатие",
        "label_sunone_speed": "Кривая скорости",
        "label_sunone_min_speed": "Мин. скорость",
        "label_sunone_max_speed": "Макс. скорость",
        "label_sunone_snap_radius": "Радиус прилипания",
        "label_sunone_near_radius": "Ближний радиус",
        "label_sunone_curve_exp": "Экспонента кривой",
        "label_sunone_snap_boost": "Усиление прилипания",
        "label_prediction_preview": "Предпросмотр предсказания",
        "label_speed_curve_preview": "Предпросмотр кривой скорости",
        "label_class_settings": "Настройки классов",
        "label_class_priority": "Приоритет классов",
        "label_class_priority_hint": "Формат: 0-1-2-3",
        "label_infer_classes": "Классы инференса",
        "label_class_aim_config": "Конфигурация прицела класса",
        "label_select_class": "Выбрать класс",
        "label_class_names_file": "Файл имён классов",
        "label_load_class_names": "Загрузить имена",
        "label_class_aim_preview": "Предпросмотр прицела",
        "label_class_names_manual": "Имена классов (вручную)",
        "label_apply_class_names": "Применить имена",
        "label_class_names_list": "Все классы",
        "help_aim_controller": "Выбор контроллера прицеливания. PID — классический; Sunone добавляет сглаживание, Калман и предсказание.",
        "help_sunone_settings": "Сглаживание и Калман в Sunone для стабильного трекинга.",
        "help_prediction": "Предсказывает движение цели. Standard использует скорость; Kalman фильтрует скорость; combined использует оба метода.",
        "help_trigger": "Авто-выстрел, когда цель внутри триггер-зоны. Continuous удерживает ЛКМ, пока кнопка зажата.",
        "help_class_settings": "Управление классами модели: включение/отключение целей и приоритетов.",
        "help_class_aim_config": "Выбор класса и настройка точки прицеливания. 0 = верх тела, 1 = низ.",
        "help_aim_position": "Позиция прицела внутри бокса: 0.0 — верх, 1.0 — низ.",
        "help_aim_position2": "Вторая точка прицеливания для двухэтапного прицеливания/сглаживания.",
        "help_small_target_settings": "Улучшает стабильность для маленьких/дальних целей; может снизить производительность.",
        "help_mouse_re": "Компенсация отдачи с использованием файлов траекторий mouse_re.",
        "help_capture_source": "Выбор источника захвата: Standard (экран), OBS или карта захвата.",
        "help_capture_offsets": "Смещения сдвигают область захвата от центра экрана.",
        "help_capture_size": "Переопределяет размер стандартного захвата (например, 320x320 или 640x640). Auto использует размер модели.",
        "label_buttons": "Кнопки",
        "label_targeting_buttons": "Кнопки прицеливания",
        "label_triggerbot_buttons": "Кнопки триггербота",
        "label_disable_headshot_buttons": "Кнопки отключения хедшота",
        "label_disable_headshot_class": "ID класса без хедшота",
        "label_add_button": "Добавить кнопку",
        "label_remove_button": "Удалить",
        "label_button_none": "Нет",
        "label_disable_headshot_status": "Отключить хедшот",
        "help_smart_target": "Фиксирует текущую цель для стабильности; улучшает трекинг движущихся целей.",
        "help_target_lock_distance": "Max distance (px) to keep the locked target.",
        "help_target_lock_reacquire_time": "Сколько секунд удерживать фиксацию после пропажи цели; 0 = мгновенное переключение.",
        "help_aim_weights": "Веса выбора цели: дистанция, центр и размер влияют на приоритет. Tie-break применяет веса только когда цели в пределах выбранного % от ближайшей.",
        "help_aim_weight_tiebreak": "Использовать веса только когда цели в пределах этого % от ближайшей цели. 0 = выкл.",
        "help_speed_curve": "Кривая скорости определяет скорость движения прицела в зависимости от расстояния до цели.",
        "help_kalman": "Фильтр Калмана сглаживает шумные позиции цели для более стабильного трекинга.",
        "help_prediction_lead": "Время упреждения (мс) добавляет опережение к позиции прицела.",
        "help_velocity_smoothing": "Сглаживает скорость цели для уменьшения дрожания при предсказании.",
        "help_long_press_no_lock_y": "Долгое нажатие оставляет ось Y разблокированной; полезно для вертикального контроля.",
        "help_long_press_threshold": "Длительность (мс), после которой нажатие считается долгим.",
        "help_trigger_only": "Только триггер: без движения прицела, только авто-выстрел для этой клавиши.",
        "help_trigger_recoil": "Применять компенсацию отдачи при срабатывании триггербота.",
        "help_small_target_enhancement": "Усиливает стабильность детекции маленьких целей.",
        "help_small_target_smoothing": "Сглаживает позиции маленьких целей между кадрами.",
        "help_small_target_boost": "Дополнительный вес для маленьких целей при выборе.",
        "help_small_target_history": "Количество кадров для сглаживания маленьких целей.",
        "help_small_target_threshold": "Порог размера, считающийся маленькой целью.",
        "help_medium_target_threshold": "Порог для средних целей в сглаживании маленьких целей.",
        "help_adaptive_nms": "Адаптивный NMS настраивает подавление для маленьких целей.",
        "label_help_sunone": "Информация о Sunone",
        "label_help_prediction": "Информация о предсказании",
        "label_help_speed_curve": "Информация о кривой скорости",
        "help_driver": "Драйвер управляет отправкой движений мыши (Makcu).",
        "help_bypass": "Обход маскирует физический ввод при активном драйвере.",
        "help_strafe": "Вкладка стрейфа хранит профили отдачи/оружия и смещения движения.",
        "label_key_bindings": "Назначение клавиш",
        "label_bind_key": "Назначить клавишу",
        "label_key_preset": "Пресет клавиш",
        "label_key_name": "Название клавиши",
        "label_key_add": "Добавить клавишу",
        "label_key_delete": "Удалить клавишу",
        "label_model_params": "Настройки модели",
        "label_group_name": "Имя группы",
        "label_add_group": "Добавить группу",
        "label_delete_group": "Удалить группу",
        "label_infer_model": "Модель инференса",
        "label_select_model": "Выбрать модель",
        "label_yolo_format": "Формат YOLO",
        "label_yolo_auto": "Авто",
        "label_yolo_v5": "YOLOv5/v7",
        "label_yolo_v8": "YOLOv8/v10/v11",
        "label_capture_status": "Текущий захват",
        "label_capture_source": "Источник захвата",
        "label_capture_standard": "Стандартный",
        "label_capture_obs": "OBS",
        "label_capture_card": "Карта захвата",
        "label_capture_bettercam": "BetterCam (Рабочий стол)",
        "label_obs_ip": "IP OBS",
        "label_obs_port": "Порт OBS",
        "label_obs_fps": "FPS OBS",
        "label_capture_device": "Устройство захвата",
        "label_capture_fps": "FPS захвата",
        "label_capture_resolution": "Разрешение захвата",
        "label_capture_crop": "Размер обрезки захвата",
        "label_capture_size": "Размер захвата",
        "label_video_codec": "Видеокодек",
        "label_capture_offset_x": "Смещение захвата X",
        "label_capture_offset_y": "Смещение захвата Y",
        "label_gui_dpi_scale": "Масштаб DPI интерфейса",
        "label_auto_detect": "Автоопределение",
        "label_infer_window": "Окно инференса",
        "label_print_fps": "Показывать FPS",
        "label_show_motion_speed": "Показывать скорость движения",
        "label_show_curve": "Показывать кривую",
        "label_show_infer_time": "Показывать время инференса",
        "label_screenshot_separation": "Разделение скриншотов (многопоточно)",
        "label_small_target_enhancement": "Включить улучшение маленьких целей",
        "label_small_target_smoothing": "Включить сглаживание маленьких целей",
        "label_adaptive_nms": "Адаптивный NMS",
        "label_small_target_boost": "Усиление маленьких целей",
        "label_small_target_history": "Кадры истории сглаживания",
        "label_small_target_threshold": "Порог маленькой цели",
        "label_medium_target_threshold": "Порог средней цели",
        "label_turbo_mode": "Турбо-режим",
        "label_skip_frame_processing": "Пропуск обработки кадров",
        "label_recoil_debug": "Отладка отдачи",
        "label_class_priority_debug": "Отладка приоритета классов",
        "label_show_aim_scope": "Показывать область прицеливания",
        "label_aim_weights": "Веса прицеливания",
        "label_aim_weight_tiebreak": "Тай-брейк (%)",
        "label_distance_weight": "Вес дистанции",
        "label_center_weight": "Вес центра",
        "label_size_weight": "Вес размера",
        "label_move_curve": "Кривая движения",
        "label_compensation_curve": "Кривая компенсации",
        "label_horizontal_boundary": "Горизонтальная граница",
        "label_vertical_boundary": "Вертикальная граница",
        "label_control_points": "Контрольные точки",
        "label_distortion_mean": "Среднее искажение",
        "label_distortion_stddev": "Станд. отклонение искажения",
        "label_distortion_frequency": "Частота искажения",
        "label_path_points_total": "Всего точек траектории",
        "label_com_port": "COM-порт",
        "label_move_method": "Метод движения",
        "label_mask_left": "Маска левой кнопки",
        "label_mask_right": "Маска правой кнопки",
        "label_mask_middle": "Маска средней кнопки",
        "label_mask_side1": "Маска боковой 1",
        "label_mask_side2": "Маска боковой 2",
        "label_mask_x_axis": "Маска оси X",
        "label_mask_y_axis": "Маска оси Y",
        "label_aim_mask_x": "Маска прицела X",
        "label_aim_mask_y": "Маска прицела Y",
        "label_mask_wheel": "Маска колеса",
        "label_check_right_button": "Проверять правую кнопку",
        "label_delete_game": "Удалить игру",
        "label_game_name": "Название игры",
        "label_add_game": "Добавить игру",
        "label_delete_gun": "Удалить оружие",
        "label_gun_name": "Название оружия",
        "label_add_gun": "Добавить оружие",
        "label_count": "Количество",
        "label_axis_x": "X",
        "label_axis_y": "Y",
        "label_delete_index": "Удалить индекс",
        "label_add_index": "Добавить индекс",
        "label_mouse_re_header": "Траектория отдачи mouse_re",
        "label_mouse_re_enable": "Включить траекторную отдачу mouse_re",
        "label_replay_speed": "Скорость воспроизведения",
        "label_pixel_enhance_ratio": "Коэффициент усиления пикселей",
        "label_import_trajectory_file": "Импорт файла траектории",
        "label_clear_mapping": "Очистить сопоставление",
        "label_trt": "TRT",
        "label_long_press_no_lock_y": "Долгое нажатие без блокировки Y",
        "label_long_press_threshold": "Порог долгого нажатия",
        "label_target_switch_delay": "Задержка смены цели (мс)",
        "label_target_reference_class": "Референсный класс цели",
        "label_min_offset": "Мин. смещение",
        "label_aim_scope": "Область прицеливания",
        "label_dynamic_scope": "Умная фиксация цели",
        "label_min_scope": "Мин. область",
        "label_shrink_duration": "Длительность сжатия",
        "label_recover_duration": "Длительность восстановления",
        "label_target_lock_distance": "Дистанция фиксации",
        "label_target_lock_reacquire_time": "Время удержания (с)",
        "label_pid_params": "Параметры PID-контроллера",
        "label_pid_x_p": "Пропорциональный X",
        "label_pid_x_i": "Интегральный X",
        "label_pid_x_d": "Дифференциальный X",
        "label_pid_y_p": "Пропорциональный Y",
        "label_pid_y_i": "Интегральный Y",
        "label_pid_y_d": "Дифференциальный Y",
        "label_pid_x_limit": "Ограничение X",
        "label_pid_x_smooth": "Сглаживание X",
        "label_smooth_algorithm": "Алгоритм сглаживания",
        "label_pid_y_limit": "Ограничение Y",
        "label_pid_y_smooth": "Сглаживание Y",
        "label_smooth_deadzone": "Мёртвая зона сглаживания",
        "label_move_deadzone": "Мёртвая зона движения",
        "label_trigger_config": "Настройки триггера",
        "label_auto_trigger": "Авто-триггер",
        "label_continuous_trigger": "Непрерывный триггер",
        "label_trigger_recoil": "Отдача триггера",
        "label_trigger_only": "Только триггер",
        "label_trigger_delay": "Задержка триггера",
        "label_press_duration": "Длительность нажатия",
        "label_trigger_cooldown": "Кулдаун триггера",
        "label_random_delay": "Случайная задержка",
        "label_x_trigger_scope": "Область триггера X",
        "label_y_trigger_scope": "Область триггера Y",
        "label_x_trigger_offset": "Смещение триггера X",
        "label_y_trigger_offset": "Смещение триггера Y",
        "label_confidence_threshold": "Порог уверенности",
        "label_iou": "IOU",
        "label_aim_position": "Позиция прицела",
        "label_aim_position2": "Позиция прицела 2",
        "label_game": "Игра",
        "label_gun": "Оружие",
        "label_index": "Индекс",
        "label_ui_scale_detail": "Настройка масштаба интерфейса (текущее определение системы: {scale:.2f}, вступит в силу после перезапуска)",
        "label_small_target_settings": "Настройки улучшения маленьких целей",
        "label_small_target_note": "Примечание: улучшение маленьких целей может повысить стабильность детекции для дальних или малых целей",
        "label_mouse_re_config": "Конфигурация отдачи mouse_re:",
        "label_mouse_re_note": "Примечание: поддерживается загрузка JSON-файлов, сгенерированных mouse_re.py; удерживайте левую кнопку для воспроизведения траектории отдачи",
        "label_current_status": "Текущий статус:",
        "label_switch_prefix": "Переключатель",
        "label_switch_on": "ВКЛ",
        "label_switch_off": "ВЫКЛ",
        "label_mapping_file_prefix": "Файл сопоставления",
        "label_trajectory_points_prefix": "Точки траектории",
        "label_none": "Нет",
        "label_start": "Старт",
        "label_stop": "Стоп",
        "label_save_config": "Сохранить конфиг",
        "label_do_not_click": "НЕ НАЖИМАТЬ!!!"
    }
}

SUNONE_VARIANT_KEYS = [
    ("yolo8", "label_sunone_variant_yolo8"),
    ("yolo9", "label_sunone_variant_yolo9"),
    ("yolo10", "label_sunone_variant_yolo10"),
    ("yolo11", "label_sunone_variant_yolo11"),
    ("yolo12", "label_sunone_variant_yolo12"),
]


try:
    if TENSORRT_AVAILABLE:
        from src.inference_engine import ensure_engine_from_memory
    else:
        ensure_engine_from_memory = None
except ImportError:
    ensure_engine_from_memory = None


class GuiMixin:
    """Mixin class for GUI interface and callback methods."""

    def tr(self, key):
        lang = "en"
        try:
            lang = self.config.get("ui_language", "en")
        except Exception:
            lang = "en"
        lang_map = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
        return lang_map.get(key, TRANSLATIONS["en"].get(key, key))

    def get_key_presets(self):
        return [
            "",
            "mouse_left",
            "mouse_right",
            "mouse_middle",
            "mouse_x1",
            "mouse_x2",
            "mouse_side1",
            "mouse_side2",
            "m",
        ]

    def attach_tooltip(self, item, text, wrap=None):
        if item is None:
            return
        if wrap is None:
            wrap = int(self.scaled_width_xlarge * 1.3)
        with dpg.tooltip(item):
            dpg.add_text(text, wrap=wrap)

    def update_button_lists(self):
        if not hasattr(self, "targeting_button_combo"):
            return
        group_cfg = self.config["groups"][self.group]
        options = self._get_button_options()
        dpg.configure_item(self.targeting_button_combo, items=options)
        dpg.configure_item(self.triggerbot_button_combo, items=options)
        dpg.configure_item(self.disable_headshot_button_combo, items=options)
        dpg.set_value(
            self.targeting_button_combo,
            self._format_button_value(group_cfg.get("targeting_button_key", "")),
        )
        dpg.set_value(
            self.triggerbot_button_combo,
            self._format_button_value(group_cfg.get("triggerbot_button_key", "")),
        )
        dpg.set_value(
            self.disable_headshot_button_combo,
            self._format_button_value(group_cfg.get("disable_headshot_button_key", "")),
        )
        if hasattr(self, "disable_headshot_class_input"):
            dpg.set_value(
                self.disable_headshot_class_input,
                int(group_cfg.get("disable_headshot_class_id", -1)),
            )
        disable_key = group_cfg.get("disable_headshot_button_key", "")
        group_cfg["disable_headshot_keys"] = [disable_key] if disable_key else []
        self.update_disable_headshot_status()

    def update_disable_headshot_status(self):
        if not hasattr(self, "disable_headshot_status_text"):
            return
        enabled = bool(self.config["groups"][self.group].get("disable_headshot", False))
        status = self.tr("label_switch_on") if enabled else self.tr("label_switch_off")
        dpg.set_value(
            self.disable_headshot_status_text,
            f"{self.tr('label_disable_headshot_status')}: {status}",
        )

    def _ensure_aim_key(self, key_name):
        if not key_name:
            return False
        if key_name not in self.config["groups"][self.group]["aim_keys"]:
            template_key = self.select_key or next(
                iter(self.config["groups"][self.group]["aim_keys"].keys()), None
            )
            if template_key is None:
                return False
            self.config["groups"][self.group]["aim_keys"][key_name] = copy.deepcopy(
                self.config["groups"][self.group]["aim_keys"][template_key]
            )
            self.init_class_aim_positions_for_key(key_name)
        return True

    def _remove_aim_key(self, key_name):
        if not key_name:
            return False
        if key_name not in self.config["groups"][self.group]["aim_keys"]:
            return False
        if len(self.config["groups"][self.group]["aim_keys"]) <= 1:
            return False
        del self.config["groups"][self.group]["aim_keys"][key_name]
        return True

    def _format_button_value(self, key_name):
        if not key_name:
            return self.tr("label_button_none")
        return key_name

    def _get_button_options(self):
        options = [self.tr("label_button_none")]
        options.extend([k for k in self.get_key_presets() if k])
        return options

    def _refresh_aim_keys_after_button_change(self):
        self.aim_keys_dist = self.config["groups"][self.group]["aim_keys"]
        self.aim_key = list(self.aim_keys_dist.keys())
        self.render_key_combo()
        self.update_button_lists()

    def on_targeting_button_change(self, sender, app_data):
        key = "" if app_data == self.tr("label_button_none") else app_data
        group_cfg = self.config["groups"][self.group]
        prev = group_cfg.get("targeting_button_key", "")
        group_cfg["targeting_button_key"] = key
        if key:
            if not self._ensure_aim_key(key):
                return
        other = group_cfg.get("triggerbot_button_key", "")
        if prev and prev != key and prev != other:
            self._remove_aim_key(prev)
        self._refresh_aim_keys_after_button_change()

    def on_triggerbot_button_change(self, sender, app_data):
        key = "" if app_data == self.tr("label_button_none") else app_data
        group_cfg = self.config["groups"][self.group]
        prev = group_cfg.get("triggerbot_button_key", "")
        group_cfg["triggerbot_button_key"] = key
        if key:
            if not self._ensure_aim_key(key):
                return
        other = group_cfg.get("targeting_button_key", "")
        if prev and prev != key and prev != other:
            self._remove_aim_key(prev)
        self._refresh_aim_keys_after_button_change()

    def on_disable_headshot_button_change(self, sender, app_data):
        key = "" if app_data == self.tr("label_button_none") else app_data
        group_cfg = self.config["groups"][self.group]
        group_cfg["disable_headshot_button_key"] = key
        group_cfg["disable_headshot_keys"] = [key] if key else []
        self.update_button_lists()

    def on_disable_headshot_class_change(self, sender, app_data):
        try:
            self.config["groups"][self.group]["disable_headshot_class_id"] = int(app_data)
        except Exception:
            self.config["groups"][self.group]["disable_headshot_class_id"] = -1

    def on_key_preset_change(self, sender, app_data):
        if not app_data:
            return
        self.add_key_name = app_data
        try:
            dpg.set_value("aim_key_name_input", app_data)
        except Exception:
            return

    def get_capture_source_items(self):
        return [
            self.tr("label_capture_standard"),
            self.tr("label_capture_obs"),
            self.tr("label_capture_card"),
        ]

    def get_capture_source_label(self):
        if self.config.get("is_obs", False):
            return self.tr("label_capture_obs")
        if self.config.get("is_cjk", False):
            return self.tr("label_capture_card")
        return self.tr("label_capture_standard")

    def parse_capture_source_label(self, label):
        if label in (self.tr("label_capture_obs"), "OBS"):
            return "obs"
        if label in (self.tr("label_capture_card"), "Capture Card"):
            return "cjk"
        return "standard"

    def on_capture_source_change(self, sender, app_data):
        mode = self.parse_capture_source_label(app_data)
        is_obs = mode == "obs"
        is_cjk = mode == "cjk"
        self.config["is_obs"] = is_obs
        self.config["is_cjk"] = is_cjk
        if self.screenshot_manager:
            self.screenshot_manager.update_config("is_obs", is_obs)
            self.screenshot_manager.update_config("is_cjk", is_cjk)
        self.update_capture_source_visibility()
        self.update_capture_region()
        self.update_capture_status_text()

    def update_capture_source_visibility(self):
        if hasattr(self, "obs_settings_group"):
            if self.config.get("is_obs", False):
                dpg.show_item(self.obs_settings_group)
            else:
                dpg.hide_item(self.obs_settings_group)
        if hasattr(self, "cjk_settings_group"):
            if self.config.get("is_cjk", False):
                dpg.show_item(self.cjk_settings_group)
            else:
                dpg.hide_item(self.cjk_settings_group)
        if hasattr(self, "standard_capture_group"):
            if self.config.get("is_obs", False) or self.config.get("is_cjk", False):
                dpg.hide_item(self.standard_capture_group)
            else:
                dpg.show_item(self.standard_capture_group)

    def get_sunone_variant_items(self):
        return [self.tr(key) for _, key in SUNONE_VARIANT_KEYS]

    def get_sunone_variant_label(self, variant):
        key = dict(SUNONE_VARIANT_KEYS).get(variant, "label_sunone_variant_yolo11")
        return self.tr(key)

    def parse_sunone_variant_label(self, label):
        if not label:
            return "yolo11"
        for variant, key in SUNONE_VARIANT_KEYS:
            if label == self.tr(key):
                return variant
        return "yolo11"

    def on_sunone_variant_change(self, sender, app_data):
        variant = self.parse_sunone_variant_label(app_data)
        self.config["groups"][self.group]["sunone_model_variant"] = variant
        self.config["groups"][self.group]["yolo_version"] = variant
        print(f"Sunone variant set to: {variant}")

    def sync_sunone_variant_ui(self):
        if not hasattr(self, "sunone_variant_combo"):
            return
        group_cfg = self.config["groups"][self.group]
        variant = group_cfg.get("yolo_version", group_cfg.get("sunone_model_variant", "yolo11"))
        dpg.set_value(self.sunone_variant_combo, self.get_sunone_variant_label(variant))

    def on_use_sunone_processing_change(self, sender, app_data):
        flag = bool(app_data)
        self.config["groups"][self.group]["use_sunone_processing"] = flag
        self.update_group_inputs()
        self.sync_sunone_variant_ui()
        state = "enabled" if flag else "disabled"
        print(f"Sunone processing {state}")

    def _parse_capture_size(self):
        try:
            value = str(self.config.get("capture_size", "auto")).strip().lower()
        except Exception:
            return None
        if not value or value == "auto":
            return None
        value = value.replace(" ", "")
        parts = value.split("x")
        if len(parts) != 2:
            return None
        if not parts[0].isdigit() or not parts[1].isdigit():
            return None
        w = int(parts[0])
        h = int(parts[1])
        if w <= 0 or h <= 0:
            return None
        return (w, h)

    def _get_engine_input_size(self):
        override = self._parse_capture_size()
        if override:
            return override
        if self.config.get("dynamic_shape", False):
            return (640, 640)
        if hasattr(self, "engine") and self.engine is not None:
            try:
                shape = self.engine.get_input_shape()
                return (int(shape[3]), int(shape[2]))
            except Exception:
                pass
        return (640, 640)

    def update_capture_status_text(self):
        if not hasattr(self, "capture_status_text") or self.capture_status_text is None:
            return
        if self.config.get("is_obs", False):
            details = (
                f"{self.tr('label_capture_obs')} "
                f"{self.config.get('obs_ip', '')}:{self.config.get('obs_port', '')} "
                f"({self.config.get('obs_fps', 0)} FPS)"
            )
        elif self.config.get("is_cjk", False):
            details = (
                f"{self.tr('label_capture_card')} "
                f"id {self.config.get('cjk_device_id', 0)} "
                f"{self.config.get('cjk_resolution', '')} "
                f"crop {self.config.get('cjk_crop_size', '')}"
            )
        else:
            offset_x = int(self.config.get("capture_offset_x", 0))
            offset_y = int(self.config.get("capture_offset_y", 0))
            size_label = ""
            region_w, region_h = self._get_engine_input_size()
            size_label = f" {region_w}x{region_h}"
            details = (
                f"{self.tr('label_capture_bettercam')}{size_label} "
                f"offset ({offset_x}, {offset_y})"
            )
        dpg.set_value(
            self.capture_status_text,
            f"{self.tr('label_capture_status')}: {details}",
        )

    def add_help_marker(self, text, wrap=None, same_line=True):
        tip = dpg.add_text("(?)", color=(150, 150, 150))
        if wrap is None:
            wrap = int(self.scaled_width_xlarge * 1.3)
        with dpg.tooltip(tip):
            dpg.add_text(text, wrap=wrap)

    def update_capture_region(self):
        if not hasattr(self, "engine") or self.engine is None:
            return
        region_w, region_h = self._get_engine_input_size()
        offset_x = int(self.config.get("capture_offset_x", 0))
        offset_y = int(self.config.get("capture_offset_y", 0))
        left = int((self.screen_width - region_w) // 2 + offset_x)
        top = int((self.screen_height - region_h) // 2 + offset_y)
        left = max(0, min(left, self.screen_width - region_w))
        top = max(0, min(top, self.screen_height - region_h))
        self.identify_rect_left = left
        self.identify_rect_top = top

    def get_system_dpi_scale(self):
        """Get system DPI scale ratio"""
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
            ctypes.windll.user32.ReleaseDC(0, hdc)
            scale = dpi / 96.0
            scale = max(1.0, min(scale, 3.0))
            return scale
        except Exception as e:
            print(f"Failed to get DPI scale, using default: {e}")
            return 1.0

    def get_dpi_aware_screen_size(self):
        """Get DPI-aware actual available screen size"""
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return (width, height)
        except Exception as e:
            width = win32api.GetSystemMetrics(0)
            height = win32api.GetSystemMetrics(1)
            return (width, height)

    def update_combo_methods(self):
        self.render_group_combo()
        self.update_target_reference_class_combo()

    def update_target_reference_class_combo(self):
        """Update target reference class combo box options"""
        if (
            not hasattr(self, "target_reference_class_combo")
            or self.target_reference_class_combo is None
        ):
            return None
        try:
            class_num = self.get_current_class_num()
            items = [self.format_class_label(i) for i in range(class_num)]
            import dearpygui.dearpygui as dpg

            dpg.configure_item(self.target_reference_class_combo, items=items)
            current_reference_class = self.pressed_key_config.get(
                "target_reference_class", 0
            )
            if current_reference_class < 0 or current_reference_class >= class_num:
                current_reference_class = 0
                self.config["groups"][self.group]["aim_keys"][self.select_key][
                    "target_reference_class"
                ] = 0
            dpg.set_value(
                self.target_reference_class_combo,
                self.format_class_label(current_reference_class),
            )
        except Exception as e:
            print(f"Failed to update target reference class combo: {e}")

    def get_gradient_color(base_color, step):
        """Generate color gradient based on base color"""
        r, g, b, a = base_color
        factor = 1 + step
        r = min(int(r * factor), 255)
        g = min(int(g * factor), 255)
        b = min(int(b * factor), 255)
        return (r, g, b, a)

    def gui(self):
        title = "".join(random.sample(string.ascii_letters + string.digits, 8)).join(
            VERSION
        )
        dpg.create_context()
        gradient_path = create_gradient_image(
            self.gui_window_width, self.scaled_bar_height
        )
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(8, 12, [0] * 384, tag="checkbox_texture")
        with dpg.texture_registry():
            width, height, channels, data = dpg.load_image(gradient_path)
            texture_id = dpg.add_static_texture(width, height, data)
        with dpg.font_registry():
            windir = os.environ.get("WINDIR", "C:\\Windows")
            font_candidates = [
                os.path.join(windir, "Fonts", "segoeui.ttf"),
                os.path.join(windir, "Fonts", "arial.ttf"),
                os.path.join(windir, "Fonts", "arialuni.ttf"),
                "msyh.ttc",
                "ChillBitmap_16px.ttf",
            ]
            font_path = "ChillBitmap_16px.ttf"
            for candidate in font_candidates:
                if os.path.exists(candidate):
                    font_path = candidate
                    break
            with dpg.font(font_path, self.scaled_font_size_main) as msyh:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
                dpg.bind_font(msyh)
            custom_font = dpg.add_font("undefeated.ttf", self.scaled_font_size_custom)
        with dpg.theme() as tab_bar_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_Tab, (45, 45, 45, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (45, 45, 45, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Tab, (45, 45, 45, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, (45, 45, 45, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (45, 45, 45, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (45, 45, 45, 255))
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 0)
                if hasattr(dpg, "mvStyleVar_TabPadding"):
                    dpg.add_theme_style(dpg.mvStyleVar_TabPadding, 4, 2)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 4, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 2)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 2)
        with dpg.theme() as skeet_theme:
            with dpg.theme_component(dpg.mvCheckbox):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (75, 75, 75, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (95, 95, 95, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (154, 197, 39, 255))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (255, 255, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (203, 203, 203, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 1, 1)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0)
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (35, 35, 35, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (65, 65, 65, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (25, 25, 25, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (50, 50, 50, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (203, 203, 203, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (150, 150, 150, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 5, 3)
            with dpg.theme_component(dpg.mvInputInt):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (45, 45, 45, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (65, 65, 65, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (85, 85, 85, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (203, 203, 203, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 4)
            with dpg.theme_component(dpg.mvCombo):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (45, 45, 45, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (65, 65, 65, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (85, 85, 85, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (203, 203, 203, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 255))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (35, 35, 35, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 4)
        dpg.create_viewport(
            title=title, width=self.gui_window_width, height=self.gui_window_height
        )
        dpg.setup_dearpygui()

        def switch_tab(sender, app_data, user_data):
            """Switch to corresponding tab"""
            dpg.set_value("tab_bar", user_data)
            if hasattr(self, "tab_bar_container"):
                dpg.set_x_scroll(self.tab_bar_container, 0)

        with dpg.window(
            label=title,
            no_title_bar=True,
            no_resize=True,
            no_move=True,
            width=self.gui_window_width,
            height=self.gui_window_height,
        ) as self.window_tag:
            dpg.draw_image(
                texture_id, (0, 0), (self.gui_window_width, self.scaled_bar_height)
            )
            with dpg.group():
                with dpg.child_window(
                    width=self.gui_window_width,
                    height=self.gui_window_height - 120,
                ) as tab_bar_container:
                    self.tab_bar_container = tab_bar_container
                    dpg.bind_item_theme(tab_bar_container, tab_bar_theme)
                    dpg.bind_theme(skeet_theme)
                    with dpg.tab_bar(tag="tab_bar", callback=self.on_tab_change):
                        with dpg.tab(tag="system_settings", label=self.tr("tab_system")):
                            with dpg.group(horizontal=True):
                                self.dpi_scale_slider = dpg.add_slider_float(
                                    label=self.tr("label_gui_dpi_scale"),
                                    default_value=self.dpi_scale,
                                    min_value=0.5,
                                    max_value=3.0,
                                    format="%.2f",
                                    callback=self.on_gui_dpi_scale_change,
                                    width=self.scaled_width_xlarge,
                                )
                                dpg.add_button(
                                    label=self.tr("label_auto_detect"),
                                    callback=self.on_reset_dpi_scale_click,
                                )
                            with dpg.group(horizontal=True):
                                lang_default = (
                                    "Русский"
                                    if self.config.get("ui_language", "en") == "ru"
                                    else "English"
                                )
                                dpg.add_combo(
                                    label=self.tr("label_ui_language"),
                                    items=["English", "Русский"],
                                    default_value=lang_default,
                                    callback=self.on_ui_language_change,
                                    width=self.scaled_width_large,
                                )
                                dpg.add_slider_float(
                                    label=self.tr("label_ui_font_scale"),
                                    default_value=self.config.get("gui_font_scale", 1.0),
                                    min_value=0.8,
                                    max_value=2.5,
                                    format="%.2f",
                                    callback=self.on_gui_font_scale_change,
                                    width=self.scaled_width_large,
                                )
                                dpg.add_slider_float(
                                    label=self.tr("label_ui_width_scale"),
                                    default_value=self.config.get("gui_width_scale", 1.0),
                                    min_value=0.8,
                                    max_value=1.6,
                                    format="%.2f",
                                    callback=self.on_gui_width_scale_change,
                                    width=self.scaled_width_xlarge,
                                )
                            dpg.add_text(
                                self.tr("label_ui_width_hint"),
                                color=(150, 150, 150),
                            )
                            dpg.add_text(
                                self.tr("label_ui_scale_detail").format(
                                    scale=self.get_system_dpi_scale()
                                )
                            )
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(
                                    label=self.tr("label_infer_window"),
                                    default_value=self.config["infer_debug"],
                                    callback=self.on_infer_debug_change,
                                )
                                dpg.add_checkbox(
                                    label=self.tr("label_print_fps"),
                                    default_value=self.config["print_fps"],
                                    callback=self.on_print_fps_change,
                                )
                                dpg.add_checkbox(
                                    label=self.tr("label_show_motion_speed"),
                                    default_value=self.config["show_motion_speed"],
                                    callback=self.on_show_motion_speed_change,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(
                                    label=self.tr("label_show_curve"),
                                    default_value=self.config["is_show_curve"],
                                    callback=self.on_is_show_curve_change,
                                )
                                dpg.add_checkbox(
                                    label=self.tr("label_show_infer_time"),
                                    default_value=self.config.get(
                                        "show_infer_time", True
                                    ),
                                    callback=self.on_show_infer_time_change,
                                )
                                dpg.add_checkbox(
                                    label=self.tr("label_screenshot_separation"),
                                    default_value=self.config.get(
                                        "enable_parallel_processing", True
                                    ),
                                    callback=self.on_enable_parallel_processing_change,
                                )
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_text(
                                    self.tr("label_small_target_settings"),
                                    color=(100, 200, 255),
                                )
                                self.add_help_marker(
                                    self.tr("help_small_target_settings"),
                                    same_line=False,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(
                                    label=self.tr("label_small_target_enhancement"),
                                    tag="small_target_enabled_checkbox",
                                    default_value=self.config[
                                        "small_target_enhancement"
                                    ]["enabled"],
                                    callback=self.on_small_target_enabled_change,
                                )
                                self.attach_tooltip(
                                    "small_target_enabled_checkbox",
                                    self.tr("help_small_target_enhancement"),
                                )
                                dpg.add_checkbox(
                                    label=self.tr("label_small_target_smoothing"),
                                    tag="small_target_smooth_checkbox",
                                    default_value=self.config[
                                        "small_target_enhancement"
                                    ]["smooth_enabled"],
                                    callback=self.on_small_target_smooth_change,
                                )
                                self.attach_tooltip(
                                    "small_target_smooth_checkbox",
                                    self.tr("help_small_target_smoothing"),
                                )
                                dpg.add_checkbox(
                                    label=self.tr("label_adaptive_nms"),
                                    tag="small_target_nms_checkbox",
                                    default_value=self.config[
                                        "small_target_enhancement"
                                    ]["adaptive_nms"],
                                    callback=self.on_small_target_nms_change,
                                )
                                self.attach_tooltip(
                                    "small_target_nms_checkbox",
                                    self.tr("help_adaptive_nms"),
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_input_float(
                                    label=self.tr("label_small_target_boost"),
                                    tag="small_target_boost_input",
                                    default_value=self.config[
                                        "small_target_enhancement"
                                    ]["boost_factor"],
                                    min_value=1.0,
                                    max_value=5.0,
                                    step=0.1,
                                    callback=self.on_small_target_boost_change,
                                    width=self.scaled_width_normal,
                                )
                                self.attach_tooltip(
                                    "small_target_boost_input",
                                    self.tr("help_small_target_boost"),
                                )
                                dpg.add_input_int(
                                    label=self.tr("label_small_target_history"),
                                    tag="small_target_frames_input",
                                    default_value=self.config[
                                        "small_target_enhancement"
                                    ]["smooth_frames"],
                                    min_value=2,
                                    max_value=10,
                                    callback=self.on_small_target_frames_change,
                                    width=self.scaled_width_medium,
                                )
                                self.attach_tooltip(
                                    "small_target_frames_input",
                                    self.tr("help_small_target_history"),
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_input_float(
                                    label=self.tr("label_small_target_threshold"),
                                    tag="small_target_threshold_input",
                                    default_value=self.config[
                                        "small_target_enhancement"
                                    ]["threshold"],
                                    min_value=0.001,
                                    max_value=0.1,
                                    step=0.001,
                                    format="%.3f",
                                    callback=self.on_small_target_threshold_change,
                                    width=self.scaled_width_normal,
                                )
                                self.attach_tooltip(
                                    "small_target_threshold_input",
                                    self.tr("help_small_target_threshold"),
                                )
                                dpg.add_input_float(
                                    label=self.tr("label_medium_target_threshold"),
                                    tag="medium_target_threshold_input",
                                    default_value=self.config[
                                        "small_target_enhancement"
                                    ]["medium_threshold"],
                                    min_value=0.01,
                                    max_value=0.2,
                                    step=0.01,
                                    format="%.3f",
                                    callback=self.on_medium_target_threshold_change,
                                    width=self.scaled_width_normal,
                                )
                                self.attach_tooltip(
                                    "medium_target_threshold_input",
                                    self.tr("help_medium_target_threshold"),
                                )
                            dpg.add_text(
                                self.tr("label_small_target_note"),
                                color=(150, 150, 150),
                                wrap=self.scaled_width_xlarge,
                            )
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(
                                    label=self.tr("label_turbo_mode"),
                                    default_value=self.config.get("turbo_mode", True),
                                    callback=self.on_turbo_mode_change,
                                )
                                dpg.add_checkbox(
                                    label=self.tr("label_skip_frame_processing"),
                                    default_value=self.config.get(
                                        "skip_frame_processing", True
                                    ),
                                    callback=self.on_skip_frame_processing_change,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(
                                    label=self.tr("label_recoil_debug"),
                                    default_value=self.config["is_show_down"],
                                    callback=self.on_is_show_down_change,
                                )
                                dpg.add_checkbox(
                                    label=self.tr("label_class_priority_debug"),
                                    default_value=self.config.get(
                                        "is_show_priority_debug", False
                                    ),
                                    callback=self.on_is_show_priority_debug_change,
                                )
                                dpg.add_checkbox(
                                    label=self.tr("label_show_aim_scope"),
                                    default_value=self.config.get("show_fov", True),
                                    callback=self.on_show_fov_change,
                                )
                            with dpg.group(horizontal=True):
                                self.capture_source_combo = dpg.add_combo(
                                    label=self.tr("label_capture_source"),
                                    items=self.get_capture_source_items(),
                                    default_value=self.get_capture_source_label(),
                                    callback=self.on_capture_source_change,
                                    width=self.scaled_width_large,
                                )
                                self.add_help_marker(
                                    self.tr("help_capture_source"),
                                    same_line=False,
                                )
                            self.standard_capture_group = dpg.add_group()
                            with dpg.group(
                                horizontal=True, parent=self.standard_capture_group
                            ):
                                dpg.add_input_int(
                                    label=self.tr("label_capture_offset_x"),
                                    default_value=self.config.get("capture_offset_x", 0),
                                    callback=self.on_capture_offset_x_change,
                                    width=self.scaled_width_normal,
                                )
                                dpg.add_input_int(
                                    label=self.tr("label_capture_offset_y"),
                                    default_value=self.config.get("capture_offset_y", 0),
                                    callback=self.on_capture_offset_y_change,
                                    width=self.scaled_width_normal,
                                )
                                self.add_help_marker(
                                    self.tr("help_capture_offsets"),
                                    same_line=False,
                                )
                            with dpg.group(
                                horizontal=True, parent=self.standard_capture_group
                            ):
                                self.capture_size_input = dpg.add_input_text(
                                    label=self.tr("label_capture_size"),
                                    default_value=self.config.get("capture_size", "auto"),
                                    callback=self.on_capture_size_change,
                                    width=self.scaled_width_medium,
                                    hint="auto/320x320/640x640",
                                )
                                self.add_help_marker(
                                    self.tr("help_capture_size"),
                                    same_line=False,
                                )
                            self.obs_settings_group = dpg.add_group()
                            with dpg.group(parent=self.obs_settings_group):
                                with dpg.group(horizontal=True):
                                    dpg.add_input_text(
                                        label=self.tr("label_obs_ip"),
                                        default_value=self.config["obs_ip"],
                                        callback=self.on_obs_ip_change,
                                        width=self.scaled_width_normal,
                                    )
                                    dpg.add_input_int(
                                        label=self.tr("label_obs_port"),
                                        default_value=self.config["obs_port"],
                                        callback=self.on_obs_port_change,
                                        width=self.scaled_width_normal,
                                    )
                                    dpg.add_input_int(
                                        label=self.tr("label_obs_fps"),
                                        default_value=self.config["obs_fps"],
                                        callback=self.on_obs_fps_change,
                                        width=self.scaled_width_normal,
                                    )
                            self.cjk_settings_group = dpg.add_group()
                            with dpg.group(parent=self.cjk_settings_group):
                                with dpg.group(horizontal=True):
                                    dpg.add_input_int(
                                        label=self.tr("label_capture_device"),
                                        default_value=self.config["cjk_device_id"],
                                        callback=self.on_cjk_device_id_change,
                                        width=self.scaled_width_normal,
                                    )
                                    dpg.add_input_int(
                                        label=self.tr("label_capture_fps"),
                                        default_value=self.config["cjk_fps"],
                                        callback=self.on_cjk_fps_change,
                                        width=self.scaled_width_normal,
                                    )
                                with dpg.group(horizontal=True):
                                    dpg.add_input_text(
                                        label=self.tr("label_capture_resolution"),
                                        default_value=self.config["cjk_resolution"],
                                        callback=self.on_cjk_resolution_change,
                                        width=self.scaled_width_medium,
                                    )
                                    dpg.add_input_text(
                                        label=self.tr("label_capture_crop"),
                                        default_value=self.config["cjk_crop_size"],
                                        callback=self.on_cjk_crop_size_change,
                                        width=self.scaled_width_medium,
                                    )
                                dpg.add_input_text(
                                    label=self.tr("label_video_codec"),
                                    default_value=self.config.get(
                                        "cjk_fourcc_format", "NV12"
                                    ),
                                    callback=self.on_cjk_fourcc_format_change,
                                    width=self.scaled_width_medium,
                                    hint="e.g.: NV12, MJPG, YUYV",
                                )
                            self.update_capture_source_visibility()
                            self.capture_status_text = dpg.add_text(
                                "",
                                color=(150, 150, 150),
                                wrap=self.scaled_width_xlarge,
                            )
                            self.update_capture_status_text()
                        with dpg.tab(tag="aim_settings", label=self.tr("tab_aim")):
                            self.build_aim_settings_ui()
                        with dpg.tab(tag="class_settings", label=self.tr("tab_classes")):
                            self.build_class_settings_ui()
                        with dpg.tab(tag="driver_settings", label=self.tr("tab_driver")):
                            dpg.add_text(
                                self.tr("help_driver"),
                                color=(150, 150, 150),
                                wrap=self.scaled_width_xlarge,
                            )
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(
                                    label=self.tr("label_move_curve"),
                                    default_value=self.config["is_curve"],
                                    callback=self.on_is_curve_change,
                                )
                                dpg.add_checkbox(
                                    label=self.tr("label_compensation_curve"),
                                    default_value=self.config["is_curve_uniform"],
                                    callback=self.on_is_curve_uniform_change,
                                )
                                dpg.add_input_int(
                                    label=self.tr("label_horizontal_boundary"),
                                    default_value=self.config["offset_boundary_x"],
                                    callback=self.on_offset_boundary_x_change,
                                    width=self.scaled_width_normal,
                                )
                                dpg.add_input_int(
                                    label=self.tr("label_vertical_boundary"),
                                    default_value=self.config["offset_boundary_y"],
                                    callback=self.on_offset_boundary_y_change,
                                    width=self.scaled_width_normal,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_input_int(
                                    label=self.tr("label_control_points"),
                                    default_value=self.config["knots_count"],
                                    callback=self.on_knots_count_change,
                                    width=self.scaled_width_normal,
                                )
                                dpg.add_input_float(
                                    label=self.tr("label_distortion_mean"),
                                    default_value=self.config["distortion_mean"],
                                    callback=self.on_distortion_mean_change,
                                    width=self.scaled_width_normal,
                                )
                                dpg.add_input_float(
                                    label=self.tr("label_distortion_stddev"),
                                    default_value=self.config["distortion_st_dev"],
                                    callback=self.on_distortion_st_dev_change,
                                    width=self.scaled_width_normal,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_input_float(
                                    label=self.tr("label_distortion_frequency"),
                                    default_value=self.config["distortion_frequency"],
                                    callback=self.on_distortion_frequency_change,
                                    width=self.scaled_width_normal,
                                )
                                dpg.add_input_int(
                                    label=self.tr("label_path_points_total"),
                                    default_value=self.config["target_points"],
                                    callback=self.on_target_points_change,
                                    width=self.scaled_width_normal,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_combo(
                                    label=self.tr("label_move_method"),
                                    items=[
                                        "makcu",
                                    ],
                                    default_value=self.config["move_method"],
                                    callback=self.on_move_method_change,
                                    width=self.scaled_width_large,
                                )
                        with dpg.tab(tag="bypass_settings", label=self.tr("tab_bypass")):
                            dpg.add_text(
                                self.tr("help_bypass"),
                                color=(150, 150, 150),
                                wrap=self.scaled_width_xlarge,
                            )
                            with dpg.group(horizontal=True):
                                self.mask_left_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_mask_left"),
                                    default_value=self.config["mask_left"],
                                    callback=self.on_mask_left_change,
                                )
                                self.mask_right_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_mask_right"),
                                    default_value=self.config["mask_right"],
                                    callback=self.on_mask_right_change,
                                )
                                self.mask_middle_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_mask_middle"),
                                    default_value=self.config["mask_middle"],
                                    callback=self.on_mask_middle_change,
                                )
                                self.mask_side1_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_mask_side1"),
                                    default_value=self.config["mask_side1"],
                                    callback=self.on_mask_side1_change,
                                )
                                self.mask_side2_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_mask_side2"),
                                    default_value=self.config["mask_side2"],
                                    callback=self.on_mask_side2_change,
                                )
                            with dpg.group(horizontal=True):
                                self.mask_x_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_mask_x_axis"),
                                    default_value=self.config["mask_x"],
                                    callback=self.on_mask_x_change,
                                )
                                self.mask_y_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_mask_y_axis"),
                                    default_value=self.config["mask_y"],
                                    callback=self.on_mask_y_change,
                                )
                                self.aim_mask_x_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_aim_mask_x"),
                                    default_value=self.config["aim_mask_x"],
                                    callback=self.on_aim_mask_x_change,
                                )
                                self.aim_mask_y_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_aim_mask_y"),
                                    default_value=self.config["aim_mask_y"],
                                    callback=self.on_aim_mask_y_change,
                                )
                                self.mask_wheel_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_mask_wheel"),
                                    default_value=self.config["mask_wheel"],
                                    callback=self.on_mask_wheel_change,
                                )
                        with dpg.tab(tag="strafe_settings", label=self.tr("tab_strafe")):
                            dpg.add_text(
                                self.tr("help_strafe"),
                                color=(150, 150, 150),
                                wrap=self.scaled_width_xlarge,
                            )
                            self.right_down_checkbox = dpg.add_checkbox(
                                label=self.tr("label_check_right_button"),
                                callback=self.on_right_down_change,
                            )
                            with dpg.group(horizontal=True):
                                with dpg.group() as self.dpg_games_tag:
                                    self.render_games_combo()
                                dpg.add_button(
                                    label=self.tr("label_delete_game"),
                                    callback=self.on_delete_game_click,
                                    width=self.scaled_width_60,
                                )
                                dpg.add_input_text(
                                    label=self.tr("label_game_name"),
                                    callback=self.on_game_name_change,
                                    width=self.scaled_width_60,
                                )
                                dpg.add_button(
                                    label=self.tr("label_add_game"),
                                    callback=self.on_add_game_click,
                                    width=self.scaled_width_60,
                                )
                            with dpg.group(horizontal=True):
                                with dpg.group() as self.dpg_guns_tag:
                                    self.render_guns_combo()
                                dpg.add_button(
                                    label=self.tr("label_delete_gun"),
                                    callback=self.on_delete_gun_click,
                                    width=self.scaled_width_60,
                                )
                                dpg.add_input_text(
                                    label=self.tr("label_gun_name"),
                                    callback=self.on_gun_name_change,
                                    width=self.scaled_width_60,
                                )
                                dpg.add_button(
                                    label=self.tr("label_add_gun"),
                                    callback=self.on_add_gun_click,
                                    width=self.scaled_width_60,
                                )
                            with dpg.group(horizontal=True):
                                with dpg.group() as self.dpg_stages_tag:
                                    self.render_stages_combo()
                                number = self.config["games"][self.picked_game][
                                    self.picked_gun
                                ][int(self.picked_stage)]["number"]
                                x = self.config["games"][self.picked_game][
                                    self.picked_gun
                                ][int(self.picked_stage)]["offset"][0]
                                y = self.config["games"][self.picked_game][
                                    self.picked_gun
                                ][int(self.picked_stage)]["offset"][1]
                            with dpg.group(horizontal=True):
                                self.number_input = dpg.add_input_int(
                                    label=self.tr("label_count"),
                                    callback=self.on_number_change,
                                    default_value=number,
                                    width=self.scaled_width_normal,
                                )
                                self.x_input = dpg.add_input_float(
                                    label=self.tr("label_axis_x"),
                                    step=0.01,
                                    callback=self.on_x_change,
                                    default_value=x,
                                    width=self.scaled_width_normal,
                                )
                                self.y_input = dpg.add_input_float(
                                    label=self.tr("label_axis_y"),
                                    step=0.01,
                                    callback=self.on_y_change,
                                    default_value=y,
                                    width=self.scaled_width_normal,
                                )
                                dpg.add_button(
                                    label=self.tr("label_delete_index"),
                                    callback=self.on_delete_stage_click,
                                    width=self.scaled_width_60,
                                )
                                dpg.add_button(
                                    label=self.tr("label_add_index"),
                                    callback=self.on_add_stage_click,
                                    width=self.scaled_width_60,
                                )
                            dpg.add_separator()
                            mouse_re_header = dpg.add_collapsing_header(
                                label=self.tr("label_mouse_re_header"), default_open=True
                            )
                            self.add_help_marker(self.tr("help_mouse_re"))
                            with dpg.group(parent=mouse_re_header):
                                with dpg.group(horizontal=True):
                                    dpg.add_checkbox(
                                        label=self.tr("label_mouse_re_enable"),
                                        default_value=self.config["recoil"][
                                            "use_mouse_re_trajectory"
                                        ],
                                        callback=self.on_use_mouse_re_trajectory_change,
                                    )
                                    dpg.add_input_float(
                                        label=self.tr("label_replay_speed"),
                                        default_value=self.config["recoil"][
                                            "replay_speed"
                                        ],
                                        step=0.1,
                                        min_value=0.1,
                                        max_value=5.0,
                                        format="%.2f",
                                        width=self.scaled_width_normal,
                                        callback=self.on_mouse_re_replay_speed_change,
                                    )
                                    dpg.add_input_float(
                                        label=self.tr("label_pixel_enhance_ratio"),
                                        default_value=self.config["recoil"][
                                            "pixel_enhancement_ratio"
                                        ],
                                        step=0.1,
                                        min_value=0.1,
                                        max_value=3.0,
                                        format="%.2f",
                                        width=self.scaled_width_normal,
                                        callback=self.on_mouse_re_pixel_enhancement_change,
                                    )
                                dpg.add_text(
                                    self.tr("label_mouse_re_config"),
                                    color=(150, 150, 150),
                                )
                                with dpg.group(
                                    horizontal=True, tag="mouse_re_combos_group"
                                ):
                                    pass
                                with dpg.group(horizontal=True):
                                    dpg.add_button(
                                        label=self.tr("label_import_trajectory_file"),
                                        callback=self.on_import_mouse_re_trajectory_click,
                                        width=self.scaled_width_normal,
                                    )
                                    dpg.add_button(
                                        label=self.tr("label_clear_mapping"),
                                        callback=self.on_clear_mouse_re_mapping_click,
                                        width=self.scaled_width_normal,
                                    )
                                dpg.add_text(self.tr("label_mouse_re_note"))
                                dpg.add_separator()
                                dpg.add_text(
                                    self.tr("label_current_status"),
                                    color=(150, 150, 150),
                                )
                                dpg.add_text(
                                    f"{self.tr('label_switch_prefix')}: {self.tr('label_switch_off')}",
                                    tag="mouse_re_switch_text",
                                )
                                dpg.add_text(
                                    f"{self.tr('label_mapping_file_prefix')}: {self.tr('label_none')}",
                                    wrap=self.scaled_width_xlarge,
                                    tag="mouse_re_file_text",
                                )
                                dpg.add_text(
                                    f"{self.tr('label_trajectory_points_prefix')}: 0",
                                    tag="mouse_re_points_text",
                                )
                        with dpg.tab(tag="config_settings", label=self.tr("tab_config")):
                            buttons_group = dpg.add_collapsing_header(
                                label=self.tr("label_buttons"), default_open=True
                            )
                            with dpg.group(parent=buttons_group):
                                with dpg.group(horizontal=True):
                                    dpg.add_text(self.tr("label_targeting_buttons"))
                                    self.targeting_button_combo = dpg.add_combo(
                                        items=self._get_button_options(),
                                        default_value=self.tr("label_button_none"),
                                        width=self.scaled_width_normal,
                                        callback=self.on_targeting_button_change,
                                    )
                                with dpg.group(horizontal=True):
                                    dpg.add_text(self.tr("label_triggerbot_buttons"))
                                    self.triggerbot_button_combo = dpg.add_combo(
                                        items=self._get_button_options(),
                                        default_value=self.tr("label_button_none"),
                                        width=self.scaled_width_normal,
                                        callback=self.on_triggerbot_button_change,
                                    )
                                with dpg.group(horizontal=True):
                                    dpg.add_text(self.tr("label_disable_headshot_buttons"))
                                    self.disable_headshot_button_combo = dpg.add_combo(
                                        items=self._get_button_options(),
                                        default_value=self.tr("label_button_none"),
                                        width=self.scaled_width_normal,
                                        callback=self.on_disable_headshot_button_change,
                                    )
                                with dpg.group(horizontal=True):
                                    dpg.add_text(self.tr("label_disable_headshot_class"))
                                    self.disable_headshot_class_input = dpg.add_input_int(
                                        default_value=self.config["groups"][self.group].get(
                                            "disable_headshot_class_id", -1
                                        ),
                                        min_value=-1,
                                        max_value=999,
                                        callback=self.on_disable_headshot_class_change,
                                        width=self.scaled_width_normal,
                                    )
                                self.disable_headshot_status_text = dpg.add_text("")
                                self.update_disable_headshot_status()

                            self.aim_key_combo_group = None
                            self.key_tag = None
                            self.key_preset_combo = None
                            self.key_name_input = None
                            model_params_group = dpg.add_collapsing_header(
                                label=self.tr("label_model_params"), default_open=True
                            )
                            with dpg.group(horizontal=True, parent=model_params_group):
                                with dpg.group() as self.dpg_group_tag:
                                    self.render_group_combo()
                                dpg.add_button(
                                    label=self.tr("label_delete_group"),
                                    callback=self.on_delete_group_click,
                                    width=self.scaled_width_60,
                                )
                                dpg.add_input_text(
                                    label=self.tr("label_group_name"),
                                    callback=self.on_group_name_change,
                                    width=self.scaled_width_60,
                                )
                                dpg.add_button(
                                    label=self.tr("label_add_group"),
                                    callback=self.on_add_group_click,
                                    width=self.scaled_width_60,
                                )
                            with dpg.group(horizontal=True, parent=model_params_group):
                                self.is_trt_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_trt"),
                                    callback=self.on_is_trt_change,
                                )
                                self.use_sunone_processing_checkbox = dpg.add_checkbox(
                                    label=self.tr("label_use_sunone_processing"),
                                    tag="use_sunone_processing_checkbox",
                                    default_value=self.config["groups"][self.group].get(
                                        "use_sunone_processing", False
                                    ),
                                    callback=self.on_use_sunone_processing_change,
                                )
                                self.sunone_variant_combo = dpg.add_combo(
                                    label=self.tr("label_sunone_variant"),
                                    items=self.get_sunone_variant_items(),
                                    default_value=self.get_sunone_variant_label(
                                        self.config["groups"][self.group].get(
                                            "sunone_model_variant", "yolo11"
                                        )
                                    ),
                                    callback=self.on_sunone_variant_change,
                                    width=self.scaled_width_normal,
                                )
                                self.attach_tooltip(
                                    self.sunone_variant_combo,
                                    self.tr("help_sunone_variant"),
                                )
                            with dpg.group(horizontal=True, parent=model_params_group):
                                self.infer_model_input = dpg.add_input_text(
                                    label=self.tr("label_infer_model"),
                                    default_value=self.config["groups"][self.group][
                                        "infer_model"
                                    ],
                                    callback=self.on_infer_model_change,
                                    width=self.scaled_width_xlarge + 50,
                                )
                                dpg.add_button(
                                    label=self.tr("label_select_model"),
                                    callback=self.on_select_model_click,
                                    width=100,
                                )
                            self.update_group_inputs()
                            self.sync_sunone_variant_ui()
                    with dpg.group(horizontal=True):
                        self.start_button_tag = dpg.add_button(
                            label=self.tr("label_start"),
                            callback=self.on_start_button_click,
                            width=self.scaled_width_normal,
                        )
                        dpg.add_button(
                            label=self.tr("label_save_config"),
                            callback=self.on_save_button_click,
                            width=self.scaled_width_normal,
                        )
                        dpg.add_text("", tag="output_text")
            with dpg.group():
                dpg.add_spacer(height=5)
                with dpg.group(horizontal=True):
                    version_text = dpg.add_text(f"Version: {VERSION} {UPDATE_TIME}")
                    with dpg.theme() as version_theme, dpg.theme_component(dpg.mvText):
                        dpg.add_theme_color(dpg.mvThemeCol_Text, (150, 150, 150, 255))
                    dpg.bind_item_theme(version_text, version_theme)
            dpg.show_viewport()
            dpg.set_primary_window(self.window_tag, True)
            self.update_sensitivity_display()
            self.render_mouse_re_games_combo()
            self.render_mouse_re_guns_combo()
            self.update_mouse_re_ui_status()
            self.update_controller_visibility()
            self.start_preview_animation()
            self.update_button_lists()
            if hasattr(self, "tab_bar_container"):
                dpg.set_x_scroll(self.tab_bar_container, 0)
            dpg.start_dearpygui()
        self.running = False
        self.disconnect_device()
        self.close_screenshot()
        print("Exit")
        dpg.destroy_context()

    def build_aim_settings_ui(self):
        controller_items = [
            self.tr("label_controller_pid"),
            self.tr("label_controller_sunone"),
        ]
        current_controller = self.config.get("aim_controller", "pid")
        controller_default = (
            controller_items[1]
            if current_controller == "sunone"
            else controller_items[0]
        )
        self.aim_controller_combo = dpg.add_combo(
            label=f"{self.tr('label_aim_controller')} (?)",
            items=controller_items,
            default_value=controller_default,
            width=self.scaled_width_large,
            callback=self.on_aim_controller_change,
        )
        self.attach_tooltip(self.aim_controller_combo, self.tr("help_aim_controller"))

        self.sunone_settings_group = dpg.add_collapsing_header(
            label=self.tr("label_sunone_settings"), default_open=True
        )
        with dpg.group(parent=self.sunone_settings_group):
            with dpg.group(horizontal=True):
                dpg.add_text(self.tr("label_help_sunone"))
                self.add_help_marker(self.tr("help_sunone_settings"))
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label=self.tr("label_sunone_smoothing"),
                    tag="sunone_use_smoothing",
                    default_value=self.config["sunone"]["use_smoothing"],
                    callback=self.on_change,
                )
                dpg.add_input_int(
                    label=self.tr("label_sunone_smoothness"),
                    tag="sunone_smoothness",
                    default_value=self.config["sunone"]["smoothness"],
                    min_value=1,
                    max_value=50,
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                dpg.add_checkbox(
                    label=self.tr("label_sunone_tracking"),
                    tag="sunone_tracking_smoothing",
                    default_value=self.config["sunone"]["tracking_smoothing"],
                    callback=self.on_change,
                )
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label=self.tr("label_sunone_kalman"),
                    tag="sunone_use_kalman",
                    default_value=self.config["sunone"]["use_kalman"],
                    callback=self.on_change,
                )
                self.add_help_marker(self.tr("help_kalman"))
            self.sunone_kalman_group = dpg.add_group()
            with dpg.group(horizontal=True, parent=self.sunone_kalman_group):
                dpg.add_input_float(
                    label=self.tr("label_sunone_kalman_q"),
                    tag="sunone_kalman_process_noise",
                    default_value=self.config["sunone"]["kalman_process_noise"],
                    min_value=0.0001,
                    max_value=10.0,
                    format="%.4f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                dpg.add_input_float(
                    label=self.tr("label_sunone_kalman_r"),
                    tag="sunone_kalman_measurement_noise",
                    default_value=self.config["sunone"]["kalman_measurement_noise"],
                    min_value=0.0001,
                    max_value=10.0,
                    format="%.4f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
            with dpg.group(horizontal=True, parent=self.sunone_kalman_group):
                dpg.add_input_float(
                    label=self.tr("label_sunone_kalman_mul_x"),
                    tag="sunone_kalman_speed_multiplier_x",
                    default_value=self.config["sunone"]["kalman_speed_multiplier_x"],
                    min_value=0.1,
                    max_value=5.0,
                    step=0.1,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                dpg.add_input_float(
                    label=self.tr("label_sunone_kalman_mul_y"),
                    tag="sunone_kalman_speed_multiplier_y",
                    default_value=self.config["sunone"]["kalman_speed_multiplier_y"],
                    min_value=0.1,
                    max_value=5.0,
                    step=0.1,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label=self.tr("label_sunone_reset_threshold"),
                    tag="sunone_reset_threshold",
                    default_value=self.config["sunone"]["reset_threshold"],
                    min_value=0.5,
                    max_value=50.0,
                    step=0.5,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )

            dpg.add_separator()
            self.sunone_prediction_group = dpg.add_collapsing_header(
                label=self.tr("label_sunone_prediction"), default_open=True
            )
            with dpg.group(horizontal=True, parent=self.sunone_prediction_group):
                dpg.add_text(self.tr("label_help_prediction"))
                self.add_help_marker(self.tr("help_prediction"))
            self.sunone_prediction_container = dpg.add_group(
                parent=self.sunone_prediction_group, horizontal=True
            )
            self.sunone_prediction_left = dpg.add_group(
                parent=self.sunone_prediction_container
            )
            self.sunone_prediction_preview_group = dpg.add_group(
                parent=self.sunone_prediction_container
            )
            prediction_modes = ["Standard", "Kalman", "Kalman + Standard"]
            mode_idx = int(self.config["sunone"]["prediction"]["mode"])
            mode_idx = max(0, min(mode_idx, len(prediction_modes) - 1))
            self.sunone_prediction_mode_combo = dpg.add_combo(
                label=self.tr("label_sunone_prediction_mode"),
                items=prediction_modes,
                default_value=prediction_modes[mode_idx],
                width=self.scaled_width_large,
                callback=self.on_sunone_prediction_mode_change,
                parent=self.sunone_prediction_left,
            )
            self.sunone_pred_standard_group = dpg.add_group(
                parent=self.sunone_prediction_left
            )
            with dpg.group(horizontal=True, parent=self.sunone_pred_standard_group):
                dpg.add_input_float(
                    label=self.tr("label_sunone_prediction_interval"),
                    tag="sunone_prediction_interval",
                    default_value=self.config["sunone"]["prediction"]["interval"],
                    min_value=0.0,
                    max_value=0.5,
                    step=0.01,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
            self.sunone_pred_kalman_group = dpg.add_group(
                parent=self.sunone_prediction_left
            )
            with dpg.group(horizontal=True, parent=self.sunone_pred_kalman_group):
                self.sunone_prediction_lead_input = dpg.add_input_float(
                    label=self.tr("label_sunone_prediction_lead"),
                    tag="sunone_prediction_kalman_lead_ms",
                    default_value=self.config["sunone"]["prediction"]["kalman_lead_ms"],
                    min_value=0.0,
                    max_value=150.0,
                    step=1.0,
                    format="%.0f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                self.sunone_prediction_max_lead_input = dpg.add_input_float(
                    label=self.tr("label_sunone_prediction_max_lead"),
                    tag="sunone_prediction_kalman_max_lead_ms",
                    default_value=self.config["sunone"]["prediction"][
                        "kalman_max_lead_ms"
                    ],
                    min_value=0.0,
                    max_value=250.0,
                    step=1.0,
                    format="%.0f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                self.add_help_marker(self.tr("help_prediction_lead"))
                self.attach_tooltip(
                    self.sunone_prediction_lead_input,
                    self.tr("help_prediction_lead"),
                )
            with dpg.group(horizontal=True, parent=self.sunone_pred_kalman_group):
                self.sunone_prediction_velocity_smoothing_input = dpg.add_input_float(
                    label=self.tr("label_sunone_velocity_smoothing"),
                    tag="sunone_prediction_velocity_smoothing",
                    default_value=self.config["sunone"]["prediction"][
                        "velocity_smoothing"
                    ],
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                self.sunone_prediction_velocity_scale_input = dpg.add_input_float(
                    label=self.tr("label_sunone_velocity_scale"),
                    tag="sunone_prediction_velocity_scale",
                    default_value=self.config["sunone"]["prediction"]["velocity_scale"],
                    min_value=0.1,
                    max_value=3.0,
                    step=0.1,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                self.add_help_marker(self.tr("help_velocity_smoothing"))
                self.attach_tooltip(
                    self.sunone_prediction_velocity_smoothing_input,
                    self.tr("help_velocity_smoothing"),
                )
            with dpg.group(horizontal=True, parent=self.sunone_pred_kalman_group):
                dpg.add_input_float(
                    label=self.tr("label_sunone_prediction_q"),
                    tag="sunone_prediction_kalman_process_noise",
                    default_value=self.config["sunone"]["prediction"][
                        "kalman_process_noise"
                    ],
                    min_value=0.0001,
                    max_value=10.0,
                    format="%.4f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                dpg.add_input_float(
                    label=self.tr("label_sunone_prediction_r"),
                    tag="sunone_prediction_kalman_measurement_noise",
                    default_value=self.config["sunone"]["prediction"][
                        "kalman_measurement_noise"
                    ],
                    min_value=0.0001,
                    max_value=10.0,
                    format="%.4f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
            with dpg.group(horizontal=True, parent=self.sunone_prediction_left):
                dpg.add_input_int(
                    label=self.tr("label_sunone_future_positions"),
                    tag="sunone_prediction_future_positions",
                    default_value=self.config["sunone"]["prediction"]["future_positions"],
                    min_value=1,
                    max_value=40,
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                dpg.add_checkbox(
                    label=self.tr("label_sunone_draw_future"),
                    tag="sunone_prediction_draw_future_positions",
                    default_value=self.config["sunone"]["prediction"][
                        "draw_future_positions"
                    ],
                    callback=self.on_change,
                )
            with dpg.group(parent=self.sunone_prediction_preview_group):
                dpg.add_text(self.tr("label_prediction_preview"))
                self.prediction_preview_drawlist = dpg.add_drawlist(
                    width=int(self.scaled_width_large * 1.6),
                    height=int(self.scaled_height_normal * 1.6),
                )
            self.update_prediction_preview()

            dpg.add_separator()
            self.sunone_speed_group = dpg.add_collapsing_header(
                label=self.tr("label_sunone_speed"), default_open=True
            )
            with dpg.group(horizontal=True, parent=self.sunone_speed_group):
                dpg.add_text(self.tr("label_help_speed_curve"))
                self.add_help_marker(self.tr("help_speed_curve"))
            with dpg.group(horizontal=True, parent=self.sunone_speed_group):
                dpg.add_input_float(
                    label=self.tr("label_sunone_min_speed"),
                    tag="sunone_speed_min_multiplier",
                    default_value=self.config["sunone"]["speed"]["min_multiplier"],
                    min_value=0.01,
                    max_value=2.0,
                    step=0.01,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                dpg.add_input_float(
                    label=self.tr("label_sunone_max_speed"),
                    tag="sunone_speed_max_multiplier",
                    default_value=self.config["sunone"]["speed"]["max_multiplier"],
                    min_value=0.01,
                    max_value=2.0,
                    step=0.01,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
            with dpg.group(horizontal=True, parent=self.sunone_speed_group):
                dpg.add_input_float(
                    label=self.tr("label_sunone_snap_radius"),
                    tag="sunone_speed_snap_radius",
                    default_value=self.config["sunone"]["speed"]["snap_radius"],
                    min_value=0.1,
                    max_value=100.0,
                    step=0.1,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                dpg.add_input_float(
                    label=self.tr("label_sunone_near_radius"),
                    tag="sunone_speed_near_radius",
                    default_value=self.config["sunone"]["speed"]["near_radius"],
                    min_value=1.0,
                    max_value=400.0,
                    step=1.0,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
            with dpg.group(horizontal=True, parent=self.sunone_speed_group):
                dpg.add_input_float(
                    label=self.tr("label_sunone_curve_exp"),
                    tag="sunone_speed_curve_exponent",
                    default_value=self.config["sunone"]["speed"]["speed_curve_exponent"],
                    min_value=1.0,
                    max_value=20.0,
                    step=0.5,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
                dpg.add_input_float(
                    label=self.tr("label_sunone_snap_boost"),
                    tag="sunone_speed_snap_boost_factor",
                    default_value=self.config["sunone"]["speed"]["snap_boost_factor"],
                    min_value=0.1,
                    max_value=6.0,
                    step=0.1,
                    format="%.2f",
                    callback=self.on_change,
                    width=self.scaled_width_normal,
                )
            with dpg.group(parent=self.sunone_speed_group):
                dpg.add_text(self.tr("label_speed_curve_preview"))
                self.speed_curve_preview_drawlist = dpg.add_drawlist(
                    width=int(self.scaled_width_large * 1.4),
                    height=int(self.scaled_height_normal * 1.4),
                )
            self.update_speed_curve_preview()

            dpg.add_separator()
            self.sunone_debug_group = dpg.add_collapsing_header(
                label=self.tr("label_sunone_debug"), default_open=False
            )
            with dpg.group(horizontal=True, parent=self.sunone_debug_group):
                dpg.add_checkbox(
                    label=self.tr("label_sunone_debug_pred"),
                    tag="sunone_debug_show_prediction",
                    default_value=self.config["sunone"]["debug"]["show_prediction"],
                    callback=self.on_change,
                )
                dpg.add_checkbox(
                    label=self.tr("label_sunone_debug_step"),
                    tag="sunone_debug_show_step",
                    default_value=self.config["sunone"]["debug"]["show_step"],
                    callback=self.on_change,
                )
                dpg.add_checkbox(
                    label=self.tr("label_sunone_debug_future"),
                    tag="sunone_debug_show_future",
                    default_value=self.config["sunone"]["debug"]["show_future"],
                    callback=self.on_change,
                )

        dpg.add_separator()
        dpg.add_text(self.tr("label_long_press_section"))
        with dpg.group(horizontal=True):
            self.auto_y_checkbox = dpg.add_checkbox(
                label=self.tr("label_long_press_no_lock_y"),
                callback=self.on_auto_y_change,
            )
            self.attach_tooltip(
                self.auto_y_checkbox, self.tr("help_long_press_no_lock_y")
            )
            self.long_press_duration_input = dpg.add_input_int(
                label=self.tr("label_long_press_threshold"),
                default_value=self.config["groups"][self.group][
                    "long_press_duration"
                ],
                callback=self.on_long_press_duration_change,
                width=self.scaled_width_normal,
            )
            self.attach_tooltip(
                self.long_press_duration_input,
                self.tr("help_long_press_threshold"),
            )
        with dpg.group(horizontal=True):
            self.target_switch_delay_input = dpg.add_input_int(
                label=self.tr("label_target_switch_delay"),
                default_value=0,
                min_value=0,
                max_value=2000,
                callback=self.on_target_switch_delay_change,
                width=self.scaled_width_normal,
            )
            self.target_reference_class_combo = dpg.add_combo(
                label=self.tr("label_target_reference_class"),
                items=["Class0"],
                default_value="Class0",
                callback=self.on_target_reference_class_change,
                width=self.scaled_width_normal,
            )
        self.update_target_reference_class_combo()
        dpg.add_separator()
        with dpg.group(horizontal=True):
            self.min_position_offset_input = dpg.add_input_int(
                label=self.tr("label_min_offset"),
                callback=self.on_min_position_offset_change,
                width=self.scaled_width_normal,
            )
            self.aim_bot_scope_input = dpg.add_input_int(
                label=self.tr("label_aim_scope"),
                callback=self.on_aim_bot_scope_change,
                width=self.scaled_width_normal,
            )
        with dpg.group(horizontal=True):
            self.dynamic_scope_enabled_input = dpg.add_checkbox(
                label=self.tr("label_dynamic_scope"),
                callback=self.on_dynamic_scope_enabled_change,
            )
            self.attach_tooltip(
                self.dynamic_scope_enabled_input, self.tr("help_smart_target")
            )
        with dpg.group(horizontal=True):
            self.target_lock_distance_input = dpg.add_input_int(
                label=self.tr("label_target_lock_distance"),
                min_value=1,
                max_value=2000,
                callback=self.on_target_lock_distance_change,
                width=self.scaled_width_normal,
            )
            self.attach_tooltip(
                self.target_lock_distance_input, self.tr("help_target_lock_distance")
            )
            self.target_lock_reacquire_time_input = dpg.add_input_float(
                label=self.tr("label_target_lock_reacquire_time"),
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                format="%.2f",
                callback=self.on_target_lock_reacquire_time_change,
                width=self.scaled_width_normal,
            )
            self.attach_tooltip(
                self.target_lock_reacquire_time_input,
                self.tr("help_target_lock_reacquire_time"),
            )
        with dpg.group(horizontal=True):
            self.move_deadzone_input = dpg.add_input_float(
                label=self.tr("label_move_deadzone"),
                default_value=1.0,
                min_value=0.0,
                max_value=10.0,
                step=0.0001,
                format="%.4f",
                callback=self.on_move_deadzone_change,
                width=self.scaled_width_normal,
            )
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text(self.tr("label_aim_weights"))
            self.add_help_marker(self.tr("help_aim_weights"))
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label=self.tr("label_distance_weight"),
                default_value=self.config["distance_scoring_weight"],
                min_value=0.0,
                step=0.05,
                callback=self.on_distance_scoring_weight_change,
                width=self.scaled_width_normal,
            )
            dpg.add_input_float(
                label=self.tr("label_center_weight"),
                default_value=self.config["center_scoring_weight"],
                min_value=0.0,
                step=0.05,
                callback=self.on_center_scoring_weight_change,
                width=self.scaled_width_normal,
            )
            dpg.add_input_float(
                label=self.tr("label_size_weight"),
                default_value=self.config["size_scoring_weight"],
                min_value=0.0,
                step=0.05,
                callback=self.on_size_scoring_weight_change,
                width=self.scaled_width_normal,
            )
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label=self.tr("label_aim_weight_tiebreak"),
                default_value=float(self.config.get("aim_weight_tiebreak_ratio", 0.1)) * 100.0,
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format="%.0f",
                callback=self.on_aim_weight_tiebreak_change,
                width=self.scaled_width_normal,
            )
            self.add_help_marker(self.tr("help_aim_weight_tiebreak"))
        self.pid_params_group = dpg.add_collapsing_header(
            label=self.tr("label_pid_params"), default_open=True
        )
        dpg.add_text(self.tr("label_pid_params"), parent=self.pid_params_group)
        with dpg.group(horizontal=True, parent=self.pid_params_group):
            self.pid_kp_x_input = dpg.add_input_float(
                label=self.tr("label_pid_x_p"),
                default_value=0.4,
                step=0.0001,
                format="%.4f",
                callback=self.on_pid_kp_x_change,
                width=self.scaled_width_normal,
            )
            self.pid_ki_x_input = dpg.add_input_float(
                label=self.tr("label_pid_x_i"),
                default_value=0.0,
                step=0.0001,
                format="%.4f",
                callback=self.on_pid_ki_x_change,
                width=self.scaled_width_normal,
            )
            self.pid_kd_x_input = dpg.add_input_float(
                label=self.tr("label_pid_x_d"),
                default_value=0.002,
                step=0.0001,
                format="%.4f",
                callback=self.on_pid_kd_x_change,
                width=self.scaled_width_normal,
            )
        with dpg.group(horizontal=True, parent=self.pid_params_group):
            self.pid_kp_y_input = dpg.add_input_float(
                label=self.tr("label_pid_y_p"),
                default_value=0.4,
                step=0.0001,
                format="%.4f",
                callback=self.on_pid_kp_y_change,
                width=self.scaled_width_normal,
            )
            self.pid_ki_y_input = dpg.add_input_float(
                label=self.tr("label_pid_y_i"),
                default_value=0,
                step=0.0001,
                format="%.4f",
                callback=self.on_pid_ki_y_change,
                width=self.scaled_width_normal,
            )
            self.pid_kd_y_input = dpg.add_input_float(
                label=self.tr("label_pid_y_d"),
                default_value=0.002,
                step=0.0001,
                format="%.4f",
                callback=self.on_pid_kd_y_change,
                width=self.scaled_width_normal,
            )
        with dpg.group(horizontal=True, parent=self.pid_params_group):
            self.pid_integral_limit_x_input = dpg.add_input_float(
                label=self.tr("label_pid_x_limit"),
                default_value=0.0,
                min_value=0.0,
                max_value=100.0,
                step=0.0001,
                format="%.4f",
                callback=self.on_pid_integral_limit_x_change,
                width=self.scaled_width_normal,
            )
            self.smooth_x_input = dpg.add_input_float(
                label=self.tr("label_pid_x_smooth"),
                default_value=0,
                min_value=0.0,
                max_value=1000.0,
                step=0.0001,
                format="%.4f",
                callback=self.on_smooth_x_change,
                width=self.scaled_width_normal,
            )
            self.smooth_algorithm_input = dpg.add_input_float(
                label=self.tr("label_smooth_algorithm"),
                default_value=1.0,
                min_value=0.1,
                max_value=10.0,
                step=0.0001,
                format="%.4f",
                callback=self.on_smooth_algorithm_change,
                width=self.scaled_width_normal,
            )
        with dpg.group(horizontal=True, parent=self.pid_params_group):
            self.pid_integral_limit_y_input = dpg.add_input_float(
                label=self.tr("label_pid_y_limit"),
                default_value=0.0,
                min_value=0.0,
                max_value=100.0,
                step=0.0001,
                format="%.4f",
                callback=self.on_pid_integral_limit_y_change,
                width=self.scaled_width_normal,
            )
            self.smooth_y_input = dpg.add_input_float(
                label=self.tr("label_pid_y_smooth"),
                default_value=0,
                min_value=0.0,
                max_value=1000.0,
                step=0.0001,
                format="%.4f",
                callback=self.on_smooth_y_change,
                width=self.scaled_width_normal,
            )
            self.smooth_deadzone_input = dpg.add_input_float(
                label=self.tr("label_smooth_deadzone"),
                default_value=0.0,
                min_value=0.0,
                max_value=50.0,
                step=0.0001,
                format="%.4f",
                callback=self.on_smooth_deadzone_change,
                width=self.scaled_width_normal,
            )
        trigger_setting_tag = dpg.add_collapsing_header(
            label=f"{self.tr('label_trigger_config')} (?)", default_open=True
        )
        self.attach_tooltip(trigger_setting_tag, self.tr("help_trigger"))
        with dpg.group(parent=trigger_setting_tag):
            with dpg.group(horizontal=True):
                self.status_input = dpg.add_checkbox(
                    label=self.tr("label_auto_trigger"),
                    callback=self.on_status_change,
                )
                self.continuous_trigger_input = dpg.add_checkbox(
                    label=self.tr("label_continuous_trigger"),
                    callback=self.on_continuous_trigger_change,
                )
        with dpg.group(horizontal=True):
            self.start_delay_input = dpg.add_input_int(
                label=self.tr("label_trigger_delay"),
                min_value=0,
                max_value=1000,
                callback=self.on_start_delay_change,
                width=self.scaled_width_normal,
            )
            self.press_delay_input = dpg.add_input_int(
                label=self.tr("label_press_duration"),
                min_value=0,
                max_value=1000,
                callback=self.on_press_delay_change,
                width=self.scaled_width_normal,
            )
        with dpg.group(horizontal=True):
            self.end_delay_input = dpg.add_input_int(
                label=self.tr("label_trigger_cooldown"),
                min_value=0,
                max_value=1000,
                callback=self.on_end_delay_change,
                width=self.scaled_width_normal,
            )
            self.random_delay_input = dpg.add_input_int(
                label=self.tr("label_random_delay"),
                min_value=0,
                max_value=1000,
                callback=self.on_random_delay_change,
                width=self.scaled_width_normal,
            )
        with dpg.group(horizontal=True):
            self.x_trigger_scope_input = dpg.add_input_float(
                label=self.tr("label_x_trigger_scope"),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                callback=self.on_x_trigger_scope_change,
                width=self.scaled_width_normal,
            )
            self.y_trigger_scope_input = dpg.add_input_float(
                label=self.tr("label_y_trigger_scope"),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                callback=self.on_y_trigger_scope_change,
                width=self.scaled_width_normal,
            )
        with dpg.group(horizontal=True):
            self.x_trigger_offset_input = dpg.add_input_float(
                label=self.tr("label_x_trigger_offset"),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                callback=self.on_x_trigger_offset_change,
                width=self.scaled_width_normal,
            )
            self.y_trigger_offset_input = dpg.add_input_float(
                label=self.tr("label_y_trigger_offset"),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                callback=self.on_y_trigger_offset_change,
                width=self.scaled_width_normal,
            )
        with dpg.drawlist(
            width=self.scaled_width_small,
            height=self.scaled_height_normal,
        ):
            dpg.draw_rectangle((0, 0), (50, 100), color=(255, 255, 255))
            x_trigger_offset = self.config["groups"][self.group]["aim_keys"][
                self.select_key
            ]["trigger"]["x_trigger_offset"]
            y_trigger_offset = self.config["groups"][self.group]["aim_keys"][
                self.select_key
            ]["trigger"]["y_trigger_offset"]
            x_trigger_scope = self.config["groups"][self.group]["aim_keys"][
                self.select_key
            ]["trigger"]["x_trigger_scope"]
            y_trigger_scope = self.config["groups"][self.group]["aim_keys"][
                self.select_key
            ]["trigger"]["y_trigger_scope"]
            x_trigger_offset = x_trigger_offset * 50
            y_trigger_offset = y_trigger_offset * 100
            dpg.draw_rectangle(
                (x_trigger_offset, y_trigger_offset),
                (
                    x_trigger_offset + 50 * x_trigger_scope,
                    y_trigger_offset + 100 * y_trigger_scope,
                ),
                fill=(255, 0, 0),
                tag="small_rect",
            )
        self.update_sunone_visibility()
        self.update_controller_visibility()

    def build_class_settings_ui(self):
        class_group = dpg.add_collapsing_header(
            label=f"{self.tr('label_class_settings')} (?)", default_open=True
        )
        with dpg.group(horizontal=True, parent=class_group):
            self.class_names_file_input = dpg.add_input_text(
                label=self.tr("label_class_names_file"),
                default_value=self.config.get("class_names_file", ""),
                readonly=True,
                width=self.scaled_width_xlarge,
            )
            dpg.add_button(
                label=self.tr("label_load_class_names"),
                callback=self.on_select_class_names_click,
                width=self.scaled_width_normal,
            )
        with dpg.group(horizontal=True, parent=class_group):
            with dpg.group():
                self.class_names_manual_input = dpg.add_input_text(
                    label=self.tr("label_class_names_manual"),
                    default_value="\n".join(self.config.get("class_names", []) or []),
                    multiline=True,
                    height=int(self.scaled_height_normal * 1.2),
                    width=int(self.scaled_width_large * 1.05),
                )
                dpg.add_button(
                    label=self.tr("label_apply_class_names"),
                    callback=self.on_apply_class_names_click,
                    width=self.scaled_width_normal,
                )
        with dpg.group(horizontal=True, parent=class_group):
            self.class_priority_input = dpg.add_input_text(
                label=self.tr("label_class_priority"),
                hint=self.tr("label_class_priority_hint"),
                default_value="",
                callback=self.on_class_priority_change,
                width=self.scaled_width_large,
            )
        dpg.add_text(self.tr("label_infer_classes"), parent=class_group)
        self.checkbox_group_tag = dpg.add_child_window(
            parent=class_group,
            width=int(self.scaled_width_large * 1.15),
            height=int(self.scaled_height_normal * 1.2),
            border=True,
        )
        class_num = self.get_current_class_num()
        class_ary = list(range(class_num))
        self.create_checkboxes(class_ary)
        self.update_target_reference_class_combo()
        with dpg.group(horizontal=True, parent=class_group):
            class_aim_label = dpg.add_text(f"{self.tr('label_class_aim_config')} (?)")
            self.attach_tooltip(class_aim_label, self.tr("help_class_aim_config"))
            self.class_aim_combo = dpg.add_combo(
                items=[],
                label=self.tr("label_select_class"),
                callback=self.on_class_aim_combo_change,
                width=self.scaled_width_normal,
                default_value="",
            )
        self.update_class_aim_combo()
        with dpg.group(horizontal=True, parent=class_group):
            self.confidence_threshold_input = dpg.add_input_float(
                label=self.tr("label_confidence_threshold"),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                callback=self.on_confidence_threshold_change,
                width=self.scaled_width_normal,
            )
            self.iou_t_input = dpg.add_input_float(
                label=self.tr("label_iou"),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                callback=self.on_iou_t_change,
                width=self.scaled_width_normal,
            )
        with dpg.group(horizontal=True, parent=class_group):
            with dpg.group():
                self.aim_bot_position_input = dpg.add_input_float(
                    label=self.tr("label_aim_position"),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    callback=self.on_aim_bot_position_change,
                    width=self.scaled_width_normal,
                )
                self.attach_tooltip(
                    self.aim_bot_position_input, self.tr("help_aim_position")
                )
                self.aim_bot_position2_input = dpg.add_input_float(
                    label=self.tr("label_aim_position2"),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    callback=self.on_aim_bot_position2_change,
                    width=self.scaled_width_normal,
                )
                self.attach_tooltip(
                    self.aim_bot_position2_input, self.tr("help_aim_position2")
                )
            with dpg.group():
                dpg.add_text(self.tr("label_class_aim_preview"))
                self.class_aim_preview_drawlist = dpg.add_drawlist(
                    width=int(self.scaled_width_large * 1.4),
                    height=int(self.scaled_height_normal * 1.8),
                )
        self.update_class_aim_preview()

    def on_start_button_click(self, sender, app_data):
        if not getattr(self, "running", False):
            dpg.configure_item(sender, label=self.tr("label_do_not_click"))
            self.running = True
            if (
                self.config["groups"][self.group].get("is_trt", False)
                and TENSORRT_AVAILABLE
            ):
                print("TRT mode detected, checking engine file...")
                current_model = self.config["groups"][self.group]["infer_model"]
                engine_path = os.path.splitext(current_model)[0] + ".engine"
                if not os.path.exists(engine_path):
                    print(f"Engine file does not exist: {engine_path}")
                    print("Starting TRT engine conversion...")
                    dpg.set_value(
                        "output_text", "Converting TRT engine, please wait..."
                    )
                    if current_model.endswith(".onnx"):
                        print("Converting TRT engine from ONNX file...")
                        from src.inference_engine import auto_convert_engine

                        if auto_convert_engine(current_model):
                            print(f"TRT engine conversion successful: {engine_path}")
                            self.config["groups"][self.group]["infer_model"] = (
                                engine_path
                            )
                            dpg.set_value(self.infer_model_input, engine_path)
                        else:
                            print(
                                "TRT engine conversion failed, will use original model"
                            )
                            self.config["groups"][self.group]["is_trt"] = False
                            dpg.set_value(self.is_trt_checkbox, False)
                else:
                    print(f"Found existing engine file: {engine_path}")
                    self.config["groups"][self.group]["infer_model"] = engine_path
                    dpg.set_value(self.infer_model_input, engine_path)
            if self.go():
                dpg.configure_item(sender, label=self.tr("label_stop"))
            else:
                self.running = False
                dpg.configure_item(sender, label=self.tr("label_start"))
        else:
            dpg.configure_item(sender, label=self.tr("label_do_not_click"))
            self.running = False
            if self.timer_id != 0:
                self.time_kill_event(self.timer_id)
                self.timer_id = 0
            if self.timer_id2 != 0:
                self.time_kill_event(self.timer_id2)
                self.timer_id2 = 0
            self.close_screenshot()
            self.unmask_all()
            self.stop_listen()
            dpg.configure_item(sender, label=self.tr("label_start"))

    def on_save_button_click(self, sender, app_data):
        self.save_config_callback()

    def on_game_sensitivity_change(self, sender, app_data):
        return

    def on_mouse_dpi_change(self, sender, app_data):
        return

    def update_sensitivity_display(self):
        return

    def calculate_sensitivity_multiplier(self):
        return 1.0

    def on_infer_debug_change(self, sender, app_data):
        self.config["infer_debug"] = app_data
        print(f"changed to: {self.config['infer_debug']}")

    def on_is_curve_change(self, sender, app_data):
        self.config["is_curve"] = app_data
        print(f"changed to: {self.config['is_curve']}")

    def on_is_curve_uniform_change(self, sender, app_data):
        self.config["is_curve_uniform"] = app_data
        print(f"changed to: {self.config['is_curve_uniform']}")

    def on_distance_scoring_weight_change(self, sender, app_data):
        self.config["distance_scoring_weight"] = app_data
        self.init_target_priority()
        print(f"changed to: {self.config['distance_scoring_weight']}")

    def on_center_scoring_weight_change(self, sender, app_data):
        self.config["center_scoring_weight"] = app_data
        self.init_target_priority()
        print(f"changed to: {self.config['center_scoring_weight']}")

    def on_size_scoring_weight_change(self, sender, app_data):
        self.config["size_scoring_weight"] = app_data
        self.init_target_priority()
        print(f"changed to: {self.config['size_scoring_weight']}")

    def on_aim_weight_tiebreak_change(self, sender, app_data):
        try:
            ratio = float(app_data) / 100.0
        except Exception:
            ratio = 0.1
        ratio = max(0.0, min(1.0, ratio))
        self.config["aim_weight_tiebreak_ratio"] = ratio
        print(f"changed to: {self.config['aim_weight_tiebreak_ratio']}")

    def on_print_fps_change(self, sender, app_data):
        self.config["print_fps"] = app_data
        print(f"changed to: {self.config['print_fps']}")

    def on_show_motion_speed_change(self, sender, app_data):
        self.config["show_motion_speed"] = app_data
        self.refresh_controller_params()
        print(f"changed to: {self.config['show_motion_speed']}")

    def on_is_show_curve_change(self, sender, app_data):
        self.config["is_show_curve"] = app_data
        print(f"changed to: {self.config['is_show_curve']}")

    def on_enable_parallel_processing_change(self, sender, app_data):
        self.config["enable_parallel_processing"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config(
                "enable_parallel_processing", app_data
            )
        print(f"changed to: {self.config['enable_parallel_processing']}")

    def on_turbo_mode_change(self, sender, app_data):
        self.config["turbo_mode"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("turbo_mode", app_data)
        print(f"Turbo mode: {('enabled' if app_data else 'disabled')}")

    def on_skip_frame_processing_change(self, sender, app_data):
        self.config["skip_frame_processing"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("skip_frame_processing", app_data)
        print(f"Skip frame processing: {('enabled' if app_data else 'disabled')}")

    def on_is_show_down_change(self, sender, app_data):
        self.config["is_show_down"] = app_data
        print(f"changed to: {self.config['is_show_down']}")

    def on_is_obs_change(self, sender, app_data):
        self.config["is_obs"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("is_obs", app_data)
        print(f"changed to: {self.config['is_obs']}")
        if hasattr(self, "capture_source_combo"):
            dpg.set_value(self.capture_source_combo, self.get_capture_source_label())
        self.update_capture_source_visibility()
        self.update_capture_region()
        self.update_capture_status_text()

    def on_is_cjk_change(self, sender, app_data):
        self.config["is_cjk"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("is_cjk", app_data)
        print(f"changed to: {self.config['is_cjk']}")
        if hasattr(self, "capture_source_combo"):
            dpg.set_value(self.capture_source_combo, self.get_capture_source_label())
        self.update_capture_source_visibility()
        self.update_capture_region()
        self.update_capture_status_text()

    def on_obs_ip_change(self, sender, app_data):
        self.config["obs_ip"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("obs_ip", app_data)
        print(f"changed to: {self.config['obs_ip']}")
        self.update_capture_status_text()

    def on_cjk_device_id_change(self, sender, app_data):
        self.config["cjk_device_id"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("cjk_device_id", app_data)
        print(f"changed to: {self.config['cjk_device_id']}")
        self.update_capture_status_text()

    def on_cjk_fps_change(self, sender, app_data):
        self.config["cjk_fps"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("cjk_fps", app_data)
        print(f"changed to: {self.config['cjk_fps']}")
        self.update_capture_status_text()

    def on_cjk_resolution_change(self, sender, app_data):
        self.config["cjk_resolution"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("cjk_resolution", app_data)
        print(f"changed to: {self.config['cjk_resolution']}")
        self.update_capture_status_text()

    def on_cjk_crop_size_change(self, sender, app_data):
        self.config["cjk_crop_size"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("cjk_crop_size", app_data)
        print(f"changed to: {self.config['cjk_crop_size']}")
        self.update_capture_status_text()

    def on_capture_offset_x_change(self, sender, app_data):
        self.config["capture_offset_x"] = int(app_data)
        if self.screenshot_manager:
            self.screenshot_manager.update_config("capture_offset_x", int(app_data))
        self.update_capture_region()
        self.update_capture_status_text()

    def on_capture_offset_y_change(self, sender, app_data):
        self.config["capture_offset_y"] = int(app_data)
        if self.screenshot_manager:
            self.screenshot_manager.update_config("capture_offset_y", int(app_data))
        self.update_capture_region()
        self.update_capture_status_text()

    def on_capture_size_change(self, sender, app_data):
        value = str(app_data).strip().lower()
        normalized = "auto"
        if value and value != "auto":
            value = value.replace(" ", "")
            parts = value.split("x")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                normalized = f"{int(parts[0])}x{int(parts[1])}"
            else:
                try:
                    dpg.set_value(sender, self.config.get("capture_size", "auto"))
                except Exception:
                    pass
                return
        self.config["capture_size"] = normalized
        if self.screenshot_manager:
            self.screenshot_manager.update_config("capture_size", normalized)
        self.update_capture_region()
        self.update_capture_status_text()
        print(f"changed to: {self.config['capture_size']}")

    def on_cjk_fourcc_format_change(self, sender, app_data):
        self.config["cjk_fourcc_format"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("cjk_fourcc_format", app_data)
        print(f"Capture card video codec format set to: {app_data}")
        self.update_capture_status_text()

    def on_obs_fps_change(self, sender, app_data):
        self.config["obs_fps"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("obs_fps", app_data)
        print(f"changed to: {self.config['obs_fps']}")
        self.update_capture_status_text()

    def on_obs_port_change(self, sender, app_data):
        self.config["obs_port"] = app_data
        if self.screenshot_manager:
            self.screenshot_manager.update_config("obs_port", app_data)
        print(f"changed to: {self.config['obs_port']}")
        self.update_capture_status_text()

    def on_offset_boundary_x_change(self, sender, app_data):
        self.config["offset_boundary_x"] = app_data
        print(f"changed to: {self.config['offset_boundary_x']}")

    def on_offset_boundary_y_change(self, sender, app_data):
        self.config["offset_boundary_y"] = app_data
        print(f"changed to: {self.config['offset_boundary_y']}")

    def on_knots_count_change(self, sender, app_data):
        self.config["knots_count"] = app_data
        print(f"changed to: {self.config['knots_count']}")

    def on_distortion_mean_change(self, sender, app_data):
        self.config["distortion_mean"] = app_data
        print(f"changed to: {self.config['distortion_mean']}")

    def on_distortion_st_dev_change(self, sender, app_data):
        self.config["distortion_st_dev"] = app_data
        print(f"changed to: {self.config['distortion_st_dev']}")

    def on_distortion_frequency_change(self, sender, app_data):
        self.config["distortion_frequency"] = app_data
        print(f"changed to: {self.config['distortion_frequency']}")

    def on_target_points_change(self, sender, app_data):
        self.config["target_points"] = app_data
        print(f"changed to: {self.config['target_points']}")

    def on_move_method_change(self, sender, app_data):
        self.config["move_method"] = app_data
        print(f"changed to: {self.config['move_method']}")

    def on_group_change(self, sender, app_data):
        self.select_key = ""
        self.config["group"] = app_data
        self.group = app_data
        self.refresh_engine()
        self.update_capture_region()
        self.update_capture_status_text()
        class_num = self.get_current_class_num()
        class_ary = list(range(class_num))
        self.create_checkboxes(class_ary)
        self.update_class_aim_combo()
        self.update_target_reference_class_combo()
        self.aim_keys_dist = self.config["groups"][app_data]["aim_keys"]
        self.aim_key = list(self.aim_keys_dist.keys())
        self.render_key_combo()
        self.update_group_inputs()
        self.sync_sunone_variant_ui()
        self.update_button_lists()
        self.update_auto_flashbang_ui_state()
        print(f"changed to: {self.config['group']}")

    def on_confidence_threshold_change(self, sender, app_data):
        if (
            "class_aim_positions"
            not in self.config["groups"][self.group]["aim_keys"][self.select_key]
        ):
            self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ] = {}
        if (
            self.current_selected_class
            not in self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ]
        ):
            self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ][self.current_selected_class] = {
                "aim_bot_position": 0.0,
                "aim_bot_position2": 0.0,
                "confidence_threshold": 0.5,
                "iou_t": 1.0,
            }
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "class_aim_positions"
        ][self.current_selected_class]["confidence_threshold"] = round(app_data, 4)
        print(
            f"Class {self.current_selected_class} confidence threshold changed to: {round(app_data, 4)}"
        )

    def on_iou_t_change(self, sender, app_data):
        if (
            "class_aim_positions"
            not in self.config["groups"][self.group]["aim_keys"][self.select_key]
        ):
            self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ] = {}
        if (
            self.current_selected_class
            not in self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ]
        ):
            self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ][self.current_selected_class] = {
                "aim_bot_position": 0.0,
                "aim_bot_position2": 0.0,
                "confidence_threshold": 0.5,
                "iou_t": 1.0,
            }
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "class_aim_positions"
        ][self.current_selected_class]["iou_t"] = round(app_data, 4)
        print(
            f"Class {self.current_selected_class} IOU threshold changed to: {round(app_data, 4)}"
        )

    def on_infer_model_change(self, sender, app_data):
        if app_data != "" and os.path.exists(app_data):
            if app_data.endswith(".onnx"):
                self.config["groups"][self.group]["original_infer_model"] = app_data
            elif app_data.endswith(".engine"):
                onnx_path = os.path.splitext(app_data)[0] + ".onnx"
                if os.path.exists(onnx_path):
                    self.config["groups"][self.group]["original_infer_model"] = (
                        onnx_path
                    )
                else:
                    print(
                        f"Warning: Cannot find corresponding ONNX model: {onnx_path}, TRT mode switching may not work properly"
                    )
            self.config["groups"][self.group]["infer_model"] = app_data
            self.refresh_engine()
            self.update_capture_region()
            self.update_capture_status_text()
            class_num = self.get_current_class_num()
            class_ary = list(range(class_num))
            self.create_checkboxes(class_ary)
            self.update_class_aim_combo()
            self.update_target_reference_class_combo()
            self.update_auto_flashbang_ui_state()
            print(app_data + " model file exists, updated")
        else:
            print(app_data + " model file does not exist, please check the path")

    def on_select_model_click(self, sender, app_data):
        """Select model file callback function"""
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            filetypes = [
                ("ONNX Model", "*.onnx"),
                ("TensorRT Engine", "*.engine"),
                ("All Files", "*.*"),
            ]
            file_path = filedialog.askopenfilename(
                title="Select Model File", filetypes=filetypes, parent=root
            )
            root.destroy()
            if file_path:
                valid_extensions = [".onnx", ".engine"]
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in valid_extensions:
                    if (
                        hasattr(self, "is_trt_checkbox")
                        and self.is_trt_checkbox is not None
                    ):
                        current_trt_value = dpg.get_value(self.is_trt_checkbox)
                        if current_trt_value:
                            dpg.set_value(self.is_trt_checkbox, False)
                            self.config["groups"][self.group]["is_trt"] = False
                    dpg.set_value(self.infer_model_input, file_path)
                    self.on_infer_model_change(self.infer_model_input, file_path)
                else:
                    print(f"Unsupported file format: {file_ext}")
                    print("Supported formats: .onnx, .engine")
        except Exception as e:
            print(f"Error selecting model file: {e}")

    def on_aim_bot_position_change(self, sender, app_data):
        if (
            "class_aim_positions"
            not in self.config["groups"][self.group]["aim_keys"][self.select_key]
        ):
            self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ] = {}
        if (
            self.current_selected_class
            not in self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ]
        ):
            self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ][self.current_selected_class] = {
                "aim_bot_position": 0.0,
                "aim_bot_position2": 0.0,
            }
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "class_aim_positions"
        ][self.current_selected_class]["aim_bot_position"] = round(app_data, 4)
        self.update_class_aim_preview()
        print(
            f"Class {self.current_selected_class} aim position 1 changed to: {round(app_data, 4)}"
        )

    def on_aim_bot_position2_change(self, sender, app_data):
        if (
            "class_aim_positions"
            not in self.config["groups"][self.group]["aim_keys"][self.select_key]
        ):
            self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ] = {}
        if (
            self.current_selected_class
            not in self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ]
        ):
            self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_aim_positions"
            ][self.current_selected_class] = {
                "aim_bot_position": 0.0,
                "aim_bot_position2": 0.0,
            }
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "class_aim_positions"
        ][self.current_selected_class]["aim_bot_position2"] = round(app_data, 4)
        self.update_class_aim_preview()
        print(
            f"Class {self.current_selected_class} aim position 2 changed to: {round(app_data, 4)}"
        )

    def on_class_priority_change(self, sender, app_data):
        """Class priority input callback function"""
        priority_text = app_data.strip()
        print(f"Class priority input: {priority_text}")
        priority_order = self.parse_class_priority(priority_text)
        if priority_order is not None:
            self.config["groups"][self.group]["aim_keys"][self.select_key][
                "class_priority_order"
            ] = priority_order
            print(f"Class priority updated: {priority_order}")
        else:
            print(f"Class priority format error: {priority_text}")

    def parse_class_priority(self, priority_text):
        """Parse class priority string"""
        if not priority_text:
            return []
        try:
            import re

            parts = re.split("[-,\\s]+", priority_text.strip())
            priority_order = []
            seen = set()
            for part in parts:
                if part.strip():
                    try:
                        class_id = int(part.strip())
                        if class_id not in seen:
                            priority_order.append(class_id)
                            seen.add(class_id)
                    except ValueError:
                        return
            else:
                return priority_order
        except Exception:
            return None

    def format_class_priority(self, priority_order):
        """Format priority list to string"""
        return "-".join(map(str, priority_order)) if priority_order else ""

    def get_class_priority_order(self):
        """Get current key's class priority order"""
        try:
            key_config = self.config["groups"][self.group]["aim_keys"][self.select_key]
            return key_config.get("class_priority_order", [])
        except:
            return []

    def on_class_aim_combo_change(self, sender, app_data):
        """Class selection combo callback function"""
        if app_data:
            self.current_selected_class = str(self.parse_class_label(app_data))
            print(f"Current selected class: {self.current_selected_class}")
            self.update_class_aim_inputs()

    def update_class_aim_inputs(self):
        """Update aim position input values based on currently selected class"""
        if (
            not hasattr(self, "aim_bot_position_input")
            or self.aim_bot_position_input is None
        ):
            return None
        key_cfg = self.config["groups"][self.group]["aim_keys"][self.select_key]
        cap = key_cfg.get("class_aim_positions", {})
        if isinstance(cap, list):
            converted = {}
            for i, item in enumerate(cap):
                if isinstance(item, dict):
                    converted[str(i)] = {
                        "aim_bot_position": float(item.get("aim_bot_position", 0.0)),
                        "aim_bot_position2": float(item.get("aim_bot_position2", 0.0)),
                        "confidence_threshold": float(
                            item.get("confidence_threshold", 0.5)
                        ),
                        "iou_t": float(item.get("iou_t", 1.0)),
                    }
                else:
                    converted[str(i)] = {
                        "aim_bot_position": 0.0,
                        "aim_bot_position2": 0.0,
                        "confidence_threshold": 0.5,
                        "iou_t": 1.0,
                    }
            key_cfg["class_aim_positions"] = converted
        else:
            if not isinstance(cap, dict):
                key_cfg["class_aim_positions"] = {}
        if (
            not self.current_selected_class
            or not str(self.current_selected_class).isdigit()
        ):
            self.current_selected_class = "0"
        if self.current_selected_class not in key_cfg["class_aim_positions"]:
            key_cfg["class_aim_positions"][self.current_selected_class] = {
                "aim_bot_position": 0.0,
                "aim_bot_position2": 0.0,
                "confidence_threshold": 0.5,
                "iou_t": 1.0,
            }
        class_config = key_cfg["class_aim_positions"][self.current_selected_class]
        import dearpygui.dearpygui as dpg

        dpg.set_value(
            self.aim_bot_position_input,
            float(class_config.get("aim_bot_position", 0.0)),
        )
        dpg.set_value(
            self.aim_bot_position2_input,
            float(class_config.get("aim_bot_position2", 0.0)),
        )
        if (
            hasattr(self, "confidence_threshold_input")
            and self.confidence_threshold_input is not None
        ):
            dpg.set_value(
                self.confidence_threshold_input,
                float(class_config.get("confidence_threshold", 0.5)),
            )
        if hasattr(self, "iou_t_input") and self.iou_t_input is not None:
            dpg.set_value(self.iou_t_input, float(class_config.get("iou_t", 1.0)))
        self.update_class_aim_preview()

    def update_class_aim_preview(self):
        if (
            not hasattr(self, "class_aim_preview_drawlist")
            or self.class_aim_preview_drawlist is None
        ):
            return
        try:
            import dearpygui.dearpygui as dpg

            dpg.delete_item(self.class_aim_preview_drawlist, children_only=True)
            width = int(self.scaled_width_large * 1.15)
            height = int(self.scaled_height_normal * 1.8)
            dpg.configure_item(
                self.class_aim_preview_drawlist, width=width, height=height
            )
            group_cfg = self.config.get("groups", {}).get(self.group, {})
            aim_keys = group_cfg.get("aim_keys", {})
            key_id = self.select_key or (next(iter(aim_keys), None))
            if key_id is None:
                return
            if not self.select_key:
                self.select_key = key_id
            key_cfg = aim_keys.get(key_id, {})
            if not self.current_selected_class:
                self.current_selected_class = "0"
            class_config = (
                key_cfg.get("class_aim_positions", {})
                .get(str(self.current_selected_class), {})
            )
            pos1 = float(class_config.get("aim_bot_position", 0.0))
            pos2 = float(class_config.get("aim_bot_position2", 0.0))
            pos1 = max(0.0, min(1.0, pos1))
            pos2 = max(0.0, min(1.0, pos2))

            cx = width * 0.5
            head_r = min(width, height) * 0.12
            head_y = height * 0.18
            body_top = head_y + head_r + 6
            body_bottom = height * 0.9
            body_w = width * 0.22
            left = cx - body_w * 0.5
            right = cx + body_w * 0.5
            bbox_top = head_y - head_r
            bbox_bottom = body_bottom

            dpg.draw_rectangle(
                (2, 2),
                (width - 2, height - 2),
                color=(60, 60, 60, 255),
                parent=self.class_aim_preview_drawlist,
            )
            dpg.draw_circle(
                (cx, head_y),
                head_r,
                color=(120, 120, 120, 255),
                fill=(80, 80, 80, 255),
                parent=self.class_aim_preview_drawlist,
            )
            dpg.draw_rectangle(
                (left, body_top),
                (right, body_bottom),
                color=(120, 120, 120, 255),
                fill=(70, 70, 70, 255),
                parent=self.class_aim_preview_drawlist,
            )
            line_y1 = bbox_top + (bbox_bottom - bbox_top) * pos1
            line_y2 = bbox_top + (bbox_bottom - bbox_top) * pos2
            dpg.draw_line(
                (left, line_y1),
                (right, line_y1),
                color=(255, 80, 80, 255),
                thickness=2,
                parent=self.class_aim_preview_drawlist,
            )
            dpg.draw_line(
                (left, line_y2),
                (right, line_y2),
                color=(255, 220, 80, 255),
                thickness=2,
                parent=self.class_aim_preview_drawlist,
            )
        except Exception:
            return

    def update_prediction_preview(self):
        if (
            not hasattr(self, "prediction_preview_drawlist")
            or self.prediction_preview_drawlist is None
        ):
            return
        try:
            dpg.delete_item(self.prediction_preview_drawlist, children_only=True)
            width = int(self.scaled_width_large * 1.6)
            height = int(self.scaled_height_normal * 1.6)
            dpg.configure_item(
                self.prediction_preview_drawlist, width=width, height=height
            )
            cx = width * 0.5
            cy = height * 0.55

            pred_cfg = self.config.get("sunone", {}).get("prediction", {})
            interval = float(pred_cfg.get("interval", 0.05))
            lead_ms = float(pred_cfg.get("kalman_lead_ms", 0.0))
            lead_s = max(0.0, lead_ms / 1000.0)
            vel_smoothing = float(pred_cfg.get("velocity_smoothing", 0.0))
            vel_scale = float(pred_cfg.get("velocity_scale", 1.0))
            mode = int(pred_cfg.get("mode", 0))
            q = float(pred_cfg.get("kalman_process_noise", 0.01))
            r = float(pred_cfg.get("kalman_measurement_noise", 0.1))

            state = getattr(self, "_pred_demo_state", None)
            if state is None:
                state = {
                    "angle": 0.0,
                    "last_t": time.perf_counter(),
                    "prev_raw_x": cx,
                    "prev_raw_y": cy,
                    "kf_x": DemoKalman1D(q, r),
                    "kf_y": DemoKalman1D(q, r),
                    "last_q": q,
                    "last_r": r,
                    "sm_vx": 0.0,
                    "sm_vy": 0.0,
                    "sm_init": False,
                }
                self._pred_demo_state = state

            now = time.perf_counter()
            dt = now - state["last_t"]
            state["last_t"] = now
            dt = max(1e-4, min(dt, 0.1))

            if state["last_q"] != q or state["last_r"] != r:
                state["kf_x"] = DemoKalman1D(q, r)
                state["kf_y"] = DemoKalman1D(q, r)
                state["last_q"] = q
                state["last_r"] = r
                state["sm_init"] = False

            state["angle"] += dt * 1.2
            if state["angle"] > math.tau:
                state["angle"] -= math.tau

            radius = width * 0.35
            raw_x = cx + math.cos(state["angle"]) * radius
            raw_y = cy + math.sin(state["angle"] * 1.3) * (radius * 0.7)

            raw_vx = (raw_x - state["prev_raw_x"]) / dt
            raw_vy = (raw_y - state["prev_raw_y"]) / dt
            state["prev_raw_x"] = raw_x
            state["prev_raw_y"] = raw_y

            kal_x = state["kf_x"].update(raw_x, dt)
            kal_y = state["kf_y"].update(raw_y, dt)

            alpha = max(0.0, min(1.0, vel_smoothing))
            if not state["sm_init"]:
                state["sm_vx"] = state["kf_x"].v
                state["sm_vy"] = state["kf_y"].v
                state["sm_init"] = True
            else:
                state["sm_vx"] += (state["kf_x"].v - state["sm_vx"]) * alpha
                state["sm_vy"] += (state["kf_y"].v - state["sm_vy"]) * alpha

            scale = max(0.0, vel_scale)
            std_px = raw_x + raw_vx * interval
            std_py = raw_y + raw_vy * interval
            kal_px = kal_x + state["sm_vx"] * scale * lead_s
            kal_py = kal_y + state["sm_vy"] * scale * lead_s

            if mode == 0:
                final_x = std_px
                final_y = std_py
            elif mode == 1:
                final_x = kal_px
                final_y = kal_py
            else:
                final_x = kal_px + raw_vx * interval
                final_y = kal_py + raw_vy * interval

            dpg.draw_rectangle(
                (2, 2),
                (width - 2, height - 2),
                color=(60, 60, 60, 255),
                parent=self.prediction_preview_drawlist,
            )
            dpg.draw_line(
                (cx, cy), (cx, height - 6), color=(50, 50, 50, 255), parent=self.prediction_preview_drawlist
            )
            dpg.draw_line(
                (6, cy), (width - 6, cy), color=(50, 50, 50, 255), parent=self.prediction_preview_drawlist
            )

            dpg.draw_line(
                (raw_x, raw_y),
                (final_x, final_y),
                color=(100, 255, 100, 200),
                thickness=2,
                parent=self.prediction_preview_drawlist,
            )
            dpg.draw_circle(
                (raw_x, raw_y),
                4,
                color=(255, 255, 255, 220),
                fill=(255, 255, 255, 220),
                parent=self.prediction_preview_drawlist,
            )
            dpg.draw_circle(
                (final_x, final_y),
                4,
                color=(100, 255, 100, 220),
                fill=(100, 255, 100, 220),
                parent=self.prediction_preview_drawlist,
            )

            steps = max(1, min(int(pred_cfg.get("future_positions", 6)), 12))
            preview_vx = raw_vx if mode == 0 else state["sm_vx"] * scale
            preview_vy = raw_vy if mode == 0 else state["sm_vy"] * scale
            fps = max(1.0, float(self.config.get("obs_fps", 60)))
            frame_time = 1.0 / fps
            for i in range(1, steps + 1):
                t = frame_time * i
                pt_x = final_x + preview_vx * t
                pt_y = final_y + preview_vy * t
                alpha_pt = int(200 - i * (140.0 / steps))
                dpg.draw_circle(
                    (pt_x, pt_y),
                    2.5,
                    color=(80, 180, 255, alpha_pt),
                    fill=(80, 180, 255, alpha_pt),
                    parent=self.prediction_preview_drawlist,
                )

            dpg.draw_text(
                (8, 8),
                "W=Raw  G=Prediction",
                color=(150, 150, 150, 255),
                parent=self.prediction_preview_drawlist,
            )
        except Exception:
            return

    def update_speed_curve_preview(self):
        if (
            not hasattr(self, "speed_curve_preview_drawlist")
            or self.speed_curve_preview_drawlist is None
        ):
            return
        try:
            dpg.delete_item(self.speed_curve_preview_drawlist, children_only=True)
            width = int(self.scaled_width_large * 1.4)
            height = int(self.scaled_height_normal * 1.4)
            dpg.configure_item(
                self.speed_curve_preview_drawlist, width=width, height=height
            )
            cx = width * 0.5
            cy = height * 0.55
            dpg.draw_rectangle(
                (2, 2),
                (width - 2, height - 2),
                color=(60, 60, 60, 255),
                parent=self.speed_curve_preview_drawlist,
            )

            speed_cfg = self.config.get("sunone", {}).get("speed", {})
            min_speed = float(speed_cfg.get("min_multiplier", 0.5))
            max_speed = float(speed_cfg.get("max_multiplier", 0.7))
            snap_radius = float(speed_cfg.get("snap_radius", 3.2))
            near_radius = float(speed_cfg.get("near_radius", 40.0))
            curve_exp = max(1.0, float(speed_cfg.get("speed_curve_exponent", 10.0)))
            snap_boost = max(0.1, float(speed_cfg.get("snap_boost_factor", 4.0)))

            scale = 4.0
            near_px = max(10.0, min(near_radius * scale, width * 0.45))
            snap_px = max(6.0, min(snap_radius * scale, near_px - 4.0))

            dpg.draw_circle(
                (cx, cy),
                near_px,
                color=(80, 120, 255, 180),
                thickness=2,
                parent=self.speed_curve_preview_drawlist,
            )
            dpg.draw_circle(
                (cx, cy),
                snap_px,
                color=(255, 100, 100, 180),
                thickness=2,
                parent=self.speed_curve_preview_drawlist,
            )

            state = getattr(self, "_speed_curve_demo", None)
            if state is None:
                state = {
                    "dist_px": near_px,
                    "vel_px": 0.0,
                    "last_t": time.perf_counter(),
                }
                self._speed_curve_demo = state

            now = time.perf_counter()
            dt = now - state["last_t"]
            state["last_t"] = now
            dt = max(1e-4, min(dt, 0.1))

            dist_units = state["dist_px"] / scale
            if dist_units < snap_radius:
                speed_mult = min_speed * snap_boost
            elif dist_units < near_radius:
                t = dist_units / max(near_radius, 1e-6)
                crv = 1.0 - math.pow(1.0 - t, curve_exp)
                speed_mult = min_speed + (max_speed - min_speed) * crv
            else:
                norm = max(0.0, min(1.0, dist_units / max(near_radius, 1e-6)))
                speed_mult = min_speed + (max_speed - min_speed) * norm

            base_px_s = 60.0
            state["vel_px"] = base_px_s * speed_mult
            state["dist_px"] -= state["vel_px"] * dt
            if state["dist_px"] <= 0.0:
                state["dist_px"] = near_px

            dot_x = cx - state["dist_px"]
            dot_y = cy
            dpg.draw_circle(
                (dot_x, dot_y),
                4,
                color=(255, 255, 80, 255),
                fill=(255, 255, 80, 255),
                parent=self.speed_curve_preview_drawlist,
            )
        except Exception:
            return

    def _schedule_preview_tick(self):
        if not getattr(self, "_preview_anim_active", False):
            return
        try:
            frame = dpg.get_frame_count() + 1
            dpg.set_frame_callback(frame, self._preview_tick)
        except Exception:
            return

    def _preview_tick(self, sender, app_data, user_data):
        if not getattr(self, "_preview_anim_active", False):
            return
        self._prediction_preview_time = time.perf_counter()
        self.update_prediction_preview()
        self.update_speed_curve_preview()
        self._schedule_preview_tick()

    def start_preview_animation(self):
        if getattr(self, "_preview_anim_active", False):
            return
        self._preview_anim_active = True
        self._schedule_preview_tick()

    def update_class_aim_combo(self):
        """Update class combo options"""
        if not hasattr(self, "class_aim_combo") or self.class_aim_combo is None:
            return None
        try:
            class_num = self.get_current_class_num()
            class_num = int(class_num)
            class_items = [self.format_class_label(i) for i in range(class_num)]
            import dearpygui.dearpygui as dpg

            dpg.configure_item(self.class_aim_combo, items=class_items)
            if class_items:
                try:
                    current_class_int = (
                        int(self.current_selected_class)
                        if self.current_selected_class
                        and self.current_selected_class.isdigit()
                        else (-1)
                    )
                    if (
                        not self.current_selected_class
                        or current_class_int < 0
                        or current_class_int >= class_num
                    ):
                        self.current_selected_class = "0"
                except (ValueError, TypeError):
                    self.current_selected_class = "0"
                dpg.set_value(
                    self.class_aim_combo,
                    self.format_class_label(int(self.current_selected_class)),
                )
                self.update_class_aim_inputs()
        except Exception as e:
            import traceback

            traceback.print_exc()

    def on_aim_bot_scope_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "aim_bot_scope"
        ] = app_data
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['aim_bot_scope']}"
        )

    def on_dynamic_scope_enabled_change(self, sender, app_data):
        key_cfg = self.config["groups"][self.group]["aim_keys"][self.select_key]
        if "dynamic_scope" not in key_cfg:
            key_cfg["dynamic_scope"] = {}
        key_cfg["dynamic_scope"]["enabled"] = bool(app_data)

    def on_dynamic_scope_min_ratio_change(self, sender, app_data):
        key_cfg = self.config["groups"][self.group]["aim_keys"][self.select_key]
        if "dynamic_scope" not in key_cfg:
            key_cfg["dynamic_scope"] = {}
        try:
            v = float(app_data)
        except Exception:
            v = 0.5
        v = max(0.0, min(1.0, v))
        key_cfg["dynamic_scope"]["min_ratio"] = v

    def on_dynamic_scope_min_scope_change(self, sender, app_data):
        key_cfg = self.config["groups"][self.group]["aim_keys"][self.select_key]
        if "dynamic_scope" not in key_cfg:
            key_cfg["dynamic_scope"] = {}
        try:
            v = int(app_data)
        except Exception:
            v = 0
        key_cfg["dynamic_scope"]["min_scope"] = max(0, v)

    def on_dynamic_scope_shrink_ms_change(self, sender, app_data):
        key_cfg = self.config["groups"][self.group]["aim_keys"][self.select_key]
        if "dynamic_scope" not in key_cfg:
            key_cfg["dynamic_scope"] = {}
        try:
            v = int(app_data)
        except Exception:
            v = 300
        key_cfg["dynamic_scope"]["shrink_duration_ms"] = max(0, v)

    def on_dynamic_scope_recover_ms_change(self, sender, app_data):
        key_cfg = self.config["groups"][self.group]["aim_keys"][self.select_key]
        if "dynamic_scope" not in key_cfg:
            key_cfg["dynamic_scope"] = {}
        try:
            v = int(app_data)
        except Exception:
            v = 300
        key_cfg["dynamic_scope"]["recover_duration_ms"] = max(0, v)

    def on_target_lock_distance_change(self, sender, app_data):
        key_cfg = self.config["groups"][self.group]["aim_keys"][self.select_key]
        try:
            v = int(app_data)
        except Exception:
            v = 100
        key_cfg["target_lock_distance"] = max(1, v)

    def on_target_lock_reacquire_time_change(self, sender, app_data):
        key_cfg = self.config["groups"][self.group]["aim_keys"][self.select_key]
        try:
            v = float(app_data)
        except Exception:
            v = 0.3
        key_cfg["target_lock_reacquire_time"] = max(0.0, v)

    def on_min_position_offset_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "min_position_offset"
        ] = app_data
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['min_position_offset']}"
        )

    def on_smoothing_factor_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "smoothing_factor"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['smoothing_factor']}"
        )

    def on_base_step_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["base_step"] = (
            round(app_data, 3)
        )
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['base_step']}"
        )

    def on_distance_weight_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "distance_weight"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['distance_weight']}"
        )

    def on_fov_angle_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["fov_angle"] = (
            round(app_data, 3)
        )
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['fov_angle']}"
        )

    def on_history_size_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "history_size"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['history_size']}"
        )

    def on_deadzone_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["deadzone"] = (
            round(app_data, 3)
        )
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['deadzone']}"
        )

    def on_smoothing_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["smoothing"] = (
            round(app_data, 3)
        )
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['smoothing']}"
        )

    def on_velocity_decay_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "velocity_decay"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['velocity_decay']}"
        )

    def on_current_frame_weight_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "current_frame_weight"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['current_frame_weight']}"
        )

    def on_last_frame_weight_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "last_frame_weight"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['last_frame_weight']}"
        )

    def on_output_scale_x_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "output_scale_x"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['output_scale_x']}"
        )

    def on_output_scale_y_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "output_scale_y"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['output_scale_y']}"
        )

    def on_uniform_threshold_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "uniform_threshold"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['uniform_threshold']}"
        )

    def on_min_velocity_threshold_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "min_velocity_threshold"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['min_velocity_threshold']}"
        )

    def on_max_velocity_threshold_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "max_velocity_threshold"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['max_velocity_threshold']}"
        )

    def on_compensation_factor_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "compensation_factor"
        ] = round(app_data, 3)
        self.refresh_controller_params()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['compensation_factor']}"
        )

    def on_overshoot_threshold_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "overshoot_threshold"
        ] = round(app_data, 1)
        self.refresh_controller_params()
        print(
            f"Overshoot detection threshold: {self.config['groups'][self.group]['aim_keys'][self.select_key]['overshoot_threshold']}"
        )

    def on_overshoot_x_factor_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "overshoot_x_factor"
        ] = round(app_data, 2)
        self.refresh_controller_params()
        print(
            f"X-axis overshoot suppression factor: {self.config['groups'][self.group]['aim_keys'][self.select_key]['overshoot_x_factor']}"
        )

    def on_overshoot_y_factor_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "overshoot_y_factor"
        ] = round(app_data, 2)
        self.refresh_controller_params()
        print(
            f"Y-axis overshoot suppression factor: {self.config['groups'][self.group]['aim_keys'][self.select_key]['overshoot_y_factor']}"
        )

    def on_status_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["trigger"][
            "status"
        ] = app_data
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['trigger']['status']}"
        )

    def on_continuous_trigger_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["trigger"][
            "continuous"
        ] = app_data
        print(f"Continuous trigger set to: {app_data}")

    def on_start_delay_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["trigger"][
            "start_delay"
        ] = app_data
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['trigger']['start_delay']}"
        )

    def on_long_press_duration_change(self, sender, app_data):
        self.config["groups"][self.group]["long_press_duration"] = app_data
        print(f"changed to: {self.config['groups'][self.group]['long_press_duration']}")

    def on_press_delay_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["trigger"][
            "press_delay"
        ] = app_data
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['trigger']['press_delay']}"
        )

    def on_end_delay_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["trigger"][
            "end_delay"
        ] = app_data
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['trigger']['end_delay']}"
        )

    def on_random_delay_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["trigger"][
            "random_delay"
        ] = app_data
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['trigger']['random_delay']}"
        )

    def on_x_trigger_scope_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["trigger"][
            "x_trigger_scope"
        ] = app_data
        self.update_rect()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['trigger']['x_trigger_scope']}"
        )

    def on_y_trigger_scope_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["trigger"][
            "y_trigger_scope"
        ] = app_data
        self.update_rect()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['trigger']['y_trigger_scope']}"
        )

    def on_x_trigger_offset_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["trigger"][
            "x_trigger_offset"
        ] = app_data
        self.update_rect()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['trigger']['x_trigger_offset']}"
        )

    def on_y_trigger_offset_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["trigger"][
            "y_trigger_offset"
        ] = app_data
        self.update_rect()
        print(
            f"changed to: {self.config['groups'][self.group]['aim_keys'][self.select_key]['trigger']['y_trigger_offset']}"
        )

    def update_rect(self):
        x_ratio = self.config["groups"][self.group]["aim_keys"][self.select_key][
            "trigger"
        ]["x_trigger_offset"]
        y_ratio = self.config["groups"][self.group]["aim_keys"][self.select_key][
            "trigger"
        ]["y_trigger_offset"]
        w_ratio = self.config["groups"][self.group]["aim_keys"][self.select_key][
            "trigger"
        ]["x_trigger_scope"]
        h_ratio = self.config["groups"][self.group]["aim_keys"][self.select_key][
            "trigger"
        ]["y_trigger_scope"]
        x = x_ratio * 50
        y = y_ratio * 100
        w = w_ratio * 50
        h = h_ratio * 100
        dpg.configure_item("small_rect", pmin=[x, y], pmax=[x + w, y + h])

    def render_games_combo(self):
        if self.games_combo is not None:
            dpg.delete_item(self.games_combo)
        self.games_combo = dpg.add_combo(
            label=self.tr("label_game"),
            items=list(self.config["games"].keys()),
            default_value=self.config["picked_game"],
            callback=self.on_games_change,
            width=self.scaled_width_large,
            parent=self.dpg_games_tag,
        )

    def render_guns_combo(self):
        if self.guns_combo is not None:
            dpg.delete_item(self.guns_combo)
        guns = list(self.config["games"][self.picked_game].keys())
        if self.picked_gun not in guns:
            self.picked_gun = guns[0]
        self.guns_combo = dpg.add_combo(
            label=self.tr("label_gun"),
            items=guns,
            default_value=self.picked_gun,
            callback=self.on_guns_change,
            width=self.scaled_width_large,
            parent=self.dpg_guns_tag,
        )

    def render_stages_combo(self):
        if self.stages_combo is not None:
            dpg.delete_item(self.stages_combo)
        stages = self.config["games"][self.picked_game][self.picked_gun]
        stage_len = len(self.config["games"][self.picked_game][self.picked_gun])
        stages_obj = {}
        for i in range(stage_len):
            stages_obj[str(i)] = stages[i]
        if self.picked_stage not in stages_obj:
            self.picked_stage = "0"
        self.stages_combo = dpg.add_combo(
            label=self.tr("label_index"),
            items=list(stages_obj.keys()),
            default_value=self.picked_stage,
            callback=self.on_stages_change,
            width=self.scaled_width_large,
            parent=self.dpg_stages_tag,
        )

    def on_delete_game_click(self, sender, app_data):
        if len(self.config["games"]) > 1:
            del self.config["games"][self.picked_game]
            self.picked_game = list(self.config["games"].keys())[0]
            self.config["picked_game"] = self.picked_game
            self.render_games_combo()

    def on_delete_gun_click(self, sender, app_data):
        if len(self.config["games"][self.picked_game]) > 1:
            del self.config["games"][self.picked_game][self.picked_gun]
            self.picked_gun = list(self.config["games"][self.picked_game].keys())[0]
            self.render_guns_combo()

    def on_delete_stage_click(self, sender, app_data):
        if len(self.config["games"][self.picked_game][self.picked_gun]) > 1:
            del self.config["games"][self.picked_game][self.picked_gun][
                int(self.picked_stage)
            ]
            self.render_stages_combo()

    def on_game_name_change(self, sender, app_data):
        self.add_game_name = app_data

    def on_gun_name_change(self, sender, app_data):
        self.add_gun_name = app_data

    def on_number_change(self, sender, app_data):
        self.config["games"][self.picked_game][self.picked_gun][int(self.picked_stage)][
            "number"
        ] = app_data

    def on_x_change(self, sender, app_data):
        self.config["games"][self.picked_game][self.picked_gun][int(self.picked_stage)][
            "offset"
        ][0] = round(app_data, 3)

    def on_y_change(self, sender, app_data):
        self.config["games"][self.picked_game][self.picked_gun][int(self.picked_stage)][
            "offset"
        ][1] = round(app_data, 3)

    def on_add_game_click(self, sender, app_data):
        if self.add_game_name not in self.config["games"] and self.add_game_name != "":
            self.config["games"][self.add_game_name] = copy.deepcopy(
                self.config["games"][self.picked_game]
            )
            self.picked_game = self.add_game_name
            self.config["picked_game"] = self.picked_game
            self.render_games_combo()

    def on_add_stage_click(self, sender, app_data):
        self.config["games"][self.picked_game][self.picked_gun].append(
            {"number": 0, "offset": [0, 0]}
        )
        self.render_stages_combo()

    def on_add_gun_click(self, sender, app_data):
        if (
            self.add_gun_name not in self.config["games"][self.picked_game]
            and self.add_gun_name != ""
        ):
            self.config["games"][self.picked_game][self.add_gun_name] = copy.deepcopy(
                self.config["games"][self.picked_game][self.picked_gun]
            )
            self.picked_gun = self.add_gun_name
            self.render_guns_combo()
            self.render_stages_combo()
            self.refresh_stage()

    def on_games_change(self, sender, app_data):
        self.picked_game = app_data
        self.config["picked_game"] = self.picked_game
        self.render_guns_combo()
        self.render_stages_combo()
        self.refresh_stage()
        self._current_mouse_re_points = None
        if self.config.get("recoil", {}).get("use_mouse_re_trajectory", False):
            self._current_mouse_re_points = self._load_mouse_re_trajectory_for_current()

    def on_guns_change(self, sender, app_data):
        self.picked_gun = app_data
        self.render_stages_combo()
        self.refresh_stage()
        self._current_mouse_re_points = None
        if self.config.get("recoil", {}).get("use_mouse_re_trajectory", False):
            self._current_mouse_re_points = self._load_mouse_re_trajectory_for_current()

    def on_stages_change(self, sender, app_data):
        self.picked_stage = app_data
        self.refresh_stage()

    def refresh_stage(self):
        number = self.config["games"][self.picked_game][self.picked_gun][
            int(self.picked_stage)
        ]["number"]
        x = self.config["games"][self.picked_game][self.picked_gun][
            int(self.picked_stage)
        ]["offset"][0]
        y = self.config["games"][self.picked_game][self.picked_gun][
            int(self.picked_stage)
        ]["offset"][1]
        dpg.set_value(self.number_input, number)
        dpg.set_value(self.x_input, x)
        dpg.set_value(self.y_input, y)

    def reset_down_status(self):
        if self.config["is_show_down"]:
            print(self.now_stage, self.now_num)
        self.now_num = 0
        self.now_stage = 0
        self.decimal_x = 0
        self.decimal_y = 0
        self.end = False

    def close_screenshot(self):
        """Close and release screenshot resources"""
        if self.screenshot_manager is not None:
            self.screenshot_manager.close()
            self.screenshot_manager = None

    def on_mask_left_change(self, sender, app_data):
        self.config["mask_left"] = app_data

    def on_mask_right_change(self, sender, app_data):
        self.config["mask_right"] = app_data

    def on_mask_middle_change(self, sender, app_data):
        self.config["mask_middle"] = app_data

    def on_mask_side1_change(self, sender, app_data):
        self.config["mask_side1"] = app_data

    def on_mask_side2_change(self, sender, app_data):
        self.config["mask_side2"] = app_data

    def on_mask_x_change(self, sender, app_data):
        self.config["mask_x"] = app_data

    def on_mask_y_change(self, sender, app_data):
        self.config["mask_y"] = app_data

    def on_mask_wheel_change(self, sender, app_data):
        self.config["mask_wheel"] = app_data

    def on_aim_mask_x_change(self, sender, app_data):
        self.config["aim_mask_x"] = app_data

    def on_aim_mask_y_change(self, sender, app_data):
        self.config["aim_mask_y"] = app_data

    def _init_makcu_locks(self):
        """Initialize makcu button and axis lock states"""
        if self.makcu is None:
            return
        try:
            if self.config["mask_left"]:
                self.makcu.lock(MouseButton.LEFT)
            if self.config["mask_right"]:
                self.makcu.lock(MouseButton.RIGHT)
            if self.config["mask_middle"]:
                self.makcu.lock(MouseButton.MIDDLE)
            if self.config["mask_side1"]:
                self.makcu.lock(MouseButton.MOUSE4)
            if self.config["mask_side2"]:
                self.makcu.lock(MouseButton.MOUSE5)
            if self.config["mask_x"]:
                self.makcu.lock("X")
            if self.config["mask_y"]:
                self.makcu.lock("Y")
            if self.config["mask_wheel"]:
                return
        except Exception as e:
            print(f"Failed to initialize Makcu lock states: {e}")

    def on_is_show_priority_debug_change(self, sender, app_data):
        self.config["is_show_priority_debug"] = app_data
        print(f"Class priority debug: {app_data}")

    def on_is_trt_change(self, sender, app_data):
        self.config["groups"][self.group]["is_trt"] = app_data
        if app_data:
            if not TENSORRT_AVAILABLE:
                print("TensorRT environment not installed or unavailable")
                print("Please install the following components:")
                print("1. CUDA Toolkit")
                print("2. cuDNN")
                print("3. TensorRT")
                self.config["groups"][self.group]["is_trt"] = False
                dpg.set_value(self.is_trt_checkbox, False)
                return
            current_model = self.config["groups"][self.group]["infer_model"]
            if current_model.endswith(".onnx"):
                self.config["groups"][self.group]["original_infer_model"] = (
                    current_model
                )
                engine_path = os.path.splitext(current_model)[0] + ".engine"
                if os.path.exists(engine_path):
                    self.config["groups"][self.group]["infer_model"] = engine_path
                    dpg.set_value(self.infer_model_input, engine_path)
                    print(f"TRT engine found, using: {engine_path}")
                    self.refresh_engine()
                    return
                print(
                    "TRT mode enabled, will detect and convert engine file on startup"
                )
            elif current_model.endswith(".engine"):
                onnx_path = self.config["groups"][self.group].get(
                    "original_infer_model", None
                )
                if not onnx_path:
                    possible_onnx = os.path.splitext(current_model)[0] + ".onnx"
                    if os.path.exists(possible_onnx):
                        self.config["groups"][self.group]["original_infer_model"] = (
                            possible_onnx
                        )
                        print(
                            f"Automatically inferred and set original model path: {possible_onnx}"
                        )
                    else:
                        print(
                            "Warning: Cannot find corresponding ONNX model, TRT switching may not work properly"
                        )
            else:
                print(
                    "Current model is not onnx or engine format, cannot properly handle TRT mode."
                )
                self.config["groups"][self.group]["is_trt"] = False
                dpg.set_value(self.is_trt_checkbox, False)
                return
            self.refresh_engine()
        else:
            current_model = self.config["groups"][self.group]["infer_model"]
            if current_model.endswith(".engine"):
                onnx_path = self.config["groups"][self.group].get(
                    "original_infer_model", None
                )
                if onnx_path and os.path.exists(onnx_path):
                    self.config["groups"][self.group]["infer_model"] = onnx_path
                    self.refresh_engine()
                    dpg.set_value(self.infer_model_input, onnx_path)
                    print("Switched back to ONNX Runtime inference")
                else:
                    print(
                        "Original ONNX model path not found, please check configuration."
                    )
        class_num = self.get_current_class_num()
        class_ary = list(range(class_num))
        self.create_checkboxes(class_ary)
        self.update_class_aim_combo()
        self.update_target_reference_class_combo()

    def on_show_infer_time_change(self, sender, app_data):
        self.config["show_infer_time"] = app_data
        print(f"Show inference time: {app_data}")

    def on_show_fov_change(self, sender, app_data):
        self.config["show_fov"] = app_data
        print(f"Show aim scope: {app_data}")

    def on_small_target_enabled_change(self, sender, app_data):
        self.config["small_target_enhancement"]["enabled"] = app_data
        print(
            f"Small target recognition enhancement: {('enabled' if app_data else 'disabled')}"
        )

    def on_small_target_smooth_change(self, sender, app_data):
        self.config["small_target_enhancement"]["smooth_enabled"] = app_data
        print(f"Small target smoothing: {('enabled' if app_data else 'disabled')}")

    def on_small_target_nms_change(self, sender, app_data):
        self.config["small_target_enhancement"]["adaptive_nms"] = app_data
        print(f"Adaptive NMS: {('enabled' if app_data else 'disabled')}")

    def on_small_target_boost_change(self, sender, app_data):
        self.config["small_target_enhancement"]["boost_factor"] = app_data
        print(f"Small target boost factor set to: {app_data}")

    def on_small_target_frames_change(self, sender, app_data):
        self.config["small_target_enhancement"]["smooth_frames"] = app_data
        self.target_history_max_frames = app_data
        print(f"Smooth history frames set to: {app_data}")

    def on_small_target_threshold_change(self, sender, app_data):
        self.config["small_target_enhancement"]["threshold"] = app_data
        print(f"Small target threshold set to: {app_data:.3f}")

    def on_medium_target_threshold_change(self, sender, app_data):
        self.config["small_target_enhancement"]["medium_threshold"] = app_data
        print(f"Medium target threshold set to: {app_data:.3f}")

    def _init_config_handlers(self):
        """
        Initialize configuration change handlers, register config items and handler functions
        """
        basic_group = ConfigItemGroup(self.config_handler)
        basic_group.register_item("infer_debug", "infer_debug", bool)
        basic_group.register_item("is_curve", "is_curve", bool)
        basic_group.register_item("is_curve_uniform", "is_curve_uniform", bool)
        basic_group.register_item("print_fps", "print_fps", bool)
        basic_group.register_item(
            "show_motion_speed",
            "show_motion_speed",
            bool,
            self.refresh_controller_params,
        )
        basic_group.register_item("is_show_curve", "is_show_curve", bool)
        basic_group.register_item("is_show_down", "is_show_down", bool)
        basic_group.register_item("game_sensitivity", "game_sensitivity", float)
        basic_group.register_item("mouse_dpi", "mouse_dpi", int)
        basic_group.register_item("is_v8", "is_v8", bool)
        basic_group.register_item("right_down", "right_down", bool)
        scoring_group = ConfigItemGroup(self.config_handler)
        scoring_group.register_item(
            "distance_scoring_weight",
            "distance_scoring_weight",
            float,
            self.init_target_priority,
        )
        scoring_group.register_item(
            "center_scoring_weight",
            "center_scoring_weight",
            float,
            self.init_target_priority,
        )
        scoring_group.register_item(
            "size_scoring_weight",
            "size_scoring_weight",
            float,
            self.init_target_priority,
        )
        scoring_group.register_item(
            "aim_weight_tiebreak_ratio",
            "aim_weight_tiebreak_ratio",
            float,
        )
        screenshot_group = ConfigItemGroup(self.config_handler)
        screenshot_group.register_item(
            "is_obs",
            "is_obs",
            bool,
            lambda: self.screenshot_manager.update_config(
                "is_obs", self.config["is_obs"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "is_cjk",
            "is_cjk",
            bool,
            lambda: self.screenshot_manager.update_config(
                "is_cjk", self.config["is_cjk"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "obs_ip",
            "obs_ip",
            str,
            lambda: self.screenshot_manager.update_config(
                "obs_ip", self.config["obs_ip"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "obs_port",
            "obs_port",
            int,
            lambda: self.screenshot_manager.update_config(
                "obs_port", self.config["obs_port"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "obs_fps",
            "obs_fps",
            int,
            lambda: self.screenshot_manager.update_config(
                "obs_fps", self.config["obs_fps"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "cjk_device_id",
            "cjk_device_id",
            int,
            lambda: self.screenshot_manager.update_config(
                "cjk_device_id", self.config["cjk_device_id"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "cjk_fps",
            "cjk_fps",
            int,
            lambda: self.screenshot_manager.update_config(
                "cjk_fps", self.config["cjk_fps"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "cjk_resolution",
            "cjk_resolution",
            int,
            lambda: self.screenshot_manager.update_config(
                "cjk_resolution", self.config["cjk_resolution"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "cjk_crop_size",
            "cjk_crop_size",
            int,
            lambda: self.screenshot_manager.update_config(
                "cjk_crop_size", self.config["cjk_crop_size"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "capture_size",
            "capture_size",
            str,
            lambda: self.screenshot_manager.update_config(
                "capture_size", self.config["capture_size"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "enable_parallel_processing",
            "enable_parallel_processing",
            bool,
            lambda: self.screenshot_manager.update_config(
                "enable_parallel_processing", self.config["enable_parallel_processing"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "turbo_mode",
            "turbo_mode",
            bool,
            lambda: self.screenshot_manager.update_config(
                "turbo_mode", self.config["turbo_mode"]
            )
            if self.screenshot_manager
            else None,
        )
        screenshot_group.register_item(
            "skip_frame_processing",
            "skip_frame_processing",
            bool,
            lambda: self.screenshot_manager.update_config(
                "skip_frame_processing", self.config["skip_frame_processing"]
            )
            if self.screenshot_manager
            else None,
        )
        curve_group = ConfigItemGroup(self.config_handler)
        curve_group.register_item("offset_boundary_x", "offset_boundary_x", int)
        curve_group.register_item("offset_boundary_y", "offset_boundary_y", int)
        curve_group.register_item("knots_count", "knots_count", int)
        curve_group.register_item("distortion_mean", "distortion_mean", float)
        curve_group.register_item("distortion_st_dev", "distortion_st_dev", float)
        curve_group.register_item("distortion_frequency", "distortion_frequency", float)
        curve_group.register_item("target_points", "target_points", int)
        move_group = ConfigItemGroup(self.config_handler)
        move_group.register_item("move_method", "move_method", str)
        key_group = ConfigItemGroup(self.config_handler)
        key_group.register_item("group", "group", str, self.update_group_inputs)
        aim_key_group = ConfigItemGroup(
            self.config_handler, "groups.{group}.aim_keys.{key}"
        )
        aim_key_group.register_item(
            "confidence_threshold", "confidence_threshold", float
        )
        aim_key_group.register_item("iou_t", "iou_t", float)
        aim_key_group.register_item("aim_bot_position", "aim_bot_position", float)
        aim_key_group.register_item("aim_bot_position2", "aim_bot_position2", float)
        aim_key_group.register_item("aim_bot_scope", "aim_bot_scope", int)
        aim_key_group.register_item("min_position_offset", "min_position_offset", int)
        aim_key_group.register_item(
            "smoothing_factor",
            "smoothing_factor",
            float,
            self.refresh_controller_params,
        )
        aim_key_group.register_item(
            "base_step", "base_step", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "distance_weight", "distance_weight", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "fov_angle", "fov_angle", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "history_size", "history_size", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "deadzone", "deadzone", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "smoothing", "smoothing", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "velocity_decay", "velocity_decay", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "current_frame_weight",
            "current_frame_weight",
            float,
            self.refresh_controller_params,
        )
        aim_key_group.register_item(
            "last_frame_weight",
            "last_frame_weight",
            float,
            self.refresh_controller_params,
        )
        aim_key_group.register_item(
            "output_scale_x", "output_scale_x", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "output_scale_y", "output_scale_y", float, self.refresh_controller_params
        )
        aim_key_group.register_item("uniform_threshold", "uniform_threshold", float)
        aim_key_group.register_item(
            "min_velocity_threshold", "min_velocity_threshold", float
        )
        aim_key_group.register_item(
            "max_velocity_threshold", "max_velocity_threshold", float
        )
        aim_key_group.register_item("compensation_factor", "compensation_factor", float)
        aim_key_group.register_item("auto_y", "auto_y", bool)
        aim_key_group.register_item(
            "pid_kp_x", "pid_kp_x", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "pid_ki_x", "pid_ki_x", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "pid_kd_x", "pid_kd_x", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "pid_kp_y", "pid_kp_y", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "pid_ki_y", "pid_ki_y", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "pid_kd_y", "pid_kd_y", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "pid_integral_limit_x",
            "pid_integral_limit_x",
            float,
            self.refresh_controller_params,
        )
        aim_key_group.register_item(
            "pid_integral_limit_y",
            "pid_integral_limit_y",
            float,
            self.refresh_controller_params,
        )
        aim_key_group.register_item(
            "smooth_x", "smooth_x", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "smooth_y", "smooth_y", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "smooth_deadzone", "smooth_deadzone", float, self.refresh_controller_params
        )
        aim_key_group.register_item(
            "smooth_algorithm",
            "smooth_algorithm",
            float,
            self.refresh_controller_params,
        )
        aim_key_group.register_item("move_deadzone", "move_deadzone", float)
        aim_key_group.register_item("target_switch_delay", "target_switch_delay", int)
        aim_key_group.register_item(
            "target_reference_class", "target_reference_class", int
        )
        aim_key_group.register_item(
            "target_lock_distance", "target_lock_distance", int
        )
        aim_key_group.register_item(
            "target_lock_reacquire_time", "target_lock_reacquire_time", float
        )
        aim_key_group.register_item(
            "dynamic_scope.enabled", "dynamic_scope.enabled", bool
        )
        aim_key_group.register_item(
            "dynamic_scope.min_ratio", "dynamic_scope.min_ratio", float
        )
        aim_key_group.register_item(
            "dynamic_scope.min_scope", "dynamic_scope.min_scope", int
        )
        aim_key_group.register_item(
            "dynamic_scope.shrink_duration_ms", "dynamic_scope.shrink_duration_ms", int
        )
        aim_key_group.register_item(
            "dynamic_scope.recover_duration_ms",
            "dynamic_scope.recover_duration_ms",
            int,
        )
        aim_key_group.register_item("status", "status", bool)
        aim_key_group.register_item("start_delay", "start_delay", float)
        aim_key_group.register_item("long_press_duration", "long_press_duration", int)
        aim_key_group.register_item("press_delay", "press_delay", float)
        aim_key_group.register_item("end_delay", "end_delay", float)
        aim_key_group.register_item("random_delay", "random_delay", float)
        aim_key_group.register_item("x_trigger_scope", "x_trigger_scope", int)
        aim_key_group.register_item("y_trigger_scope", "y_trigger_scope", int)
        aim_key_group.register_item("x_trigger_offset", "x_trigger_offset", int)
        aim_key_group.register_item("y_trigger_offset", "y_trigger_offset", int)
        sunone_group = ConfigItemGroup(self.config_handler, "sunone")
        sunone_group.register_item(
            "sunone_use_smoothing",
            "use_smoothing",
            bool,
            self.update_sunone_visibility,
        )
        sunone_group.register_item(
            "sunone_smoothness", "smoothness", int, self.update_sunone_visibility
        )
        sunone_group.register_item(
            "sunone_tracking_smoothing",
            "tracking_smoothing",
            bool,
            self.update_sunone_visibility,
        )
        sunone_group.register_item(
            "sunone_use_kalman",
            "use_kalman",
            bool,
            self.update_sunone_visibility,
        )
        sunone_group.register_item(
            "sunone_kalman_process_noise", "kalman_process_noise", float
        )
        sunone_group.register_item(
            "sunone_kalman_measurement_noise", "kalman_measurement_noise", float
        )
        sunone_group.register_item(
            "sunone_kalman_speed_multiplier_x", "kalman_speed_multiplier_x", float
        )
        sunone_group.register_item(
            "sunone_kalman_speed_multiplier_y", "kalman_speed_multiplier_y", float
        )
        sunone_group.register_item("sunone_reset_threshold", "reset_threshold", float)
        sunone_group.register_item(
            "sunone_prediction_interval", "prediction.interval", float
        )
        sunone_group.register_item(
            "sunone_prediction_kalman_lead_ms",
            "prediction.kalman_lead_ms",
            float,
        )
        sunone_group.register_item(
            "sunone_prediction_kalman_max_lead_ms",
            "prediction.kalman_max_lead_ms",
            float,
        )
        sunone_group.register_item(
            "sunone_prediction_velocity_smoothing",
            "prediction.velocity_smoothing",
            float,
        )
        sunone_group.register_item(
            "sunone_prediction_velocity_scale",
            "prediction.velocity_scale",
            float,
        )
        sunone_group.register_item(
            "sunone_prediction_kalman_process_noise",
            "prediction.kalman_process_noise",
            float,
        )
        sunone_group.register_item(
            "sunone_prediction_kalman_measurement_noise",
            "prediction.kalman_measurement_noise",
            float,
        )
        sunone_group.register_item(
            "sunone_prediction_future_positions",
            "prediction.future_positions",
            int,
        )
        sunone_group.register_item(
            "sunone_prediction_draw_future_positions",
            "prediction.draw_future_positions",
            bool,
        )
        sunone_group.register_item(
            "sunone_speed_min_multiplier", "speed.min_multiplier", float
        )
        sunone_group.register_item(
            "sunone_speed_max_multiplier", "speed.max_multiplier", float
        )
        sunone_group.register_item("sunone_speed_snap_radius", "speed.snap_radius", float)
        sunone_group.register_item("sunone_speed_near_radius", "speed.near_radius", float)
        sunone_group.register_item(
            "sunone_speed_curve_exponent", "speed.speed_curve_exponent", float
        )
        sunone_group.register_item(
            "sunone_speed_snap_boost_factor", "speed.snap_boost_factor", float
        )
        sunone_group.register_item(
            "sunone_debug_show_prediction", "debug.show_prediction", bool
        )
        sunone_group.register_item("sunone_debug_show_step", "debug.show_step", bool)
        sunone_group.register_item("sunone_debug_show_future", "debug.show_future", bool)
        infer_group = ConfigItemGroup(self.config_handler)
        infer_group.register_item("is_trt", "is_trt", bool, None, self.on_is_trt_change)
        infer_group.register_item("show_infer_time", "show_infer_time", bool)
        mask_group = ConfigItemGroup(self.config_handler)
        mask_group.register_item(
            "mask_left", "mask_left", bool, None, self.on_mask_left_change
        )
        mask_group.register_item(
            "mask_right", "mask_right", bool, None, self.on_mask_right_change
        )
        mask_group.register_item(
            "mask_middle", "mask_middle", bool, None, self.on_mask_middle_change
        )
        mask_group.register_item(
            "mask_side1", "mask_side1", bool, None, self.on_mask_side1_change
        )
        mask_group.register_item(
            "mask_side2", "mask_side2", bool, None, self.on_mask_side2_change
        )
        mask_group.register_item("mask_x", "mask_x", bool, None, self.on_mask_x_change)
        mask_group.register_item("mask_y", "mask_y", bool, None, self.on_mask_y_change)
        mask_group.register_item(
            "mask_wheel", "mask_wheel", bool, None, self.on_mask_wheel_change
        )
        mask_group.register_item("aim_mask_x", "aim_mask_x", int)
        mask_group.register_item("aim_mask_y", "aim_mask_y", int)
        self.config_handler.register_config_item(
            "infer_model",
            "groups.{group}.infer_model",
            None,
            None,
            self.on_infer_model_change,
        )
        self.config_handler.register_config_item(
            "key", "key", None, None, self.on_key_change
        )
        self.config_handler.register_config_item(
            "games", "games", None, None, self.on_games_change
        )
        self.config_handler.register_config_item(
            "guns", "guns", None, None, self.on_guns_change
        )
        self.config_handler.register_config_item(
            "stages", "stages", None, None, self.on_stages_change
        )

    def on_gui_dpi_scale_change(self, sender, app_data):
        """DPI scale change callback"""
        self.config["gui_dpi_scale"] = app_data
        print(f"DPI scale changed to: {app_data:.2f}, will take effect after restart")

    def on_gui_width_scale_change(self, sender, app_data):
        self.config["gui_width_scale"] = app_data
        print("UI width scale changed, will take effect after restart")

    def on_gui_font_scale_change(self, sender, app_data):
        self.config["gui_font_scale"] = app_data
        print("Font scale changed, will take effect after restart")

    def on_ui_language_change(self, sender, app_data):
        if app_data == "Русский":
            self.config["ui_language"] = "ru"
        else:
            self.config["ui_language"] = "en"
        print("UI language changed, will take effect after restart")

    def on_reset_dpi_scale_click(self, sender, app_data):
        """Reset DPI scale to auto-detected value"""
        auto_scale = self.get_system_dpi_scale()
        self.config["gui_dpi_scale"] = 0.0
        dpg.set_value(self.dpi_scale_slider, auto_scale)
        print(
            f"DPI scale reset to auto-detection: {auto_scale:.2f}, will take effect after restart"
        )

    def on_aim_controller_change(self, sender, app_data):
        if app_data == self.tr("label_controller_sunone"):
            self.config["aim_controller"] = "sunone"
        else:
            self.config["aim_controller"] = "pid"
        self.update_controller_visibility()
        if hasattr(self, "reset_pid"):
            self.reset_pid()

    def on_sunone_prediction_mode_change(self, sender, app_data):
        modes = {"Standard": 0, "Kalman": 1, "Kalman + Standard": 2}
        self.config["sunone"]["prediction"]["mode"] = modes.get(app_data, 0)
        self.update_sunone_visibility()

    def update_controller_visibility(self):
        controller = self.config.get("aim_controller", "pid")
        if hasattr(self, "sunone_settings_group") and self.sunone_settings_group:
            if controller == "sunone":
                dpg.show_item(self.sunone_settings_group)
            else:
                dpg.hide_item(self.sunone_settings_group)
        if hasattr(self, "pid_params_group") and self.pid_params_group:
            if controller == "pid":
                dpg.show_item(self.pid_params_group)
            else:
                dpg.hide_item(self.pid_params_group)

    def update_sunone_visibility(self):
        if not hasattr(self, "sunone_pred_standard_group"):
            return
        use_smoothing = bool(self.config["sunone"].get("use_smoothing", False))
        if use_smoothing:
            dpg.show_item("sunone_smoothness")
            dpg.show_item("sunone_tracking_smoothing")
        else:
            dpg.hide_item("sunone_smoothness")
            dpg.hide_item("sunone_tracking_smoothing")
        use_kalman = bool(self.config["sunone"].get("use_kalman", False))
        if hasattr(self, "sunone_kalman_group") and self.sunone_kalman_group:
            if use_kalman:
                dpg.show_item(self.sunone_kalman_group)
            else:
                dpg.hide_item(self.sunone_kalman_group)
        mode = int(self.config["sunone"]["prediction"].get("mode", 0))
        if mode in (0, 2):
            dpg.show_item(self.sunone_pred_standard_group)
        else:
            dpg.hide_item(self.sunone_pred_standard_group)
        if mode in (1, 2):
            dpg.show_item(self.sunone_pred_kalman_group)
        else:
            dpg.hide_item(self.sunone_pred_kalman_group)
        self.update_prediction_preview()

    def on_select_class_names_click(self, sender, app_data):
        try:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select class names file",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
            root.destroy()
        except Exception:
            file_path = ""
        if not file_path:
            return
        names = self.load_class_names_from_file(file_path)
        self.config["class_names_file"] = file_path
        self.config["class_names"] = names
        if hasattr(self, "class_names_file_input") and self.class_names_file_input:
            dpg.set_value(self.class_names_file_input, file_path)
        self.refresh_class_names()
        class_num = self.get_current_class_num()
        class_ary = list(range(class_num))
        self.create_checkboxes(class_ary)
        self.update_class_aim_combo()
        self.update_target_reference_class_combo()

    def on_apply_class_names_click(self, sender, app_data):
        raw = ""
        try:
            if hasattr(self, "class_names_manual_input"):
                raw = dpg.get_value(self.class_names_manual_input) or ""
        except Exception:
            raw = ""
        cleaned = raw.replace(";", "\n").replace(",", "\n")
        names = [name.strip() for name in cleaned.splitlines() if name.strip()]
        if not names:
            return
        self.config["class_names"] = names
        self.config["class_names_file"] = ""
        if hasattr(self, "class_names_file_input") and self.class_names_file_input:
            dpg.set_value(self.class_names_file_input, "")
        self.refresh_class_names()
        class_num = self.get_current_class_num()
        class_ary = list(range(class_num))
        self.create_checkboxes(class_ary)
        self.update_class_aim_combo()
        self.update_target_reference_class_combo()

    def on_change(self, sender, app_data):
        """
        Generic configuration change handler, forwards events to ConfigChangeHandler

        Args:
            sender: Sender ID
            app_data: New configuration value
        """
        self.config_handler.handle_change(sender, app_data)

    def on_tab_change(self, sender, app_data):
        if hasattr(self, "tab_bar_container"):
            dpg.set_x_scroll(self.tab_bar_container, 0)

    def on_controller_type_change(self, sender, app_data):
        """Controller type switch"""
        print("Current version only supports PID controller")

    def on_pid_kp_x_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["pid_kp_x"] = (
            round(app_data, 4)
        )
        self._update_pid_params()

    def on_pid_ki_x_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["pid_ki_x"] = (
            round(app_data, 4)
        )
        self._update_pid_params()

    def on_pid_kd_x_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["pid_kd_x"] = (
            round(app_data, 4)
        )
        self._update_pid_params()

    def on_pid_kp_y_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["pid_kp_y"] = (
            round(app_data, 4)
        )
        self._update_pid_params()

    def on_pid_ki_y_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["pid_ki_y"] = (
            round(app_data, 4)
        )
        self._update_pid_params()

    def on_pid_kd_y_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["pid_kd_y"] = (
            round(app_data, 4)
        )
        self._update_pid_params()

    def on_pid_integral_limit_x_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "pid_integral_limit_x"
        ] = round(app_data, 4)
        self._update_pid_params()

    def on_pid_integral_limit_y_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "pid_integral_limit_y"
        ] = round(app_data, 4)
        self._update_pid_params()

    def on_smooth_x_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["smooth_x"] = (
            round(app_data, 4)
        )
        self._update_pid_params()

    def on_smooth_y_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key]["smooth_y"] = (
            round(app_data, 4)
        )
        self._update_pid_params()

    def on_smooth_deadzone_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "smooth_deadzone"
        ] = round(app_data, 4)
        self._update_pid_params()

    def on_smooth_algorithm_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "smooth_algorithm"
        ] = round(app_data, 4)
        self._update_pid_params()

    def on_move_deadzone_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "move_deadzone"
        ] = round(app_data, 4)

    def on_target_switch_delay_change(self, sender, app_data):
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "target_switch_delay"
        ] = app_data

    def on_target_reference_class_change(self, sender, app_data):
        class_id = self.parse_class_label(app_data)
        self.config["groups"][self.group]["aim_keys"][self.select_key][
            "target_reference_class"
        ] = class_id

    def _update_pid_params(self):
        """Update dual-axis PID controller parameters"""
        if hasattr(self, "dual_pid"):
            self.refresh_controller_params()

    def _register_control_callback(self, control_id):
        """
        Register callback function for control

        Args:
            control_id: Control ID
        """
        dpg.set_item_callback(control_id, self.on_change)
