import json
import threading
import time

import dearpygui.dearpygui as dpg


class RecoilMixin:
    """Mixin class for mouse recoil trajectory methods."""

    def update_mouse_re_ui_status(self):
        """Refresh mouse_re status panel"""
        try:
            tr = getattr(self, "tr", None)
            switch_on = tr("label_switch_on") if callable(tr) else "ON"
            switch_off = tr("label_switch_off") if callable(tr) else "OFF"
            switch_prefix = tr("label_switch_prefix") if callable(tr) else "Switch"
            mapping_prefix = (
                tr("label_mapping_file_prefix") if callable(tr) else "Mapping file"
            )
            points_prefix = (
                tr("label_trajectory_points_prefix")
                if callable(tr)
                else "Trajectory points"
            )
            none_text = tr("label_none") if callable(tr) else "None"
            switch_text = (
                switch_on
                if self.config.get("recoil", {}).get("use_mouse_re_trajectory", False)
                and getattr(self, "down_switch", False)
                else switch_off
            )
            key = f"{getattr(self, 'mouse_re_picked_game', '')}:{getattr(self, 'mouse_re_picked_gun', '')}"
            recoil_config = self.config.get("recoil", {})
            mapping = recoil_config.get("mapping", {})
            if not isinstance(mapping, dict):
                print(
                    f"[Warning] mapping is not dict type: {type(mapping)}, resetting to empty dict"
                )
                mapping = {}
                self.config["recoil"]["mapping"] = {}
            file_text = none_text
            if key and key != ":":
                entry = mapping.get(key)
                if entry and isinstance(entry, dict) and ("path" in entry):
                    file_text = entry["path"] or none_text
            points_cnt = (
                len(self._current_mouse_re_points)
                if self._current_mouse_re_points
                else 0
            )
            if dpg.does_item_exist("mouse_re_switch_text"):
                dpg.set_value(
                    "mouse_re_switch_text", f"{switch_prefix}: {switch_text}"
                )
            if dpg.does_item_exist("mouse_re_file_text"):
                dpg.set_value(
                    "mouse_re_file_text", f"{mapping_prefix}: {file_text}"
                )
            if dpg.does_item_exist("mouse_re_points_text"):
                dpg.set_value(
                    "mouse_re_points_text", f"{points_prefix}: {points_cnt}"
                )
        except Exception as e:
            print(f"Refresh mouse_re status failed: {e}")
            import traceback

            traceback.print_exc()

    def on_use_mouse_re_trajectory_change(self, sender, app_data):
        """Enable/disable mouse_re trajectory recoil"""
        try:
            enabled = bool(app_data)
            self.config["recoil"]["use_mouse_re_trajectory"] = enabled
            print(
                f"[mouse_re] Trajectory recoil: {('enabled' if enabled else 'disabled')}"
            )
            self.update_mouse_re_ui_status()
        except Exception as e:
            print(f"Update mouse_re trajectory recoil switch failed: {e}")

    def on_mouse_re_replay_speed_change(self, sender, app_data):
        """Update mouse_re trajectory replay speed"""
        try:
            speed = float(app_data)
            if speed <= 0:
                speed = 1.0
            self.config["recoil"]["replay_speed"] = speed
            print(f"[mouse_re] Replay speed set to {speed}x")
        except Exception as e:
            print(f"Update mouse_re replay speed failed: {e}")

    def on_mouse_re_pixel_enhancement_change(self, sender, app_data):
        """Update mouse_re trajectory pixel enhancement ratio"""
        try:
            ratio = float(app_data)
            if ratio <= 0:
                ratio = 1.0
            self.config["recoil"]["pixel_enhancement_ratio"] = ratio
            print(f"[mouse_re] Pixel enhancement ratio set to {ratio}x")
        except Exception as e:
            print(f"Update mouse_re pixel enhancement ratio failed: {e}")

    def on_import_mouse_re_trajectory_click(self, sender, app_data):
        """Import trajectory file for mouse_re selected game/gun"""
        try:
            if not self.mouse_re_picked_game or not self.mouse_re_picked_gun:
                print("[mouse_re] Please select game and gun first")
                return
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select mouse_re trajectory JSON",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            root.destroy()
            if not file_path:
                return
            key = f"{self.mouse_re_picked_game}:{self.mouse_re_picked_gun}"
            self.config["recoil"]["mapping"][key] = {"path": file_path}
            print(f"[mouse_re] Bound trajectory for {key}: {file_path}")
            self._current_mouse_re_points = self._load_mouse_re_trajectory_for_current()
            self.update_mouse_re_ui_status()
        except Exception as e:
            print(f"Import mouse_re trajectory failed: {e}")

    def on_clear_mouse_re_mapping_click(self, sender, app_data):
        """Clear mapping for mouse_re selected game/gun"""
        try:
            if not self.mouse_re_picked_game or not self.mouse_re_picked_gun:
                print("[mouse_re] Please select game and gun first")
                return
            key = f"{self.mouse_re_picked_game}:{self.mouse_re_picked_gun}"
            if key in self.config["recoil"]["mapping"]:
                del self.config["recoil"]["mapping"][key]
                print(f"[mouse_re] Cleared mapping: {key}")
            self._current_mouse_re_points = None
            self.update_mouse_re_ui_status()
        except Exception as e:
            print(f"Clear mouse_re mapping failed: {e}")

    def _load_mouse_re_trajectory_for_current(self):
        # Load trajectory bound to mouse_re selected game/gun as incremental point sequence
        try:
            if not (
                getattr(self, "mouse_re_picked_game", "")
                and getattr(self, "mouse_re_picked_gun", "")
            ):
                return None
            key = f"{self.mouse_re_picked_game}" + ":" + f"{self.mouse_re_picked_gun}"
            recoil_config = self.config.get("recoil", {})
            mapping = recoil_config.get("mapping", {})
            if not isinstance(mapping, dict):
                print("[Warning] mapping is not dict type: " + f"{type(mapping)}")
                return None
            entry = mapping.get(key)
            if not entry or not isinstance(entry, dict) or "path" not in entry:
                print("[mouse_re] Mapping not found: " + f"{key}")
                return None
            path = entry["path"]
            return self._parse_mouse_re_json(path)
        except Exception as e:
            print("Load mouse_re trajectory failed: " + f"{e}")
            import traceback

            traceback.print_exc()

    def _parse_mouse_re_json(self, path):
        """Parse JSON saved by mouse_re.py, convert to incremental (dx,dy,dt_ms) sequence"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                print(
                    f"[mouse_re] JSON format error: root object is not dict, is {type(data)}"
                )
                return
            moves = data.get("movements")
            if not moves:
                print("[mouse_re] No 'movements' field found in JSON")
                return
            if not isinstance(moves, list):
                print(f"[mouse_re] movements field is not list, is {type(moves)}")
                return
            if len(moves) < 2:
                print(f"[mouse_re] Too few trajectory points: {len(moves)}")
                return
            points = []
            prev = moves[0]
            if not isinstance(prev, dict):
                print(f"[mouse_re] First trajectory point is not dict: {type(prev)}")
                return
            prev_t = float(prev.get("timestamp", 0.0))
            prev_x = float(prev.get("x", 0.0))
            prev_y = float(prev.get("y", 0.0))
            for i in range(1, len(moves)):
                m = moves[i]
                if not isinstance(m, dict):
                    print(f"[mouse_re] Trajectory point {i} is not dict: {type(m)}")
                    continue
                cur_t = float(m.get("timestamp", prev_t))
                cur_x = float(m.get("x", prev_x))
                cur_y = float(m.get("y", prev_y))
                dx = cur_x - prev_x
                dy = cur_y - prev_y
                dt_ms = max(0.0, (cur_t - prev_t) * 1000.0)
                points.append({"dx": dx, "dy": dy, "dt_ms": dt_ms})
                prev_t, prev_x, prev_y = (cur_t, cur_x, cur_y)
            print(
                f"[mouse_re] Successfully parsed trajectory: {len(points)} incremental points"
            )
            return points
        except Exception as e:
            print(f"Parse mouse_re JSON failed: {e}")
            import traceback

            traceback.print_exc()

    def _start_mouse_re_recoil(self):
        """Start mouse_re trajectory replay thread"""
        if self._recoil_is_replaying:
            return
        if self.config["groups"][self.group]["right_down"] and (not self.right_pressed):
            return
        if self._current_mouse_re_points is None:
            self._current_mouse_re_points = self._load_mouse_re_trajectory_for_current()
        if not self._current_mouse_re_points:
            print("[mouse_re] No available trajectory, cannot start replay")
            return
        self._recoil_is_replaying = True
        self._recoil_replay_thread = threading.Thread(
            target=self._recoil_replay_worker,
            args=(self._current_mouse_re_points,),
            daemon=True,
        )
        self._recoil_replay_thread.start()

    def _stop_mouse_re_recoil(self):
        """Stop mouse_re trajectory replay"""
        self._recoil_is_replaying = False

    def _recoil_replay_worker(self, points):
        """Replay relative movements in background according to time sequence"""
        try:
            speed = float(self.config.get("recoil", {}).get("replay_speed", 1.0))
            speed = 1.0 if speed <= 0 else speed
            enhancement_ratio = float(
                self.config.get("recoil", {}).get("pixel_enhancement_ratio", 1.0)
            )
            enhancement_ratio = 1.0 if enhancement_ratio <= 0 else enhancement_ratio
            for step in points:
                left_press_valid = self.left_pressed and self.down_switch
                trigger_press_valid = self.trigger_recoil_pressed
                if not self._recoil_is_replaying or (
                    not left_press_valid and (not trigger_press_valid)
                ):
                    break
                if self.config["groups"][self.group]["right_down"] and (
                    not self.right_pressed
                ):
                    break
                dx = step["dx"] * enhancement_ratio
                dy = step["dy"] * enhancement_ratio
                ix = int(round(dx))
                iy = int(round(dy))
                if ix != 0 or iy != 0:
                    try:
                        self.move_r(ix, iy)
                    except Exception as e:
                        print(f"[mouse_re] move_r call failed: {e}")
                        break
                dt_ms = step.get("dt_ms", 0.0) / speed
                remaining = max(0.0, dt_ms) / 1000.0
                if remaining <= 0.0005:
                    continue
                if remaining <= 0.003:
                    time.sleep(remaining)
                    continue
                while remaining > 0 and self._recoil_is_replaying:
                    left_press_valid = self.left_pressed and self.down_switch
                    trigger_press_valid = self.trigger_recoil_pressed
                    if not left_press_valid and (not trigger_press_valid):
                        break
                    chunk = 0.003 if remaining > 0.006 else remaining
                    time.sleep(chunk)
                    remaining -= chunk
        except Exception as e:
            print(f"mouse_re replay thread error: {e}")
        finally:
            self._recoil_is_replaying = False

    def render_mouse_re_games_combo(self):
        """Render mouse_re game selection dropdown"""
        try:
            if self.mouse_re_games_combo is not None:
                dpg.delete_item(self.mouse_re_games_combo)
            games_config = self.config.get("games", {})
            if not isinstance(games_config, dict):
                print(f"[Warning] games config is not dict: {type(games_config)}")
                games = []
            else:
                games = list(games_config.keys())
            if not self.mouse_re_picked_game or self.mouse_re_picked_game not in games:
                self.mouse_re_picked_game = games[0] if games else ""
            self.mouse_re_games_combo = dpg.add_combo(
                label="mouse_re Game",
                items=games,
                default_value=self.mouse_re_picked_game,
                callback=self.on_mouse_re_games_change,
                width=150,
                parent="mouse_re_combos_group",
            )
        except Exception as e:
            print(f"Render mouse_re game dropdown failed: {e}")
            import traceback

            traceback.print_exc()

    def render_mouse_re_guns_combo(self):
        """Render mouse_re gun selection dropdown"""
        try:
            if self.mouse_re_guns_combo is not None:
                dpg.delete_item(self.mouse_re_guns_combo)
            if not self.mouse_re_picked_game:
                self.mouse_re_guns_combo = dpg.add_combo(
                    label="mouse_re Gun",
                    items=[],
                    default_value="",
                    callback=self.on_mouse_re_guns_change,
                    width=150,
                    parent="mouse_re_combos_group",
                )
                return
            games_config = self.config.get("games", {})
            if (
                not isinstance(games_config, dict)
                or self.mouse_re_picked_game not in games_config
            ):
                guns = []
            else:
                game_guns = games_config[self.mouse_re_picked_game]
                if isinstance(game_guns, dict):
                    guns = list(game_guns.keys())
                else:
                    guns = []
            if not self.mouse_re_picked_gun or self.mouse_re_picked_gun not in guns:
                self.mouse_re_picked_gun = guns[0] if guns else ""
            self.mouse_re_guns_combo = dpg.add_combo(
                label="mouse_re Gun",
                items=guns,
                default_value=self.mouse_re_picked_gun,
                callback=self.on_mouse_re_guns_change,
                width=150,
                parent="mouse_re_combos_group",
            )
        except Exception as e:
            print(f"Render mouse_re gun dropdown failed: {e}")
            import traceback

            traceback.print_exc()

    def on_mouse_re_games_change(self, sender, app_data):
        """mouse_re game selection changed"""
        self.mouse_re_picked_game = app_data
        self.mouse_re_picked_gun = ""
        self.render_mouse_re_guns_combo()
        self._current_mouse_re_points = self._load_mouse_re_trajectory_for_current()
        self.update_mouse_re_ui_status()

    def on_mouse_re_guns_change(self, sender, app_data):
        """mouse_re gun selection changed"""
        self.mouse_re_picked_gun = app_data
        self._current_mouse_re_points = self._load_mouse_re_trajectory_for_current()
        self.update_mouse_re_ui_status()

    def on_card_change(self, sender, app_data):
        print("Card key is read-only, cannot modify")
