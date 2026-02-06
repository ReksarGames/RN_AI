import math
import threading
import time


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


class Kalman1D:
    def __init__(self, process_noise, measurement_noise):
        self.Q = max(process_noise, 1e-6)
        self.R = max(measurement_noise, 1e-6)
        self.x = 0.0
        self.v = 0.0
        self.P = 1.0

    def reset(self, x=0.0, v=0.0):
        self.x = x
        self.v = v
        self.P = 1.0

    def update(self, z, dt):
        self.x += self.v * dt
        self.P += self.Q * dt
        k = self.P / (self.P + self.R)
        self.x += k * (z - self.x)
        self.P *= (1.0 - k)
        self.v = (1.0 - k) * self.v + k * ((z - self.x) / max(dt, 1e-8))
        return self.x


class SunoneAimController:
    def __init__(self):
        self._debug_lock = threading.Lock()
        self._prediction_debug = None
        self._prediction_step = None
        self._future_positions = []
        self._last_prediction_q = None
        self._last_prediction_r = None
        self.reset()

    def reset(self):
        self.prev_time = None
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_velocity_x = 0.0
        self.prev_velocity_y = 0.0
        self.last_target_time = 0.0
        self.target_detected = False
        self.reset_prediction_state()
        self.reset_kalman_state()
        self.reset_smoothing_state()
        self._clear_debug()

    def _clear_debug(self):
        with self._debug_lock:
            self._prediction_debug = None
            self._prediction_step = None
            self._future_positions = []

    def reset_prediction_state(self):
        self.prediction_prev_time = None
        self.prediction_prev_x = 0.0
        self.prediction_prev_y = 0.0
        self.prediction_smoothed_velocity_x = 0.0
        self.prediction_smoothed_velocity_y = 0.0
        self.prediction_velocity_x = 0.0
        self.prediction_velocity_y = 0.0
        self.prediction_initialized = False
        self.prediction_kalman_time = None
        self.prediction_kalman_initialized = False
        self._last_prediction_mode = None
        q = getattr(self, "_last_prediction_q", None) or 0.01
        r = getattr(self, "_last_prediction_r", None) or 0.1
        self.prediction_kf_x = Kalman1D(q, r)
        self.prediction_kf_y = Kalman1D(q, r)
        self._last_raw_velocity_x = 0.0
        self._last_raw_velocity_y = 0.0

    def reset_kalman_state(self):
        self.kf_x = Kalman1D(0.01, 0.1)
        self.kf_y = Kalman1D(0.01, 0.1)
        self.prev_kalman_time = None
        self.kalman_smoothing_initialized = False
        self.last_raw_kalman_x = 0.0
        self.last_raw_kalman_y = 0.0
        self.last_kx = 0.0
        self.last_ky = 0.0
        self._last_kalman_q = None
        self._last_kalman_r = None

    def reset_smoothing_state(self):
        self.move_overflow_x = 0.0
        self.move_overflow_y = 0.0
        self._smooth_frame = 0
        self._smooth_start_x = 0.0
        self._smooth_start_y = 0.0
        self._smooth_prev_x = 0.0
        self._smooth_prev_y = 0.0
        self._smooth_last_tx = 0.0
        self._smooth_last_ty = 0.0
        self._tracking_initialized = False
        self._track_x = 0.0
        self._track_y = 0.0
        self._track_prev_x = 0.0
        self._track_prev_y = 0.0
        self._track_time = None
        self._last_tracking_mode = None

    def mark_target_seen(self):
        self.last_target_time = time.perf_counter()
        self.target_detected = True

    def reset_if_stale(self, timeout_s=0.5):
        if not self.target_detected:
            return
        if time.perf_counter() - self.last_target_time > timeout_s:
            self.reset()

    def get_debug_snapshot(self):
        with self._debug_lock:
            pred = self._prediction_debug
            step = self._prediction_step
            future = list(self._future_positions)
        return pred, step, future

    def _set_prediction_debug(self, x, y):
        with self._debug_lock:
            self._prediction_debug = (x, y)

    def _set_prediction_step(self, x, y):
        with self._debug_lock:
            self._prediction_step = (x, y)

    def _set_future_positions(self, positions):
        with self._debug_lock:
            self._future_positions = positions

    def _ensure_prediction_kalman(self, q, r):
        q = max(float(q), 1e-6)
        r = max(float(r), 1e-6)
        if q != self._last_prediction_q or r != self._last_prediction_r:
            self.prediction_kf_x = Kalman1D(q, r)
            self.prediction_kf_y = Kalman1D(q, r)
            self._last_prediction_q = q
            self._last_prediction_r = r
            self.reset_prediction_state()

    def _ensure_kalman(self, q, r):
        q = max(float(q), 1e-6)
        r = max(float(r), 1e-6)
        if q != self._last_kalman_q or r != self._last_kalman_r:
            self.kf_x = Kalman1D(q, r)
            self.kf_y = Kalman1D(q, r)
            self._last_kalman_q = q
            self._last_kalman_r = r
            self.kalman_smoothing_initialized = False
            self.prev_kalman_time = None

    def _update_standard_velocity(self, target_x, target_y):
        now = time.perf_counter()
        if self.prev_time is None or not self.target_detected:
            self.prev_time = now
            self.prev_x = target_x
            self.prev_y = target_y
            self.prev_velocity_x = 0.0
            self.prev_velocity_y = 0.0
            return 0.0, 0.0
        dt = max(now - self.prev_time, 1e-8)
        vx = (target_x - self.prev_x) / dt
        vy = (target_y - self.prev_y) / dt
        self.prev_time = now
        self.prev_x = target_x
        self.prev_y = target_y
        self.prev_velocity_x = _clamp(vx, -20000.0, 20000.0)
        self.prev_velocity_y = _clamp(vy, -20000.0, 20000.0)
        return self.prev_velocity_x, self.prev_velocity_y

    def _update_prediction_state(self, pivot_x, pivot_y, settings):
        mode = int(settings["prediction"]["mode"])
        if mode != self._last_prediction_mode:
            self.reset_prediction_state()
            self._last_prediction_mode = mode

        now = time.perf_counter()
        max_gap = 0.25
        if self.prediction_prev_time is None:
            self.prediction_prev_time = now
            self.prediction_prev_x = pivot_x
            self.prediction_prev_y = pivot_y
            if mode != 0:
                self.prediction_kf_x.reset(pivot_x, 0.0)
                self.prediction_kf_y.reset(pivot_y, 0.0)
                self.prediction_kalman_time = now
                self.prediction_kalman_initialized = True
            return pivot_x, pivot_y

        dt = now - self.prediction_prev_time
        if dt > max_gap:
            self.reset_prediction_state()
            self.prediction_prev_time = now
            self.prediction_prev_x = pivot_x
            self.prediction_prev_y = pivot_y
            if mode != 0:
                self.prediction_kf_x.reset(pivot_x, 0.0)
                self.prediction_kf_y.reset(pivot_y, 0.0)
                self.prediction_kalman_time = now
                self.prediction_kalman_initialized = True
            return pivot_x, pivot_y

        dt = max(dt, 1e-8)
        raw_vx = (pivot_x - self.prediction_prev_x) / dt
        raw_vy = (pivot_y - self.prediction_prev_y) / dt
        raw_vx = _clamp(raw_vx, -20000.0, 20000.0)
        raw_vy = _clamp(raw_vy, -20000.0, 20000.0)
        self.prediction_prev_x = pivot_x
        self.prediction_prev_y = pivot_y
        self.prediction_prev_time = now
        self._last_raw_velocity_x = raw_vx
        self._last_raw_velocity_y = raw_vy

        base_vx = raw_vx
        base_vy = raw_vy
        filt_x = pivot_x
        filt_y = pivot_y

        if mode != 0:
            reset_threshold = float(settings.get("reset_threshold", 8.0))
            if (
                not self.prediction_kalman_initialized
                or math.hypot(pivot_x - self.prediction_kf_x.x, pivot_y - self.prediction_kf_y.x)
                > reset_threshold
            ):
                self.prediction_kf_x.reset(pivot_x, 0.0)
                self.prediction_kf_y.reset(pivot_y, 0.0)
                self.prediction_kalman_time = now
                self.prediction_kalman_initialized = True

            if self.prediction_kalman_time is None:
                kdt = dt
            else:
                kdt = max(now - self.prediction_kalman_time, 1e-8)
            self.prediction_kalman_time = now
            filt_x = self.prediction_kf_x.update(pivot_x, kdt)
            filt_y = self.prediction_kf_y.update(pivot_y, kdt)
            base_vx = self.prediction_kf_x.v
            base_vy = self.prediction_kf_y.v

        alpha = _clamp(float(settings["prediction"]["velocity_smoothing"]), 0.0, 1.0)
        if not self.prediction_initialized:
            self.prediction_smoothed_velocity_x = base_vx
            self.prediction_smoothed_velocity_y = base_vy
            self.prediction_initialized = True
        else:
            self.prediction_smoothed_velocity_x += (
                base_vx - self.prediction_smoothed_velocity_x
            ) * alpha
            self.prediction_smoothed_velocity_y += (
                base_vy - self.prediction_smoothed_velocity_y
            ) * alpha

        scale = max(float(settings["prediction"]["velocity_scale"]), 0.0)
        self.prediction_velocity_x = self.prediction_smoothed_velocity_x * scale
        self.prediction_velocity_y = self.prediction_smoothed_velocity_y * scale

        if mode == 0:
            return pivot_x, pivot_y

        lead_ms = max(float(settings["prediction"]["kalman_lead_ms"]), 0.0)
        max_lead_ms = max(float(settings["prediction"]["kalman_max_lead_ms"]), 0.0)
        if max_lead_ms > 0.0 and lead_ms > max_lead_ms:
            lead_ms = max_lead_ms
        lead_s = lead_ms / 1000.0
        return (
            filt_x + self.prediction_velocity_x * lead_s,
            filt_y + self.prediction_velocity_y * lead_s,
        )

    def _predict_future_positions(self, pivot_x, pivot_y, frames, fps):
        results = []
        if frames <= 0:
            return results
        frame_time = 1.0 / max(fps, 1.0)
        if self.prediction_prev_time is None:
            return results
        dt = time.perf_counter() - self.prediction_prev_time
        if dt > 0.5:
            return results
        vx = self.prediction_velocity_x if self.prediction_initialized else 0.0
        vy = self.prediction_velocity_y if self.prediction_initialized else 0.0
        for i in range(1, frames + 1):
            t = frame_time * i
            results.append((pivot_x + vx * t, pivot_y + vy * t))
        return results

    def _speed_multiplier(self, distance, max_distance, settings):
        min_speed = float(settings["speed"]["min_multiplier"])
        max_speed = float(settings["speed"]["max_multiplier"])
        snap_radius = float(settings["speed"]["snap_radius"])
        near_radius = float(settings["speed"]["near_radius"])
        curve = float(settings["speed"]["speed_curve_exponent"])
        snap_boost = float(settings["speed"]["snap_boost_factor"])

        if distance < snap_radius:
            return min_speed * snap_boost
        if distance < near_radius:
            t = distance / max(near_radius, 1e-6)
            curve_val = 1.0 - math.pow(1.0 - t, curve)
            return min_speed + (max_speed - min_speed) * curve_val
        norm = _clamp(distance / max(max_distance, 1e-6), 0.0, 1.0)
        return min_speed + (max_speed - min_speed) * norm

    def _calc_movement(self, target_x, target_y, center_x, center_y, screen_w, screen_h, fps, settings):
        offx = target_x - center_x
        offy = target_y - center_y
        dist = math.hypot(offx, offy)
        max_distance = math.hypot(screen_w, screen_h) / 2.0
        speed = self._speed_multiplier(dist, max_distance, settings)
        corr = 1.0
        if fps > 30.0:
            corr = 30.0 / fps
        move_x = offx * speed * corr
        move_y = offy * speed * corr
        self._set_prediction_step(center_x + move_x, center_y + move_y)
        return move_x, move_y

    def _add_overflow(self, dx, dy):
        frac_x, int_x = math.modf(dx + self.move_overflow_x)
        frac_y, int_y = math.modf(dy + self.move_overflow_y)
        if abs(frac_x) > 1.0:
            frac_x, extra = math.modf(frac_x)
            int_x += extra
        if abs(frac_y) > 1.0:
            frac_y, extra = math.modf(frac_y)
            int_y += extra
        self.move_overflow_x = frac_x
        self.move_overflow_y = frac_y
        return int(int_x), int(int_y)

    def _move_with_smoothing(self, target_x, target_y, center_x, center_y, fps, settings):
        smoothness = max(int(settings.get("smoothness", 1)), 1)
        tracking = bool(settings.get("tracking_smoothing", False))

        if self._last_tracking_mode is None or self._last_tracking_mode != tracking:
            self.reset_smoothing_state()
            self._last_tracking_mode = tracking

        if tracking:
            now = time.perf_counter()
            if self._track_time is None:
                dt = 1.0 / max(fps, 1.0)
            else:
                dt = max(now - self._track_time, 1e-8)
            dt = min(dt, 0.25)
            self._track_time = now
            if not self._tracking_initialized:
                self._track_x = center_x
                self._track_y = center_y
                self._track_prev_x = self._track_x
                self._track_prev_y = self._track_y
                self._tracking_initialized = True

            delta = math.hypot(target_x - self._track_x, target_y - self._track_y)
            reset_threshold = max(float(settings.get("reset_threshold", 8.0)), 1.0)
            base_alpha = 1.0 - math.exp(-dt * max(fps, 1.0) / smoothness)
            catchup = _clamp(delta / max(reset_threshold, 1e-3), 0.0, 1.0)
            alpha = _clamp(base_alpha + (0.65 - base_alpha) * catchup, base_alpha, 0.65)
            self._track_x += (target_x - self._track_x) * alpha
            self._track_y += (target_y - self._track_y) * alpha

            dx = self._track_x - self._track_prev_x
            dy = self._track_y - self._track_prev_y
            self._set_prediction_step(self._track_x, self._track_y)
            ix, iy = self._add_overflow(dx, dy)
            self._track_prev_x = self._track_x
            self._track_prev_y = self._track_y
            self._smooth_last_tx = target_x
            self._smooth_last_ty = target_y
            return ix, iy

        if self._smooth_frame == 0 or math.hypot(
            target_x - self._smooth_last_tx, target_y - self._smooth_last_ty
        ) > 1.0:
            self._smooth_start_x = center_x
            self._smooth_start_y = center_y
            self._smooth_prev_x = self._smooth_start_x
            self._smooth_prev_y = self._smooth_start_y
            self._smooth_frame = 0

        self._smooth_last_tx = target_x
        self._smooth_last_ty = target_y
        self._smooth_frame = min(self._smooth_frame + 1, smoothness)
        t = float(self._smooth_frame) / smoothness
        p = -0.5 * (math.cos(math.pi * t) - 1.0)
        cur_x = self._smooth_start_x + (target_x - self._smooth_start_x) * p
        cur_y = self._smooth_start_y + (target_y - self._smooth_start_y) * p
        dx = cur_x - self._smooth_prev_x
        dy = cur_y - self._smooth_prev_y
        self._set_prediction_step(cur_x, cur_y)
        ix, iy = self._add_overflow(dx, dy)
        self._smooth_prev_x = cur_x
        self._smooth_prev_y = cur_y
        return ix, iy

    def _move_with_kalman(self, target_x, target_y, center_x, center_y, screen_w, screen_h, fps, settings):
        reset_threshold = max(float(settings.get("reset_threshold", 8.0)), 1.0)
        if (
            self.prev_kalman_time is None
            or math.hypot(target_x - self.last_raw_kalman_x, target_y - self.last_raw_kalman_y)
            > reset_threshold
        ):
            self.kf_x.reset(target_x, 0.0)
            self.kf_y.reset(target_y, 0.0)
            self.prev_kalman_time = time.perf_counter()
        self.last_raw_kalman_x = target_x
        self.last_raw_kalman_y = target_y

        now = time.perf_counter()
        if self.prev_kalman_time is None:
            dt = 1.0 / max(fps, 1.0)
        else:
            dt = max(now - self.prev_kalman_time, 1e-8)
        self.prev_kalman_time = now

        filt_x = self.kf_x.update(target_x, dt)
        filt_y = self.kf_y.update(target_y, dt)
        mv_x, mv_y = self._calc_movement(
            filt_x, filt_y, center_x, center_y, screen_w, screen_h, fps, settings
        )
        mv_x *= float(settings.get("kalman_speed_multiplier_x", 1.0))
        mv_y *= float(settings.get("kalman_speed_multiplier_y", 1.0))
        return int(round(mv_x)), int(round(mv_y))

    def _move_with_kalman_and_smoothing(
        self, target_x, target_y, center_x, center_y, screen_w, screen_h, fps, settings
    ):
        reset_threshold = max(float(settings.get("reset_threshold", 8.0)), 1.0)
        now = time.perf_counter()
        max_dt = 0.25
        need_reset = not self.kalman_smoothing_initialized
        if not need_reset:
            jump = math.hypot(target_x - self.last_raw_kalman_x, target_y - self.last_raw_kalman_y)
            if jump > reset_threshold:
                need_reset = True
            if self.prev_kalman_time is not None:
                if now - self.prev_kalman_time > max_dt:
                    need_reset = True
        if need_reset or self.prev_kalman_time is None:
            self.kf_x.reset(target_x, 0.0)
            self.kf_y.reset(target_y, 0.0)
            self.prev_kalman_time = now
            self.last_kx = target_x
            self.last_ky = target_y
            self.move_overflow_x = 0.0
            self.move_overflow_y = 0.0
            self.kalman_smoothing_initialized = True

        self.last_raw_kalman_x = target_x
        self.last_raw_kalman_y = target_y
        if self.prev_kalman_time is None or need_reset:
            dt = 1.0 / max(fps, 1.0)
        else:
            dt = now - self.prev_kalman_time
        self.prev_kalman_time = now
        dt = _clamp(dt, 1e-8, max_dt)

        filt_x = self.kf_x.update(target_x, dt)
        filt_y = self.kf_y.update(target_y, dt)
        smoothness = max(int(settings.get("smoothness", 1)), 1)
        base_alpha = 1.0 - math.exp(-dt * max(fps, 1.0) / smoothness)
        delta = math.hypot(filt_x - self.last_kx, filt_y - self.last_ky)
        catchup = _clamp(delta / max(reset_threshold, 1e-3), 0.0, 1.0)
        max_alpha = 0.65 if bool(settings.get("tracking_smoothing", False)) else 0.45
        alpha = _clamp(base_alpha + (max_alpha - base_alpha) * catchup, base_alpha, max_alpha)
        if delta > reset_threshold:
            self.last_kx = filt_x
            self.last_ky = filt_y
        else:
            self.last_kx += (filt_x - self.last_kx) * alpha
            self.last_ky += (filt_y - self.last_ky) * alpha

        mv_x, mv_y = self._calc_movement(
            self.last_kx, self.last_ky, center_x, center_y, screen_w, screen_h, fps, settings
        )
        mv_x *= float(settings.get("kalman_speed_multiplier_x", 1.0))
        mv_y *= float(settings.get("kalman_speed_multiplier_y", 1.0))
        return self._add_overflow(mv_x, mv_y)

    def compute_move(
        self,
        target_x,
        target_y,
        center_x,
        center_y,
        screen_w,
        screen_h,
        fps,
        infer_latency_ms,
        settings,
    ):
        self.mark_target_seen()
        self._ensure_prediction_kalman(
            settings["prediction"]["kalman_process_noise"],
            settings["prediction"]["kalman_measurement_noise"],
        )
        self._ensure_kalman(
            settings.get("kalman_process_noise", 0.01),
            settings.get("kalman_measurement_noise", 0.1),
        )
        pred_x, pred_y = self._update_prediction_state(target_x, target_y, settings)
        mode = int(settings["prediction"]["mode"])
        interval = max(float(settings["prediction"]["interval"]), 0.0)
        if mode == 0:
            vx, vy = self._update_standard_velocity(target_x, target_y)
            latency_s = max(float(infer_latency_ms or 0.0), 0.0) / 1000.0
            pred_x = target_x + vx * (interval + latency_s)
            pred_y = target_y + vy * (interval + latency_s)
        elif mode == 2:
            pred_x += self._last_raw_velocity_x * interval
            pred_y += self._last_raw_velocity_y * interval

        latency_s = max(float(infer_latency_ms or 0.0), 0.0) / 1000.0
        extra_lead_s = latency_s
        if mode == 1:
            extra_lead_s += interval
        if mode in (1, 2) and extra_lead_s > 0.0:
            pred_x += self.prediction_velocity_x * extra_lead_s
            pred_y += self.prediction_velocity_y * extra_lead_s

        self._set_prediction_debug(pred_x, pred_y)
        use_future_for_aim = bool(settings["prediction"].get("use_future_for_aim", False))
        need_future = use_future_for_aim or bool(settings["prediction"].get("draw_future_positions", False)) or bool(
            settings.get("debug", {}).get("show_future", False)
        )
        if need_future:
            future = self._predict_future_positions(
                pred_x,
                pred_y,
                int(settings["prediction"]["future_positions"]),
                fps,
            )
            if use_future_for_aim and future:
                pred_x, pred_y = future[-1]
            self._set_future_positions(future)
        else:
            self._set_future_positions([])

        use_kalman = bool(settings.get("use_kalman", False))
        use_smoothing = bool(settings.get("use_smoothing", False))
        if use_kalman and use_smoothing:
            dx, dy = self._move_with_kalman_and_smoothing(
                pred_x, pred_y, center_x, center_y, screen_w, screen_h, fps, settings
            )
        elif use_kalman:
            dx, dy = self._move_with_kalman(
                pred_x, pred_y, center_x, center_y, screen_w, screen_h, fps, settings
            )
        elif use_smoothing:
            dx, dy = self._move_with_smoothing(
                pred_x, pred_y, center_x, center_y, fps, settings
            )
        else:
            mv_x, mv_y = self._calc_movement(
                pred_x, pred_y, center_x, center_y, screen_w, screen_h, fps, settings
            )
            dx = int(round(mv_x))
            dy = int(round(mv_y))
        return dx, dy
