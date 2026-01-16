import time
import math
from collections import deque


class PID:
    """Position-based PID controller class"""

    def __init__(self, kp, ki, kd, output_limits=None):
        """
        Initialize PID parameters
        :param kp: Proportional coefficient
        :param ki: Integral coefficient
        :param kd: Derivative coefficient
        :param output_limits: Output limits, tuple (min, max)
        """
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.setpoint = 0
        self.actual_value = 0
        self.error = 0
        self.last_error = 0
        self.integral = 0
        self.output = 0
        self.output_limits = output_limits
        self.sample_time = 0
        self.last_time = time.time()

    def compute(self, actual_value):
        """
        Calculate PID output
        :param actual_value: Current process measurement value
        :return: Controller output
        """
        current_time = time.time()
        delta_time = current_time - self.last_time
        if delta_time < self.sample_time:
            return self.output
        self.actual_value = actual_value
        self.error = self.setpoint - self.actual_value
        self.integral += self.error * delta_time
        derivative = (self.error - self.last_error) / delta_time if delta_time > 0 else 0
        self.output = self.Kp * self.error + self.Ki * self.integral + self.Kd * derivative
        if self.output_limits is not None:
            min_output, max_output = self.output_limits
            self.output = max(min_output, min(self.output, max_output))
            if self.output == min_output or self.output == max_output:
                self.integral -= self.error * delta_time
        self.last_error = self.error
        self.last_time = current_time
        return self.output

    def set_sample_time(self, sample_time):
        """Set sample time"""
        self.sample_time = sample_time

    def set_setpoint(self, setpoint):
        """Set target value"""
        self.setpoint = setpoint

    def reset(self):
        """Reset PID controller state"""
        self.error = 0
        self.last_error = 0
        self.integral = 0
        self.output = 0
        self.last_time = time.time()


class DualAxisPID:
    """Dual-axis PID controller class, supports independent X and Y axis control with smoothing"""
    pass
    pass
    pass

    def __init__(self, kp, ki, kd, windup_guard, smooth_params=None, output_limits_x=None, output_limits_y=None,
                 anti_windup_mode='freeze', backcalc_gain=None):
        """
        Initialize dual-axis PID parameters
        :param kp: Proportional coefficient [kp_x, kp_y]
        :param ki: Integral coefficient [ki_x, ki_y]
        :param kd: Derivative coefficient [kd_x, kd_y]
        :param windup_guard: Integral windup limit [windup_x, windup_y]
        :param smooth_params: Smoothing parameters [smooth_x, smooth_y, smooth_deadzone]
        """
        self.kp = {'x': kp[0], 'y': kp[1]}
        self.ki = {'x': ki[0], 'y': ki[1]}
        self.kd = {'x': kd[0], 'y': kd[1]}
        self.windup_guard = {'x': windup_guard[0], 'y': windup_guard[1]}
        self.output_limits = {'x': output_limits_x, 'y': output_limits_y}
        self.anti_windup_mode = anti_windup_mode
        if backcalc_gain is None:
            backcalc_gain = [0.0, 0.0]
        elif not isinstance(backcalc_gain, (list, tuple)):
            backcalc_gain = [backcalc_gain, backcalc_gain]
        self.backcalc_gain = {'x': backcalc_gain[0], 'y': backcalc_gain[1]}
        if smooth_params is None:
            smooth_params = [0.0, 0.0, 0.0, 1.0]
        self.smooth_x = smooth_params[0]
        self.smooth_y = smooth_params[1]
        self.smooth_deadzone = smooth_params[2]
        self.smooth_algorithm = smooth_params[3] if len(smooth_params) > 3 else 1.0
        self._smooth_history_x = []
        self._smooth_history_y = []
        self.history_size = 20
        self.error_history = deque(maxlen=self.history_size)
        self.time_history = deque(maxlen=self.history_size)
        self.uniform_threshold = 1.5
        self.min_velocity_threshold = 10.0
        self.max_velocity_threshold = 100.0
        self.compensation_factor = 2.0
        self.reset()

    def reset(self):
        """Reset controller state"""
        self._last_time = time.time()
        self._last_error = {'x': 0, 'y': 0}
        self._p_term = {'x': 0, 'y': 0}
        self._i_term = {'x': 0, 'y': 0}
        self._d_term = {'x': 0, 'y': 0}
        self._i_max = {'x': self.windup_guard['x'], 'y': self.windup_guard['y']}
        self._i_min = {'x': -self.windup_guard['x'], 'y': -self.windup_guard['y']}
        self._last_integral_increment = {'x': 0, 'y': 0}
        self._smooth_history_x = []
        self._smooth_history_y = []
        self.error_history = deque(maxlen=self.history_size)
        self.time_history = deque(maxlen=self.history_size)

    def _calculate_output(self, axis, error, delta_time):
        """
        Calculate PID output for specified axis

        Args:
            axis (str): Axis name ('x' or 'y')
            error (float): Error value
            delta_time (float): Time difference

        Returns:
            float: PID output value
        """
        self._p_term[axis] = self.kp[axis] * error
        integral_increment = self.ki[axis] * error * delta_time
        self._i_term[axis] += integral_increment
        self._last_integral_increment[axis] = integral_increment
        if self.windup_guard[axis] > 0:
            if self._i_term[axis] > self._i_max[axis]:
                self._i_term[axis] = self._i_max[axis]
            elif self._i_term[axis] < self._i_min[axis]:
                self._i_term[axis] = self._i_min[axis]
        if delta_time > 0:
            self._d_term[axis] = self.kd[axis] * ((error - self._last_error[axis]) / delta_time)
        else:
            self._d_term[axis] = 0
        return self._p_term[axis] + self._i_term[axis] + self._d_term[axis]

    def _apply_smoothing(self, x_output, y_output, error_x, error_y):
        """
        Apply old controller style smoothing:
        - Maintain error/time history to determine if motion is uniform
        - For uniform motion: output = step * compensation_factor (skip smoothing)
        - For non-uniform motion: final = s*step + (1-s)*(error_delta)
          where step is the raw PID output for this cycle, error_delta = current error - previous error
        """
        error_distance = (error_x ** 2 + error_y ** 2) ** 0.5
        if error_distance <= self.smooth_deadzone:
            return (x_output, y_output)
        current_time = time.time()
        self.error_history.append((error_x, error_y))
        self.time_history.append(current_time)

        def is_uniform_motion():
            if len(self.error_history) < min(3, self.history_size):
                return False
            velocities = []
            for i in range(1, len(self.error_history)):
                dt = self.time_history[i] - self.time_history[i - 1]
                if dt <= 0:
                    continue
                dx = self.error_history[i][0] - self.error_history[i - 1][0]
                dy = self.error_history[i][1] - self.error_history[i - 1][1]
                v = math.sqrt(dx / dt * (dx / dt) + dy / dt * (dy / dt))
                velocities.append(v)
            if not velocities:
                return False
            avg_v = sum(velocities) / len(velocities)
            if not self.min_velocity_threshold <= avg_v <= self.max_velocity_threshold:
                return False
            var_v = sum(((v - avg_v) * (v - avg_v) for v in velocities)) / len(velocities)
            return var_v < self.uniform_threshold

        step_x, step_y = (x_output, y_output)
        if is_uniform_motion():
            comp = 1.0
            try:
                comp = float(self.smooth_algorithm)
            except:
                comp = 1.0
            comp = max(1.0, comp)
            final_x = step_x * comp
            final_y = step_y * comp
            return (final_x, final_y)
        if len(self.error_history) >= 2:
            last_error = self.error_history[-2]
            dx_err = error_x - last_error[0]
            dy_err = error_y - last_error[1]
        else:
            dx_err = 0.0
            dy_err = 0.0
        try:
            s_x = float(self.smooth_x)
        except:
            s_x = 0.0
        try:
            s_y = float(self.smooth_y)
        except:
            s_y = 0.0
        s_x = max(0.0, min(1.0, s_x))
        s_y = max(0.0, min(1.0, s_y))
        final_x = s_x * step_x + (1.0 - s_x) * dx_err
        final_y = s_y * step_y + (1.0 - s_y) * dy_err
        return (final_x, final_y)

    def compute(self, error_x, error_y):
        """
        Calculate dual-axis PID output

        Args:
            error_x (float): X-axis error value
            error_y (float): Y-axis error value

        Returns:
            tuple: (x_output, y_output) Control output for X and Y axes
        """
        current_time = time.time()
        delta_time = current_time - self._last_time
        x_output_unsat = self._calculate_output('x', error_x, delta_time)
        y_output_unsat = self._calculate_output('y', error_y, delta_time)

        def _apply_limits_and_anti_windup(axis, unsat_value):
            value = unsat_value
            saturated = False
            limits = self.output_limits.get(axis)
            if limits is not None:
                min_out, max_out = limits
                if value > max_out:
                    value = max_out
                    saturated = True
                elif value < min_out:
                    value = min_out
                    saturated = True
            if saturated:
                if self.anti_windup_mode == 'backcalc':
                    self._i_term[axis] += self.backcalc_gain[axis] * (value - unsat_value)
                else:
                    self._i_term[axis] -= self._last_integral_increment[axis]
                if self.windup_guard[axis] > 0:
                    if self._i_term[axis] > self._i_max[axis]:
                        self._i_term[axis] = self._i_max[axis]
                        return value
                    if self._i_term[axis] < self._i_min[axis]:
                        self._i_term[axis] = self._i_min[axis]
            return value

        x_output = _apply_limits_and_anti_windup('x', x_output_unsat)
        y_output = _apply_limits_and_anti_windup('y', y_output_unsat)
        x_output, y_output = self._apply_smoothing(x_output, y_output, error_x, error_y)
        error_magnitude = (error_x ** 2 + error_y ** 2) ** 0.5
        if error_magnitude < 5.0:
            deadzone_factor = max(0.1, error_magnitude / 5.0)
            x_output *= deadzone_factor
            y_output *= deadzone_factor

        def _final_clamp(axis, value):
            limits = self.output_limits.get(axis)
            if limits is None:
                return value
            min_out, max_out = limits
            if value > max_out:
                return max_out
            if value < min_out:
                return min_out
            return value

        x_output = _final_clamp('x', x_output)
        y_output = _final_clamp('y', y_output)
        self._last_error['x'] = error_x
        self._last_error['y'] = error_y
        self._last_time = current_time
        return (x_output, y_output)

    def set_windup_guard(self, windup_guard):
        """
        Set integral windup limit

        Args:
            windup_guard: Integral windup limit [windup_x, windup_y]
        """
        self.windup_guard = {'x': windup_guard[0], 'y': windup_guard[1]}
        self._i_max = {'x': self.windup_guard['x'], 'y': self.windup_guard['y']}
        self._i_min = {'x': -self.windup_guard['x'], 'y': -self.windup_guard['y']}

    def set_pid_params(self, kp=None, ki=None, kd=None):
        """
        Dynamically set PID parameters

        Args:
            kp: Proportional coefficient [kp_x, kp_y] (optional)
            ki: Integral coefficient [ki_x, ki_y] (optional)
            kd: Derivative coefficient [kd_x, kd_y] (optional)
        """
        if kp is not None:
            self.kp = {'x': kp[0], 'y': kp[1]}
        if ki is not None:
            self.ki = {'x': ki[0], 'y': ki[1]}
        if kd is not None:
            self.kd = {'x': kd[0], 'y': kd[1]}

    def set_output_limits(self, x_limits=None, y_limits=None):
        """Set output limits for each axis, format (min, max) or None"""
        if x_limits is not None:
            self.output_limits['x'] = x_limits
        if y_limits is not None:
            self.output_limits['y'] = y_limits

    def set_anti_windup(self, mode='freeze', backcalc_gain=None):
        """Set anti-windup strategy and back-calculation gain"""
        self.anti_windup_mode = mode
        if backcalc_gain is not None:
            if not isinstance(backcalc_gain, (list, tuple)):
                backcalc_gain = [backcalc_gain, backcalc_gain]
            self.backcalc_gain = {'x': backcalc_gain[0], 'y': backcalc_gain[1]}

    def set_smooth_params(self, smooth_x=None, smooth_y=None, smooth_deadzone=None, smooth_algorithm=None):
        """
        Set smoothing parameters

        Args:
            smooth_x: X-axis smoothing value (optional)
            smooth_y: Y-axis smoothing value (optional)
            smooth_deadzone: Smoothing deadzone radius (optional)
            smooth_algorithm: Smoothing algorithm strength (optional)
        """
        if smooth_x is not None:
            self.smooth_x = smooth_x
        if smooth_y is not None:
            self.smooth_y = smooth_y
        if smooth_deadzone is not None:
            self.smooth_deadzone = smooth_deadzone
        if smooth_algorithm is not None:
            self.smooth_algorithm = smooth_algorithm

    def get_components(self):
        """
        Get current PID component values

        Returns:
            dict: Dictionary containing P, I, D components for each axis
        """
        return {'x': {'p': self._p_term['x'], 'i': self._i_term['x'], 'd': self._d_term['x']},
                'y': {'p': self._p_term['y'], 'i': self._i_term['y'], 'd': self._d_term['y']}}


if __name__ == '__main__':
    print('=== Single-axis PID Controller Example ===')
    pid = PID(kp=1.0, ki=0.1, kd=0.05, output_limits=(-100, 100))
    pid.set_setpoint(50)
    actual_value = 0
    for i in range(10):
        output = pid.compute(actual_value)
        actual_value += output * 0.01
        print(f'Step {i + 1}: Target={pid.setpoint}, Current={actual_value:.2f}, Output={output:.2f}')
    print('\n=== Dual-axis PID Controller Example ===')
    dual_pid = DualAxisPID(kp=[0.8, 0.6], ki=[0.01, 0.008], kd=[0.05, 0.06], windup_guard=[50, 50],
                           smooth_params=[0.0, 0.0, 2.0, 1.0])
    target_x, target_y = (100, 80)
    current_x, current_y = (0, 0)
    for i in range(10):
        error_x = target_x - current_x
        error_y = target_y - current_y
        output_x, output_y = dual_pid.compute(error_x, error_y)
        current_x += output_x * 0.01 - current_x * 0.01
        current_y += output_y * 0.01 - current_y * 0.01
        print(
            f'Step {i + 1}: X-axis({current_x:.2f}/{target_x}), Y-axis({current_y:.2f}/{target_y}), Output X={output_x:.2f}, Output Y={output_y:.2f}')
    components = dual_pid.get_components()
    print('\nFinal PID Components:')
    print(f"X-axis - P:{components['x']['p']:.3f}, I:{components['x']['i']:.3f}, D:{components['x']['d']:.3f}")
    print(f"Y-axis - P:{components['y']['p']:.3f}, I:{components['y']['i']:.3f}, D:{components['y']['d']:.3f}")