# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: screenshot_manager.py
# Bytecode version: 3.10.0rc2 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

"""
截图分离管理模块

该模块负责管理不同截图源的初始化、获取和显示，包括BetterCam直接截屏、OBS截图和采集卡截图。
提供统一的接口，使得调用方不需要关心具体的截图源实现细节。

支持两种模式:
- 分离模式: 多线程截图分离架构，内存池复用，智能缓存策略
- 简单模式: 传统同步截图模式，保持向后兼容
"""


def draw_move_deadzone_visualization(screenshot, center_x, center_y, move_deadzone):
    """绘制移动死区可视化"""
    if move_deadzone <= 0:
        return
    deadzone_radius = int(move_deadzone)
    cv2.circle(screenshot, (center_x, center_y), deadzone_radius, (0, 0, 255), 2)


def draw_smooth_deadzone_visualization(screenshot, center_x, center_y, smooth_deadzone):
    """绘制平滑禁区可视化"""
    if smooth_deadzone <= 0:
        return
    smooth_deadzone_radius = int(smooth_deadzone)
    cv2.circle(screenshot, (center_x, center_y), smooth_deadzone_radius, (255, 0, 0), 2)


import time
import traceback
import queue
import threading
import gc
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Optional, Tuple, Dict
import cv2
import numpy as np

try:
    import bettercam

    bettercam_available = True
except Exception as e:
    bettercam_available = False
    with open('error_log.txt', 'a', encoding='utf-8') as f:
        f.write('[BetterCam加载失败] ' + str(e) + '\n')
    print('BetterCam 截屏不可用: ', e)
GPU_AVAILABLE = False


class MemoryPool:
    """内存池管理器，减少内存分配开销"""

    def __init__(self, pool_size: int = 30, frame_shape: Tuple[int, int, int] = (640, 640, 3)):
        self.pool_size = pool_size
        self.frame_shape = frame_shape
        self.available_frames = queue.Queue(maxsize=pool_size)
        self.allocated_frame_ids = set()
        self._initialize_pool()

    def _initialize_pool(self):
        """初始化内存池"""
        for _ in range(self.pool_size):
            frame = np.empty(self.frame_shape, dtype=np.uint8)
            self.available_frames.put(frame)

    def get_frame(self) -> np.ndarray:
        """从内存池获取帧"""
        try:
            frame = self.available_frames.get_nowait()
            self.allocated_frame_ids.add(id(frame))
            return frame
        except queue.Empty:
            frame = np.empty(self.frame_shape, dtype=np.uint8)
            self.allocated_frame_ids.add(id(frame))
            return frame

    def return_frame(self, frame: np.ndarray):
        """归还帧到内存池"""
        try:
            frame_id = id(frame)
            if frame_id in self.allocated_frame_ids:
                self.allocated_frame_ids.discard(frame_id)
                if frame.shape == self.frame_shape and frame.dtype == np.uint8:
                    self.available_frames.put_nowait(frame)
        except queue.Full:
            return None

    def resize_pool(self, new_shape: Tuple[int, int, int]):
        """调整内存池大小"""
        if new_shape != self.frame_shape:
            self.frame_shape = new_shape
            while not self.available_frames.empty():
                try:
                    self.available_frames.get_nowait()
                except queue.Empty:
                    break
            self._initialize_pool()


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.capture_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.last_report_time = time.perf_counter()

    def record_frame_time(self, duration: float):
        self.frame_times.append(duration)

    def record_capture_time(self, duration: float):
        self.capture_times.append(duration)

    def record_processing_time(self, duration: float):
        self.processing_times.append(duration)

    def get_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        stats = {}
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            stats['avg_frame_time'] = avg_frame_time * 1000
            stats['fps'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        if self.capture_times:
            stats['avg_capture_time'] = sum(self.capture_times) / len(self.capture_times) * 1000
        if self.processing_times:
            stats['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times) * 1000
        return stats

    def should_report(self, interval: float = 5.0) -> bool:
        """是否应该报告性能"""
        current_time = time.perf_counter()
        if current_time - self.last_report_time >= interval:
            self.last_report_time = current_time
            return True
        return False

    def get_performance_stats(self) -> Dict[str, float]:
        """获取详细性能统计"""
        return self.get_stats()

    def optimize_performance(self):
        """根据性能数据优化设置"""
        stats = self.get_stats()
        current_fps = stats.get('fps', 0)
        if current_fps > 120:
            return 'high_performance'
        if current_fps > 60:
            return 'balanced'
        return 'power_saving'


    def print_performance_report(self):
        """打印性能报告"""
        stats = self.get_stats()
        if not stats:
            return
        print('\n=== 截图分离性能报告 ===')
        if 'fps' in stats:
            print(f"平均FPS: {stats['fps']:.1f}")
        if 'avg_frame_time' in stats:
            print(f"平均帧时间: {stats['avg_frame_time']:.2f}ms")
        if 'avg_capture_time' in stats:
            print(f"平均捕获时间: {stats['avg_capture_time']:.2f}ms")
        if 'avg_processing_time' in stats:
            print(f"平均处理时间: {stats['avg_processing_time']:.2f}ms")
        print(
            f"运行模式: {('多线程分离' if hasattr(self, 'enable_parallel_processing') and self.enable_parallel_processing else '单线程')}")

    def set_performance_level(self, level: str):
        """设置性能级别"""
        performance_configs = {'high_performance': {'max_workers': 6, 'queue_size': 2, 'cache_duration': 0.0},
                               'balanced': {'max_workers': 4, 'queue_size': 3, 'cache_duration': 0.001},
                               'power_saving': {'max_workers': 2, 'queue_size': 5, 'cache_duration': 0.002}}
        config = performance_configs.get(level, performance_configs['balanced'])
        return config


def check_performance_requirements():
    """检查性能要求"""
    import psutil
    import platform
    try:
        cpu_count = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory().total / 1073741824
        system = platform.system()
        if cpu_count >= 4 and memory >= 8:
            return True
        return False
    except Exception:
        return True


class ScreenshotManager:
    """
    截图分离管理器

    管理不同截图源的初始化、获取和显示，提供统一的接口。
    支持的截图源：
    - BetterCam (直接截屏)
    - OBS (通过OBS WebSocket API)
    - CJK (采集卡)

    支持两种模式:
    - 分离模式: 多线程截图分离架构，内存池复用，智能缓存策略，性能监控，GPU加速（可选）
    - 简单模式: 传统同步截图模式，保持向后兼容
    """

    def __init__(self, config, engine=None):
        """
        初始化截图管理器

        Args:
            config: 配置字典
            engine: 推理引擎实例，用于获取输入尺寸等信息
        """
        self.config = config
        self.engine = engine
        self.screen_width = self.config.get('screen_width', 1920)
        self.screen_height = self.config.get('screen_height', 1080)
        self.bettercam_capture = None
        self.obs = None
        self.cjk_device = None
        self.enable_parallel_processing = config.get('enable_parallel_processing', True)
        if self.enable_parallel_processing:
            self.performance_monitor = PerformanceMonitor()
            self.memory_pool = None
            self.executor = None
            self.running = True
            self.capture_thread = None
            self.processing_threads = []
            self.display_thread = None
            self.raw_screenshot_queue = Queue(maxsize=1)
            self.processed_screenshot_queue = Queue(maxsize=1)
            self.screenshot_queue = Queue(maxsize=1)
            self.enable_memory_pool = config.get('enable_memory_pool', False)
            self.max_workers = config.get('max_workers', 1)
            self._cached_screenshot = None
            self._last_screenshot_time = 0
            self._screenshot_cache_duration = 0.001
            self._region_cache = {}
            self.adaptive_cache = config.get('adaptive_cache', False)
            self.min_cache_duration = 0.0
            self.max_cache_duration = 0.005
            self.turbo_mode = config.get('turbo_mode', True)
            self.skip_frame_processing = config.get('skip_frame_processing', True)
            self.frame_reuse_stats = {'total_requests': 0, 'new_frames': 0, 'reused_frames': 0, 'reuse_rate': 0.0}
            self._obs_latest_frame = None
            self._bettercam_latest_frame = None
            self._cjk_latest_frame = None
            self._init_high_performance_components()
        else:
            self.running = True
            self.screenshot_queue = Queue(maxsize=1)
            self.display_thread = None
            self._cached_screenshot = None
            self._last_screenshot_time = 0
            self._screenshot_cache_duration = 0.002
            self._region_cache = {}

    def _init_high_performance_components(self):
        """初始化高性能组件（仅分离模式）"""
        try:
            if self.enable_memory_pool and self.engine:
                try:
                    input_shape = self.engine.get_input_shape()
                    frame_shape = (input_shape[2], input_shape[3], 3)
                    self.memory_pool = MemoryPool(frame_shape=frame_shape)
                except Exception as e:
                    self.memory_pool = None
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        except Exception as e:
            return None

    def _start_capture_pipeline(self):
        """启动截图处理管道（仅分离模式）"""
        if not self.enable_parallel_processing:
            return
        try:
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            for i in range(self.max_workers):
                thread = threading.Thread(target=self._processing_loop, daemon=True)
                thread.start()
                self.processing_threads.append(thread)
        except Exception as e:
            print(f'启动截图管道失败: {e}')

    def _capture_loop(self):
        """截图捕获循环 - 专门负责获取原始截图（仅分离模式）"""
        while self.running:
            try:
                if self.turbo_mode:
                    # Turbo：取一帧→丢到队列（覆盖旧帧），不做计时统计
                    raw_screenshot = self._get_raw_screenshot()
                    if raw_screenshot is None:
                        continue
                    while not self.raw_screenshot_queue.empty():
                        try:
                            self.raw_screenshot_queue.get_nowait()
                        except queue.Empty:
                            break
                    try:
                        self.raw_screenshot_queue.put_nowait((raw_screenshot, 0))
                    except queue.Full:
                        pass
                    continue
                else:
                    # 非 Turbo：计时 + 投递
                    start_time = time.perf_counter()
                    raw_screenshot = self._get_raw_screenshot()
                    if raw_screenshot is None:
                        time.sleep(0.001)
                        continue
                    try:
                        self.raw_screenshot_queue.put_nowait((raw_screenshot, start_time))
                    except queue.Full:
                        try:
                            self.raw_screenshot_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.raw_screenshot_queue.put_nowait((raw_screenshot, start_time))
                        except queue.Full:
                            pass
                    capture_time = time.perf_counter() - start_time
                    self.performance_monitor.record_capture_time(capture_time)
            except Exception as e:
                print(f'截图捕获异常: {e}')
                if not self.turbo_mode:
                    time.sleep(0.001)

    def _processing_loop(self):
        """图像处理循环 - 专门负责处理截图（仅分离模式）- 强制提速版本"""
        while self.running:
            try:
                raw_screenshot, capture_start_time = self.raw_screenshot_queue.get(timeout=1)
                if self.turbo_mode:
                    if self.skip_frame_processing:
                        processed_screenshot = raw_screenshot
                    else:
                        processed_screenshot = self._process_screenshot(raw_screenshot)
                    if processed_screenshot is not None:
                        while not self.processed_screenshot_queue.empty():
                            try:
                                self.processed_screenshot_queue.get_nowait()
                            except queue.Empty:
                                break
                        try:
                            self.processed_screenshot_queue.put_nowait((processed_screenshot, 0))
                        except queue.Full:
                            pass
                else:
                    start_time = time.perf_counter()
                    processed_screenshot = self._process_screenshot(raw_screenshot)
                    if processed_screenshot is not None:
                        try:
                            total_time = time.perf_counter() - capture_start_time if capture_start_time > 0 else 0
                            self.processed_screenshot_queue.put_nowait((processed_screenshot, total_time))
                        except queue.Full:
                            try:
                                self.processed_screenshot_queue.get_nowait()
                                self.processed_screenshot_queue.put_nowait((processed_screenshot, total_time))
                            except queue.Empty:
                                pass
                    if capture_start_time > 0:
                        processing_time = time.perf_counter() - start_time
                        self.performance_monitor.record_processing_time(processing_time)
            except queue.Empty:
                continue
            except Exception as e:
                print(f'图像处理异常: {e}')

    def _get_raw_screenshot(self) -> Optional[np.ndarray]:
        """获取原始截图 - 统一接口，支持帧复用"""
        try:
            if self.config.get('is_cjk', False) and self.cjk_device:
                return self._get_cjk_frame_separated()
            if self.config.get('is_obs', False) and self.obs:
                return self._get_obs_frame_non_blocking()
            if self.bettercam_capture:
                return self.bettercam_capture.get_latest_frame()
        except Exception as e:
            return None

    def _get_obs_frame_non_blocking(self) -> Optional[np.ndarray]:
        """非阻塞获取OBS帧，支持帧复用以突破FPS限制（仅分离模式）"""
        try:
            try:
                new_frame = self.obs.frame_queue.get_nowait()
                self._obs_latest_frame = new_frame
                if hasattr(self, '_update_frame_stats'):
                    self._update_frame_stats(is_new_frame=True)
                return self._obs_latest_frame
            except queue.Empty:
                if hasattr(self, '_obs_latest_frame'):
                    if self._obs_latest_frame is not None:
                        if hasattr(self, '_update_frame_stats'):
                            self._update_frame_stats(is_new_frame=False)
                        return self._obs_latest_frame.copy()
                try:
                    frame = self.obs.frame_queue.get(timeout=0.1)
                    self._obs_latest_frame = frame
                    if hasattr(self, '_update_frame_stats'):
                        self._update_frame_stats(is_new_frame=True)
                    return frame
                except queue.Empty:
                    return
        except Exception as e:
            print(f'OBS帧获取异常: {e}')

    def _get_cjk_frame_separated(self) -> Optional[np.ndarray]:
        """采集卡真正分离架构：支持推理线程无限制FPS（参考OBS架构）"""
        try:
            frame = self.cjk_device.get_frame_non_blocking()
            if frame is not None:
                if hasattr(self, '_update_frame_stats'):
                    self._update_frame_stats(is_new_frame=True)
                return frame
            if hasattr(self, '_cjk_latest_frame') and self._cjk_latest_frame is not None:
                if hasattr(self, '_update_frame_stats'):
                    self._update_frame_stats(is_new_frame=False)
                if getattr(self, 'turbo_mode', False):
                    return self._cjk_latest_frame
                return self._cjk_latest_frame.copy()
            return None
        except Exception as e:
            print(f'采集卡分离架构帧获取异常: {e}')
            return None

    def _update_frame_stats(self, is_new_frame: bool):
        """更新帧复用统计（仅分离模式）"""
        self.frame_reuse_stats['total_requests'] += 1
        if is_new_frame:
            self.frame_reuse_stats['new_frames'] += 1
        else:
            self.frame_reuse_stats['reused_frames'] += 1
        total = self.frame_reuse_stats['total_requests']
        if total > 0:
            self.frame_reuse_stats['reuse_rate'] = self.frame_reuse_stats['reused_frames'] / total * 100

    def _process_screenshot(self, raw_screenshot: np.ndarray) -> Optional[np.ndarray]:
        """处理截图 - 支持CPU/GPU加速（仅分离模式）"""
        try:
            if raw_screenshot is None:
                return
            return self._process_screenshot_internal(raw_screenshot)
        except Exception as e:
            print(f'截图处理失败: {e}')
            return None

    def _process_screenshot_internal(self, screenshot: np.ndarray) -> np.ndarray:
        """截图处理内部实现"""
        return screenshot

    def init_sources(self):
        """
        根据配置初始化所有启用的截图源并启动处理管道
        """
        success = False
        if self.config.get('is_cjk', False):
            success = self.init_cjk_device()
        elif self.config.get('is_obs', False):
            success = self.init_obs()
        else:
            success = self.init_bettercam()
        if success:
            if self.enable_parallel_processing:
                self._start_capture_pipeline()
                turbo_status = '强制提速' if getattr(self, 'turbo_mode', False) else '标准'
            elif False:
                pass
            if self.config.get('infer_debug', False):
                if self.enable_parallel_processing:
                    self.display_thread = Thread(target=self.display_screenshot_separated)
                else:
                    self.display_thread = Thread(target=self.display_screenshot_simple)
                self.display_thread.daemon = True
                self.display_thread.start()
        return success

    def init_bettercam(self):
        """
        初始化BetterCam截图

        Returns:
            bool: 初始化是否成功
        """
        try:
            if not bettercam_available:
                pass
                self.bettercam_capture = None
                return False
            if self.bettercam_capture is not None:
                self.bettercam_capture.stop()
                del self.bettercam_capture
                self.bettercam_capture = None
            if self.engine is None:
                return False
            width = self.engine.get_input_shape()[3]
            height = self.engine.get_input_shape()[2]
            if width <= 0 or height <= 0:
                print(f'BetterCam初始化失败: 无效的模型输入尺寸 {width}x{height}')
                return False
            if width > self.screen_width or height > self.screen_height:
                print(
                    f'BetterCam初始化失败: 模型输入尺寸 {width}x{height} 超过屏幕尺寸 {self.screen_width}x{self.screen_height}')
                return False
            offset_x = int(self.config.get("capture_offset_x", 0))
            offset_y = int(self.config.get("capture_offset_y", 0))
            left = (self.screen_width - width) // 2 + offset_x
            top = (self.screen_height - height) // 2 + offset_y
            right = left + width
            bottom = top + height
            left = max(0, left)
            top = max(0, top)
            right = min(self.screen_width, right)
            bottom = min(self.screen_height, bottom)
            actual_width = right - left
            actual_height = bottom - top
            if actual_width <= 0 or actual_height <= 0:
                print(f'BetterCam初始化失败: 计算得到的区域无效 ({left}, {top}, {right}, {bottom})')
                return False
            region = (left, top, right, bottom)
            print(f'BetterCam区域: {region}, 屏幕尺寸: {self.screen_width}x{self.screen_height}')
            try:
                buffer_len = 4 if self.enable_parallel_processing else 16
                self.bettercam_capture = bettercam.create(output_color='BGR', max_buffer_len=buffer_len, region=region)
                self.bettercam_capture.start(target_fps=0, video_mode=True)
                return True
            except Exception as e:
                if hasattr(self, 'bettercam_capture') and self.bettercam_capture is not None:
                    try:
                        if hasattr(self.bettercam_capture, 'stop'):
                            self.bettercam_capture.stop()
                        del self.bettercam_capture
                    except:
                        pass
                self.bettercam_capture = None
                with open('error_log.txt', 'a', encoding='utf-8') as f:
                    f.write(
                        f'[BetterCam初始化失败] 区域:{region}, 屏幕:{self.screen_width}x{self.screen_height}, 错误: {str(e)}\n')
                print(f'BetterCam初始化失败: {e}')
                return False
        except Exception as e:
            self.bettercam_capture = None
            with open('error_log.txt', 'a', encoding='utf-8') as f:
                f.write(f'[BetterCam初始化异常] 屏幕:{self.screen_width}x{self.screen_height}, 错误: {str(e)}\n')
            print(f'BetterCam初始化异常: {e}')
            return False

    def init_obs(self):
        """
        初始化OBS WebSocket连接

        Returns:
            bool: 初始化是否成功
        """
        from src.obs import OBSVideoStream
        if self.obs is not None:
            self.obs.close()
            del self.obs
        self.obs = OBSVideoStream(ip=self.config['obs_ip'], port=self.config['obs_port'], fps=self.config['obs_fps'])
        return self.obs.start()

    def init_cjk_device(self):
        """
        初始化采集卡设备

        Returns:
            bool: 初始化是否成功
        """
        from src.cjk_get import VideoCaptureDevice
        if self.config['is_cjk']:
            fourcc_format = self.config.get('cjk_fourcc_format', None)
            self.cjk_device = VideoCaptureDevice(device_id=self.config['cjk_device_id'], fps=self.config['cjk_fps'],
                                                 resolution=tuple(map(int, self.config['cjk_resolution'].split('x'))),
                                                 crop_size=tuple(map(int, self.config['cjk_crop_size'].split('x'))),
                                                 fourcc_format=fourcc_format)
            return self.cjk_device.start()
        return False

    def get_screenshot(self, region=None):
        """
        获取截图 - 自动选择分离或简单模式

        Args:
            region: 截图区域 (left, top, right, bottom)，仅用于BetterCam源

        Returns:
            numpy.ndarray: 截图图像，如果失败则返回None
        """
        if self.enable_parallel_processing:
            return self._get_screenshot_separated(region)
        return self._get_screenshot_simple(region)

    def _get_screenshot_separated(self, region=None):
        """获取截图 - 分离版本"""
        current_time = time.perf_counter()
        if self._cached_screenshot is not None and current_time - self._last_screenshot_time < self._screenshot_cache_duration:
            return self._cached_screenshot
        screenshot = None
        try:
            screenshot, total_time = self.processed_screenshot_queue.get_nowait()
            self.performance_monitor.record_frame_time(total_time)
        except queue.Empty:
            screenshot = self._get_raw_screenshot()
        if screenshot is not None:
            self._cached_screenshot = screenshot.copy() if self.enable_memory_pool else screenshot
            self._last_screenshot_time = current_time
            if self.adaptive_cache:
                self._adjust_cache_duration()
        return screenshot

    def _get_screenshot_simple(self, region=None):
        """获取截图，根据配置自动选择截图源 - 简单版本（兼容老版本）"""
        current_time = time.perf_counter()
        if self._cached_screenshot is not None and current_time - self._last_screenshot_time < self._screenshot_cache_duration:
            return self._cached_screenshot
        screenshot = None
        if self.config.get('is_cjk', False):
            screenshot = self.get_cjk_screenshot()
        elif self.config.get('is_obs', False):
            screenshot = self.get_obs_screenshot_simple()
        else:
            if region is None and self.engine is not None:
                region_key = 'default'
                if region_key not in self._region_cache:
                    input_shape_weight = self.engine.get_input_shape()[3]
                    input_shape_height = self.engine.get_input_shape()[2]
                    offset_x = int(self.config.get("capture_offset_x", 0))
                    offset_y = int(self.config.get("capture_offset_y", 0))
                    left = (self.screen_width - input_shape_weight) // 2 + offset_x
                    top = (self.screen_height - input_shape_height) // 2 + offset_y
                    right = left + input_shape_weight
                    bottom = top + input_shape_height
                    self._region_cache[region_key] = (left, top, right, bottom)
                region = self._region_cache[region_key]
            screenshot = self.get_bettercam_screenshot()
        if screenshot is not None:
            self._cached_screenshot = screenshot
            self._last_screenshot_time = current_time
        return screenshot

    def _adjust_cache_duration(self):
        """动态调整缓存时间（仅分离模式）"""
        stats = self.performance_monitor.get_stats()
        current_fps = stats.get('fps', 0)
        if current_fps > 120:
            self._screenshot_cache_duration = max(self.min_cache_duration, self._screenshot_cache_duration * 0.95)
        elif current_fps < 60:
            self._screenshot_cache_duration = min(self.max_cache_duration, self._screenshot_cache_duration * 1.05)

    def get_bettercam_screenshot(self):
        """
        使用BetterCam获取屏幕截图 - 通用版本
        Returns:
            numpy.ndarray | None
        """
        try:
            cap = self.bettercam_capture
            if cap is None:
                return None

            if self.enable_parallel_processing:
                frame = cap.get_latest_frame()
                if frame is not None:
                    self._bettercam_latest_frame = frame
                    return frame

                # 复用上一帧
                last = getattr(self, '_bettercam_latest_frame', None)
                if last is not None:
                    if getattr(self, 'turbo_mode', False):
                        return last
                    return last.copy()
                return None
            else:
                return cap.get_latest_frame()
        except Exception as e:
            if not getattr(self, 'turbo_mode', False):
                with open('error_log.txt', 'a', encoding='utf-8') as f:
                    f.write('[BetterCam截图异常] ' + str(e) + '\n')
                print(f'BetterCam截图异常: {e}')
            return None

    def get_obs_screenshot_simple(self):
        """
        从OBS获取截图 - 简单版本（兼容老版本）

        Returns:
            numpy.ndarray: 截图图像，如果失败则返回None
        """
        try:
            return self.obs.frame_queue.get(timeout=1)
        except Exception as e:
            print(f'获取OBS画面失败: {e}')

    def get_cjk_screenshot(self):
        """
        从采集卡设备获取截图 - 使用真正分离架构

        Returns:
            numpy.ndarray: 截图图像，如果失败则返回None
        """
        if not self.cjk_device:
            return
        try:
            if self.enable_parallel_processing:
                frame = self._get_cjk_frame_separated()
                if frame is not None:
                    self._cjk_latest_frame = frame
                    return frame
                return None
            return self.cjk_device.get_frame_non_blocking()
        except Exception as e:
            if not getattr(self, 'turbo_mode', False):
                print(f'采集卡获取帧异常: {e}')
            return None

    def display_screenshot_separated(self):
        """
        显示截图和检测结果 - 分离模式

        在单独的线程中运行，从screenshot_queue队列中获取截图和检测结果并显示。
        """
        window_created = False
        window_name = 'screenshot'
        while self.running:
            try:
                result = self.screenshot_queue.get(timeout=1)
                sunone_debug = None
                if len(result) == 13:
                    screenshot, boxes, scores, classes, fps_text, infer_time_ms, current_key, current_scope, aim_key_status, is_v8, pid_deadzone, smooth_deadzone, sunone_debug = result
                else:
                    screenshot, boxes, scores, classes, fps_text, infer_time_ms, current_key, current_scope, aim_key_status, is_v8, pid_deadzone, smooth_deadzone = result
                if not window_created:
                    try:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                        if self.engine is not None:
                            cv2.resizeWindow(window_name, self.engine.get_input_shape()[3],
                                             self.engine.get_input_shape()[2])
                        window_created = True
                    except Exception as e:
                        print(f'创建OpenCV窗口失败: {e}')
                        with open('error_log.txt', 'a', encoding='utf-8') as f:
                            f.write(f'[OpenCV窗口创建失败] {str(e)}\n')
                        continue
                from src.infer_function import draw_boxes, draw_boxes_v8
                if is_v8:
                    screenshot = draw_boxes_v8(screenshot, boxes, scores, classes)
                else:
                    screenshot = draw_boxes(screenshot, boxes, scores, classes)
                height, width = screenshot.shape[:2]
                center_x, center_y = (width // 2, height // 2)
                if aim_key_status and current_scope > 0 and self.config.get('show_fov', True):
                    scope_radius = int(current_scope)
                    cv2.circle(screenshot, (center_x, center_y), scope_radius, (0, 255, 255), 2)
                if pid_deadzone > 0:
                    draw_move_deadzone_visualization(screenshot, center_x, center_y, pid_deadzone)
                if smooth_deadzone > 0:
                    draw_smooth_deadzone_visualization(screenshot, center_x, center_y, smooth_deadzone)
                if sunone_debug:
                    pred_pt = sunone_debug.get('prediction')
                    step_pt = sunone_debug.get('step')
                    future_pts = sunone_debug.get('future') or []
                    if step_pt:
                        sp = (int(round(step_pt[0])), int(round(step_pt[1])))
                        cv2.line(screenshot, (center_x, center_y), sp, (0, 255, 255), 1)
                        cv2.circle(screenshot, sp, 4, (0, 255, 255), -1)
                    if pred_pt:
                        pp = (int(round(pred_pt[0])), int(round(pred_pt[1])))
                        cv2.circle(screenshot, pp, 5, (0, 255, 0), 2)
                    for fp in future_pts:
                        fpv = (int(round(fp[0])), int(round(fp[1])))
                        cv2.circle(screenshot, fpv, 2, (255, 255, 0), -1)
                cv2.putText(screenshot, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if self.config.get('show_infer_time', True):
                    latency_text = f'latency: {infer_time_ms:.2f}ms' if isinstance(infer_time_ms,
                                                                                   (int, float)) else str(infer_time_ms)
                    cv2.putText(screenshot, latency_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                try:
                    cv2.imshow(window_name, screenshot)
                    cv2.waitKey(1)
                except Exception as e:
                    print(f'显示图像失败: {e}')
                    window_created = False
            except queue.Empty:
                pass
            except Exception as e:
                with open('error_log.txt', 'a', encoding='utf-8') as f:
                    f.write('[显示截图异常] ' + str(e) + '\n' + traceback.format_exc() + '\n')
                print(f'显示截图异常: {e}')
        if window_created:
            try:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
            except Exception:
                return

    def display_screenshot_simple(self):
        """
        显示截图和检测结果 - 简单模式（兼容老版本）

        在单独的线程中运行，从screenshot_queue队列中获取截图和检测结果并显示。
        """
        while self.running:
            try:
                result = self.screenshot_queue.get(timeout=1)
                sunone_debug = None
                if len(result) == 9:
                    screenshot, boxes, scores, classes, fps_text, infer_time_ms, current_key, current_scope, aim_key_status = result
                    is_v8 = False
                    pid_deadzone = 0.0
                    smooth_deadzone = 0.0
                elif len(result) == 10:
                    screenshot, boxes, scores, classes, fps_text, infer_time_ms, current_key, current_scope, aim_key_status, is_v8 = result
                    pid_deadzone = 0.0
                    smooth_deadzone = 0.0
                elif len(result) == 11:
                    screenshot, boxes, scores, classes, fps_text, infer_time_ms, current_key, current_scope, aim_key_status, is_v8, pid_deadzone = result
                    smooth_deadzone = 0.0
                elif len(result) == 12:
                    screenshot, boxes, scores, classes, fps_text, infer_time_ms, current_key, current_scope, aim_key_status, is_v8, pid_deadzone, smooth_deadzone = result
                else:
                    screenshot, boxes, scores, classes, fps_text, infer_time_ms, current_key, current_scope, aim_key_status, is_v8, pid_deadzone, smooth_deadzone, sunone_debug = result
                cv2.namedWindow('screenshot', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                cv2.setWindowProperty('screenshot', cv2.WND_PROP_TOPMOST, 1)
                if self.engine is not None:
                    cv2.resizeWindow('screenshot', self.engine.get_input_shape()[3], self.engine.get_input_shape()[2])
                from src.infer_function import draw_boxes, draw_boxes_v8
                if is_v8:
                    screenshot = draw_boxes_v8(screenshot, boxes, scores, classes)
                else:
                    screenshot = draw_boxes(screenshot, boxes, scores, classes)
                height, width = screenshot.shape[:2]
                center_x, center_y = (width // 2, height // 2)
                if aim_key_status and current_scope > 0 and self.config.get('show_fov', True):
                    scope_radius = int(current_scope)
                    cv2.circle(screenshot, (center_x, center_y), scope_radius, (0, 255, 255), 2)
                if pid_deadzone > 0:
                    draw_move_deadzone_visualization(screenshot, center_x, center_y, pid_deadzone)
                if smooth_deadzone > 0:
                    draw_smooth_deadzone_visualization(screenshot, center_x, center_y, smooth_deadzone)
                if sunone_debug:
                    pred_pt = sunone_debug.get('prediction')
                    step_pt = sunone_debug.get('step')
                    future_pts = sunone_debug.get('future') or []
                    if step_pt:
                        sp = (int(round(step_pt[0])), int(round(step_pt[1])))
                        cv2.line(screenshot, (center_x, center_y), sp, (0, 255, 255), 1)
                        cv2.circle(screenshot, sp, 4, (0, 255, 255), -1)
                    if pred_pt:
                        pp = (int(round(pred_pt[0])), int(round(pred_pt[1])))
                        cv2.circle(screenshot, pp, 5, (0, 255, 0), 2)
                    for fp in future_pts:
                        fpv = (int(round(fp[0])), int(round(fp[1])))
                        cv2.circle(screenshot, fpv, 2, (255, 255, 0), -1)
                cv2.putText(screenshot, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if self.config.get('show_infer_time', True):
                    latency_text = f'latency: {infer_time_ms:.2f}ms' if isinstance(infer_time_ms,
                                                                                   (int, float)) else str(infer_time_ms)
                    cv2.putText(screenshot, latency_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('screenshot', screenshot)
                cv2.waitKey(1)
            except queue.Empty:
                pass
            except Exception as e:
                with open('error_log.txt', 'a', encoding='utf-8') as f:
                    f.write('[显示截图异常] ' + str(e) + '\n' + traceback.format_exc() + '\n')
                print(f'显示截图异常: {e}')

    pass

    def put_screenshot_result(self, screenshot, boxes, scores, classes, fps_text, infer_time_ms, current_key='',
                              current_scope=0, aim_key_status=False, is_v8=False, pid_deadzone=0.0,
                              smooth_deadzone=0.0, sunone_debug=None):
        """
        将检测结果放入显示队列 - 兼容两种模式

        Args:
            screenshot: 截图图像
            boxes: 检测框
            scores: 检测分数
            classes: 检测类别
            fps_text: FPS文本
            infer_time_ms: 推理时间(毫秒)
            current_key: 当前按下的按键
            current_scope: 当前按键的瞄准范围(像素)
            aim_key_status: 瞄准按键状态
            is_v8: 是否为v8模型
            pid_deadzone: 移动死区半径(像素)
            smooth_deadzone: 平滑禁区半径(像素)
        """
        try:
            if self.screenshot_queue.full():
                try:
                    self.screenshot_queue.get_nowait()
                except queue.Empty:
                    pass
            if self.enable_parallel_processing:
                self.screenshot_queue.put(
                    (screenshot, boxes, scores, classes, fps_text, infer_time_ms, current_key, current_scope,
                     aim_key_status, is_v8, pid_deadzone, smooth_deadzone, sunone_debug))
            else:
                self.screenshot_queue.put(
                    (screenshot, boxes, scores, classes, fps_text, infer_time_ms, current_key, current_scope,
                     aim_key_status, is_v8, pid_deadzone, smooth_deadzone, sunone_debug))
        except Exception as e:
            print(f'放入截图结果异常: {e}')

    def close(self):
        """
        关闭并释放所有截图资源
        """
        self.running = False
        if self.enable_parallel_processing:
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1)
            for thread in self.processing_threads:
                if thread.is_alive():
                    thread.join(timeout=1)
            self._clear_all_queues()
        if self.bettercam_capture is not None:
            try:
                self.bettercam_capture.stop()
                del self.bettercam_capture
            except Exception as e:
                with open('error_log.txt', 'a', encoding='utf-8') as f:
                    f.write('[BetterCam释放异常] ' + str(e) + '\n')
                print(f'BetterCam释放异常: {e}')
            self.bettercam_capture = None
        if self.obs is not None:
            self.obs.close()
            self.obs = None
        if self.cjk_device:
            self.cjk_device.close()
            self.cjk_device = None
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1)
        try:
            cv2.destroyWindow('screenshot')
        except Exception:
            print("OpenCV窗口 'screenshot' 不存在或已被销毁")
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception as e:
            print(f'清理所有OpenCV窗口异常: {e}')
        if self.enable_parallel_processing and self.memory_pool:
            del self.memory_pool
            self.memory_pool = None
        gc.collect()
        print('截图管理器已关闭')

    def _clear_all_queues(self):
        """清空所有队列（仅分离模式）"""
        if not self.enable_parallel_processing:
            return
        queues = [self.raw_screenshot_queue, self.processed_screenshot_queue, self.screenshot_queue]
        for q in queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def set_turbo_mode(self, enabled=True):
        """
        设置强制提速模式

        Args:
            enabled: 是否启用强制提速模式
        """
        if hasattr(self, 'turbo_mode'):
            self.turbo_mode = enabled
            self.skip_frame_processing = enabled
            print(f"强制提速模式: {('开启' if enabled else '关闭')}")
            if enabled:
                print('提速优化: 最小队列延迟、减少内存拷贝、跳过非必要处理')
            else:
                print('标准模式: 完整处理流程、安全内存管理')

    def get_performance_info(self):
        """获取当前性能信息"""
        if not self.enable_parallel_processing:
            return {'mode': '简单模式', 'status': '单线程同步处理', 'optimization': '传统稳定模式'}
        info = {'mode': '分离模式', 'turbo_mode': getattr(self, 'turbo_mode', False),
                'queue_sizes': {'raw_queue': self.raw_screenshot_queue.qsize(),
                                'processed_queue': self.processed_screenshot_queue.qsize(),
                                'display_queue': self.screenshot_queue.qsize()}, 'workers': self.max_workers,
                'memory_pool': self.enable_memory_pool, 'frame_reuse_rate': self.frame_reuse_stats.get('reuse_rate', 0)}
        if self.config.get('is_cjk', False):
            info['screenshot_source'] = 'CJK采集卡 (支持帧复用)'
            info['frame_reuse_support'] = True
        elif self.config.get('is_obs', False):
            info['screenshot_source'] = 'OBS (支持帧复用)'
            info['frame_reuse_support'] = True
        else:
            info['screenshot_source'] = 'BetterCam (直接截图)'
            info['frame_reuse_support'] = False
        if hasattr(self, 'performance_monitor'):
            stats = self.performance_monitor.get_stats()
            info.update(stats)
        return info

    def update_config(self, config_key, value):
        """
        更新配置并在必要时重新初始化截图源

        Args:
            config_key: 配置键
            value: 新的配置值

        Returns:
            bool: 更新是否成功
        """
        if config_key not in self.config:
            self.config[config_key] = value
            return True
        if self.config[config_key] == value:
            return True
        self.config[config_key] = value
        need_reinit = False
        if config_key in ['is_cjk', 'is_obs']:
            need_reinit = True
        elif config_key.startswith('cjk_') and self.config.get('is_cjk', False):
            need_reinit = True
        elif config_key.startswith('obs_') and self.config.get('is_obs', False):
            need_reinit = True
        elif config_key in ['capture_offset_x', 'capture_offset_y']:
            self._region_cache = {}
            if not self.config.get('is_cjk', False) and not self.config.get('is_obs', False):
                need_reinit = True
        elif config_key == 'enable_parallel_processing':
            need_reinit = True
            self.enable_parallel_processing = value
            print(f"截图模式切换为: {('分离模式' if value else '简单模式')}")
        elif config_key == 'turbo_mode':
            if self.enable_parallel_processing:
                self.set_turbo_mode(value)
                return True
        if need_reinit:
            self.close()
            return self.init_sources()
        return True
