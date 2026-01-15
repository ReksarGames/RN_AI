"""
采集卡视频捕获模块

提供基于 OpenCV 的视频采集卡封装，支持高性能帧获取和非阻塞操作。
"""

import cv2
import threading
import queue
import time
import numpy as np
from typing import Optional, Tuple
from collections import deque


class VideoCaptureDevice:
    """
    视频采集卡设备类
    
    封装 OpenCV VideoCapture，提供非阻塞帧获取、自动裁剪等功能。
    支持多种视频编码格式（NV12, MJPG, YUYV等）。
    """
    
    def __init__(self, device_id: int, fps: int, resolution: Tuple[int, int], 
                 crop_size: Tuple[int, int], fourcc_format: Optional[str] = None):
        """
        初始化采集卡设备
        
        Args:
            device_id: 设备ID（通常是0, 1, 2等）
            fps: 目标帧率
            resolution: 采集分辨率 (width, height)
            crop_size: 裁剪尺寸 (width, height)
            fourcc_format: 视频编码格式，如 'NV12', 'MJPG', 'YUYV' 等
        """
        self.device_id = device_id
        self.fps = fps
        self.resolution = resolution  # (width, height)
        self.crop_size = crop_size    # (width, height)
        self.fourcc_format = fourcc_format
        
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)  # 限制队列大小，避免内存堆积
        self.running = False
        self.capture_thread = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        
        # 真实捕获FPS计数器（在采集线程中统计）
        self._fps_timestamps = deque(maxlen=30)  # 滑动窗口30帧
        self._real_fps = 0.0
        self._fps_lock = threading.Lock()
        
        print(f"[CJK] 初始化采集卡: 设备ID={device_id}, FPS={fps}, 分辨率={resolution}, 裁剪={crop_size}, 编码={fourcc_format}")
    
    def start(self) -> bool:
        """
        启动采集卡
        
        Returns:
            bool: 启动是否成功
        """
        try:
            # 使用 DirectShow 后端（Windows）以获得更好的性能
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                print(f"[CJK] 错误: 无法打开设备 {self.device_id}")
                return False
            
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # 设置帧率
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 设置视频编码格式
            if self.fourcc_format:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*self.fourcc_format)
                    self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                    print(f"[CJK] 设置编码格式: {self.fourcc_format}")
                except Exception as e:
                    print(f"[CJK] 警告: 设置编码格式失败: {e}")
            
            # 低延迟配置
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲区，减少延迟
            
            # 验证实际设置的参数
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"[CJK] 实际参数: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # 启动采集线程
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            print(f"[CJK] 采集卡启动成功")
            return True
            
        except Exception as e:
            print(f"[CJK] 启动失败: {e}")
            return False
    
    def _capture_loop(self):
        """
        采集线程主循环
        
        持续从采集卡读取帧，进行裁剪后放入队列
        """
        frame_count = 0
        last_fps_time = time.time()
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print(f"[CJK] 警告: 读取帧失败")
                    time.sleep(0.001)
                    continue
                
                # 裁剪到目标尺寸（中心裁剪）
                cropped_frame = self._crop_frame(frame)
                
                if cropped_frame is None:
                    continue
                
                # 更新最新帧（用于帧复用）
                with self._frame_lock:
                    self._latest_frame = cropped_frame
                
                # 统计真实捕获FPS（在采集线程中，每帧都统计）
                self._update_real_fps()
                
                # 非阻塞放入队列（如果队列满了就丢弃旧帧）
                try:
                    self.frame_queue.put_nowait(cropped_frame)
                except queue.Full:
                    # 队列满了，清空并放入新帧
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.frame_queue.put_nowait(cropped_frame)
                    except queue.Full:
                        pass
                
                # FPS 统计（每100帧打印一次）
                frame_count += 1
                if frame_count % 100 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_fps_time
                    if elapsed > 0:
                        capture_fps = 100 / elapsed
                        # print(f"[CJK] 采集FPS: {capture_fps:.1f}")
                        last_fps_time = current_time
                
            except Exception as e:
                print(f"[CJK] 采集循环异常: {e}")
                time.sleep(0.01)
    
    def _crop_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        裁剪帧到目标尺寸（中心裁剪）
        
        Args:
            frame: 原始帧
            
        Returns:
            裁剪后的帧，如果失败则返回 None
        """
        try:
            h, w = frame.shape[:2]
            crop_w, crop_h = self.crop_size
            
            # 计算中心裁剪的起始位置
            x = max(0, (w - crop_w) // 2)
            y = max(0, (h - crop_h) // 2)
            
            # 确保不超出边界
            x_end = min(w, x + crop_w)
            y_end = min(h, y + crop_h)
            
            cropped = frame[y:y_end, x:x_end]
            
            # 如果裁剪后尺寸不对，进行缩放
            if cropped.shape[0] != crop_h or cropped.shape[1] != crop_w:
                cropped = cv2.resize(cropped, (crop_w, crop_h))
            
            return cropped
            
        except Exception as e:
            print(f"[CJK] 裁剪帧失败: {e}")
            return None
    
    def _update_real_fps(self):
        """更新真实捕获FPS（在采集线程中调用）"""
        try:
            current_time = time.perf_counter()
            with self._fps_lock:
                self._fps_timestamps.append(current_time)
                
                # 至少需要2个时间戳才能计算FPS
                if len(self._fps_timestamps) >= 2:
                    time_span = self._fps_timestamps[-1] - self._fps_timestamps[0]
                    if time_span > 0:
                        # 使用整个窗口的时间跨度计算FPS
                        self._real_fps = (len(self._fps_timestamps) - 1) / time_span
        except Exception as e:
            pass  # 静默处理，避免影响采集线程
    
    def get_real_fps(self) -> float:
        """获取真实的采集帧率（从采集线程统计）"""
        with self._fps_lock:
            return self._real_fps
    
    def get_frame_non_blocking(self) -> Optional[np.ndarray]:
        """
        非阻塞获取帧
        
        优先从队列获取新帧，如果队列为空则返回最新缓存的帧。
        这种设计支持推理线程以高于采集卡FPS的速度运行（帧复用）。
        
        Returns:
            帧数据，如果没有可用帧则返回 None
        """
        try:
            # 尝试从队列获取新帧（非阻塞）
            frame = self.frame_queue.get_nowait()
            return frame
        except queue.Empty:
            # 队列为空，返回最新缓存的帧
            with self._frame_lock:
                if self._latest_frame is not None:
                    # 返回副本以避免多线程问题
                    return self._latest_frame.copy()
            return None
    
    def get_frame_blocking(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        阻塞获取帧
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            帧数据，如果超时则返回 None
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None
    
    def close(self):
        """
        关闭采集卡并释放资源
        """
        print(f"[CJK] 正在关闭采集卡...")
        
        self.running = False
        
        # 等待采集线程结束
        if hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        with self._frame_lock:
            self._latest_frame = None
        
        print(f"[CJK] 采集卡已关闭")
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        if hasattr(self, 'cap'):
            self.close()


def list_devices(max_devices=10):
    """
    枚举可用的视频采集设备
    
    Args:
        max_devices: 最多检测的设备数量
        
    Returns:
        list: 可用设备ID列表
    """
    available = []
    print(f"[CJK] 正在扫描视频设备 (0-{max_devices-1})...")
    
    for i in range(max_devices):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            # 获取设备信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            available.append(i)
            print(f"[CJK] 设备 {i}: {width}x{height} @ {fps}fps")
            cap.release()
    
    if not available:
        print("[CJK] 未找到可用的视频设备")
    else:
        print(f"[CJK] 找到 {len(available)} 个可用设备: {available}")
    
    return available


# 测试代码
if __name__ == "__main__":
    print("=== 采集卡测试 ===\n")
    
    # 1. 检测可用设备
    print("步骤 1: 检测可用的视频设备")
    available_devices = list_devices(max_devices=10)
    print()
    
    # 2. 选择设备
    if not available_devices:
        print("✗ 未找到可用的视频设备，无法继续测试")
        print("提示: 请检查采集卡是否正确连接")
        exit(1)
    
    device_id = available_devices[0]
    print(f"步骤 2: 使用设备 {device_id}")
    print()
    
    # 3. 创建采集卡实例
    print("步骤 3: 创建采集卡实例")
    device = VideoCaptureDevice(
        device_id=device_id,
        fps=60,
        resolution=(1920, 1080),
        crop_size=(640, 640),
        fourcc_format='MJPG'
    )
    print()
    
    # 4. 启动采集卡
    print("步骤 4: 启动采集卡")
    if device.start():
        print("✓ 采集卡启动成功")
        print("按 'q' 键退出预览窗口")
        print()
        
        # 测试帧获取
        cv2.namedWindow('CJK Test', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        while True:
            frame = device.get_frame_non_blocking()
            
            if frame is not None:
                # 添加帧计数显示
                frame_count += 1
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('CJK Test', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print(f"\n总共捕获 {frame_count} 帧")
    else:
        print("✗ 采集卡启动失败")
    
    # 关闭采集卡
    print("\n步骤 5: 关闭采集卡")
    device.close()
    print("\n=== 测试完成 ===")
