import time
import cv2
import queue
from concurrent.futures import ThreadPoolExecutor

class OBSVideoStream:

    def __init__(self, ip='0.0.0.0', port=4455, fps=240):
        self.ip = ip
        self.port = port
        self.buffer_size = 0
        self.fps = fps
        self.udp_url = f'udp://{self.ip}:{self.port}'
        self.executor = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.screenshot_queue = queue.Queue(maxsize=1)
        self.cap = None
        self.running = False
        self.latency_stats = {'frames_received': 0, 'frames_dropped': 0, 'decode_times': []}

    def print_latency_report(self):
        """Print latency statistics report"""
        try:
            total_frames = self.latency_stats['frames_received']
            dropped_frames = self.latency_stats['frames_dropped']
            decode_times = self.latency_stats['decode_times']
            if total_frames == 0:
                print('OBS latency report: No frames received')
                return
            drop_rate = dropped_frames / total_frames * 100 if total_frames > 0 else 0
            if decode_times:
                avg_decode = sum(decode_times) / len(decode_times)
                min_decode = min(decode_times)
                max_decode = max(decode_times)
                print(f'Decode latency: avg {avg_decode:.2f}ms, min {min_decode:.2f}ms, max {max_decode:.2f}ms')
            print('========================\n')
        except Exception as e:
            print(f'Failed to print latency report: {e}')

    def _read_frame(self):
        """Read video frames and put them in queue - latency optimized version"""
        consecutive_failures = 0
        max_failures = 10
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    print('OBS video stream closed, exiting read thread')
                    break
                start_time = time.perf_counter()
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f'Failed {max_failures} consecutive times, stopping OBS reading')
                        break
                    print('Unable to read video frame, retrying...')
                    time.sleep(0.1)
                    continue
                consecutive_failures = 0
                decode_time = time.perf_counter() - start_time
                self.latency_stats['decode_times'].append(decode_time * 1000)
                if len(self.latency_stats['decode_times']) > 100:
                    self.latency_stats['decode_times'].pop(0)
                self.latency_stats['frames_received'] += 1
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self.latency_stats['frames_dropped'] += 1
                    except queue.Empty:
                        pass
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    self.latency_stats['frames_dropped'] += 1
            except Exception as e:
                if self.cap is None:
                    print('OBS read thread: Video stream closed')
                    break
                error_str = str(e)
                if 'Unknown C++ exception from OpenCV code' in error_str or 'NoneType' in error_str:
                    print('OBS read thread: OpenCV stream close exception detected, exiting thread')
                    break
                print(f'OBS read frame exception: {e}')
                time.sleep(0.01)
        print('OBS read thread exited')

    def start(self):
        """Start video stream reading and display"""
        self.cap = cv2.VideoCapture(self.udp_url)
        low_latency_configs = [(cv2.CAP_PROP_BUFFERSIZE, 1), (cv2.CAP_PROP_FPS, self.fps)]
        for prop, value in low_latency_configs:
            try:
                self.cap.set(prop, value)
            except Exception as e:
                print(f'OBS config {prop}={value} failed: {e}')
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        buffer_size = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
        if not self.cap.isOpened():
            print('Unable to open OBS UDP stream')
            return False
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.executor.submit(self._read_frame)
        return True

    def close(self):
        """Close video stream and release resources"""
        print('Closing OBS video stream...')
        self.running = False
        if self.executor:
            try:
                self.executor.shutdown(wait=True)
                print('OBS thread pool closed safely')
            except Exception as e:
                print(f'Exception closing OBS thread pool: {e}')
            self.executor = None
        try:
            if self.latency_stats['frames_received'] > 0:
                self.print_latency_report()
        except Exception as e:
            print(f'Failed to print OBS latency report: {e}')
        if self.cap:
            try:
                self.cap.release()
                print('OBS video stream closed')
            except Exception as e:
                print(f'Exception closing video stream: {e}')
            finally:
                self.cap = None