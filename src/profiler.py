from typing import Dict, Optional

class FrameProfiler:

    def __init__(self, report_interval_frames: int=20, ema_alpha: float=0.2):
        self.report_interval_frames = max(1, int(report_interval_frames))
        self.ema_alpha = float(ema_alpha)
        self._frame_idx = 0
        self._last_ms = {}
        self._ema_ms = {}

    def begin_frame(self):
        self._frame_idx += 1

    def record(self, name: str, value_ms: float):
        self._last_ms[name] = float(value_ms)
        if name not in self._ema_ms:
            self._ema_ms[name] = float(value_ms)
        else:
            a = self.ema_alpha
            self._ema_ms[name] = a * float(value_ms) + (1 - a) * self._ema_ms[name]

    def should_report(self) -> bool:
        return self._frame_idx % self.report_interval_frames == 0

    def format_report(self, trt_profile: Optional[Dict[str, float]]=None) -> str:
        cap = self._last_ms.get('cap', 0.0)
        pre = self._last_ms.get('pre', 0.0)
        inf = self._last_ms.get('infer', 0.0)
        post = self._last_ms.get('post', 0.0)
        if trt_profile and all((k in trt_profile for k in ['htod_ms', 'exec_ms', 'dtoh_ms', 'total_ms'])):
            return f"[PROFILE] cap:{cap:.3f}ms pre:{pre:.3f}ms infer:{inf:.3f}ms (TRT h2d:{trt_profile['htod_ms']:.3f} exec:{trt_profile['exec_ms']:.3f} d2h:{trt_profile['dtoh_ms']:.3f} total:{trt_profile['total_ms']:.3f}) post:{post:.3f}ms"
        return f'[PROFILE] cap:{cap:.3f}ms pre:{pre:.3f}ms infer:{inf:.3f}ms post:{post:.3f}ms'

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        return {'last_ms': dict(self._last_ms), 'ema_ms': dict(self._ema_ms), 'frame_idx': self._frame_idx}