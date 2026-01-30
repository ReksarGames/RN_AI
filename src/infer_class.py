import onnxruntime as rt
from src.infer_function import *
from src.function import *
import threading
import warnings
import time
from typing import Optional, Tuple, List
import numpy as np


def build_onnx(path):
    key = b'fOWPyk6AOO5FW5yjs96xZ9MTTcgMYvag4kNbdY8396k='
    f = Fernet(key)
    model = open(path, 'rb').read()
    model = f.decrypt(model)
    return model


class NMSProcessor:
    """
    非极大值抑制(NMS)处理器 - 支持多种 NMS 算法的统一接口
    实现延迟初始化和智能算法选择
    """

    def __init__(self):
        self._nms_cache = {}
        self._performance_stats = {}
        self._algorithm_preference = None

    pass
    pass
    pass
    pass

    def process(self, pred: np.ndarray, conf_thres: float = 0.5, iou_thres: float = 0.45,
                class_num: Optional[int] = None, algorithm: str = 'auto') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行 NMS 处理

        Args:
            pred: 模型预测输出
            conf_thres: 置信度阈值
            iou_thres: IoU阈值
            class_num: 类别数量(仅对v5算法需要)
            algorithm: NMS算法选择 ('auto', 'v8', 'v5', 'standard')

        Returns:
            boxes, scores, classes: 处理后的检测结果
        """
        start_time = time.perf_counter()
        if algorithm == 'auto':
            algorithm = self._select_optimal_algorithm(pred, class_num)
        cache_key = self._generate_cache_key(pred, conf_thres, iou_thres, algorithm)
        if cache_key in self._nms_cache:
            return self._nms_cache[cache_key]
        try:
            if algorithm == 'v8':
                boxes, scores, classes = nms_v8(pred, conf_thres, iou_thres)
            else:
                if algorithm == 'v5':
                    if class_num is None:
                        raise ValueError('class_num is required for v5 algorithm')
                    boxes, scores, classes = nms_v5(pred, conf_thres, iou_thres, class_num)
                elif algorithm == 'standard':
                    if class_num is None:
                        raise ValueError('class_num is required for standard algorithm')
                    boxes, scores, classes = nms(pred, conf_thres, iou_thres, class_num)
                else:
                    raise ValueError(f'Unsupported NMS algorithm: {algorithm}')
            processing_time = time.perf_counter() - start_time
            self._update_performance_stats(algorithm, processing_time, len(boxes) if len(boxes) > 0 else 0)
            if len(self._nms_cache) < 100:
                self._nms_cache[cache_key] = (boxes, scores, classes)
            return (boxes, scores, classes)
        except Exception as e:
            print(f'[NMS] Algorithm {algorithm} failed: {e}, falling back to v8')
            if algorithm != 'v8':
                return self.process(pred, conf_thres, iou_thres, class_num, 'v8')
            raise

    def _select_optimal_algorithm(self, pred: np.ndarray, class_num: Optional[int]) -> str:
        """基于输入特征和性能统计选择最优算法"""
        pred_shape = pred.shape if hasattr(pred, 'shape') else None
        if pred_shape and len(pred_shape) >= 2:
            if pred_shape[-2] == 84 or (len(pred_shape) == 3 and pred_shape[1] == 84):
                return 'v8'
            if class_num is not None:
                return 'v5'
        if self._performance_stats:
            best_algo = min(self._performance_stats.keys(), key=lambda k: self._performance_stats[k]['avg_time'])
            return best_algo
        return 'v8'

    def _generate_cache_key(self, pred: np.ndarray, conf_thres: float, iou_thres: float, algorithm: str) -> str:
        """生成缓存键"""
        pred_hash = hash(pred.tobytes()) if hasattr(pred, 'tobytes') else hash(str(pred))
        return f'{algorithm}_{conf_thres}_{iou_thres}_{pred_hash}'

    def _update_performance_stats(self, algorithm: str, processing_time: float, result_count: int):
        """更新性能统计"""
        if algorithm not in self._performance_stats:
            self._performance_stats[algorithm] = {'total_time': 0.0, 'call_count': 0, 'avg_time': 0.0,
                                                  'total_results': 0}
        stats = self._performance_stats[algorithm]
        stats['total_time'] += processing_time
        stats['call_count'] += 1
        stats['avg_time'] = stats['total_time'] / stats['call_count']
        stats['total_results'] += result_count

    def clear_cache(self):
        """清除缓存"""
        self._nms_cache.clear()

    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        return self._performance_stats.copy()


class OnnxRuntimeDmlEngine:

    def __init__(self, model_path, is_onnx_engine=False, is_trt=False, lazy_init=True):
        """
        初始化 ONNX 推理引擎

        Args:
            model_path: 模型文件路径
            is_onnx_engine: 是否为原始ONNX文件
            is_trt: 是否使用TensorRT
            lazy_init: 是否启用延迟初始化
        """
        warnings.filterwarnings('ignore', message='.*pagelocked_host_allocation.*')
        warnings.filterwarnings('ignore', message='.*device_allocation.*')
        warnings.filterwarnings('ignore', message='.*stream.*')
        self.model_path = model_path
        self.is_onnx_engine = is_onnx_engine
        self.is_trt = is_trt
        self.lazy_init = lazy_init
        self._session = None
        self._model_bytes = None
        self._input_name = None
        self._input_shape = None
        self._output_names = None
        self._providers = None
        self._lock = threading.Lock()
        self._nms_processor = None
        self._inference_count = 0
        self._total_inference_time = 0.0
        if not lazy_init:
            self._initialize_session()

    def _initialize_session(self):
        """初始化推理会话"""
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
                return
            self._providers = self._determine_providers()
            print(f'[ONNX] 当前可用推理后端: {rt.get_available_providers()}，实际使用: {self._providers}')
            if self.is_onnx_engine:
                self._model_bytes = open(self.model_path, 'rb').read()
            else:
                self._model_bytes = build_onnx(self.model_path)
            self._session = rt.InferenceSession(self._model_bytes, providers=self._providers)
            self._input_name = self._session.get_inputs()[0].name
            self._input_shape = self._session.get_inputs()[0].shape
            self._output_names = [out.name for out in self._session.get_outputs()]

    def _determine_providers(self) -> List[str]:
        """确定推理提供商优先级"""
        available = rt.get_available_providers()
        providers = []
        if self.is_trt:
            priority_providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'DmlExecutionProvider',
                                  'AzureExecutionProvider', 'CPUExecutionProvider']
        else:
            priority_providers = ['DmlExecutionProvider', 'CUDAExecutionProvider', 'AzureExecutionProvider',
                                  'CPUExecutionProvider']
        for provider in priority_providers:
            if provider in available:
                providers.append(provider)
        if not providers:
            providers = available
        return providers

    @property
    def nms_processor(self) -> NMSProcessor:
        """延迟初始化 NMS 处理器"""
        if self._nms_processor is None:
            self._nms_processor = NMSProcessor()
        return self._nms_processor

    def infer(self, img_input: np.ndarray) -> List[np.ndarray]:
        """
        执行推理

        Args:
            img_input: 输入图像数据

        Returns:
            模型输出结果
        """
        if self._session is None:
            self._initialize_session()
        start_time = time.perf_counter()
        with self._lock:
            outputs = self._session.run(self._output_names, {self._input_name: img_input})
        inference_time = time.perf_counter() - start_time
        self._inference_count += 1
        self._total_inference_time += inference_time
        return outputs

    pass
    pass
    pass

    def infer_with_nms(self, img_input: np.ndarray, conf_thres: float = 0.5, iou_thres: float = 0.45,
                       nms_algorithm: str = 'auto') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行推理并应用 NMS 处理

        Args:
            img_input: 输入图像数据
            conf_thres: 置信度阈值
            iou_thres: IoU阈值
            nms_algorithm: NMS算法选择

        Returns:
            boxes, scores, classes: 检测结果
        """
        outputs = self.infer(img_input)
        pred = outputs[0]
        class_num = self.get_class_num() if nms_algorithm in ['v5', 'standard'] else None
        return self.nms_processor.process(pred, conf_thres, iou_thres, class_num, nms_algorithm)

    def get_input_shape(self) -> Tuple:
        """获取输入形状"""
        if self._input_shape is None:
            self._initialize_session()
        return self._input_shape

    def get_class_num(self) -> int:
        """获取类别数量(YOLOv5格式)"""
        if self._session is None:
            self._initialize_session()
        outputs_meta = self._session.get_outputs()
        output_shapes = outputs_meta[0].shape
        try:
            return int(output_shapes[2]) - 5
        except Exception:
            return 0

    def get_class_num_v8(self) -> int:
        """获取类别数量(YOLOv8格式)"""
        if self._session is None:
            self._initialize_session()
        outputs_meta = self._session.get_outputs()
        output_shapes = outputs_meta[0].shape
        try:
            return int(output_shapes[1]) - 4
        except Exception:
            return 0

    def get_performance_info(self) -> dict:
        """获取性能信息"""
        avg_inference_time = self._total_inference_time / self._inference_count if self._inference_count > 0 else 0.0
        return {'inference_count': self._inference_count, 'total_inference_time': self._total_inference_time,
                'avg_inference_time': avg_inference_time, 'providers': self._providers,
                'nms_stats': self.nms_processor.get_performance_stats() if self._nms_processor else {}}

    def clear_caches(self):
        """清除所有缓存"""
        if self._nms_processor:
            self._nms_processor.clear_cache()

    def reset_performance_stats(self):
        """重置性能统计"""
        self._inference_count = 0
        self._total_inference_time = 0.0
        if self._nms_processor:
            self._nms_processor._performance_stats.clear()

    def warmup(self, input_shape: Optional[Tuple] = None, warmup_iterations: int = 3):
        """
        预热模型以提升后续推理性能

        Args:
            input_shape: 输入形状，如果为None则使用模型默认输入形状
            warmup_iterations: 预热迭代次数
        """
        if input_shape is None:
            input_shape = self.get_input_shape()
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        for i in range(warmup_iterations):
            _ = self.infer(dummy_input)

    def __del__(self):
        """析构函数，确保资源被正确清理"""
        try:
            if hasattr(self, '_session') and self._session:
                del self._session
            if hasattr(self, '_nms_processor') and self._nms_processor:
                self._nms_processor.clear_cache()
        except:
            return None
