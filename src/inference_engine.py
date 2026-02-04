"""
Inference Engine Module

TensorRT inference engine implementation with PyTorch CUDA memory management.
Replaces pycuda/cupy with torch for CUDA operations while keeping native TensorRT inference.
"""
import os
import traceback

import numpy as np
import torch


class TensorRTInferenceEngine:
    """
    TensorRT Inference Engine (PyTorch CUDA backend)

    Loads and executes TensorRT engine models for inference.
    Supports automatic conversion from ONNX to TensorRT format.
    Supports CUDA Graph acceleration.
    """

    def __init__(self, engine_path):
        try:
            import tensorrt as trt
        except (ImportError, OSError) as e:
            raise RuntimeError(
                f'TensorRT not detected. Install TensorRT or switch to ONNX Runtime. Error: {e}')

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available, cannot use TensorRT acceleration.')

        self.device = torch.device('cuda:0')
        torch.cuda.set_device(self.device)

        self.logger = trt.Logger(trt.Logger.INFO)
        self.stream = torch.cuda.Stream(device=self.device)

        # Load engine
        with open(engine_path, 'rb') as f:
            with trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            # Try to regenerate from ONNX using Python API
            os.remove(engine_path)
            onnx_path = os.path.splitext(engine_path)[0] + '.onnx'
            if not os.path.exists(onnx_path):
                raise RuntimeError(f'Cannot find corresponding ONNX file: {onnx_path}')

            print(f'[TRT] Building engine from {onnx_path}...')
            with open(onnx_path, 'rb') as f:
                onnx_bytes = f.read()
            success, _ = auto_convert_engine_from_memory(onnx_bytes, engine_path)
            if not success:
                raise RuntimeError('TensorRT engine build failed!')

            with open(engine_path, 'rb') as f2:
                with trt.Runtime(self.logger) as runtime2:
                    self.engine = runtime2.deserialize_cuda_engine(f2.read())
            if self.engine is None:
                raise RuntimeError('TensorRT engine load failed!')

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()

        # CUDA Graph flags (matching original pycuda/cupy implementation)
        self._graph_supported = False
        self._use_cuda_graph = False
        self._use_cupy_graph = False  # Kept for compatibility, now uses torch
        self._graph = None
        self._graph_exec = None
        self._graph_captured = False
        self._graph_warmed = False
        self._cupy_stream = None  # Kept for compatibility, not used

        # GPU preprocessing support
        self._preprocess_input = None
        self._preprocess_target_size = None

        # Check CUDA Graph support
        try:
            cuda_version = torch.version.cuda
            if cuda_version:
                major = int(cuda_version.split('.')[0])
                if major >= 10:
                    self._graph_supported = True
        except Exception:
            self._graph_supported = False

    def _allocate_buffers(self):
        """
        Allocate CUDA memory for model inputs/outputs using PyTorch

        Returns:
            tuple: (inputs, outputs, bindings)
        """
        import tensorrt as trt

        inputs = []
        outputs = []
        bindings = []

        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))

            # Convert numpy dtype to torch dtype
            torch_dtype = torch.float32
            np_dtype = np.float32
            if dtype == np.float16:
                torch_dtype = torch.float16
                np_dtype = np.float16
            elif dtype == np.int32:
                torch_dtype = torch.int32
                np_dtype = np.int32
            elif dtype == np.int8:
                torch_dtype = torch.int8
                np_dtype = np.int8

            # Allocate pinned host memory and device memory
            # pin_memory=True is equivalent to pycuda.pagelocked_empty
            host_mem = torch.empty(size, dtype=torch_dtype, pin_memory=True)
            device_mem = torch.empty(size, dtype=torch_dtype, device=self.device)

            # data_ptr() returns the memory address, equivalent to int(pycuda_mem)
            bindings.append(device_mem.data_ptr())

            buffer_info = {
                'name': binding,
                'host': host_mem,
                'device': device_mem,
                'shape': shape,
                'dtype': dtype,
                'np_dtype': np_dtype,
                'torch_dtype': torch_dtype,
                'size': size
            }

            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append(buffer_info)
            else:
                outputs.append(buffer_info)

        return inputs, outputs, bindings

    def enable_cuda_graph(self, force=False):
        """
        Enable CUDA Graph functionality

        Args:
            force (bool): Force enable even if compatibility issues detected

        Returns:
            bool: Whether successfully enabled
        """
        if not self._graph_supported and not force:
            print('CUDA Graph not supported (requires CUDA 10.0+)')
            return False

        try:
            cuda_version = torch.version.cuda
            if cuda_version:
                major = int(cuda_version.split('.')[0])
                if major >= 10 or force:
                    self._graph_supported = True
                    self._use_cuda_graph = True
                    self._use_cupy_graph = True  # Compatibility flag
                    self._graph_captured = False
                    self._graph = None
                    print(f'CUDA Graph enabled (CUDA {cuda_version})')
                    return True
            print(f'CUDA version too low, requires 10.0+')
            return False
        except Exception as e:
            print(f'Failed to enable CUDA Graph: {e}')
            return False

    def disable_cuda_graph(self):
        """Disable CUDA Graph functionality"""
        self._graph_supported = False
        self._use_cuda_graph = False
        self._use_cupy_graph = False
        self._graph = None
        self._graph_exec = None
        self._graph_captured = False
        print('CUDA Graph disabled')

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """Explicit CUDA resource cleanup"""
        if hasattr(self, '_graph') and self._graph is not None:
            try:
                self._graph = None
            except Exception:
                pass
        if hasattr(self, '_cupy_stream') and self._cupy_stream is not None:
            try:
                # Compatibility: was cupy stream, now just None
                self._cupy_stream = None
            except Exception:
                pass
        if hasattr(self, '_graph_exec') and self._graph_exec is not None:
            try:
                self._graph_exec = None
            except Exception:
                pass
        # PyTorch handles CUDA context automatically, no ctx.pop() needed

    def allocate_buffers(self):
        """Backward compatibility interface"""
        return self.inputs, self.outputs, self.bindings, self.stream

    # =========================================================================
    # CORE TENSORRT EXECUTION
    # =========================================================================

    def _execute_inference(self):
        """Execute TensorRT inference (for CUDA Graph capture)"""
        try:
            for inp in self.inputs:
                self.context.set_tensor_address(inp['name'], inp['device'].data_ptr())
            for out in self.outputs:
                self.context.set_tensor_address(out['name'], out['device'].data_ptr())
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        except AttributeError:
            # Fallback for older TensorRT versions
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.cuda_stream)

    # =========================================================================
    # LEGACY INFERENCE (CPU PREPROCESSING)
    # - Accepts preprocessed numpy array from read_img()
    # - Used by ONNX fallback path in aiming.py
    # =========================================================================

    def infer(self, input_array):
        """
        Execute model inference

        Args:
            input_array: Input data array (numpy)

        Returns:
            list: Inference results
        """
        # Prepare input data
        input_flat = input_array.ravel()
        np_dtype = self.inputs[0]['np_dtype']
        if input_flat.dtype != np_dtype:
            input_flat = input_flat.astype(np_dtype)

        # Copy to pinned host memory (equivalent to np.copyto)
        input_tensor = torch.from_numpy(input_flat)
        self.inputs[0]['host'][:len(input_flat)].copy_(input_tensor)

        # Check if using CUDA Graph
        use_graph = (
            self._use_cuda_graph and
            self._graph_supported and
            self._use_cupy_graph
        )

        if use_graph:
            try:
                with torch.cuda.stream(self.stream):
                    if not self._graph_captured:
                        # Warmup run before capture (required)
                        self.inputs[0]['device'].copy_(self.inputs[0]['host'], non_blocking=True)
                        self._execute_inference()
                        self.outputs[0]['host'].copy_(self.outputs[0]['device'], non_blocking=True)
                        self.stream.synchronize()
                        self._graph_warmed = True

                        # Capture CUDA Graph
                        self._graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(self._graph, stream=self.stream):
                            self.inputs[0]['device'].copy_(self.inputs[0]['host'], non_blocking=True)
                            self._execute_inference()
                            self.outputs[0]['host'].copy_(self.outputs[0]['device'], non_blocking=True)

                        self._graph_captured = True
                        print('[TRT] CUDA Graph captured successfully')

                    # Replay the graph
                    # First update input (copy is part of graph, but we need fresh data in host)
                    self._graph.replay()
                    self.stream.synchronize()

                return [self.outputs[0]['host'].numpy()]

            except Exception as e:
                print(f'[TRT] CUDA Graph execution failed, falling back to regular path: {e}')
                self._use_cuda_graph = False
                self._graph_supported = False
                self._graph_captured = False

        # Regular inference path (no CUDA Graph)
        with torch.cuda.stream(self.stream):
            # Host -> Device (equivalent to cuda.memcpy_htod_async)
            self.inputs[0]['device'].copy_(self.inputs[0]['host'], non_blocking=True)

            # Execute TensorRT inference
            self._execute_inference()

            # Device -> Host (equivalent to cuda.memcpy_dtoh_async)
            self.outputs[0]['host'].copy_(self.outputs[0]['device'], non_blocking=True)

            self.stream.synchronize()

        return [self.outputs[0]['host'].numpy()]

    # =========================================================================
    # MODEL INFO
    # =========================================================================

    def get_input_shape(self):
        """Get model input shape"""
        if self.inputs:
            return self.inputs[0]['shape']
        binding = self.engine[0]
        return self.engine.get_tensor_shape(binding)

    def get_class_num(self):
        """Get model class count (YOLOv5 format)"""
        if self.outputs:
            return self.outputs[0]['shape'][-1] - 5
        binding = self.engine[1]
        return self.engine.get_tensor_shape(binding)[-1] - 5

    def get_class_num_v8(self):
        """Get YOLOv8 model class count"""
        if self.outputs:
            return self.outputs[0]['shape'][-1] - 4
        binding = self.engine[1]
        return self.engine.get_tensor_shape(binding)[-1] - 4

    # =========================================================================
    # GPU PREPROCESSING WITH INFERENCE - ENTRY POINT
    # =========================================================================

    def infer_with_preprocess(self, img_bgr: np.ndarray, target_size: tuple = None):
        """
        Execute inference with GPU-based preprocessing.

        Combines preprocessing and inference, enabling CUDA Graph capture
        of the entire pipeline for maximum performance.

        Args:
            img_bgr: Input BGR image as numpy array (H, W, 3) uint8
            target_size: Target size as (width, height). If None, uses model input shape.

        Returns:
            list: Inference results
        """
        if target_size is None:
            shape = self.get_input_shape()
            target_size = (shape[3], shape[2])  # (width, height)

        # Check if using CUDA Graph
        use_graph = (
            self._use_cuda_graph and
            self._graph_supported and
            self._use_cupy_graph
        )

        if use_graph:
            return self._infer_with_preprocess_graph(img_bgr, target_size)
        else:
            return self._infer_with_preprocess_regular(img_bgr, target_size)

    # =========================================================================
    # REGULAR INFERENCE PATH (NO CUDA GRAPH)
    # - Works on all CUDA devices
    # - GPU preprocessing + TensorRT inference
    # - Slightly higher kernel launch overhead
    # =========================================================================

    def _infer_with_preprocess_regular(self, img_bgr: np.ndarray, target_size: tuple):
        """
        Regular inference path with GPU preprocessing (no CUDA Graph).

        Args:
            img_bgr: Input BGR image as numpy array (H, W, 3) uint8
            target_size: Target size as (width, height)

        Returns:
            list: Inference results
        """
        with torch.cuda.stream(self.stream):
            # GPU preprocessing
            preprocessed = self._preprocess_gpu(img_bgr, target_size)

            # Copy preprocessed tensor to device input buffer (flatten)
            self.inputs[0]['device'].copy_(preprocessed.view(-1), non_blocking=True)

            # Execute TensorRT inference
            self._execute_inference()

            # Device -> Host
            self.outputs[0]['host'].copy_(self.outputs[0]['device'], non_blocking=True)

            self.stream.synchronize()

        return [self.outputs[0]['host'].numpy()]

    def _preprocess_gpu(self, img_bgr: np.ndarray, target_size: tuple) -> torch.Tensor:
        """
        GPU-based image preprocessing for regular path.

        Performs: transfer uint8 -> normalize -> BGR->RGB -> resize

        Args:
            img_bgr: Input BGR image as numpy array (H, W, 3) uint8
            target_size: Target size as (width, height)

        Returns:
            Preprocessed tensor on GPU device (1, 3, H, W) float32
        """
        target_w, target_h = target_size

        # Transfer to GPU as uint8 (4x smaller than float32)
        img_tensor = torch.from_numpy(img_bgr).to(
            device=self.device,
            dtype=torch.uint8,
            non_blocking=True
        )

        # Normalize: [0,255] -> [0,1]
        img_float = img_tensor.float().div_(255.0)

        # BGR -> RGB using flip
        img_rgb = img_float[..., [2, 1, 0]]

        # HWC -> CHW -> NCHW
        img_chw = img_rgb.permute(2, 0, 1).unsqueeze(0)

        # Resize if needed
        if img_chw.shape[2] != target_h or img_chw.shape[3] != target_w:
            img_chw = torch.nn.functional.interpolate(
                img_chw,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )

        return img_chw.contiguous()

    # =========================================================================
    # CUDA GRAPH INFERENCE PATH
    # - Requires CUDA 10.0+
    # - Captures preprocessing + inference in single graph
    # - Minimal kernel launch overhead (graph replay)
    # - Fixed input size after first capture
    # =========================================================================

    def _infer_with_preprocess_graph(self, img_bgr: np.ndarray, target_size: tuple):
        """
        CUDA Graph inference path with GPU preprocessing.
        Captures preprocessing + inference in a single graph for maximum performance.

        Args:
            img_bgr: Input BGR image as numpy array (H, W, 3) uint8
            target_size: Target size as (width, height)

        Returns:
            list: Inference results
        """
        try:
            with torch.cuda.stream(self.stream):
                if not self._graph_captured:
                    # Allocate fixed input tensor (CUDA Graph requires fixed memory addresses)
                    self._preprocess_input = torch.empty(
                        img_bgr.shape, dtype=torch.uint8, device=self.device
                    )
                    self._preprocess_target_size = target_size

                    # Warmup run before capture
                    self._preprocess_input.copy_(
                        torch.from_numpy(img_bgr).to(self.device, non_blocking=True)
                    )
                    preprocessed = self._graph_preprocess_ops(target_size)
                    self.inputs[0]['device'].copy_(preprocessed.view(-1), non_blocking=True)
                    self._execute_inference()
                    self.outputs[0]['host'].copy_(self.outputs[0]['device'], non_blocking=True)
                    self.stream.synchronize()
                    self._graph_warmed = True

                    # Capture CUDA Graph
                    self._graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self._graph, stream=self.stream):
                        preprocessed = self._graph_preprocess_ops(target_size)
                        self.inputs[0]['device'].copy_(preprocessed.view(-1), non_blocking=True)
                        self._execute_inference()
                        self.outputs[0]['host'].copy_(self.outputs[0]['device'], non_blocking=True)

                    self._graph_captured = True
                    print('[TRT] CUDA Graph with GPU preprocessing captured successfully')

                # Update input tensor and replay graph
                self._preprocess_input.copy_(
                    torch.from_numpy(img_bgr).to(self.device, non_blocking=True)
                )
                self._graph.replay()
                self.stream.synchronize()

            return [self.outputs[0]['host'].numpy()]

        except Exception as e:
            print(f'[TRT] CUDA Graph failed, falling back to regular path: {e}')
            self._use_cuda_graph = False
            self._graph_captured = False
            self._graph = None

            # Reset CUDA state after failed graph capture
            try:
                torch.cuda.synchronize()
                self.stream = torch.cuda.Stream(device=self.device)
            except Exception:
                pass

            return self._infer_with_preprocess_regular(img_bgr, target_size)

    def _graph_preprocess_ops(self, target_size: tuple) -> torch.Tensor:
        """
        Preprocessing operations for CUDA Graph capture.
        Uses fixed input tensor (_preprocess_input) for graph compatibility.

        Note: Uses flip() instead of fancy indexing [:,:,[2,1,0]] for graph compatibility.

        Args:
            target_size: Target size as (width, height)

        Returns:
            Preprocessed tensor (1, 3, H, W) float32
        """
        target_w, target_h = target_size

        # Normalize: [0,255] -> [0,1]
        img_float = self._preprocess_input.float().div_(255.0)

        # BGR -> RGB using flip (CUDA Graph compatible)
        img_rgb = torch.flip(img_float, dims=[-1])

        # HWC -> CHW -> NCHW
        img_chw = img_rgb.permute(2, 0, 1).unsqueeze(0)

        # Resize if needed
        if img_chw.shape[2] != target_h or img_chw.shape[3] != target_w:
            img_chw = torch.nn.functional.interpolate(
                img_chw,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )

        return img_chw.contiguous()


def auto_convert_engine(model_path):
    """
    Automatically convert ONNX model to TensorRT engine

    Args:
        model_path: ONNX model path

    Returns:
        bool: Whether conversion succeeded
    """
    print(f'Starting TRT conversion, model path: {model_path}')
    try:
        import tensorrt as trt
        import onnxruntime as ort
        print(f'TensorRT version: {trt.__version__}')

        if not torch.cuda.is_available():
            print('CUDA not available')
            return False
        print(f'CUDA available, GPU: {torch.cuda.get_device_name(0)}')
    except ImportError as e:
        print(f'TensorRT environment not installed: {e}')
        print('Please install:')
        print('1. CUDA Toolkit')
        print('2. cuDNN')
        print('3. TensorRT')
        print('4. PyTorch with CUDA')
        print('5. ONNX Runtime')
        return False

    base_path = os.path.splitext(model_path)[0]
    print(f'Base path: {base_path}')

    if os.path.exists(base_path + '.onnx'):
        onnx_path = base_path + '.onnx'
        print(f'Found ONNX model: {onnx_path}')
    elif os.path.exists(base_path + '.data'):
        onnx_path = base_path + '.data'
        print(f'Found DATA model: {onnx_path}')
    else:
        print(f'Original model not found: {base_path}.onnx or {base_path}.data')
        return False

    try:
        print('Validating model file...')
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider'] if 'DmlExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        print(f'Using providers: {providers}')
        sess = ort.InferenceSession(onnx_path, providers=providers)
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        print(f'Model input: name={input_name}, shape={input_shape}')
    except Exception as e:
        print(f'Model validation failed: {e}')
        return False

    engine_path = base_path + '.engine'
    print(f'Target engine path: {engine_path}')

    # Use existing auto_convert_engine_from_memory
    with open(onnx_path, 'rb') as f:
        onnx_bytes = f.read()

    success, final_path = auto_convert_engine_from_memory(
        model_bytes=onnx_bytes,
        output_engine_path=engine_path,
        use_fp16=True
    )

    if success:
        print('TRT conversion complete')
    return success


def auto_convert_engine_from_memory(
    model_bytes,
    output_engine_path,
    target_hw=None,
    use_fp16=True,
    workspace_mb=1024,
    builder_optimization_level: int = 5,
):
    """
    Convert ONNX model from memory to TensorRT engine

    Args:
        model_bytes: bytes, ONNX model in memory
        output_engine_path: Final engine file path
        target_hw: (H, W); None to infer from model/default
        use_fp16: Enable FP16 if platform supports
        workspace_mb: Build max workspace (MB)
        builder_optimization_level: 0~5
    Returns:
        (success: bool, final_engine_path: str)
    """
    print("Starting TRT engine conversion from memory")
    print(f"Output path: {output_engine_path}")
    print(f"Target resolution: {target_hw}")
    print(f"Use FP16: {use_fp16}")
    print(f"Workspace: {workspace_mb}MB")
    print(f"Optimization level: {builder_optimization_level}")

    try:
        import tensorrt as trt
        import onnxruntime as ort

        if not torch.cuda.is_available():
            print("CUDA not available")
            return (False, output_engine_path)

        print(f"TensorRT version: {trt.__version__}")
        print(f"CUDA available, GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"TensorRT environment not installed: {e}")
        print("Please install:\n1. CUDA Toolkit\n2. cuDNN\n3. TensorRT\n4. PyTorch with CUDA\n5. ONNX Runtime")
        return (False, output_engine_path)

    base, ext = os.path.splitext(output_engine_path)
    final_engine_path = output_engine_path if ext else (output_engine_path + ".engine")
    print(f"Final engine path: {final_engine_path}")

    try:
        providers = (
            ["DmlExecutionProvider", "CPUExecutionProvider"]
            if "DmlExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        print(f"Using providers: {providers}")
        sess = ort.InferenceSession(model_bytes, providers=providers)
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        print(f"ONNX input: name={input_name}, shape={input_shape}")
        del sess
    except Exception as e:
        print(f"Model validation failed: {e}")
        return (False, final_engine_path)

    if target_hw and len(target_hw) == 2:
        H_W = (int(target_hw[0]), int(target_hw[1]))
        print(f"Using specified target size: {H_W}")
    else:
        try:
            h = int(input_shape[2]) if isinstance(input_shape[2], (int, np.integer)) else 640
            w = int(input_shape[3]) if isinstance(input_shape[3], (int, np.integer)) else 640
            H_W = (h, w)
            print(f"Inferred target size from ONNX: {H_W}")
        except Exception:
            H_W = (640, 640)
            print(f"Using default target size: {H_W}")

    print(f"Target input size: {H_W[0]}x{H_W[1]}")

    if os.path.exists(final_engine_path):
        print(f"Found existing engine: {final_engine_path}")
        return (True, final_engine_path)

    try:
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        print("Parsing ONNX model...")
        if not parser.parse(model_bytes):
            print("ONNX parsing failed:")
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            return (False, final_engine_path)

        print("Creating build config...")
        config = builder.create_builder_config()

        workspace_bytes = int(workspace_mb) * 1024 * 1024
        print(f"Setting workspace: {workspace_bytes} bytes")
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
            print("Using new API for workspace")
        except Exception:
            config.max_workspace_size = workspace_bytes
            print("Using legacy API for workspace")

        try:
            config.builder_optimization_level = max(0, min(5, int(builder_optimization_level)))
            print(f"Set optimization level: {config.builder_optimization_level}")
        except Exception:
            print("Cannot set optimization level")

        # Check FP16 support using torch
        fp16_enabled = False
        if use_fp16:
            try:
                capability = torch.cuda.get_device_capability(0)
                supports_fp16 = capability[0] > 6 or (capability[0] == 6 and capability[1] >= 0)
                if supports_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    fp16_enabled = True
                    print("FP16 enabled")
                else:
                    print(f"GPU does not support FP16 (compute capability: {capability[0]}.{capability[1]})")
            except Exception:
                print("Cannot enable FP16, continuing with FP32")

        input_tensor = network.get_input(0)
        dims = list(input_tensor.shape)
        print(f"Input tensor dims: {dims}")

        has_dynamic = any(d is None or (isinstance(d, int) and d < 0) for d in dims)
        if has_dynamic:
            print("Creating optimization profile for dynamic input")
            profile = builder.create_optimization_profile()
            min_shape = (1, 3, H_W[0], H_W[1])
            opt_shape = (1, 3, H_W[0], H_W[1])
            max_shape = (1, 3, H_W[0], H_W[1])
            print(f"Profile shapes - min: {min_shape}, opt: {opt_shape}, max: {max_shape}")
            try:
                profile.set_shape(input_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
                print("Using new API for shape")
            except TypeError:
                profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
                print("Using legacy API for shape")
            config.add_optimization_profile(profile)
            print("Optimization profile added")
        else:
            if len(dims) >= 4 and (dims[2], dims[3]) != H_W:
                try:
                    print(f"Adjusting static input shape {dims} -> [1,3,{H_W[0]},{H_W[1]}]")
                    input_tensor.shape = (1, 3, H_W[0], H_W[1])
                except Exception as e:
                    print(f"Static shape adjustment failed: {e}")

        print("Building engine...")
        engine_bytes = builder.build_serialized_network(network, config)
        if not engine_bytes:
            print("TensorRT engine build failed")
            return (False, final_engine_path)

        os.makedirs(os.path.dirname(final_engine_path) or ".", exist_ok=True)
        with open(final_engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"Saved: {final_engine_path}")

        with open(final_engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                print("Engine deserialization failed")
                return (False, final_engine_path)
            print(f"Engine IO tensor count: {engine.num_io_tensors}")

        print("Memory conversion complete")
        return (True, final_engine_path)

    except Exception as e:
        print(f"TensorRT engine conversion failed: {e}")
        traceback.print_exc()
        try:
            if os.path.exists(final_engine_path):
                os.remove(final_engine_path)
                print(f"Deleted incomplete engine: {final_engine_path}")
        except Exception:
            pass
        return (False, final_engine_path)


def ensure_engine_from_memory(model_bytes, base_output_engine_path, target_hw, use_fp16=True, workspace_mb=1024,
                              builder_optimization_level: int = 5):
    """
    Build engine from memory if not exists, return final engine path.

    Args:
        model_bytes: ONNX bytes in memory
        base_output_engine_path: Base output path
        target_hw: (H, W)
    Returns:
        final_engine_path: Built or reused engine path
    """
    ok, final_path = auto_convert_engine_from_memory(
        model_bytes=model_bytes,
        output_engine_path=base_output_engine_path,
        target_hw=target_hw,
        use_fp16=use_fp16,
        workspace_mb=workspace_mb,
        builder_optimization_level=builder_optimization_level
    )
    return final_path
