# -*- coding: utf-8 -*-

# Initialize CUDA context early (before ONNX imports) so providers are detected
try:
    import torch
    _ = torch.cuda.is_available()
except ImportError:
    pass

# -*- coding: utf-8 -*-
"""
Main entry point - preloads all static dependencies to validate at startup.
Dynamic imports (kmNet, TensorRT, numba, bettercam, etc.) are handled in their modules.
"""

# === Standard library ===
import copy
import ctypes
from ctypes import *
import gc
import glob
import json
import logging
import math
import os
import queue
from queue import Queue
import random
import socket
import string
import struct
import subprocess
import sys
import threading
from threading import Thread, Timer, Event
import time
import traceback
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import tkinter as tk
from tkinter import filedialog

# === Third-party: Core ===
import cv2
import numpy as np
import onnxruntime as rt
import requests

# === Third-party: GUI ===
import dearpygui.dearpygui as dpg

# === Third-party: Windows API ===
import win32api
import win32con
import win32gui

# === Third-party: Input ===
from pynput import keyboard, mouse
from pynput.keyboard import Key, KeyCode
import pydirectinput
from pyclick import HumanCurve
import serial
import serial.tools.list_ports

# === Third-party: Image processing ===
from PIL import Image

# === Third-party: Scientific ===
import scipy
from scipy.optimize import linear_sum_assignment
import filterpy
from filterpy.kalman import KalmanFilter

# === Third-party: Cryptography ===
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

# === Application entry point ===
from app import Aassist


def global_exception_hook(exctype, value, tb):
    """
    Global exception hook to catch all unhandled exceptions.
    """
    # Write detailed exception information to log file
    with open('error_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"[Global Exception] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(''.join(traceback.format_exception(exctype, value, tb)))
        f.write('\n')
    # Display a friendly error message to the user
    print('An unhandled exception occurred. Details have been written to error_log.txt. Please report this file to the developer.')

# Set custom global exception handler
sys.excepthook = global_exception_hook

def handle_tensorrt_error():
    """
    Handle TensorRT-related import or initialization errors.
    """
    # Print detailed guidance information to console
    print('\n==================================================')
    print('TensorRT-related error detected, but program will continue using ONNX inference')
    print('If you want to use TensorRT acceleration, please install the following components:')
    print('1. CUDA Toolkit')
    print('2. cuDNN')
    print('3. TensorRT')
    print('4. For configuration tutorial, visit: https://www.yuque.com/huiyestudio/dqrld3/rislyof9zegdfira')
    print('==================================================\n')

    # Record error information to file
    with open('tensorrt_error.txt', 'a', encoding='utf-8') as f:
        f.write(f"[TensorRT Not Installed] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write('Program will continue using ONNX inference\n\n')

# Main program entry point


if __name__ == '__main__':
    valorant_instance = None
    try:
        try:
            print('Starting program...')
            try:
                # Try to start main program
                valorant_instance = Aassist()
                valorant_instance.start()
            except ImportError as e:
                # Catch import errors, especially TensorRT-related ones
                error_str = str(e).lower()
                if 'tensorrt' in error_str or 'cuda' in error_str or 'nvinfer' in error_str:
                    handle_tensorrt_error()
                    valorant_instance = Aassist()
                    valorant_instance.start()
                else:
                    raise
            except OSError as e:
                # Catch OS errors, which may also be caused by missing GPU libraries
                error_str = str(e).lower()
                if 'nvinfer' in error_str or 'cudnn' in error_str or 'cuda' in error_str:
                    handle_tensorrt_error()
                    valorant_instance = Aassist()
                    valorant_instance.start()
                else:
                    raise
        except Exception as e:
            # Catch all other exceptions during startup
            with open('error_log.txt', 'w', encoding='utf-8') as f:
                f.write(traceback.format_exc())
            print('An error occurred. Details have been written to error_log.txt. Please report this file to the developer.')
            input('Press Enter to exit...')
    finally:
        # Ensure resource cleanup when program exits
        if valorant_instance is not None:
            try:
                print('Cleaning up program resources...')
                valorant_instance._secure_cleanup()
                print('Program resource cleanup completed')
            except Exception as e:
                # Handle exceptions that may occur during cleanup
                print(f"Error cleaning up program resources: {e}")
                with open('error_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"[Resource Cleanup Exception] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Cleanup error: {str(e)}\n")
                    f.write(traceback.format_exc())
                    f.write('\n')
