# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: cat\catnet_lite.py
# Bytecode version: 3.10.0rc2 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

"""\nCatNet单文件便携版\n\n将catnet项目重构为单个文件，实现完整的Cat盒子调用功能。\n支持打包为pyd便携格式。\n\n基于极简系统设计原则：\n- 单文件部署\n- 去除复杂架构组件\n- 保持API兼容性\n- 支持便携部署\n"""
import os
import struct
import socket
import threading
import time
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
CMD_CONNECT = 1
CMD_MONITOR = 2
CMD_MOUSE_BUTTON = 3
CMD_KEYBOARD_BUTTON = 4
CMD_BLOCKED = 5
CMD_UNBLOCKED_MOUSE_ALL = 6
CMD_UNBLOCKED_KEYBOARD_ALL = 7
CMD_MOUSE_MOVE = 8
CMD_MOUSE_AUTO_MOVE = 9
AES_BLOCK_SIZE = 16
KEY_RESERVED = 0
KEY_ESC = 1
KEY_1 = 2
KEY_2 = 3
KEY_3 = 4
KEY_4 = 5
KEY_5 = 6
KEY_6 = 7
KEY_7 = 8
KEY_8 = 9
KEY_9 = 10
KEY_0 = 11
KEY_MINUS = 12
KEY_EQUAL = 13
KEY_BACKSPACE = 14
KEY_TAB = 15
KEY_Q = 16
KEY_W = 17
KEY_E = 18
KEY_R = 19
KEY_T = 20
KEY_Y = 21
KEY_U = 22
KEY_I = 23
KEY_O = 24
KEY_P = 25
KEY_LEFTBRACE = 26
KEY_RIGHTBRACE = 27
KEY_ENTER = 28
KEY_LEFTCTRL = 29
KEY_A = 30
KEY_S = 31
KEY_D = 32
KEY_F = 33
KEY_G = 34
KEY_H = 35
KEY_J = 36
KEY_K = 37
KEY_L = 38
KEY_SEMICOLON = 39
KEY_APOSTROPHE = 40
KEY_GRAVE = 41
KEY_LEFTSHIFT = 42
KEY_BACKSLASH = 43
KEY_Z = 44
KEY_X = 45
KEY_C = 46
KEY_V = 47
KEY_B = 48
KEY_N = 49
KEY_M = 50
KEY_COMMA = 51
KEY_DOT = 52
KEY_SLASH = 53
KEY_RIGHTSHIFT = 54
KEY_KPASTERISK = 55
KEY_LEFTALT = 56
KEY_SPACE = 57
KEY_CAPSLOCK = 58
KEY_F1 = 59
KEY_F2 = 60
KEY_F3 = 61
KEY_F4 = 62
KEY_F5 = 63
KEY_F6 = 64
KEY_F7 = 65
KEY_F8 = 66
KEY_F9 = 67
KEY_F10 = 68
KEY_NUMLOCK = 69
KEY_SCROLLLOCK = 70
BTN_LEFT = 272
BTN_RIGHT = 273
BTN_MIDDLE = 274
BTN_SIDE = 275
BTN_EXTRA = 276
KEY_CNT = 1024

class ErrorCode(IntEnum):
    """\n    表示NET操作的错误代码\n    \n    枚举类型用于描述不同类型的网络操作错误及其状态码。\n    """
    SUCCESS = 0
    DECRYPTION_FAILED = 100
    ENCRYPTION_FAILED = 101
    SEND_FAILED = 102
    RECEIVE_FAILED = 103
    RECEIVE_TIMEOUT = 104
    INIT_FAILED = 300
    MONITOR_CLOSE = 301
    MONITOR_OPEN = 302
    SOCKET_FAILED = 500
    SOCKET_TIMEOUT = 501

@dataclass
class CmdData:
    """Cat协议命令数据结构"""
    cmd: int = 0
    options: int = 0
    value1: int = 0
    value2: int = 0

    def pack(self) -> bytes:
        """打包为字节数据"""  # inserted
        return struct.pack('<BHHH', self.cmd & 255, self.options & 65535, self.value1 & 65535, self.value2 & 65535)

    @classmethod
    def unpack(cls, data: bytes) -> 'CmdData':
        """从字节数据解包"""  # inserted
        if len(data) < 7:
            raise ValueError('数据长度不足，无法解包CmdData')
        cmd, options, value1, value2 = struct.unpack('<BHHH', data[:7])
        value1 = value1 if value1 < 32768 else value1 - 65536
        value2 = value2 if value2 < 32768 else value2 - 65536
        return cls(cmd, options, value1, value2)

@dataclass
class MouseEvent:
    """鼠标事件数据"""
    x: int = 0
    y: int = 0
    wheel: int = 0

    def pack(self) -> bytes:
        """打包为字节数据"""  # inserted
        return struct.pack('<hhh', self.x, self.y, self.wheel)

    @classmethod
    def unpack(cls, data: bytes) -> 'MouseEvent':
        """从字节数据解包"""  # inserted
        if len(data) < 6:
            raise ValueError('数据长度不足，无法解包MouseEvent')
        x, y, wheel = struct.unpack('<hhh', data[:6])
        return cls(x, y, wheel)

@dataclass
class MouseData:
    """鼠标数据结构"""
    code: int = 0
    value: int = 0
    mouse_event: Optional[MouseEvent] = None

    def __post_init__(self):
        """初始化后处理"""  # inserted
        if self.mouse_event is None:
            self.mouse_event = MouseEvent()

    def pack(self) -> bytes:
        """打包为字节数据"""  # inserted
        mouse_bytes = self.mouse_event.pack()
        return struct.pack('<HH', self.code, self.value) + mouse_bytes

    @classmethod
    def unpack(cls, data: bytes) -> 'MouseData':
        """从字节数据解包"""  # inserted
        if len(data) < 10:
            raise ValueError('数据长度不足，无法解包MouseData')
        code, value = struct.unpack('<HH', data[:4])
        mouse_event = MouseEvent.unpack(data[4:10])
        return cls(code, value, mouse_event)

@dataclass
class KeyboardData:
    """键盘数据结构"""
    code: int = 0
    value: int = 0
    lock: int = 0

    def pack(self) -> bytes:
        """打包为字节数据"""  # inserted
        return struct.pack('<HHB', self.code, self.value, self.lock)

    @classmethod
    def unpack(cls, data: bytes) -> 'KeyboardData':
        """从字节数据解包"""  # inserted
        if len(data) < 5:
            raise ValueError('数据长度不足，无法解包KeyboardData')
        code, value, lock = struct.unpack('<HHB', data[:5])
        return cls(code, value, lock)

@dataclass
class HidData:
    """HID设备数据结构"""
    mouse_data: Optional[MouseData] = None
    keyboard_data: Optional[KeyboardData] = None

    def __post_init__(self):
        """初始化后处理"""  # inserted
        if self.mouse_data is None:
            self.mouse_data = MouseData()
        if self.keyboard_data is None:
            self.keyboard_data = KeyboardData()

    def pack(self) -> bytes:
        """打包为HID数据"""  # inserted
        mouse_bytes = self.mouse_data.pack()
        keyboard_bytes = self.keyboard_data.pack()
        return mouse_bytes + keyboard_bytes

    @classmethod
    def unpack(cls, data: bytes) -> 'HidData':
        """从字节数据解包HID数据"""  # inserted
        if len(data) < 15:
            raise ValueError('数据长度不足，无法解包HidData')
        mouse_data = MouseData.unpack(data[:10])
        keyboard_data = KeyboardData.unpack(data[10:15])
        return cls(mouse_data, keyboard_data)

class CryptoManager:
    """AES加密解密管理器"""

    def __init__(self, uuid: str):
        """\n        初始化加密管理器\n        \n        Args:\n            uuid: 盒子UUID（十六进制字符串）\n        """  # inserted
        self._key = self._expand_uuid_to_key(uuid)
        self._backend = default_backend()

    def _expand_uuid_to_key(self, uuid: str) -> bytes:
        """\n        将UUID扩展为16字节密钥\n        \n        这个方法完全复制C++版本的逻辑:\n        1. 将十六进制字符串转换为整数\n        2. 格式化为8位十六进制字符串\n        3. 扩展为16字节\n        \n        Args:\n            uuid: 十六进制UUID字符串\n            \n        Returns:\n            16字节的AES密钥\n        """  # inserted
        try:
            uuid_int = int(uuid, 16)
            uuid_str = f'{uuid_int:08x}'
            key_str = uuid_str.ljust(16, '0')[:16]
            return key_str.encode('ascii')
        except ValueError as e:
            raise RuntimeError(f'无效的UUID格式: {uuid}') from e

    def _generate_random_iv(self) -> bytes:
        """生成随机IV"""  # inserted
        return os.urandom(AES_BLOCK_SIZE)

    def encrypt(self, plaintext: bytes) -> bytes:
        """\n        AES-CBC加密\n        \n        Args:\n            plaintext: 明文数据\n            \n        Returns:\n            IV + 密文的组合\n            \n        Raises:\n            RuntimeError: 加密失败\n        """  # inserted
        try:
            iv = self._generate_random_iv()
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(plaintext)
            padded_data += padder.finalize()
            cipher = Cipher(algorithms.AES(self._key), modes.CBC(iv), backend=self._backend)
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            return iv + ciphertext
        except Exception as e:
            raise RuntimeError(f'加密失败: {str(e)}') from e

    def decrypt(self, ciphertext_with_iv: bytes) -> bytes:
        """\n        AES-CBC解密\n        \n        Args:\n            ciphertext_with_iv: IV + 密文的组合\n            \n        Returns:\n            解密后的明文\n            \n        Raises:\n            RuntimeError: 解密失败\n        """  # inserted
        try:
            if len(ciphertext_with_iv) < AES_BLOCK_SIZE * 2:
                raise ValueError('加密数据长度不足')
            iv = ciphertext_with_iv[:AES_BLOCK_SIZE]
            ciphertext = ciphertext_with_iv[AES_BLOCK_SIZE:]
            if len(ciphertext) % AES_BLOCK_SIZE!= 0:
                raise ValueError('密文长度必须是16的倍数')
            cipher = Cipher(algorithms.AES(self._key), modes.CBC(iv), backend=self._backend)
            decryptor = cipher.decryptor()
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            unpadder = padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_plaintext)
            plaintext += unpadder.finalize()
            return plaintext
        except Exception as e:
            raise RuntimeError(f'解密失败: {str(e)}') from e

    def encrypt_command(self, cmd_data: CmdData) -> bytes:
        """\n        加密命令数据\n        \n        Args:\n            cmd_data: 命令数据对象\n            \n        Returns:\n            加密后的数据\n        """  # inserted
        cmd_bytes = cmd_data.pack()
        return self.encrypt(cmd_bytes)

    def decrypt_hid_data(self, encrypted_data: bytes) -> HidData:
        """\n        解密HID数据\n        \n        Args:\n            encrypted_data: 加密的HID数据\n            \n        Returns:\n            解密后的HID数据对象\n            \n        Raises:\n            RuntimeError: 解密失败或数据格式错误\n        """  # inserted
        try:
            decrypted_bytes = self.decrypt(encrypted_data)
            return HidData.unpack(decrypted_bytes)
        except Exception as e:
            raise RuntimeError(f'解密HID数据失败: {str(e)}') from e

    def decrypt_ack(self, encrypted_ack: bytes) -> CmdData:
        """\n        解密ACK响应\n        \n        Args:\n            encrypted_ack: 加密的ACK数据\n            \n        Returns:\n            解密后的命令数据对象\n            \n        Raises:\n            RuntimeError: 解密失败或数据格式错误\n        """  # inserted
        try:
            decrypted_bytes = self.decrypt(encrypted_ack)
            return CmdData.unpack(decrypted_bytes)
        except Exception as e:
            raise RuntimeError(f'解密ACK响应失败: {str(e)}') from e

    @property
    def key_hex(self) -> str:
        """获取密钥的十六进制表示（用于调试）"""  # inserted
        return self._key.hex()

class NetworkManager:
    """同步网络通信管理器"""

    def __init__(self):
        self._send_socket = None
        self._box_endpoint = None

    def init_connection(self, box_ip: str, box_port: int) -> ErrorCode:
        """\n        初始化发送连接\n        \n        Args:\n            box_ip: 盒子IP地址\n            box_port: 盒子端口\n            \n        Returns:\n            错误代码\n        """  # inserted
        try:
            self._box_endpoint = (box_ip, box_port)
            self._send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return ErrorCode.SUCCESS
        except Exception as e:
            print(f'初始化连接失败: {e}')
            return ErrorCode.SOCKET_FAILED

    def send_and_wait_response(self, data: bytes, timeout: float=5.0) -> Tuple[ErrorCode, Optional[bytes]]:
        """\n        发送数据并等待响应\n        \n        Args:\n            data: 要发送的数据\n            timeout: 等待超时时间\n            \n        Returns:\n            (错误代码, 响应数据)\n        """  # inserted
        if not self._send_socket or not self._box_endpoint:
            return (ErrorCode.SOCKET_FAILED, None)
        try:
            self._clear_receive_buffer()
            self._send_socket.sendto(data, self._box_endpoint)
            self._send_socket.settimeout(timeout)
            try:
                response_data, addr = self._send_socket.recvfrom(4096)
                return (ErrorCode.SUCCESS, response_data)
            except socket.timeout:
                return (ErrorCode.RECEIVE_TIMEOUT, None)
            finally:  # inserted
                if self._send_socket:
                    self._send_socket.settimeout(None)
        except Exception as e:
            print(f'发送接收失败: {e}')
            return (ErrorCode.SEND_FAILED, None)

    def _clear_receive_buffer(self) -> None:
        """清空接收缓冲区"""  # inserted
        if not self._send_socket:
            return
        self._send_socket.setblocking(False)
        try:
            while True:
                try:
                    data, addr = self._send_socket.recvfrom(4096)
                    if not data:
                        break
                except socket.error:
                    break
        finally:
            self._send_socket.setblocking(True)

    def close_connections(self) -> None:
        """关闭所有连接"""  # inserted
        if self._send_socket:
            self._send_socket.close()
            self._send_socket = None

    def is_connected(self) -> bool:
        """检查连接状态"""  # inserted
        return self._send_socket is not None

class StateManager:
    """状态管理器基类"""

    def __init__(self, max_keys: int=KEY_CNT):
        """\n        初始化状态管理器\n        \n        Args:\n            max_keys: 最大键数量\n        """  # inserted
        self._max_keys = max_keys
        self._key_states = [False] * max_keys
        self._lock = threading.Lock()

    def update_key_state(self, key: int, is_pressed: bool) -> None:
        """\n        更新按键状态\n        \n        Args:\n            key: 按键代码\n            is_pressed: 是否按下\n        """  # inserted
        with self._lock:
            if 0 <= key < self._max_keys:
                self._key_states[key] = is_pressed

    def is_key_pressed(self, key: int) -> bool:
        """\n        查询按键状态\n        \n        Args:\n            key: 按键代码\n            \n        Returns:\n            是否按下\n        """  # inserted
        with self._lock:
            if 0 <= key < self._max_keys:
                return self._key_states[key]
            return False

    def clear_all_states(self) -> None:
        """清除所有按键状态"""  # inserted
        with self._lock:
            self._key_states = [False] * self._max_keys

class MouseStateManager(StateManager):
    """鼠标状态管理器"""

    def __init__(self):
        super().__init__()
        self._mouse_event = MouseEvent(0, 0, 0)

    def update_mouse_event(self, x: int, y: int, wheel: int) -> None:
        """\n        更新鼠标事件\n        \n        Args:\n            x: X坐标偏移\n            y: Y坐标偏移\n            wheel: 滚轮偏移\n        """  # inserted
        with self._lock:
            self._mouse_event.x = x
            self._mouse_event.y = y
            self._mouse_event.wheel = wheel

    @property
    def mouse_event(self) -> MouseEvent:
        """获取鼠标事件"""  # inserted
        with self._lock:
            return MouseEvent(self._mouse_event.x, self._mouse_event.y, self._mouse_event.wheel)

class KeyboardStateManager(StateManager):
    """键盘状态管理器"""

    def __init__(self):
        super().__init__()
        self._lock_state = 0

    def update_lock_state(self, lock_state: int) -> None:
        """\n        更新锁定键状态\n        \n        Args:\n            lock_state: 锁定键状态位掩码\n        """  # inserted
        with self._lock:
            self._lock_state = lock_state

    def is_lock_key_pressed(self, key: int) -> bool:
        """\n        查询锁定键状态\n        \n        Args:\n            key: 锁定键代码\n            \n        Returns:\n            是否激活\n        """  # inserted
        with self._lock:
            return self._lock_state & key!= 0

class HidStateManager:
    """HID设备状态管理器"""

    def __init__(self):
        self.mouse_state = MouseStateManager()
        self.keyboard_state = KeyboardStateManager()
        self._update_lock = threading.Lock()

    def update_from_hid_data(self, hid_data: HidData) -> None:
        """\n        从HID数据更新状态\n        \n        Args:\n            hid_data: HID数据对象\n        """  # inserted
        with self._update_lock:
            mouse_data = hid_data.mouse_data
            if mouse_data:
                self.mouse_state.update_key_state(mouse_data.code, bool(mouse_data.value))
                if mouse_data.mouse_event:
                    self.mouse_state.update_mouse_event(mouse_data.mouse_event.x, mouse_data.mouse_event.y, mouse_data.mouse_event.wheel)
            keyboard_data = hid_data.keyboard_data
            if keyboard_data:
                self.keyboard_state.update_key_state(keyboard_data.code, bool(keyboard_data.value))
                self.keyboard_state.update_lock_state(keyboard_data.lock)

    def is_mouse_pressed(self, code: int) -> bool:
        """\n        查询鼠标按键状态\n        \n        Args:\n            code: 鼠标按键代码\n            \n        Returns:\n            是否按下\n        """  # inserted
        return self.mouse_state.is_key_pressed(code)

    def is_keyboard_pressed(self, code: int) -> bool:
        """\n        查询键盘按键状态\n        \n        Args:\n            code: 键盘按键代码\n            \n        Returns:\n            是否按下\n        """  # inserted
        return self.keyboard_state.is_key_pressed(code)

    def is_lock_key_pressed(self, code: int) -> bool:
        """\n        查询锁定键状态\n        \n        Args:\n            code: 锁定键代码\n            \n        Returns:\n            是否激活\n        """  # inserted
        return self.keyboard_state.is_lock_key_pressed(code)

    def clear_all_states(self) -> None:
        """清除所有状态"""  # inserted
        with self._update_lock:
            self.mouse_state.clear_all_states()
            self.keyboard_state.clear_all_states()
            self.keyboard_state.update_lock_state(0)
            self.mouse_state.update_mouse_event(0, 0, 0)

class CatNetLite:
    """Cat协议网络控制客户端单文件版"""

    def __init__(self):
        """初始化CatNetLite实例"""  # inserted
        self._is_init = False
        self._is_monitor = False
        self._crypto_manager = None
        self._network_manager = NetworkManager()
        self._state_manager = HidStateManager()
        self._monitor_thread = None
        self._monitor_socket = None
        self._stop_monitor = threading.Event()
        self._box_ip = None
        self._box_port = None
        self._server_port = None
    pass
    def init(self, box_ip: str, box_port: int, uuid: str, timeout_ms: int=5000) -> ErrorCode:
        """\n        初始化连接到Cat盒子\n        \n        Args:\n            box_ip: 盒子IP地址\n            box_port: 盒子端口\n            uuid: 盒子UUID（十六进制字符串）\n            timeout_ms: 超时时间（毫秒）\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        try:
            self._crypto_manager = CryptoManager(uuid)
            result = self._network_manager.init_connection(box_ip, box_port)
            if result!= ErrorCode.SUCCESS:
                return result
            cmd_data = CmdData(cmd=CMD_CONNECT)
            result = self._send_cmd(cmd_data, timeout_ms)
            if result!= ErrorCode.SUCCESS:
                return result
            self._box_ip = box_ip
            self._box_port = box_port
            self._is_init = True
            return ErrorCode.SUCCESS
        except Exception as e:
            print(f'初始化失败: {e}')
            return ErrorCode.INIT_FAILED

    def monitor(self, server_port: int, timeout_ms: int=5000) -> ErrorCode:
        """\n        开启键鼠事件监听\n        \n        Args:\n            server_port: 本机监听的端口\n            timeout_ms: 初始化盒子响应ACK超时时间（毫秒）\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._is_init:
            return ErrorCode.INIT_FAILED
        if self._is_monitor:
            return ErrorCode.MONITOR_OPEN
        try:
            self._server_port = server_port
            self._stop_monitor.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_worker, args=(server_port,), daemon=True)
            self._monitor_thread.start()
            time.sleep(0.1)
            cmd_data = CmdData(cmd=CMD_MONITOR, options=server_port)
            result = self._send_cmd(cmd_data, timeout_ms)
            if result!= ErrorCode.SUCCESS:
                self.close_monitor()
                return result
            self._is_monitor = True
            return ErrorCode.SUCCESS
        except Exception as e:
            print(f'监听启动失败: {e}')
            self.close_monitor()
            return ErrorCode.SOCKET_FAILED

    def close_monitor(self) -> None:
        """关闭键鼠事件监听"""  # inserted
        if self._is_monitor:
            self._stop_monitor.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2.0)
            if self._monitor_socket:
                try:
                    self._monitor_socket.close()
                except:
                    pass
                self._monitor_socket = None
            self._monitor_thread = None
            self._is_monitor = False

    def _monitor_worker(self, server_port: int) -> None:
        """监听工作线程"""  # inserted
        try:
            self._monitor_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._monitor_socket.settimeout(1.0)
            self._monitor_socket.bind(('0.0.0.0', server_port))
            buffer = bytearray(1024)
            while not self._stop_monitor.is_set():
                try:
                    length, addr = self._monitor_socket.recvfrom_into(buffer)
                    if length > 0:
                        self._handle_hid_data(buffer[:length])
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self._stop_monitor.is_set():
                        print(f'监听接收错误: {e}')
                    break
        except Exception as e:
            print(f'监听线程错误: {e}')
        finally:  # inserted
            if self._monitor_socket:
                try:
                    self._monitor_socket.close()
                except:
                    pass
                self._monitor_socket = None

    def _handle_hid_data(self, encrypted_data: bytes) -> None:
        """处理HID数据"""  # inserted
        if not self._crypto_manager:
            return
        try:
            hid_data = self._crypto_manager.decrypt_hid_data(encrypted_data)
            self._state_manager.update_from_hid_data(hid_data)
        except Exception as e:
            print(f'HID数据处理错误: {e}')

    def _send_cmd(self, cmd_data: CmdData, timeout_ms: int=300) -> ErrorCode:
        """\n        发送命令并等待ACK\n        \n        Args:\n            cmd_data: 命令数据\n            timeout_ms: 超时时间（毫秒）\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._crypto_manager:
            return ErrorCode.INIT_FAILED
        try:
            encrypted_data = self._crypto_manager.encrypt_command(cmd_data)
            timeout_sec = timeout_ms / 1000.0
            result, response_data = self._network_manager.send_and_wait_response(encrypted_data, timeout_sec)
            if result!= ErrorCode.SUCCESS:
                return result
            if not response_data:
                return ErrorCode.RECEIVE_FAILED
            try:
                ack_data = self._crypto_manager.decrypt_ack(response_data)
                if ack_data.cmd == cmd_data.cmd:
                    return ErrorCode.SUCCESS
                return ErrorCode.RECEIVE_FAILED
            except Exception:
                return ErrorCode.DECRYPTION_FAILED
        except Exception as e:
            print(f'发送命令失败: {e}')
            return ErrorCode.SEND_FAILED

    def mouse_move(self, x: int, y: int) -> ErrorCode:
        """\n        鼠标移动\n        \n        Args:\n            x: 正值向右\n            y: 正值向下\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._is_init:
            return ErrorCode.INIT_FAILED
        cmd_data = CmdData(cmd=CMD_MOUSE_MOVE, value1=x, value2=y)
        return self._send_cmd(cmd_data)

    def mouse_move_auto(self, x: int, y: int, ms: int) -> ErrorCode:
        """\n        鼠标算法优化移动\n        \n        Args:\n            x: 正值向右\n            y: 正值向下\n            ms: 移动时间\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._is_init:
            return ErrorCode.INIT_FAILED
        cmd_data = CmdData(cmd=CMD_MOUSE_AUTO_MOVE, options=ms, value1=x, value2=y)
        return self._send_cmd(cmd_data, ms + 300)

    def mouse_button(self, code: int, value: int) -> ErrorCode:
        """\n        鼠标按键触发\n        \n        Args:\n            code: 按键值，参考event-codes\n            value: 按下1 释放0\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._is_init:
            return ErrorCode.INIT_FAILED
        cmd_data = CmdData(cmd=CMD_MOUSE_BUTTON, options=code, value1=value)
        return self._send_cmd(cmd_data)

    def tap_mouse_button(self, code: int, ms: int) -> ErrorCode:
        """\n        鼠标按下多少ms后释放\n        \n        Args:\n            code: 按键值，参考event-codes\n            ms: 毫秒\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._is_init:
            return ErrorCode.INIT_FAILED
        result = self.mouse_button(code, 1)
        if result!= ErrorCode.SUCCESS:
            return result
        time.sleep(ms / 1000.0)
        return self.mouse_button(code, 0)

    def keyboard_button(self, code: int, value: int) -> ErrorCode:
        """\n        键盘按键触发\n        \n        Args:\n            code: 按键值，参考event-codes\n            value: 按下1 释放0\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._is_init:
            return ErrorCode.INIT_FAILED
        cmd_data = CmdData(cmd=CMD_KEYBOARD_BUTTON, options=code, value1=value)
        return self._send_cmd(cmd_data)

    def tap_keyboard_button(self, code: int, ms: int) -> ErrorCode:
        """\n        键盘按键按下多少ms后释放\n        \n        Args:\n            code: 按键值，参考event-codes\n            ms: 毫秒\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._is_init:
            return ErrorCode.INIT_FAILED
        result = self.keyboard_button(code, 1)
        if result!= ErrorCode.SUCCESS:
            return result
        time.sleep(ms / 1000.0)
        return self.keyboard_button(code, 0)

    def is_mouse_pressed(self, code: int) -> bool:
        """\n        监测鼠标某个键是否按下\n        \n        Args:\n            code: 按键值，参考event-codes\n        \n        Returns:\n            按下true 释放false\n        """  # inserted
        if not self._is_monitor:
            return False
        return self._state_manager.is_mouse_pressed(code)

    def is_keyboard_pressed(self, code: int) -> bool:
        """\n        监测键盘按键某个键是否按下\n        \n        Args:\n            code: 按键值，参考event-codes\n        \n        Returns:\n            按下true 释放false\n        """  # inserted
        if not self._is_monitor:
            return False
        return self._state_manager.is_keyboard_pressed(code)

    def is_lock_key_pressed(self, code: int) -> bool:
        """\n        监测锁定键是否按下\n        \n        Args:\n            code: NumLock CapsLock ScrLOCK\n        \n        Returns:\n            按下true 释放false\n        """  # inserted
        if not self._is_monitor:
            return False
        return self._state_manager.is_lock_key_pressed(code)

    def blocked_mouse(self, code: int, value: int) -> ErrorCode:
        """\n        屏蔽鼠标某个键\n        \n        Args:\n            code: 按键值，参考event-codes\n            value: 屏蔽1 解除屏蔽0\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._is_init:
            return ErrorCode.INIT_FAILED
        cmd_data = CmdData(cmd=CMD_BLOCKED, options=1, value1=code, value2=value)
        return self._send_cmd(cmd_data)

    def blocked_keyboard(self, code: int, value: int) -> ErrorCode:
        """\n        屏蔽键盘某个键\n        \n        Args:\n            code: 按键值，参考event-codes\n            value: 屏蔽1 解除屏蔽0\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._is_init:
            return ErrorCode.INIT_FAILED
        cmd_data = CmdData(cmd=CMD_BLOCKED, options=2, value1=code, value2=value)
        return self._send_cmd(cmd_data)

    def unblocked_mouse_all(self) -> ErrorCode:
        """\n        解除鼠标所有屏蔽按键\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._is_init:
            return ErrorCode.INIT_FAILED
        cmd_data = CmdData(cmd=CMD_UNBLOCKED_MOUSE_ALL)
        return self._send_cmd(cmd_data)

    def unblocked_keyboard_all(self) -> ErrorCode:
        """\n        解除键盘所有屏蔽按键\n        \n        Returns:\n            ErrorCode: 操作结果\n        """  # inserted
        if not self._is_init:
            return ErrorCode.INIT_FAILED
        cmd_data = CmdData(cmd=CMD_UNBLOCKED_KEYBOARD_ALL)
        return self._send_cmd(cmd_data)

    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""  # inserted
        return self._is_init

    @property
    def is_monitoring(self) -> bool:
        """检查是否正在监听"""  # inserted
        return self._is_monitor

    def get_connection_info(self) -> dict:
        """获取连接信息"""  # inserted
        return {'box_ip': self._box_ip, 'box_port': self._box_port, 'server_port': self._server_port, 'is_initialized': self._is_init, 'is_monitoring': self._is_monitor}

    def __del__(self):
        """析构函数，确保资源正确释放"""  # inserted
        try:
            self.close_monitor()
            if self._network_manager:
                self._network_manager.close_connections()
        except:
            return None

def test_catnet_lite_basic():
    """基本功能测试（不需要真实硬件）"""  # inserted
    print('开始CatNetLite基本功能测试...')
    cat = CatNetLite()
    assert not cat.is_initialized, '初始状态应该未初始化'
    assert not cat.is_monitoring, '初始状态应该未监听'
    result = cat.mouse_move(10, 10)
    assert result == ErrorCode.INIT_FAILED, '未初始化时操作应该失败'
    result = cat.monitor(1234)
    assert result == ErrorCode.INIT_FAILED, '未初始化时监听应该失败'
    assert not cat.is_mouse_pressed(BTN_LEFT), '未监听时状态查询应该返图False'
    assert not cat.is_keyboard_pressed(KEY_A), '未监听时状态查询应该返图False'
    print('✓ 基本功能测试通过')
    info = cat.get_connection_info()
    assert 'is_initialized' in info, '连接信息应该包含初始化状态'
    assert 'is_monitoring' in info, '连接信息应该包含监听状态'
    print('✓ 连接信息测试通过')
    print('CatNetLite基本功能测试完成！')

def test_data_structures():
    """测试数据结构"""  # inserted
    print('开始数据结构测试...')
    cmd = CmdData(1, 2, 100, (-50))
    cmd_bytes = cmd.pack()
    assert len(cmd_bytes) == 7, f'CmdData大小错误: 期望7, 实際{len(cmd_bytes)}'
    unpacked_cmd = CmdData.unpack(cmd_bytes)
    assert cmd == unpacked_cmd, 'CmdData打包解包不一致'
    mouse_event = MouseEvent((-50), 75, (-10))
    mouse_event_bytes = mouse_event.pack()
    assert len(mouse_event_bytes) == 6, f'MouseEvent大小错误: 期望6, 实際{len(mouse_event_bytes)}'
    unpacked_mouse_event = MouseEvent.unpack(mouse_event_bytes)
    assert mouse_event == unpacked_mouse_event, 'MouseEvent打包解包不一致'
    mouse_data = MouseData(BTN_LEFT, 1, mouse_event)
    keyboard_data = KeyboardData(KEY_A, 1, 4)
    hid_data = HidData(mouse_data, keyboard_data)
    hid_bytes = hid_data.pack()
    assert len(hid_bytes) == 15, f'HidData大小错误: 期望15, 实際{len(hid_bytes)}'
    unpacked_hid_data = HidData.unpack(hid_bytes)
    assert hid_data == unpacked_hid_data, 'HidData打包解包不一致'
    print('✓ 数据结构测试通过')

def test_crypto_manager():
    """测试加密管理器"""  # inserted
    print('开始加密管理器测试...')
    crypto = CryptoManager('ad60ecf0')
    print(f'生成的密钥: {crypto.key_hex}')
    test_data = b'Hello, Cat Protocol!'
    encrypted = crypto.encrypt(test_data)
    decrypted = crypto.decrypt(encrypted)
    assert decrypted == test_data, '基本加密解密测试失败'
    print('✓ 基本加密解密测试通过')
    cmd_data = CmdData(1, 2, 100, (-50))
    encrypted_cmd = crypto.encrypt_command(cmd_data)
    decrypted_cmd = crypto.decrypt_ack(encrypted_cmd)
    assert cmd_data == decrypted_cmd, '命令数据加密解密测试失败'
    print('✓ 命令数据加密解密测试通过')
    print('加密管理器测试完成！')

def test_state_managers():
    """测试状态管理器"""  # inserted
    print('开始状态管理器测试...')
    hid_manager = HidStateManager()
    mouse_event = MouseEvent(20, 30, (-2))
    mouse_data = MouseData(BTN_LEFT, 1, mouse_event)
    keyboard_data = KeyboardData(KEY_A, 1, 3)
    hid_data = HidData(mouse_data, keyboard_data)
    hid_manager.update_from_hid_data(hid_data)
    assert hid_manager.is_mouse_pressed(BTN_LEFT), '鼠标左键应该按下'
    assert hid_manager.is_keyboard_pressed(KEY_A), 'A键应该按下'
    assert hid_manager.is_lock_key_pressed(1), 'CAPS LOCK应该激活'
    assert hid_manager.is_lock_key_pressed(2), 'NUM LOCK应该激活'
    print('✓ HID状态管理器测试通过')
    hid_manager.clear_all_states()
    assert not hid_manager.is_mouse_pressed(BTN_LEFT), '清除后鼠标左键应该未按下'
    assert not hid_manager.is_keyboard_pressed(KEY_A), '清除后A键应该未按下'
    assert not hid_manager.is_lock_key_pressed(1), '清除后锁定键应该未激活'
    print('✓ 状态清除测试通过')
    print('状态管理器测试完成！')

def run_all_tests():
    """运行所有测试"""  # inserted
    print('==================================================')
    print('CatNetLite 单文件便携版 - 全面测试')
    print('==================================================')
    try:
        test_data_structures()
        print()
        test_crypto_manager()
        print()
        test_state_managers()
        print()
        test_catnet_lite_basic()
        print()
        print('==================================================')
        print('✓ 所有测试通过！CatNetLite单文件便携版准备就绪！')
        print('==================================================')
    except Exception as e:
        print(f'✗ 测试失败: {str(e)}')
        raise
if __name__ == '__main__':
    run_all_tests()
