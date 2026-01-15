"""
CatBox 协议 Python 包装器
用于与 cat协议 C++ 库进行交互
"""
import os
import sys
from ctypes import *
from catnet_lite import CatNetLite, ErrorCode, BTN_LEFT, BTN_RIGHT, BTN_MIDDLE, BTN_SIDE, BTN_EXTRA

class CatBoxWrapper:
    """CatBox协议包装器类"""

    def __init__(self):
        self.process = None
        self.is_connected = False
        self.config = {'ip': '192.168.7.1', 'port': 8888, 'uuid': 'ad60ecf0'}
        self.cat = None
        self.cat_exe_path = os.path.join(sys._MEIPASS, 'cat.exe') if hasattr(sys, '_MEIPASS') else './cat协议/x64/Release/cat.exe'

    def init(self, ip, port, uuid):
        """
        初始化catbox连接
        Args:
            ip: catbox设备IP地址
            port: catbox设备端口
            uuid: catbox设备UUID
        Returns:
            bool: 初始化是否成功
        """
        self.config['ip'] = ip
        self.config['port'] = port
        self.config['uuid'] = uuid
        try:
            self.cat = CatNetLite()
            print(f'CatBox: 尝试连接到 {ip}:{port}, UUID: {uuid}')
            result = self.cat.init(ip, port, uuid, 5000)
            if result == ErrorCode.SUCCESS:
                self.is_connected = True
                print('CatBox: 连接成功')
                return True
            self.is_connected = False
            print(f'CatBox: 连接失败, 错误码: {int(result)}')
            return False
        except Exception as e:
            print(f'CatBox: 初始化失败: {e}')
            self.is_connected = False
            return False

    def move(self, x, y):
        """
        鼠标移动
        Args:
            x: X轴移动距离 (正值向右)
            y: Y轴移动距离 (正值向下)
        """
        if not self.is_connected:
            print('CatBox: 设备未连接')
            return
        try:
            if self.cat is not None:
                self.cat.mouse_move(int(x), int(y))
        except Exception as e:
            print(f'CatBox: 移动失败: {e}')

    def left_down(self):
        """鼠标左键按下"""
        if not self.is_connected:
            return
        try:
            if self.cat is not None:
                self.cat.mouse_button(BTN_LEFT, 1)
        except Exception as e:
            print(f'CatBox: 左键按下失败: {e}')

    def left_up(self):
        """鼠标左键释放"""
        if not self.is_connected:
            return
        try:
            if self.cat is not None:
                self.cat.mouse_button(BTN_LEFT, 0)
        except Exception as e:
            print(f'CatBox: 左键释放失败: {e}')

    def right_down(self):
        """鼠标右键按下"""
        if not self.is_connected:
            return
        try:
            if self.cat is not None:
                self.cat.mouse_button(BTN_RIGHT, 1)
        except Exception as e:
            print(f'CatBox: 右键按下失败: {e}')

    def right_up(self):
        """鼠标右键释放"""
        if not self.is_connected:
            return
        try:
            if self.cat is not None:
                self.cat.mouse_button(BTN_RIGHT, 0)
        except Exception as e:
            print(f'CatBox: 右键释放失败: {e}')

    def middle_down(self):
        """鼠标中键按下"""
        if not self.is_connected:
            return
        try:
            if self.cat is not None:
                self.cat.mouse_button(BTN_MIDDLE, 1)
        except Exception as e:
            print(f'CatBox: 中键按下失败: {e}')

    def middle_up(self):
        """鼠标中键释放"""
        if not self.is_connected:
            return
        try:
            if self.cat is not None:
                self.cat.mouse_button(BTN_MIDDLE, 0)
        except Exception as e:
            print(f'CatBox: 中键释放失败: {e}')

    def _send_move_command(self, x, y):
        """
        发送移动命令 (内部方法)
        这里需要实现真正的UDP通信或DLL调用
        """
        if abs(x) > 0 or abs(y) > 0:
            print(f'CatBox: 移动 ({x}, {y})')

    def _send_mouse_command(self, button, action):
        """
        发送鼠标按键命令 (内部方法)
        Args:
            button: 按键 (1=左键, 2=右键, 3=中键)
            action: 动作 (1=按下, 0=释放)
        """
        action_str = '按下' if action == 1 else '释放'
        button_str = {1: '左键', 2: '右键', 3: '中键'}.get(button, '未知')
        print(f'CatBox: {button_str}{action_str}')

    def mask_left(self, enable):
        """屏蔽/解除屏蔽左键"""
        if not self.is_connected:
            return
        try:
            if self.cat is not None:
                self.cat.blocked_mouse(BTN_LEFT, 1 if enable else 0)
        except Exception as e:
            print(f'CatBox: 屏蔽左键失败: {e}')

    def mask_right(self, enable):
        """屏蔽/解除屏蔽右键"""
        if not self.is_connected:
            return
        try:
            if self.cat is not None:
                self.cat.blocked_mouse(BTN_RIGHT, 1 if enable else 0)
        except Exception as e:
            print(f'CatBox: 屏蔽右键失败: {e}')

    def mask_middle(self, enable):
        """屏蔽/解除屏蔽中键"""
        if not self.is_connected:
            return
        try:
            if self.cat is not None:
                self.cat.blocked_mouse(BTN_MIDDLE, 1 if enable else 0)
        except Exception as e:
            print(f'CatBox: 屏蔽中键失败: {e}')

    def mask_side1(self, enable):
        """屏蔽/解除屏蔽侧键1"""
        if not self.is_connected:
            return
        try:
            if self.cat is not None:
                self.cat.blocked_mouse(BTN_SIDE, 1 if enable else 0)
        except Exception as e:
            print(f'CatBox: 屏蔽侧键1失败: {e}')

    def mask_side2(self, enable):
        """屏蔽/解除屏蔽侧键2"""
        if not self.is_connected:
            return
        try:
            if self.cat is not None:
                self.cat.blocked_mouse(BTN_EXTRA, 1 if enable else 0)
        except Exception as e:
            print(f'CatBox: 屏蔽侧键2失败: {e}')

    def mask_x(self, enable):
        """屏蔽/解除屏蔽X轴移动"""
        if not self.is_connected:
            return
        try:
            return
        except Exception as e:
            print(f'CatBox: 屏蔽X轴失败: {e}')

    def mask_y(self, enable):
        """屏蔽/解除屏蔽Y轴移动"""
        if not self.is_connected:
            return
        try:
            return
        except Exception as e:
            print(f'CatBox: 屏蔽Y轴失败: {e}')

    def mask_wheel(self, enable):
        """屏蔽/解除屏蔽滚轮"""
        if not self.is_connected:
            return
        try:
            return
        except Exception as e:
            print(f'CatBox: 屏蔽滚轮失败: {e}')

    def disconnect(self):
        """断开连接"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            self.process = None
        try:
            if self.cat is not None:
                self.cat.close_monitor()
        except Exception:
            pass
        finally:
            self.cat = None
        self.is_connected = False
        print('CatBox: 已断开连接')

    def __del__(self):
        """析构函数"""
        self.disconnect()
catbox = CatBoxWrapper()

def init_catbox(ip, port, uuid):
    """初始化catbox"""
    return catbox.init(ip, port, uuid)

def catbox_move(x, y):
    """catbox移动"""
    catbox.move(x, y)

def catbox_left_down():
    """catbox左键按下"""
    catbox.left_down()

def catbox_left_up():
    """catbox左键释放"""
    catbox.left_up()

def catbox_disconnect():
    """断开catbox连接"""
    catbox.disconnect()
