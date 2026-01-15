# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: pykm2.py
# Bytecode version: 3.10.0rc2 (3439)
# Source timestamp: 1970-01-01 00:00:00 UTC (0)

import serial
import serial.tools.list_ports
import ctypes
import sys
import time
import random
k1 = '`1234567890-=qwertyuiop[]\\asdfghjkl;\'zxcvbnm,./ '
k2 = '~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:\"ZXCVBNM<>?'

def CanInput(c):
    if c in k1:
        return True
    if c in k2:
        return True
    return False

def needShift(c):
    return c in k2

def getUnShiftKey(c):
    for i in range(0, 48):
        if k2[i] == c:
            return k1[i]
    else:  # inserted
        return 0

class i_KM:
    def __init__(self):
        self.dev = None
        self.version = 0
        self.model = 0
        self.vid = 0
        self.pid = 0
        self.wait_respon = False
        if sys.platform == 'win32':
            user32 = ctypes.windll.user32
            self.screenX = user32.GetSystemMetrics(78)
            self.screenY = user32.GetSystemMetrics(79)
        else:  # inserted
            import tkinter
            root = tkinter.Tk()
            self.screenX = root.winfo_vrootwidth()
            self.screenY = root.winfo_vrootheight()
            root.quit()

    def __del__(self):
        self.Close()

    def Close(self):
        if self.dev:
            self.dev.close()
            self.dev = None
        self.version = 0
        self.model = 0
        self.vid = 0
        self.pid = 0
        self.wait_respon = False

    def DelayRandom(self, min, max):
        delay = 0
        if max >= min and max > 0 and (min >= 0):
            delay = random.randint(min, max)
        else:  # inserted
            if max == 0 and min > 0:
                delay = min
        if delay > 0:
            time.sleep(delay / 1000)

    def OpenDevice(self, comname):
        dev = None
        ports_list = list(serial.tools.list_ports.comports())
        if len(ports_list) <= 0:
            return False
        for comport in ports_list:
            if list(comport)[0].lower() == comname.lower():
                dev = serial.Serial(list(comport)[0], 460800, timeout=0.1)
                if dev.isOpen():
                    break
                return False
        if dev:
            self.dev = dev
            ret = self._getVersion()
            if not ret:
                return False
            self.version = ret[1]
            self.model = ret[0]
            return True
        return False

    def IsOpen(self):
        if self.dev!= None:
            return self.dev.isOpen()
        return False

    def write_cmd(self, cmd, dat=None):
        if not self.dev:
            return (-1)
        if dat and len(dat) > 61:
            return (-2)
        buf = [1, cmd]
        if dat:
            buf[0] = len(dat) + 1
            buf.extend(dat)
        buf.extend([255] * (15 - len(buf)))
        ret = self.dev.write(bytes(buf))
        if ret < 0:
            self.Close()
        return ret

    def read_data_timeout(self, timeout=None):
        if not self.dev:
            return
        if not self.dev.isOpen():
            self.Close()
            return
        try:
            ret = self.dev.read(1)
            if len(ret) > 0:
                remain_len = ret[0]
                ret = self.dev.read(remain_len + 2)
                if len(ret) > 0:
                    return (ret[0], ret[1:remain_len])
            return None
        except OSError:
            self.Close()
            return None

    def read_data_timeout_promise(self, cmd, timeout=None):
        if not self.dev:
            return
        for i in range(0, 10):
            ret = self.read_data_timeout(timeout)
            if ret and ret[0] == cmd:
                return ret[1]
        else:  # inserted
            return None

    def _getVersion(self):
        self.write_cmd(1)
        return self.read_data_timeout_promise(1, 10)

    def GetVID(self):
        return self.vid

    def GetPID(self):
        return self.pid

    def GetVersion(self):
        return self.version

    def GetModel(self):
        return self.model

    def GetChipID(self):
        self.write_cmd(12)
        ret = self.read_data_timeout_promise(9, 10)
        if not ret:
            return (-1)
        result = int.from_bytes(ret, byteorder='little', signed=True)
        result += 113666
        return ctypes.c_int32(result).value

    def GetStorageSize(self):
        self.write_cmd(2)
        ret = self.read_data_timeout_promise(2, 10)
        if not ret:
            return (-1)
        result = int.from_bytes(ret, byteorder='little', signed=True)
        return result

    def SetWaitRespon(self, wait):
        self.wait_respon = wait
        self.write_cmd(34)
        self.read_data_timeout_promise(39, 10)

    def Reboot(self):
        self.write_cmd(20)
        ret = self.read_data_timeout_promise(39, 10)
        if not ret:
            return 1
        self.Close()
        return 0

    def SetVidPid(self, vid, pid):
        cmd = [0, 0, 0, 4, vid & 255, vid >> 8 & 255, pid & 255, pid >> 8 & 255]
        if self.write_cmd(7, cmd) == (-1):
            return 1
        ret = self.read_data_timeout_promise(7, 10)
        if ret:
            return 0
        return 1

    def SetConfigData(self, index, data):
        if index < 0 or index >= 252:
            return 2
        cmd = [255] * 6
        addr = index * 2 + 8
        cmd[0] = addr >> 24 & 255
        cmd[1] = addr >> 16 & 255
        cmd[2] = addr >> 8 & 255
        cmd[3] = addr >> 0 & 255
        cmd[4] = (data & 65280) >> 8
        cmd[5] = data & 255
        if self.write_cmd(7, cmd) == (-1):
            return 1
        ret = self.read_data_timeout_promise(7, 100)
        if ret:
            return 0
        return 1

    def GetConfigData(self, index):
        if index < 0 or index >= 252:
            return (-2)
        cmd = [255] * 5
        addr = index * 2 + 8
        cmd[0] = addr >> 24 & 255
        cmd[1] = addr >> 16 & 255
        cmd[2] = addr >> 8 & 255
        cmd[3] = addr >> 0 & 255
        cmd[4] = 2
        if self.write_cmd(6, cmd) == (-1):
            return (-1)
        ret = self.read_data_timeout_promise(6, 20)
        if not ret:
            return (-1)
        return ret[0] * 256 + ret[1]

    def SetLed(self, on):
        self.write_cmd(24, [1 if on else 0])
        ret = self.read_data_timeout_promise(39, 10)
        return 0 if ret else 1

    def RunScript(self, mode, index):
        self.write_cmd(5, [mode, index])

    def mouse_event(self, e, x=0, y=0, extra1=0, extra2=0):
        cmd = [255] * 12
        cmd[0] = e
        if e >= 1 and e <= 7:
            return None
        if e == 8:
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            screenx = self.screenX
            screeny = self.screenY
            if x >= screenx:
                x = screenx - 1
            if y >= screeny:
                y = screeny - 1
            x = int((x << 15) / screenx)
            y = int((y << 15) / screeny)
            cmd[1] = x >> 8 & 255
            cmd[2] = x & 255
            cmd[3] = y >> 8 & 255
            cmd[4] = y & 255
        else:  # inserted
            if e == 9:
                if x < (-128) or x > 127 or y < (-128) or (y > 127):
                    return None
                cmd[1] = x
                cmd[2] = y
            else:  # inserted
                if e == 91:
                    if x < (-32768) or x > 32767 or y < (-32768) or (y > 32767):
                        return None
                    cmd[1] = x >> 8 & 255
                    cmd[2] = x & 255
                    cmd[3] = y >> 8 & 255
                    cmd[4] = y & 255
                else:  # inserted
                    if e == 10:
                        if x < (-128) or x > 127:
                            return None
                        cmd[1] = x
                    else:  # inserted
                        if e == 11:
                            if x < 0:
                                x = 0
                            if y < 0:
                                y = 0
                            cmd[1] = x >> 8 & 255
                            cmd[2] = x & 255
                            cmd[3] = y >> 8 & 255
                            cmd[4] = y & 255
                            screenx = self.screenX
                            screeny = self.screenY
                            cmd[5] = screenx >> 8 & 255
                            cmd[6] = screenx & 255
                            cmd[7] = screeny >> 8 & 255
                            cmd[8] = screeny & 255
                            cmd[9] = extra1
                            cmd[10] = extra2
                        else:  # inserted
                            if e == 12:
                                cmd[1] = x >> 8 & 255
                                cmd[2] = x & 255
                                cmd[3] = y >> 8 & 255
                                cmd[4] = y & 255
                                screenx = self.screenX
                                screeny = self.screenY
                                cmd[5] = screenx >> 8 & 255
                                cmd[6] = screenx & 255
                                cmd[7] = screeny >> 8 & 255
                                cmd[8] = screeny & 255
                                cmd[9] = extra1
                                cmd[10] = extra2
                            else:  # inserted
                                if e == 13 or e == 14:
                                    cmd[1] = x
        self.write_cmd(16, cmd)
        if self.wait_respon:
            self.read_data_timeout_promise(20, 10)

    def LeftDown(self):
        self.mouse_event(1)

    def LeftUp(self):
        self.mouse_event(2)

    def LeftClick(self, min=0, max=0):
        self.LeftDown()
        self.DelayRandom(min, max)
        self.LeftUp()

    def LeftDoubleClick(self, min=0, max=0):
        self.LeftDown()
        self.DelayRandom(min, max)
        self.LeftUp()
        self.DelayRandom(min, max)
        self.LeftDown()
        self.DelayRandom(min, max)
        self.LeftUp()

    def RightDown(self):
        self.mouse_event(3)

    def RightUp(self):
        self.mouse_event(4)

    def RightClick(self, min=0, max=0):
        self.RightDown()
        self.DelayRandom(min, max)
        self.RightUp()

    def MiddleDown(self):
        self.mouse_event(5)

    def MiddleUp(self):
        self.mouse_event(6)

    def MiddleClick(self, min=0, max=0):
        self.MiddleDown()
        self.DelayRandom(min, max)
        self.MiddleUp()

    def MouseWheel(self, delta):
        self.mouse_event(10, delta)

    def MouseButtonDown(self, index):
        if index < 1 or index > 8:
            return None
        index -= 1
        self.mouse_event(13, index)

    def MouseButtonUp(self, index):
        if index < 1 or index > 8:
            return None
        index -= 1
        self.mouse_event(14, index)

    def MouseButtonClick(self, index, min=0, max=0):
        self.MouseButtonDown(index)
        self.DelayRandom(min, max)
        self.MouseButtonUp(index)

    def MouseAllUp(self):
        self.mouse_event(7)

    def MoveTo(self, x, y):
        self.mouse_event(8, x, y)

    def MoveR(self, x, y):
        self.mouse_event(91, x, y)

    def MoveD(self, x, y, delay=8, delta=10):
        self.mouse_event(11, x, y, delay, delta)

    def MoveRD(self, dx, dy, delay=8, delta=10):
        self.mouse_event(12, dx, dy, delay, delta)

    @staticmethod
    def GetScanCodeFromVirtualCode(vcode):
        keymap = {'65': 4, '66': 5, '67': 6, '68': 7, '69': 8, '70': 9, '71': 10, '72': 11, '73': 12, '74': 13, '75': 14, '76': 15, '77': 16, '78': 17, '79': 18, '80': 19, '81': 20, '82': 21, '83': 22, '84': 23, '85': 24, '86': 25, '87': 26, '88': 27, '89': 28, '90': 29, '49': 30, '50': 31, '51': 32, '52': 33, '53': 34}
        vcode = str(vcode)
        if vcode in keymap:
            return keymap[vcode]
        return 0

    @staticmethod
    def GetScanCodeFromKeyName(keyname):
        keymap = {'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8, 'f': 9, 'g': 10, 'h': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15, 'm': 16, 'n': 17, 'o': 18, 'p': 19, 'q': 20}
        keyname = keyname.lower()
        if keyname in keymap:
            return keymap[keyname]
        return 0

    def key_event(self, e, key):
        cmd = [e, 255]
        if isinstance(key, str):
            key = self.GetScanCodeFromKeyName(key)
        cmd[1] = key
        self.write_cmd(17, cmd)
        if self.wait_respon:
            self.read_data_timeout_promise(20, 10)

    def KeyDownName(self, keyname):
        self.key_event(1, keyname)

    def KeyUpName(self, keyname):
        self.key_event(2, keyname)

    def KeyPressName(self, keyname, min=0, max=0):
        self.key_event(1, keyname)
        self.DelayRandom(min, max)
        self.key_event(2, keyname)

    def KeyDownCode(self, keycode):
        self.key_event(1, keycode)

    def KeyUpCode(self, keycode):
        self.key_event(2, keycode)

    def KeyPressCode(self, keycode, min=0, max=0):
        self.key_event(1, keycode)
        self.DelayRandom(min, max)
        self.key_event(2, keycode)

    def KeyDownVirtualCode(self, keycode):
        keycode = self.GetScanCodeFromVirtualCode(keycode)
        self.KeyDownCode(keycode)

    def KeyUpVirtualCode(self, keycode):
        keycode = self.GetScanCodeFromVirtualCode(keycode)
        self.KeyUpCode(keycode)

    def KeyPressVirtualCode(self, keycode, min=0, max=0):
        keycode = self.GetScanCodeFromVirtualCode(keycode)
        self.KeyPressCode(keycode)

    def SayString(self, s, min=0, max=0):
        len_s = len(s)
        shift = False
        for i in range(0, len_s):
            c = s[i]
            need_shift = needShift(c)
            keyname = c
            if need_shift:
                keyname = getUnShiftKey(c)
            if need_shift and (not shift):
                self.KeyDownCode(225)
                self.DelayRandom(min, max)
                shift = True
            self.KeyPressName(keyname, min, max)
            self.DelayRandom(min, max)
            if i == len_s - 1:
                need_shift = False
            else:  # inserted
                need_shift = needShift(s[i + 1])
            if not need_shift and shift:
                self.KeyUpCode(225)
                self.DelayRandom(min, max)
                shift = False

    def SayStringAnsi(self, s, min=0, max=0):
        s = s.encode('gbk')
        len_s = len(s)
        i = 0
        while i < len_s:
            c = s[i]
            code = 0
            if c < 128:
                code = c
                i += 1
            else:  # inserted
                if i < len_s - 1:
                    code = c * 256 + s[i + 1]
                    i += 2
            temp = str(code)
            self.KeyDownCode(226)
            self.DelayRandom(min, max)
            for j in range(0, len(temp)):
                c2 = ord(temp[j]) - 48
                if c2 == 0:
                    c2 = 10
                keycode = 88 + c2
                self.KeyDownCode(keycode)
                self.DelayRandom(min, max)
                self.KeyUpCode(keycode)
                self.DelayRandom(min, max)
            self.KeyUpCode(226)

    def SayStringUnicode(self, s, min=0, max=0):
        len_s = len(s)
        for c in s:
            temp = str(ord(c))
            self.KeyDownCode(226)
            self.DelayRandom(min, max)
            for j in range(0, len(temp)):
                c2 = ord(temp[j]) - 48
                if c2 == 0:
                    c2 = 10
                keycode = 88 + c2
                self.KeyDownCode(keycode)
                self.DelayRandom(min, max)
                self.KeyUpCode(keycode)
                self.DelayRandom(min, max)
            self.KeyUpCode(226)

    def Lock_Mouse(self, option):
        self.write_cmd(25, [option])
        ret = self.read_data_timeout_promise(39, 10)
        return 1 if ret else 0

    def Notify_Mouse(self, option):
        self.write_cmd(26, [option])
        ret = self.read_data_timeout_promise(39, 10)
        return 1 if ret else 0

    def CombineMoveR(self, option):
        self.write_cmd(37, [option])
        ret = self.read_data_timeout_promise(39, 10)
        return 1 if ret else 0

    def Lock_KeyBoard(self, option):
        self.write_cmd(31, [option])
        ret = self.read_data_timeout_promise(39, 10)
        return 1 if ret else 0

    def Notify_KeyBoard(self, option):
        self.write_cmd(32, [option])
        ret = self.read_data_timeout_promise(39, 10)
        return 1 if ret else 0

    def Read_Notify(self, timeout):
        return self.read_data_timeout_promise(43, timeout)

    def GetKeyState(self, keycode):
        self.write_cmd(33, [keycode])
        ret = self.read_data_timeout_promise(49, 10)
        print(ret)
        return ret

    def Set_Freq(self, time):
        self.write_cmd(28, [time])
        return self.read_data_timeout_promise(39, 100)

    def Get_Freq(self):
        self.write_cmd(29)
        ret = self.read_data_timeout_promise(44, 100)
        return ret[0]
