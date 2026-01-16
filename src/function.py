import json
import math
import win32api
import win32gui
from cryptography.fernet import Fernet

def key2str(key):
    return str(key).replace("'", '').replace('Key.', '').lower()

def is_cursor_visible():
    cursor_info = win32gui.GetCursorInfo()
    if cursor_info[1] != 0:
        return True
    return False
def get_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            content = f.read()
            config = json.loads(content)
            return config
    except FileNotFoundError:
        print('配置文件未找到，请检查文件路径是否正确。')
        raise
    except json.JSONDecodeError as e:
        print(f'配置文件格式错误，无法解析JSON数据: {e}')
        raise
    except UnicodeDecodeError as e:
        print(f'解码配置文件时出错: {e}')
        raise

def get_linear_distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def get_machine_code():
    code = win32api.GetVolumeInformation('C:\\')[1]
    return str(code) if code else '00000000'