"""
GUI事件处理模块

该模块提供了GUI事件处理的统一接口，用于处理配置变更事件。
"""
import logging
from typing import Any, Callable, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigChangeHandler:
    """
    配置变更处理器

    统一处理GUI控件的值变更事件，支持简单配置和嵌套配置的更新，
    以及注册后处理回调函数。
    """

    def __init__(self, config: dict, save_callback: Callable = None):
        """
        初始化配置变更处理器

        Args:
            config: 配置字典
            save_callback: 保存配置的回调函数
        """
        self.config = config
        self.save_callback = save_callback
        self.handlers = {}
        self.post_processors = {}
        self.context_providers = {}

    def register_context_provider(self, name: str, provider: Callable) -> None:
        """
        注册上下文提供函数

        Args:
            name: 上下文名称
            provider: 提供上下文值的函数
        """
        self.context_providers[name] = provider

    def get_context(self, context_name: str) -> Any:
        """
        获取上下文值

        Args:
            context_name: 上下文名称

        Returns:
            上下文值
        """
        if context_name not in self.context_providers:
            raise ValueError(f'未注册的上下文: {context_name}')
        return self.context_providers[context_name]()

    pass
    pass
    pass

    def register_config_item(self, control_id: str, config_path: str, value_type: type = None,
                             post_processor: Callable = None, special_handler: Callable = None) -> None:
        """
        注册配置项

        Args:
            control_id: 控件ID
            config_path: 配置路径，支持点分隔的嵌套路径和上下文变量，如'groups.{group}.aim_keys.{key}.value'
            value_type: 值类型，用于类型转换
            post_processor: 后处理函数，在配置更新后调用
            special_handler: 特殊处理函数，完全接管配置更新
        """
        self.handlers[control_id] = {'path': config_path, 'type': value_type, 'post_processor': post_processor,
                                     'special_handler': special_handler}
        if post_processor:
            self.post_processors[config_path] = post_processor

    def register_post_processor(self, config_path: str, processor: Callable) -> None:
        """
        注册后处理函数

        Args:
            config_path: 配置路径
            processor: 后处理函数
        """
        self.post_processors[config_path] = processor

    def _resolve_path(self, path: str) -> List[str]:
        """
        解析配置路径，替换上下文变量

        Args:
            path: 配置路径，如'groups.{group}.aim_keys.{key}.value'

        Returns:
            解析后的路径部分列表
        """
        parts = path.split('.')
        resolved_parts = []
        for part in parts:
            if part.startswith('{') and part.endswith('}'):
                context_name = part[1:-1]
                context_value = self.get_context(context_name)
                resolved_parts.append(context_value)
            else:
                resolved_parts.append(part)
        return resolved_parts

    def _get_config_value(self, path: str) -> Any:
        """
        获取配置值

        Args:
            path: 配置路径

        Returns:
            配置值
        """
        parts = self._resolve_path(path)
        value = self.config
        for part in parts:
            if part not in value:
                return
            value = value[part]
        else:
            return value

    def _set_config_value(self, path: str, value: Any) -> None:
        """
        设置配置值

        Args:
            path: 配置路径
            value: 配置值
        """
        parts = self._resolve_path(path)
        config = self.config
        for i in range(len(parts) - 1):
            part = parts[i]
            if part not in config:
                config[part] = {}
            config = config[part]
        config[parts[-1]] = value

    def handle_change(self, sender: str, app_data: Any) -> None:
        """
        处理配置变更事件

        Args:
            sender: 发送者ID
            app_data: 新的配置值
        """
        if sender not in self.handlers:
            logger.warning(f'未注册的控件ID: {sender}')
            return
        handler = self.handlers[sender]
        if handler['special_handler']:
            handler['special_handler'](sender, app_data)
        else:
            path = handler['path']
            value_type = handler['type']
            if value_type:
                try:
                    if value_type == float:
                        app_data = round(float(app_data), 4)
                    elif value_type == int:
                        app_data = int(app_data)
                    elif value_type == bool:
                        app_data = bool(app_data)
                    elif value_type == str:
                        app_data = str(app_data)
                except (ValueError, TypeError) as e:
                    logger.error(f'类型转换失败: {e}')
                    return None
            self._set_config_value(path, app_data)
            print(f'changed to: {app_data}')
            post_processor = handler.get('post_processor')
            if post_processor:
                post_processor()

    def create_handler(self, sender: str) -> Callable:
        """
        创建特定控件的处理函数

        Args:
            sender: 控件ID

        Returns:
            处理函数
        """

        def handler(sender, app_data):
            self.handle_change(sender, app_data)

        return handler


class ConfigItemGroup:
    """
    配置项组

    用于组织和管理相关的配置项，简化配置项的注册。
    """

    def __init__(self, handler: ConfigChangeHandler, base_path: str = ''):
        """
        初始化配置项组

        Args:
            handler: 配置变更处理器
            base_path: 基础路径
        """
        self.handler = handler
        self.base_path = base_path

    pass
    pass
    pass

    def register_item(self, control_id: str, path: str, value_type: type = None, post_processor: Callable = None,
                      special_handler: Callable = None) -> None:
        """
        注册配置项

        Args:
            control_id: 控件ID
            path: 配置路径，相对于基础路径
            value_type: 值类型
            post_processor: 后处理函数
            special_handler: 特殊处理函数
        """
        full_path = f'{self.base_path}.{path}' if self.base_path else path
        self.handler.register_config_item(control_id, full_path, value_type, post_processor, special_handler)