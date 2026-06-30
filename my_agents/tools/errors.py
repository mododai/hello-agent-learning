"""
工具错误码定义(手写容易打错)
"""
class ToolErrorCode:

    # 资源相关错误
    NOT_FOUND = "NOT_FOUND"  # 资源不存在（文件、工具等）
    ACCESS_DENIED = "ACCESS_DENIED"  # 访问被拒绝
    PERMISSION_DENIED = "PERMISSION_DENIED"  # 权限不足
    IS_DIRECTORY = "IS_DIRECTORY"  # 期望文件但得到目录
    BINARY_FILE = "BINARY_FILE"  # 二进制文件无法处理

    # 参数相关错误
    INVALID_PARAM = "INVALID_PARAM"  # 参数无效或缺失
    INVALID_FORMAT = "INVALID_FORMAT"  # 格式错误

    # 执行相关错误
    EXECUTION_ERROR = "EXECUTION_ERROR"  # 执行过程中发生错误
    TIMEOUT = "TIMEOUT"  # 执行超时
    INTERNAL_ERROR = "INTERNAL_ERROR"  # 内部错误

    # 状态相关错误
    CONFLICT = "CONFLICT"  # 冲突（如乐观锁冲突）
    CIRCUIT_OPEN = "CIRCUIT_OPEN"  # 熔断器开启，拒绝执行

    # 网络相关错误
    NETWORK_ERROR = "NETWORK_ERROR"  # 网络请求失败
    API_ERROR = "API_ERROR"  # API 调用失败
    RATE_LIMIT = "RATE_LIMIT"  # 速率限制

    @classmethod
    def get_all_codes(cls) -> list[str]:
        """获取所有错误码"""
        return [
            value for name, value in vars(cls).items()
            if not name.startswith('_') and isinstance(value, str)
        ]

    @classmethod
    def is_valid_code(cls, code: str) -> bool:
        """检查是否是有效的错误码"""
        return code in cls.get_all_codes()

