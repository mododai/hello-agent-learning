"""
安全数学计算工具。

不要使用 eval 直接执行用户输入。这里使用 ast 解析表达式，
并且只允许白名单中的运算符、函数和常量。
"""

import ast
import logging
import math
import operator
from typing import Any, Callable, Dict, List

from ..base import Tool, ToolParameter
from ..errors import ToolErrorCode
from ..response import ToolResponse

logger = logging.getLogger(__name__)


class CalculatorTool(Tool):
    """安全的 Python 数学计算器工具。"""

    OPERATORS: dict[type[ast.operator | ast.AST | ast.unaryop], Callable[..., Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    FUNCTIONS: Dict[str, Callable[..., Any]] = {
        "abs": abs,
        "round": round,
        "max": max,
        "min": min,
        "sum": sum,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
    }

    CONSTANTS: Dict[str, float] = {
        "pi": math.pi,
        "e": math.e,
    }

    def __init__(self):
        super().__init__(
            name="python_calculator",
            description=(
                "执行数学计算。支持基本运算和常见数学函数。"
                "例如：2+3*4、sqrt(16)、sin(pi/2)、sum([1,2,3])。"
                f"支持的函数: {', '.join(self.FUNCTIONS.keys())}。"
                f"支持的常量: {', '.join(self.CONSTANTS.keys())}。"
            ),
        )

    def run(self, parameter: Dict[str, Any]) -> ToolResponse:
        """执行数学表达式计算。"""
        expression = parameter.get("input") or parameter.get("expression")

        if not expression:
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="计算表达式不能为空",
            )

        if not isinstance(expression, str):
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAM,
                message="计算表达式必须是字符串",
                context={"expression": expression},
            )

        logger.info(f"正在计算: {expression}")

        try:
            node = ast.parse(expression, mode="eval")
            result = self._eval_node(node.body)
            result_str = str(result)

            logger.info(f"计算结果: {result_str}")
            return ToolResponse.success(
                text=f"计算结果: {result_str}",
                data={
                    "expression": expression,
                    "result": result,
                    "result_str": result_str,
                    "result_type": type(result).__name__,
                },
            )
        except SyntaxError as e:
            error_msg = f"表达式语法错误: {e}"
            logger.error(error_msg)
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_FORMAT,
                message=error_msg,
                context={"expression": expression},
            )
        except Exception as e:
            error_msg = f"计算失败: {e}"
            logger.error(error_msg)
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=error_msg,
                context={"expression": expression},
            )

    def _eval_node(self, node: ast.AST | ast.expr):
        """递归计算 AST 节点。"""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("只支持数字常量")

        if isinstance(node, ast.BinOp):
            operator_func = self.OPERATORS.get(type(node.op))
            if not operator_func:
                raise ValueError(f"不支持的二元运算符: {type(node.op).__name__}")

            return operator_func(
                self._eval_node(node.left),
                self._eval_node(node.right),
            )

        if isinstance(node, ast.UnaryOp):
            operator_func = self.OPERATORS.get(type(node.op))
            if not operator_func:
                raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")

            return operator_func(self._eval_node(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("只支持直接调用白名单函数，例如 sqrt(16)")

            func_name = node.func.id
            if func_name not in self.FUNCTIONS:
                raise ValueError(f"不支持的函数: {func_name}")

            args = [self._eval_node(arg) for arg in node.args]
            return self.FUNCTIONS[func_name](*args)

        if isinstance(node, ast.Name):
            if node.id not in self.CONSTANTS:
                raise ValueError(f"未定义的变量: {node.id}")
            return self.CONSTANTS[node.id]

        if isinstance(node, ast.List):
            return [self._eval_node(item) for item in node.elts]

        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(item) for item in node.elts)

        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    def get_parameters(self) -> List[ToolParameter]:
        """返回 Function Calling schema 所需的参数定义。"""
        return [
            ToolParameter(
                name="input",
                type="string",
                description="要计算的数学表达式，例如：sqrt(16) + 2 * 3",
                required=True,
            )
        ]
