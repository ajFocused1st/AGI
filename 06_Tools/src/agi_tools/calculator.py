"""A small arithmetic calculator implemented with a restricted AST."""

from __future__ import annotations

import ast
import math

from agi_contracts import RiskLevel
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import BaseTool, ToolInputError


MAX_EXPRESSION_LENGTH = 256
MAX_AST_NODES = 64
MAX_AST_DEPTH = 16
MAX_ABSOLUTE_NUMBER = 1e100
MAX_ABSOLUTE_EXPONENT = 100


class CalculatorInput(BaseModel):
    """Input accepted by :class:`Calculator`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    expression: str = Field(min_length=1, max_length=MAX_EXPRESSION_LENGTH)

    @field_validator("expression")
    @classmethod
    def expression_must_be_single_line(cls, value: str) -> str:
        if any(character in value for character in ("\x00", "\r", "\n")):
            raise ValueError("expression must be a single line")
        return value


class CalculatorOutput(BaseModel):
    """A finite numeric result."""

    model_config = ConfigDict(extra="forbid")

    expression: str
    value: int | float


class Calculator(BaseTool[CalculatorInput, CalculatorOutput]):
    """Evaluate basic arithmetic without executing Python code."""

    name = "calculator"
    description = "Evaluate arithmetic containing numbers, parentheses, +, -, *, /, //, %, and **."
    risk_level = RiskLevel.LOW
    permissions_required = ("calculator.execute",)
    human_approval_required = False
    timeout_seconds = 2.0
    input_model = CalculatorInput
    output_model = CalculatorOutput

    async def run(self, tool_input: CalculatorInput) -> CalculatorOutput:
        try:
            tree = ast.parse(tool_input.expression, mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ToolInputError("expression is not valid arithmetic") from exc

        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise ToolInputError("expression is too complex")

        try:
            value = self._evaluate(tree.body, depth=0)
        except ZeroDivisionError as exc:
            raise ToolInputError("division by zero is not allowed") from exc
        except OverflowError as exc:
            raise ToolInputError("numeric result exceeds the allowed range") from exc

        self._check_number(value, label="result")
        return CalculatorOutput(expression=tool_input.expression, value=value)

    def _evaluate(self, node: ast.AST, *, depth: int) -> int | float:
        if depth > MAX_AST_DEPTH:
            raise ToolInputError("expression is too deeply nested")

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolInputError("only numeric literals are allowed")
            self._check_number(node.value, label="number")
            return node.value

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._evaluate(node.operand, depth=depth + 1)
            value = operand if isinstance(node.op, ast.UAdd) else -operand
            self._check_number(value, label="result")
            return value

        if not isinstance(node, ast.BinOp):
            raise ToolInputError("expression contains an unsupported operation")

        left = self._evaluate(node.left, depth=depth + 1)
        right = self._evaluate(node.right, depth=depth + 1)

        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.Div):
            value = left / right
        elif isinstance(node.op, ast.FloorDiv):
            value = left // right
        elif isinstance(node.op, ast.Mod):
            value = left % right
        elif isinstance(node.op, ast.Pow):
            if abs(right) > MAX_ABSOLUTE_EXPONENT:
                raise ToolInputError("exponent exceeds the allowed range")
            value = left**right
        else:
            raise ToolInputError("expression contains an unsupported operation")

        if isinstance(value, complex):
            raise ToolInputError("complex results are not supported")
        self._check_number(value, label="result")
        return value

    @staticmethod
    def _check_number(value: int | float, *, label: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolInputError(f"{label} must be numeric")
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolInputError(f"{label} must be finite")
        if abs(value) > MAX_ABSOLUTE_NUMBER:
            raise ToolInputError(f"{label} exceeds the allowed range")
