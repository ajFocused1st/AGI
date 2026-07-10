from __future__ import annotations

import unittest

from agi_contracts import RiskLevel, ToolRequest
from agi_tools import Calculator, ToolRegistry


class CalculatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(Calculator())

    async def calculate(self, expression: str):
        request = ToolRequest(
            tool_name="calculator",
            arguments={"expression": expression},
        )
        return await self.registry.execute(
            request,
            granted_permissions={"calculator.execute"},
        )

    async def test_calculates_expected_numeric_result(self) -> None:
        decision, result = await self.calculate("(17 * 6) + 9")

        self.assertTrue(decision.allowed)
        self.assertTrue(result.success)
        self.assertEqual(result.output["value"], 111)

    async def test_rejects_code_and_non_arithmetic_syntax(self) -> None:
        expressions = (
            "__import__('os').system('whoami')",
            "open('secret.txt').read()",
            "value + 1",
            "[1, 2][0]",
            "True + 1",
            "lambda: 1",
        )

        for expression in expressions:
            with self.subTest(expression=expression):
                _, result = await self.calculate(expression)
                self.assertFalse(result.success)
                self.assertEqual(result.output, {})

    async def test_rejects_unbounded_or_invalid_numeric_operations(self) -> None:
        for expression in ("1 / 0", "2 ** 101", "1e101", "(-1) ** 0.5"):
            with self.subTest(expression=expression):
                _, result = await self.calculate(expression)
                self.assertFalse(result.success)

    def test_definition_declares_schema_and_governance(self) -> None:
        definition = Calculator().definition

        self.assertEqual(definition.name, "calculator")
        self.assertEqual(definition.risk_level, RiskLevel.LOW)
        self.assertEqual(definition.permissions_required, ["calculator.execute"])
        self.assertFalse(definition.human_approval_required)
        self.assertIn("expression", definition.input_schema["properties"])


if __name__ == "__main__":
    unittest.main()
