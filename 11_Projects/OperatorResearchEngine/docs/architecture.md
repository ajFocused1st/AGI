# Vertical-Slice Architecture

`OperatorResearchEngine` is a composition root. `dependencies.py` selects concrete adapters, tools, permissions, evaluator, and storage; `runtime.py` exposes a thin project-facing facade. Shared implementation remains in its numbered ownership area.

The model proposes plans and tool requests, but the runtime owns state transitions. A deny-by-default registry makes each permission decision before tool execution. Tool output and local documents are treated as untrusted evidence, not instructions. Rule-based evaluation determines benchmark status; optional model critique is stored separately and cannot override safety checks.
