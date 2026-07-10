# Governed tools

This directory provides the first two low-risk tools through the installable
`agi_tools` package. Shared request, result, definition, risk, and permission
models come from `agi_contracts`; this package does not duplicate them.

## Public API

```python
from agi_contracts import ToolRequest
from agi_tools import (
    Calculator,
    ReadOnlyFileSearch,
    ReadOnlyFileSearchConfig,
    SearchRoot,
    ToolRegistry,
)

registry = ToolRegistry()
registry.register(Calculator())
registry.register(
    ReadOnlyFileSearch(
        ReadOnlyFileSearchConfig(
            roots=(SearchRoot(label="docs", path=r"C:\AGI\12_Research"),)
        )
    )
)

decision, result = await registry.execute(
    ToolRequest(tool_name="calculator", arguments={"expression": "37 * 14"}),
    granted_permissions={"calculator.execute"},
)
```

`ToolRegistry.execute` accepts either an `AgentProfile`-compatible object or an
explicit set of granted permissions, never both. It records and returns a
`PermissionDecision` before invoking a tool. Unknown tools, missing permissions,
unmet approval policy, and audit-recorder failures are denied without dispatch.

## Safety boundaries

- `Calculator` walks a restricted arithmetic syntax tree. It permits numeric
  literals and `+`, `-`, `*`, `/`, `//`, `%`, and bounded `**`; it cannot resolve
  names, access attributes, call functions, or execute Python source.
- `ReadOnlyFileSearch` performs literal text matching. Its request schema contains
  only `query` and `max_matches`; models cannot choose paths or roots.
- Search roots and hard caps are construction-time configuration. Results expose
  a root label and relative path, never the configured absolute path. Each match
  includes a content SHA-256 and explicit line bounds for citation verification.
- Search rejects hidden and sensitive names, certificate/key suffixes, symlinks,
  Windows reparse points, binary/non-UTF-8 files, oversized files, overlapping
  roots, and filesystem-root configuration.
- Windows containment uses normalized absolute paths, `commonpath`, resolved-path
  verification, ancestry checks, and an opened-file identity check.

Run the package-local tests with both source packages importable:

```powershell
$env:PYTHONPATH = "C:\AGI\01_Core\Contracts\src;C:\AGI\06_Tools\src"
python -m unittest discover -s C:\AGI\06_Tools\tests -v
```
