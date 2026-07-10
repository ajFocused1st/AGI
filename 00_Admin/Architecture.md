# Architecture

Records the current system boundaries and their ownership.

## Current vertical slice

The first implemented slice is `11_Projects/OperatorResearchEngine`. It composes shared packages from their owning areas instead of copying implementations into the project:

- `01_Core/Contracts` owns strict Pydantic messages and persisted records.
- `01_Core/Evaluation` owns authoritative deterministic checks.
- `02_Agents/Runtime` owns orchestration and runtime-controlled state transitions.
- `06_Tools` owns the deny-by-default registry, calculator, and read-only file search.
- `07_Models/Adapters/Ollama` owns loopback transport and provider response mapping.
- The project owns configuration and dependency composition only.

Generated state is written to `C:\AGI_Runtime`. No runtime component may silently pull a model, select a remote provider, broaden document roots, or execute an unregistered tool.

See `18_Documentation/Architecture/SystemOverview.md` for the flow and trust boundaries.
