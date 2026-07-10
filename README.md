# AGI Workspace

`C:\AGI` is a source-controlled engineering and research workspace for building governed AI systems.

The numbered folders separate administration, core runtime contracts, agent profiles, cognition, memory, knowledge, tools, models, versionable data, experiments, benchmarks, projects, research, API integrations, deployments, monitoring, security, user interfaces, documentation, and backup definitions. `Sandbox` is for disposable exploration.

Shared capabilities belong in their owning platform area rather than being copied into individual projects. New experimental work begins in `Sandbox` or `09_Experiments` and moves only after its boundary is proven. Runtime logs, databases, caches, downloaded models, uploads, and generated artifacts belong in `C:\AGI_Runtime`, not in this repository. Secrets must never be committed. Large model files must not be stored in Git. `19_Backups` holds policy and scripts only; real backups must use independent storage.
