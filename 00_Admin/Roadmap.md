# Roadmap

Tracks staged capabilities, dependencies, and review gates.

## Milestone 1 — governed local research loop

Status: implemented and deterministically verified.

Acceptance gate: one configurable agent receives a goal, creates a structured plan, invokes only approved tools, produces a grounded response, receives rule-based evaluation plus separately attributed advisory critique, and saves a schema-valid atomic run record outside Git.

The ten-case `operator-research-v1` benchmark must pass all cases. Destructive-action refusal, calculator trace, citation integrity, permission-before-execution, and run-record validity are mandatory gates.

## Deferred until evidence justifies expansion

Additional agents, mutable tools, browser automation, external providers, general memory, and self-improvement loops remain deferred. Live Ollama validation is opt-in and requires a user-selected model that is already installed.
