# Veyra Seed Archive Answer Key v1

Three invented documents about a fictional seed archive and 40 questions with known answers
(`answer_key.jsonl`). It checks two things: whether answers are right, and whether the
engine's own verdict agrees with the key.

Categories: direct, reworded, calculation, two-step, cross-document, not-in-documents,
corrected-figure, confusable-names, date-formats, table.

Each line of the key has `question`, `answer`, `accept` (other accepted wordings), `reject` (known
wrong values, flagged for review), `sources` (verbatim quotes), `calculation` and
`not_in_documents`.

Run from `11_Projects/OperatorResearchEngine`:

```
python -m operator_research_engine.answer_key check veyra_v1
python -m operator_research_engine.answer_key run veyra_v1 --config configs/real.yaml --runs 3
```

The engine settings (model, meaning search, planning) come from the config. The documents and
the profile (`researcher_veyra.yaml`) come from this dataset. Results are written under
`<runtime_root>/answer_keys/veyra_v1/`.

Generalization rule: don't tune the runtime against these names or phrasings. A fix only counts if
it would help on documents the engine has never seen.
