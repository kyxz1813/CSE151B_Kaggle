# Category Work Files

Each teammate should own one category file in this folder.

Categories:

- `statistics_probability`
- `calculus`
- `geometry_trig`
- `linear_algebra`
- `discrete_algorithm`
- `arithmetic_algebra`
- `applied_word_problem`
- `general_math`

Each category file should define:

1. `CATEGORY`
2. `DEFAULT_STRATEGY`
3. `CANDIDATE_STRATEGIES`
4. `register_category_harness()`
5. `register_category_rules()`
6. Notes about the prompt templates/strategies being tested

Prompt templates and prompt-chain strategy registration should still live in:

- `baseline/baseline3_prompts.py`
- `prompting/templates.py`
- `prompting/strategies.py`

The category files are the teammate-facing place for harnesses, rules, and experiment defaults.

Each category experiment writes its own artifacts under:

```text
results/baseline3_category_experiments/<category>/<experiment>/<strategy>/
```

Expected artifacts include:

results.jsonl
debug.jsonl
report.json
rule_one_hot.csv

The shared comparison CSV is optional and can be ignored until the final aggregation stage.
