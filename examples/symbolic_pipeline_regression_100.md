# Symbolic pipeline regression set

`symbolic_pipeline_regression_100.json` contains 100 deterministic cases across 25 common sign codes. Each sign group checks two permission questions that should match its rule and two questions that should not be routed to that sign's permission rule (or asks for sign meaning for warning signs).

The cases cover entry and vehicle restrictions, turns and U-turns, parking/date restrictions, speed, yielding and stopping, obstacle passing, lane direction, one-way roads, and warning signs. Expected intent slots, applicable rule IDs, and aggregate decisions are recorded in the JSON fixture.

The fixture fixes the clock at 2026-10-05 08:30 in Asia/Ho_Chi_Minh so date-sensitive cases are repeatable. Sign observations are high confidence; the runner does not inject a `sign_effective_scope_confirmed` fact. Some rules intentionally remain `CONDITIONAL` because their sign meaning itself is conditional or because a retained sign-provided value is missing.

Run it from the repository root:

```sh
python scripts/run_symbolic_pipeline_smoke.py examples/symbolic_pipeline_regression_100.json
```

Use `--quiet` to print only failures and the final pass count.
