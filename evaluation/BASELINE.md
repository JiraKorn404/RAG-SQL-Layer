# Evaluation baseline

The reference point for the multi-agent refactor (Phase 0). Every later phase is compared with
these numbers with `uv run rag-sql-eval compare`. Result files are local (`evaluation/results/`,
not committed); the numbers that matter are copied here.

**Setup:** `gemma4:e4b` for the chat model and for the router (`OLLAMA_ROUTER_MODEL` empty), thinking
on for the SQL writer and the answer, off for the router. Retail schema only (12 tables). 8
retail few-shot examples. Cases: `evaluation/cases.yaml`, 51 of them: 39 SQL (6 simple, 8 join, 8
trap, 4 date, 4 ranking, 3 not-vague, 6 follow-up), 6 chat, 6 vague.

## Results

| Run | Agent | Cases | Correct | SQL | Chat | Vague | Router right | p50 / p95 | Tokens/turn |
|---|---|---|---|---|---|---|---|---|---|
| A | Before the router (commit `69c33b4`) | 43 (no vague cases) | 41/43 | 36/37 | 5/6 | n/a | n/a | 7.5 s / 12.2 s | 2.0k |
| B | Router, first prompt (`c9cb460e`) | 51 | 47/51 | 38/39 | 6/6 | 3/6 | 48/51 | 7.7 s / 18.3 s | 2.4k |
| C | Router, tuned prompt (`4a215a75`), 3 repeats | 51 x 3 | 147/153 | 111/117 | 18/18 | 18/18 | 153/153 | 7.7 s / 14.7 s | 2.5k |
| C2 | Same as C, second run | 51 | 49/51 | 37/39 | 6/6 | 6/6 | 51/51 | 7.8 s / 14.8 s | 2.5k |

Router alone (`rag-sql-eval run --router-only`, 3 repeats, 0.5 s per call at p50, 0.9 s at p95):

| Prompt | Right intent | Vague cases |
|---|---|---|
| first (`c9cb460e`) | 141/153 | 6/18 |
| tuned (`4a215a75`) | 153/153 | 18/18 |

On the 43 cases that run A could load, the router agent (B) scored 42/43 against 41/43 for A: it
fixed "Delete all the orders from 2020", where the old agent wrote SQL.

## What still fails

- `trap-returned-line-percent` fails in every run, before and after the router. The model counts
  returns (`count(return_id)`) instead of distinct returned lines, which the `returns` table comment
  warns about. Work for the SQL writer or fixer (Phases 2 to 5).
- `ranking-store-most-late-shipments` passed in run B and fails in C and C2: the model groups by
  city instead of returning the store. The router's output for it is identical in both, so the
  cause is not established. The SQL model may be sensitive to earlier calls in the server's cache;
  treat a change of one SQL case as noise until it repeats.

## About the router tuning

- The first prompt sent 4 of the 6 vague cases to the SQL writer ("top products", "biggest stores",
  "how are sales doing", "is the business growing"): `gemma4:e4b` took the measure as a default.
  The tuned prompt says that only the number of rows and the period have defaults, and that a
  ranking word or an open "how is X doing" without a measure needs a question back.
- The tuned prompt was written with those failures in view, so 153/153 is not a fair measure of
  generalisation. A hold-out of 14 messages that were never used for tuning (6 vague, 6 clear
  data, 2 chat) went from 30/42 to 40/42 (vague 6/18 to 16/18), and the clear data messages stayed
  at 18/18: it does not ask back more often on clear questions.
- The eval cases repeat none of the questions in `ROUTER_PROMPT` or `few_shot.yaml` (a test
  checks it).

## Noise

- With the same prompts, two full runs gave the same outcome for every case. At temperature 0,
  `--repeat` inside one run reproduces the same answer, so it says little about noise. To test
  for it, run the evaluation again as a separate invocation.
- Across different prompts a borderline case can flip (see above), and the router itself changed
  its answer for "What are the top products?" between two runs of the same prompt. Read a change
  of one case with care, and look at the by-tag and by-kind lines.

## Reproduce

```sh
uv run rag-sql-index                                      # after changing few_shot.yaml or the schema
uv run rag-sql-eval run --models gemma4:e4b               # about 7 minutes
uv run rag-sql-eval run --models gemma4:e4b --router-only --repeat 3   # about 2 minutes
uv run rag-sql-eval compare evaluation/results/OLD.json evaluation/results/NEW.json
```
