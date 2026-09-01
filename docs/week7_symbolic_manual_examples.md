# Week 7 Manual Comparison: V1 vs V3

This table records 10 deterministic checks used while integrating symbolic
decisions into answer generation. V1 is the current RAG/VLM answer path where
the LLM decides from retrieved context. V3 adds structured facts, question
intent, manual rules, a symbolic decision, and a reasoning trace before the LLM
writes the final Vietnamese answer.

| # | Sign fact | Question intent | V1 risk | V3 symbolic result |
|---|---|---|---|---|
| 1 | `P.130` no stopping/no parking | car + parking | May cite article but soften the prohibition. | `P130_NO_STOPPING_NO_PARKING` -> `PROHIBITED` |
| 2 | `P.130` no stopping/no parking | motorbike + stopping | May confuse stopping with parking. | `P130_NO_STOPPING_NO_PARKING` -> `PROHIBITED` |
| 3 | `P.131a` no parking | car + parking | May over-explain or mix with odd/even variants. | `P131A_NO_PARKING` -> `PROHIBITED` |
| 4 | `P.131b` no parking on odd days | truck + parking | May ignore odd-day condition. | `P131B_ODD_DAY_NO_PARKING` -> `CONDITIONAL` |
| 5 | `P.131c` no parking on even days | bus + parking | May confuse even-day and odd-day variants. | `P131C_EVEN_DAY_NO_PARKING` -> `CONDITIONAL` |
| 6 | `P.123a` no left turn | car + left turn | May answer from visual impression only. | `P123A_NO_LEFT_TURN` -> `PROHIBITED` |
| 7 | `P.123b` no right turn | car + right turn | May swap left/right. | `P123B_NO_RIGHT_TURN` -> `PROHIBITED` |
| 8 | `P.102` no entry | car + entering | May answer generally without direction scope. | `P102_NO_ENTRY` -> `PROHIBITED` |
| 9 | `P.127` speed limit | speed query | May cite the rule but miss that numeric value must be read. | `P127_SPEED_LIMIT` -> `CONDITIONAL` |
| 10 | unsupported `P.999` | car + parking | May hallucinate a rule if pressured. | no rule fired -> `UNKNOWN` |

The key Week 7 check is not that V3 solves every legal case. It is that covered
cases now expose `symbolic_facts`, `question_intent`, `rules_fired`,
`symbolic_decision`, and `reasoning_trace`, while unsupported cases fall back to
`UNKNOWN` instead of inventing a confident rule.
