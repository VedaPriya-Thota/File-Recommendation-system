# Benchmark comparison (before -> after)

Same dataset, variants and thresholds; only production behaviour differs.

| variant | metric | before | after |
|---|---|---|---|
| A | precision_at_1 | 0.960 | 0.960 |
| A | precision_at_3 | 0.440 | 0.440 |
| A | recall_at_3 | 0.930 | 0.930 |
| A | MRR | 0.960 | 0.960 |
| A | success_rate | 0.968 | 0.968 |
| A | clarification_rate | 0.000 | 0.000 |
| A | no_result_rate | 0.194 | 0.194 |
| A | no_result_false_positive_rate | 0.000 | 0.000 |
| A | average_tool_calls | 1.000 | 1.000 |
| A | average_iterations | 1.000 | 1.000 |
| A | average_llm_calls | 0.000 | 0.000 |
| A | degraded_rate | 0.419 | 0.419 |
| A | scope_violations | 0 | 0 |
| B | precision_at_1 | 0.720 | 0.720 |
| B | precision_at_3 | 0.240 | 0.240 |
| B | recall_at_3 | 0.720 | 0.720 |
| B | MRR | 0.720 | 0.720 |
| B | success_rate | 0.871 | 0.871 |
| B | clarification_rate | 0.194 | 0.194 |
| B | no_result_rate | 0.387 | 0.387 |
| B | no_result_false_positive_rate | 0.000 | 0.000 |
| B | average_tool_calls | 1.839 | 1.839 |
| B | average_iterations | 1.839 | 1.839 |
| B | average_llm_calls | 0.000 | 0.000 |
| B | degraded_rate | 0.226 | 0.226 |
| B | scope_violations | 0 | 0 |
| C | precision_at_1 | 0.760 | 0.760 |
| C | precision_at_3 | 0.267 | 0.267 |
| C | recall_at_3 | 0.760 | 0.760 |
| C | MRR | 0.760 | 0.760 |
| C | success_rate | 0.903 | 0.903 |
| C | clarification_rate | 0.161 | 0.161 |
| C | no_result_rate | 0.355 | 0.355 |
| C | no_result_false_positive_rate | 0.000 | 0.000 |
| C | average_tool_calls | 1.839 | 1.839 |
| C | average_iterations | 1.839 | 1.839 |
| C | average_llm_calls | 0.000 | 0.000 |
| C | degraded_rate | 0.226 | 0.226 |
| C | scope_violations | 0 | 0 |
| D | precision_at_1 | 0.760 | 0.800 |
| D | precision_at_3 | 0.280 | 0.280 |
| D | recall_at_3 | 0.800 | 0.800 |
| D | MRR | 0.780 | 0.800 |
| D | success_rate | 0.903 | 0.935 |
| D | clarification_rate | 0.161 | 0.161 |
| D | no_result_rate | 0.355 | 0.355 |
| D | no_result_false_positive_rate | 0.000 | 0.000 |
| D | average_tool_calls | 1.903 | 1.903 |
| D | average_iterations | 1.903 | 1.903 |
| D | average_llm_calls | 1.903 | 1.839 |
| D | degraded_rate | 0.226 | 0.226 |
| D | scope_violations | 0 | 0 |

## Per-query behaviour changes (core variants)

| query | variant | before | after | success |
|---|---|---|---|---|
| H3 | D | finish [1, 2] (3 tools) | finish [2, 1] (3 tools) | False -> True |

## Scenarios

- clear_query: PASS -> PASS
- ambiguous_query: PASS -> PASS
- weak_retrieval_escalation: PASS -> PASS
- near_tie_gating: PASS -> PASS
- no_result: PASS -> PASS
- cross_user_isolation: PASS -> PASS
- personalization_end_to_end: FAIL -> FAIL
- llm_outage: PASS -> PASS
- explanation_never_affects_ranking: PASS -> PASS
