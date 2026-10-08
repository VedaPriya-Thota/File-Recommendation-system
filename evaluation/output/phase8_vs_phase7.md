# Benchmark comparison (before -> after)

Same dataset, variants and thresholds; only production behaviour differs.

| variant | metric | before | after |
|---|---|---|---|
| A | precision_at_1 | 0.960 | 0.960 |
| A | precision_at_3 | 0.440 | 0.440 |
| A | recall_at_3 | 0.930 | 0.930 |
| A | MRR | 0.960 | 0.960 |
| A | success_rate | 0.935 | 0.968 |
| A | clarification_rate | 0.000 | 0.000 |
| A | no_result_rate | 0.161 | 0.194 |
| A | no_result_false_positive_rate | 0.167 | 0.000 |
| A | average_tool_calls | 1.000 | 1.000 |
| A | average_iterations | 1.000 | 1.000 |
| A | average_llm_calls | 0.000 | 0.000 |
| A | degraded_rate | 0.387 | 0.419 |
| A | scope_violations | 0 | 0 |
| B | precision_at_1 | 0.680 | 0.720 |
| B | precision_at_3 | 0.227 | 0.240 |
| B | recall_at_3 | 0.680 | 0.720 |
| B | MRR | 0.680 | 0.720 |
| B | success_rate | 0.774 | 0.871 |
| B | clarification_rate | 0.226 | 0.194 |
| B | no_result_rate | 0.355 | 0.387 |
| B | no_result_false_positive_rate | 0.333 | 0.000 |
| B | average_tool_calls | 1.774 | 1.839 |
| B | average_iterations | 1.774 | 1.839 |
| B | average_llm_calls | 0.000 | 0.000 |
| B | degraded_rate | 0.129 | 0.226 |
| B | scope_violations | 0 | 0 |
| C | precision_at_1 | 0.680 | 0.760 |
| C | precision_at_3 | 0.227 | 0.267 |
| C | recall_at_3 | 0.680 | 0.760 |
| C | MRR | 0.680 | 0.760 |
| C | success_rate | 0.774 | 0.903 |
| C | clarification_rate | 0.226 | 0.161 |
| C | no_result_rate | 0.355 | 0.355 |
| C | no_result_false_positive_rate | 0.333 | 0.000 |
| C | average_tool_calls | 1.774 | 1.839 |
| C | average_iterations | 1.774 | 1.839 |
| C | average_llm_calls | 0.000 | 0.000 |
| C | degraded_rate | 0.129 | 0.226 |
| C | scope_violations | 0 | 0 |
| D | precision_at_1 | 0.680 | 0.760 |
| D | precision_at_3 | 0.227 | 0.280 |
| D | recall_at_3 | 0.680 | 0.800 |
| D | MRR | 0.680 | 0.780 |
| D | success_rate | 0.774 | 0.903 |
| D | clarification_rate | 0.258 | 0.161 |
| D | no_result_rate | 0.387 | 0.355 |
| D | no_result_false_positive_rate | 0.333 | 0.000 |
| D | average_tool_calls | 1.839 | 1.903 |
| D | average_iterations | 1.839 | 1.903 |
| D | average_llm_calls | 1.774 | 1.903 |
| D | degraded_rate | 0.129 | 0.226 |
| D | scope_violations | 0 | 0 |

## Per-query behaviour changes (core variants)

| query | variant | before | after | success |
|---|---|---|---|---|
| C2 | B | clarify [] (3 tools) | finish [5, 3] (3 tools) | False -> True |
| C2 | C | clarify [] (3 tools) | finish [5, 3] (3 tools) | False -> True |
| C2 | D | clarify [] (3 tools) | finish [5, 3] (3 tools) | False -> True |
| E2 | C | clarify [] (3 tools) | finish [7, 1] (3 tools) | False -> True |
| E2 | D | clarify [] (3 tools) | finish [7, 1] (3 tools) | False -> True |
| G1 | B | finish [2] (3 tools) | finish [] (2 tools) | False -> True |
| G1 | C | finish [2] (3 tools) | finish [] (2 tools) | False -> True |
| G1 | D | finish [2] (3 tools) | finish [] (2 tools) | False -> True |
| G2 | A | finish [8] (1 tools) | finish [] (1 tools) | False -> True |
| G2 | B | finish [8] (1 tools) | finish [] (4 tools) | False -> True |
| G2 | C | finish [8] (1 tools) | finish [] (4 tools) | False -> True |
| G2 | D | finish [8] (1 tools) | finish [] (4 tools) | False -> True |
| H3 | D | clarify [] (3 tools) | finish [1, 2] (3 tools) | False -> False |

## Scenarios

- clear_query: PASS -> PASS
- ambiguous_query: PASS -> PASS
- weak_retrieval_escalation: PASS -> PASS
- near_tie_gating: PASS -> PASS
- no_result: FAIL -> PASS
- cross_user_isolation: PASS -> PASS
- personalization_end_to_end: FAIL -> FAIL
- llm_outage: PASS -> PASS
- explanation_never_affects_ranking: PASS -> PASS
