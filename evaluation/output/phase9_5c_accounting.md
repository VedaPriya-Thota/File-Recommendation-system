# Phase 5C accounting

## phase8_baseline_results.json

| variant | runs | 5C eligible | 5C ran | suppressed (personalization-resolved) | order changed | rejected |
|---|---|---|---|---|---|---|
| D_full_system | 31 | 3 | 3 | 0 | 1 | 0 |
| D_llm_unavailable | 31 | 2 | 2 | 0 | 0 | 2 |
| D_llm_invalid_output | 31 | 2 | 2 | 0 | 0 | 2 |
| D_no_explanation | 31 | 3 | 3 | 0 | 1 | 0 |
| D_cap3_diagnostic | 31 | 8 | 8 | 0 | 4 | 0 |

Query IDs (D_full_system): eligible=['C2', 'E2', 'H3']; ran=['C2', 'E2', 'H3']; suppressed=-; changed=['H3']; rejected=-

## benchmark_results.json

| variant | runs | 5C eligible | 5C ran | suppressed (personalization-resolved) | order changed | rejected |
|---|---|---|---|---|---|---|
| D_full_system | 31 | 3 | 1 | 2 | 0 | 0 |
| D_llm_unavailable | 31 | 2 | 1 | 1 | 0 | 1 |
| D_llm_invalid_output | 31 | 2 | 1 | 1 | 0 | 1 |
| D_no_explanation | 31 | 3 | 1 | 2 | 0 | 0 |
| D_cap3_diagnostic | 31 | 8 | 6 | 2 | 3 | 0 |

Query IDs (D_full_system): eligible=['C2', 'E2', 'H3']; ran=['C2']; suppressed=['E2', 'H3']; changed=-; rejected=-

