# Phase 7 benchmark report

31 queries x 10 variants. Embedding: `spacy:en_core_web_md (sandbox fallback -- see module docstring)`; reranker: `lexical_overlap_fallback (NOT a cross-encoder -- see reranker.py docstring)`; semantic index: {'files_indexed': 0, 'chunks_indexed': 0}.

## Overall retrieval (core ablation)

| variant | n_scored | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | nDCG@5 | success |
|---|---|---|---|---|---|---|---|---|---|---|
| A_single_shot_metadata | 25 | 0.960 | 0.440 | 0.288 | 0.810 | 0.930 | 0.960 | 0.960 | 0.952 | 0.968 |
| B_agentic_deterministic | 25 | 0.720 | 0.240 | 0.144 | 0.720 | 0.720 | 0.720 | 0.720 | 0.720 | 0.871 |
| C_agentic_personalized | 25 | 0.760 | 0.267 | 0.160 | 0.740 | 0.760 | 0.760 | 0.760 | 0.760 | 0.903 |
| D_full_system | 25 | 0.800 | 0.280 | 0.168 | 0.780 | 0.800 | 0.800 | 0.800 | 0.800 | 0.935 |

## Abstention-aware view (core ablation)

Clarified queries return nothing and score 0 in the table above. `answered-only` restricts to queries where a ranking was returned; `pool-ranking` is a DIAGNOSTIC of how final_rank() would order the accumulated pool (not what the system returned).

| variant | abstention rate | n_answered | P@1 answered-only | MRR answered-only | MRR pool-ranking (diag.) | HR@1 | HR@1 pool-ranking (diag.) |
|---|---|---|---|---|---|---|---|
| A_single_shot_metadata | 0.000 | 25 | 0.960 | 0.960 | 0.960 | 0.900 | 0.900 |
| B_agentic_deterministic | 0.240 | 19 | 0.947 | 0.947 | 0.960 | 0.850 | 0.900 |
| C_agentic_personalized | 0.200 | 20 | 0.950 | 0.950 | 0.960 | 0.900 | 0.950 |
| D_full_system | 0.200 | 20 | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 |

## Behaviour (all variants)

| variant | success | clarify | no_result | tool calls | retr. actions | iterations | LLM calls | LLM attempts | escalation | scope viol. |
|---|---|---|---|---|---|---|---|---|---|---|
| A_single_shot_metadata | 0.968 | 0.000 | 0.194 | 1.000 | 1.000 | 1.000 | 0.000 | 2.194 | 0.000 | 0 |
| B_agentic_deterministic | 0.871 | 0.194 | 0.387 | 1.839 | 1.387 | 1.839 | 0.000 | 1.806 | 0.419 | 0 |
| C_agentic_personalized | 0.903 | 0.161 | 0.355 | 1.839 | 1.387 | 1.839 | 0.000 | 1.839 | 0.419 | 0 |
| D_full_system | 0.935 | 0.161 | 0.355 | 1.903 | 1.387 | 1.903 | 1.839 | 1.839 | 0.452 | 0 |
| D_llm_unavailable | 0.903 | 0.161 | 0.355 | 1.839 | 1.387 | 1.839 | 0.000 | 1.839 | 0.419 | 0 |
| D_llm_invalid_output | 0.903 | 0.161 | 0.355 | 1.839 | 1.387 | 1.839 | 1.839 | 1.839 | 0.419 | 0 |
| D_no_explanation | 0.935 | 0.161 | 0.355 | 1.903 | 1.387 | 1.903 | 1.194 | 1.839 | 0.452 | 0 |
| B_cap3_diagnostic | 0.968 | 0.000 | 0.194 | 1.806 | 1.355 | 1.806 | 0.000 | 2.194 | 0.419 | 0 |
| C_cap3_diagnostic | 0.968 | 0.000 | 0.194 | 1.806 | 1.355 | 1.806 | 0.000 | 2.161 | 0.419 | 0 |
| D_cap3_diagnostic | 1.000 | 0.000 | 0.194 | 1.871 | 1.355 | 1.871 | 2.161 | 2.161 | 0.452 | 0 |

## MRR / success by category

| category | A_single_shot_metadata | B_agentic_deterministic | C_agentic_personalized | D_full_system |
|---|---|---|---|---|
| A_exact | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 |
| B_filename | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 |
| C_topic | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 |
| D_ambiguous | MRR 1.000 / succ 1.000 | MRR 0.000 / succ 1.000 | MRR 0.000 / succ 1.000 | MRR 0.000 / succ 1.000 |
| E_personalization | MRR 1.000 / succ 1.000 | MRR 0.500 / succ 0.500 | MRR 0.750 / succ 0.750 | MRR 0.750 / succ 0.750 |
| F_no_result | MRR   -   / succ 1.000 | MRR   -   / succ 1.000 | MRR   -   / succ 1.000 | MRR   -   / succ 1.000 |
| G_cross_user | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 |
| H_multi_strategy | MRR 0.667 / succ 0.667 | MRR 0.333 / succ 0.333 | MRR 0.333 / succ 0.333 | MRR 0.667 / succ 0.667 |

## Scenarios

- **clear_query**: PASS -- Direct retrieval in one action, zero LLM stages, correct top result.
    - OBSERVATION `A1` no LLM call at all (including presentation-only 5D explanation): attempts=1 by_stage={'goal': 0, 'expansion': 0, 'comparison': 0, 'explanation': 1}
    - OBSERVATION `B2` no LLM call at all (including presentation-only 5D explanation): attempts=1 by_stage={'goal': 0, 'expansion': 0, 'comparison': 0, 'explanation': 1}
- **ambiguous_query**: PASS -- Ambiguity handled: clarify or flagged/low-confidence result, never a silent confident guess; LLM only past the ambiguity gate.
- **weak_retrieval_escalation**: PASS -- Strategy changes mid-query in response to flat or empty observations.
- **near_tie_gating**: PASS -- Phase 5C is consulted only when the deterministic near-tie gate fires.
- **no_result**: PASS -- Nothing relevant is accessible: no file is returned.
- **cross_user_isolation**: PASS -- No file owned by another user ever appears in a result or the candidate pool.
- **personalization_end_to_end**: FAIL -- For history-sensitive queries, personalization changes the ranking the user actually receives.
    - FAIL `E1` C returns a non-empty ranking that differs from B (personalization reaches the result): B=[]/clarify C=[]/clarify
- **llm_outage**: PASS -- With the LLM unavailable or returning garbage, results equal the deterministic system's.
- **explanation_never_affects_ranking**: PASS -- Phase 5D is presentation-only.

## Limitations

- Seed corpus has no extracted_text: the semantic index is empty, so semantic_search and the semantic half of hybrid_search cannot contribute evidence here.
- Seed corpus has no telemetry/profiles: personalization uses the explicit PERSONALIZATION_FIXTURE (user_a only); its 'intended file' labels are defined by that fixture, so E-category personalization gains are by construction a test of the mechanism, not of real-user value.
- No two users share an accessible file, so personalization is ablated within-user (history vs. disabled), not across users.
- Variant D uses a scripted provider with documented simple policies; it measures architecture behaviour under controlled LLM assistance, not real-model quality.
- Embedding/reranker providers are whatever the environment resolves (see embedding_provider / reranker_provider); a sandbox fallback is NOT the intended production model.
- Small corpus (13 files) and 31 queries: differences of a single query are within noise; treat rates as descriptive.
