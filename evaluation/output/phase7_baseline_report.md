# Phase 7 benchmark report

31 queries x 10 variants. Embedding: `spacy:en_core_web_md (sandbox fallback -- see module docstring)`; reranker: `lexical_overlap_fallback (NOT a cross-encoder -- see reranker.py docstring)`; semantic index: {'files_indexed': 0, 'chunks_indexed': 0}.

## Overall retrieval (core ablation)

| variant | n_scored | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | nDCG@5 | success |
|---|---|---|---|---|---|---|---|---|---|---|
| A_single_shot_metadata | 25 | 0.960 | 0.440 | 0.288 | 0.810 | 0.930 | 0.960 | 0.960 | 0.952 | 0.935 |
| B_agentic_deterministic | 25 | 0.680 | 0.227 | 0.136 | 0.680 | 0.680 | 0.680 | 0.680 | 0.680 | 0.774 |
| C_agentic_personalized | 25 | 0.680 | 0.227 | 0.136 | 0.680 | 0.680 | 0.680 | 0.680 | 0.680 | 0.774 |
| D_full_system | 25 | 0.680 | 0.227 | 0.136 | 0.680 | 0.680 | 0.680 | 0.680 | 0.680 | 0.774 |

## Abstention-aware view (core ablation)

Clarified queries return nothing and score 0 in the table above. `answered-only` restricts to queries where a ranking was returned; `pool-ranking` is a DIAGNOSTIC of how final_rank() would order the accumulated pool (not what the system returned).

| variant | abstention rate | n_answered | P@1 answered-only | MRR answered-only | MRR pool-ranking (diag.) | HR@1 | HR@1 pool-ranking (diag.) |
|---|---|---|---|---|---|---|---|
| A_single_shot_metadata | 0.000 | 25 | 0.960 | 0.960 | 0.960 | 0.900 | 0.900 |
| B_agentic_deterministic | 0.280 | 18 | 0.944 | 0.944 | 0.960 | 0.800 | 0.900 |
| C_agentic_personalized | 0.280 | 18 | 0.944 | 0.944 | 0.960 | 0.800 | 0.950 |
| D_full_system | 0.320 | 17 | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 |

## Behaviour (all variants)

| variant | success | clarify | no_result | tool calls | retr. actions | iterations | LLM calls | LLM attempts | escalation | scope viol. |
|---|---|---|---|---|---|---|---|---|---|---|
| A_single_shot_metadata | 0.935 | 0.000 | 0.161 | 1.000 | 1.000 | 1.000 | 0.000 | 2.226 | 0.000 | 0 |
| B_agentic_deterministic | 0.774 | 0.226 | 0.355 | 1.774 | 1.323 | 1.774 | 0.000 | 1.806 | 0.387 | 0 |
| C_agentic_personalized | 0.774 | 0.226 | 0.355 | 1.774 | 1.323 | 1.774 | 0.000 | 1.806 | 0.387 | 0 |
| D_full_system | 0.774 | 0.258 | 0.387 | 1.839 | 1.323 | 1.839 | 1.774 | 1.774 | 0.419 | 0 |
| D_llm_unavailable | 0.774 | 0.226 | 0.355 | 1.774 | 1.323 | 1.774 | 0.000 | 1.806 | 0.387 | 0 |
| D_llm_invalid_output | 0.774 | 0.226 | 0.355 | 1.774 | 1.323 | 1.774 | 1.806 | 1.806 | 0.387 | 0 |
| D_no_explanation | 0.774 | 0.258 | 0.387 | 1.839 | 1.323 | 1.839 | 1.161 | 1.774 | 0.419 | 0 |
| B_cap3_diagnostic | 0.903 | 0.000 | 0.129 | 1.774 | 1.323 | 1.774 | 0.000 | 2.258 | 0.387 | 0 |
| C_cap3_diagnostic | 0.903 | 0.000 | 0.129 | 1.774 | 1.323 | 1.774 | 0.000 | 2.258 | 0.387 | 0 |
| D_cap3_diagnostic | 0.903 | 0.000 | 0.129 | 1.839 | 1.323 | 1.839 | 2.290 | 2.290 | 0.419 | 0 |

## MRR / success by category

| category | A_single_shot_metadata | B_agentic_deterministic | C_agentic_personalized | D_full_system |
|---|---|---|---|---|
| A_exact | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 |
| B_filename | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 | MRR 1.000 / succ 1.000 |
| C_topic | MRR 1.000 / succ 1.000 | MRR 0.800 / succ 0.800 | MRR 0.800 / succ 0.800 | MRR 0.800 / succ 0.800 |
| D_ambiguous | MRR 1.000 / succ 1.000 | MRR 0.000 / succ 1.000 | MRR 0.000 / succ 1.000 | MRR 0.000 / succ 1.000 |
| E_personalization | MRR 1.000 / succ 1.000 | MRR 0.500 / succ 0.500 | MRR 0.500 / succ 0.500 | MRR 0.500 / succ 0.500 |
| F_no_result | MRR   -   / succ 1.000 | MRR   -   / succ 1.000 | MRR   -   / succ 1.000 | MRR   -   / succ 1.000 |
| G_cross_user | MRR 1.000 / succ 0.800 | MRR 1.000 / succ 0.600 | MRR 1.000 / succ 0.600 | MRR 1.000 / succ 0.600 |
| H_multi_strategy | MRR 0.667 / succ 0.667 | MRR 0.333 / succ 0.333 | MRR 0.333 / succ 0.333 | MRR 0.333 / succ 0.333 |

## Scenarios

- **clear_query**: PASS -- Direct retrieval in one action, zero LLM stages, correct top result.
    - OBSERVATION `A1` no LLM call at all (including presentation-only 5D explanation): attempts=1 by_stage={'goal': 0, 'expansion': 0, 'comparison': 0, 'explanation': 1}
    - OBSERVATION `B2` no LLM call at all (including presentation-only 5D explanation): attempts=1 by_stage={'goal': 0, 'expansion': 0, 'comparison': 0, 'explanation': 1}
- **ambiguous_query**: PASS -- Ambiguity handled: clarify or flagged/low-confidence result, never a silent confident guess; LLM only past the ambiguity gate.
- **weak_retrieval_escalation**: PASS -- Strategy changes mid-query in response to flat or empty observations.
- **near_tie_gating**: PASS -- Phase 5C is consulted only when the deterministic near-tie gate fires.
- **no_result**: FAIL -- Nothing relevant is accessible: no file is returned.
    - FAIL `A_single_shot_metadata:G2` no fabricated result: status=finish returned=[8]
    - FAIL `B_agentic_deterministic:G1` no fabricated result: status=finish returned=[2]
    - FAIL `B_agentic_deterministic:G2` no fabricated result: status=finish returned=[8]
    - FAIL `C_agentic_personalized:G1` no fabricated result: status=finish returned=[2]
    - FAIL `C_agentic_personalized:G2` no fabricated result: status=finish returned=[8]
    - FAIL `D_full_system:G1` no fabricated result: status=finish returned=[2]
    - FAIL `D_full_system:G2` no fabricated result: status=finish returned=[8]
- **cross_user_isolation**: PASS -- No file owned by another user ever appears in a result or the candidate pool.
- **personalization_end_to_end**: FAIL -- For history-sensitive queries, personalization changes the ranking the user actually receives.
    - FAIL `E1` C returns a non-empty ranking that differs from B (personalization reaches the result): B=[]/clarify C=[]/clarify
    - FAIL `E2` C returns a non-empty ranking that differs from B (personalization reaches the result): B=[]/clarify C=[]/clarify
- **llm_outage**: PASS -- With the LLM unavailable or returning garbage, results equal the deterministic system's.
- **explanation_never_affects_ranking**: PASS -- Phase 5D is presentation-only.

## Limitations

- Seed corpus has no extracted_text: the semantic index is empty, so semantic_search and the semantic half of hybrid_search cannot contribute evidence here.
- Seed corpus has no telemetry/profiles: personalization uses the explicit PERSONALIZATION_FIXTURE (user_a only); its 'intended file' labels are defined by that fixture, so E-category personalization gains are by construction a test of the mechanism, not of real-user value.
- No two users share an accessible file, so personalization is ablated within-user (history vs. disabled), not across users.
- Variant D uses a scripted provider with documented simple policies; it measures architecture behaviour under controlled LLM assistance, not real-model quality.
- Embedding/reranker providers are whatever the environment resolves (see embedding_provider / reranker_provider); a sandbox fallback is NOT the intended production model.
- Small corpus (13 files) and 31 queries: differences of a single query are within noise; treat rates as descriptive.
