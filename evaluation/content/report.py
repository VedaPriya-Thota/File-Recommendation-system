"""Markdown rendering of a Phase 11 content-evaluation result (pure formatting)."""


def _f(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def render_markdown(out: dict) -> str:
    m, s = out["metadata"], out["summary"]
    L = ["# Phase 11 real-content evaluation", "",
         f"{m['queries']} hand-labelled queries over {m['corpus_files']} ingested files. "
         f"Embedding: `{m.get('embedding_provider')}`; reranker: `{m.get('reranker_provider')}`.", "",
         f"> {m['provider_note']}", "",
         f"> Production-model evaluation: **{m['production_models']['production_model_evaluation']}** "
         f"(intended embedding `{m['production_models']['embedding']['intended']}`, intended reranker `{m['production_models']['reranker']['intended']}`).", "",
         "## Retrieval quality", "",
         "| scope | n | n_scored | P@1 | P@3 | R@3 | MRR | success | clarify | no-result |", "|---|---|---|---|---|---|---|---|---|---|"]
    rows = [("OVERALL", s["overall"])] + [(c, v) for c, v in s["by_category"].items()]
    for name, v in rows:
        r, b = v["retrieval"], v["behavior"]
        L.append(f"| {name} | {r['n']} | {r['n_scored']} | {_f(r['precision_at_1'])} | {_f(r['precision_at_3'])} | {_f(r['recall_at_3'])} | "
                 f"{_f(r['MRR'])} | {_f(r['success_rate'])} | {_f(b['clarification_rate'])} | {_f(b['no_result_rate'])} |")
    L += ["", "## Agent behaviour (averages)", "",
          "| scope | iterations | retrieval actions | bm25 | semantic | hybrid | rerank | personalization | LLM calls |", "|---|---|---|---|---|---|---|---|---|"]
    for name, v in rows:
        b = v["behavior"]
        L.append(f"| {name} | {_f(b['average_iterations'])} | {_f(b['average_retrieval_actions'])} | {_f(b['average_bm25_search_calls'])} | "
                 f"{_f(b['average_semantic_search_calls'])} | {_f(b['average_hybrid_search_calls'])} | {_f(b['average_rerank_candidates_calls'])} | "
                 f"{_f(b['average_get_user_activity_calls'])} | {_f(b['average_llm_calls'])} |")
    L += ["", "## Where a correct answer becomes incorrect (stage attribution)", "",
          "| scope | attribution (verdict:stage -> count) |", "|---|---|"]
    for name, v in rows:
        L.append(f"| {name} | {', '.join(f'{k}={n}' for k, n in v['attribution'].items())} |")
    L += ["", "## Retrieval-provider vs agent (which LAYER lost the answer)", "",
          "`semantic_provider` = the semantic stage ALONE did not rank a relevant file first (the embedding model failed); "
          "`agent_decision` = a stage that ranks it first was never consulted; `reranker` / `hybrid` / `personalization` / "
          "`final_ranking` = it was retrieved first and later lost; `no_relevance_floor` = a no-relevant-file query returned neighbours.", "",
          "| scope | layer counts |", "|---|---|"]
    for name, v in rows:
        L.append(f"| {name} | {', '.join(f'{k}={n}' for k, n in v['layers'].items())} |")
    cs = s.get("coverage_sensitivity", {}).get("by_boundary", {})
    if cs:
        L += ["", "## Weak-lexical boundary sensitivity (standalone BM25 top hits)", "",
              "| boundary | correct hits flagged weak | wrong hits flagged weak |", "|---|---|---|"]
        for t, v in cs.items():
            L.append(f"| < {t} | {v['correct_flagged_weak']} / {v['n_correct']} | {v['wrong_flagged_weak']} / {v['n_wrong']} |")
    sf = s.get("semantic_floor_evidence")
    if sf:
        L += ["", "## Is a semantic relevance floor supported? (raw top-1 cosine, this provider)", "",
              f"- no-relevant-file queries: {sf['no_relevant_top1_raw_cosine']} (n={sf['n']['no_relevant']})",
              f"- queries whose semantic top-1 is correct: {sf['correct_top1_raw_cosine']} (n={sf['n']['correct']})",
              f"- queries whose semantic top-1 is wrong: {sf['wrong_top1_raw_cosine']} (n={sf['n']['wrong']})",
              f"- ranges overlap: **{sf['ranges_overlap']}** -> floor supported: **{sf['floor_supported']}**"]
    diag = out.get("diagnostics", {}).get("results", [])
    if diag:
        L += ["", "## Diagnostics (pre-registered, reported separately, not in any metric above)", "",
              "| id | query | tools | status | basis | returned | get_user_activity calls |", "|---|---|---|---|---|---|---|"]
        for r in diag:
            L.append(f"| {r['qid']} | {r['query']} | {'>'.join(t.split('_')[0] for t in r['tool_sequence'])} | {r['status']} | {r['decision_basis']} | "
                     f"{', '.join(r['returned_files'][:2])} | {r['tool_counts']['get_user_activity']} |")
    L += ["", "## Per-query outcome", "", "| id | cat | query | success | returned (top) | tools | attribution | layer |", "|---|---|---|---|---|---|---|---|"]
    for r in out["results"]:
        a = r["attribution"]
        L.append(f"| {r['qid']} | {r['category'][:1]} | {r['query']} | {r['success']} | {(r['returned_files'][:1] or ['clarify' if r['clarified'] else '(none)'])[0]} | "
                 f"{'>'.join(t.split('_')[0] for t in r['tool_sequence'])} | {a['verdict']}{(':' + str(a['stage'])) if a['stage'] else ''} | {r['layer']['layer']} |")
    return "\n".join(L) + "\n"
