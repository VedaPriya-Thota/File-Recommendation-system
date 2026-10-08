# Phase 11 real-content evaluation

31 hand-labelled queries over {'alice': 29, 'bob': 3} ingested files. Embedding: `spacy:en_core_web_md (sandbox fallback -- see module docstring)`; reranker: `lexical_overlap_fallback (NOT a cross-encoder -- see reranker.py docstring)`.

> a 'lexical_overlap_fallback' reranker is NOT a CrossEncoder and 'spacy:' embeddings are the sandbox fallback; results do not represent production-model quality

## Retrieval quality

| scope | n | n_scored | P@1 | P@3 | R@3 | MRR | success | clarify | no-result |
|---|---|---|---|---|---|---|---|---|---|
| OVERALL | 31 | 27 | 0.741 | 0.272 | 0.741 | 0.777 | 0.677 | 0.032 | 0.032 |
| A_ambiguous | 3 | 3 | 0.333 | 0.222 | 0.333 | 0.400 | 0.667 | 0.333 | 0.333 |
| C_content | 4 | 4 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| G_goal_probe | 5 | 5 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| L_filename | 4 | 4 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| N_no_result | 3 | 0 | - | - | - | - | 0.000 | 0.000 | 0.000 |
| P_personalization | 2 | 2 | 1.000 | 0.333 | 0.500 | 1.000 | 1.000 | 0.000 | 0.000 |
| S_semantic | 6 | 6 | 0.167 | 0.111 | 0.333 | 0.296 | 0.167 | 0.000 | 0.000 |
| X_cross_user | 4 | 3 | 1.000 | 0.333 | 1.000 | 1.000 | 0.750 | 0.000 | 0.000 |

## Agent behaviour (averages)

| scope | iterations | retrieval actions | bm25 | semantic | hybrid | rerank | personalization | LLM calls |
|---|---|---|---|---|---|---|---|---|
| OVERALL | 3.065 | 2.290 | 0.645 | 0.323 | 0.290 | 0.387 | 0.387 | 0.000 |
| A_ambiguous | 4.333 | 2.333 | 0.333 | 0.333 | 0.333 | 1.000 | 1.000 | 0.000 |
| C_content | 2.000 | 2.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| G_goal_probe | 2.000 | 1.600 | 0.600 | 0.000 | 0.000 | 0.200 | 0.200 | 0.000 |
| L_filename | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| N_no_result | 6.000 | 4.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| P_personalization | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| S_semantic | 4.667 | 3.333 | 1.000 | 0.667 | 0.667 | 0.667 | 0.667 | 0.000 |
| X_cross_user | 3.000 | 2.500 | 0.750 | 0.500 | 0.250 | 0.250 | 0.250 | 0.000 |

## Where a correct answer becomes incorrect (stage attribution)

| scope | attribution (verdict:stage -> count) |
|---|---|
| OVERALL | false_positive:semantic_search=4, ok=21, retrieval_miss:semantic=2, retrieval_ordering:semantic_search=4 |
| A_ambiguous | ok=2, retrieval_ordering:semantic_search=1 |
| C_content | ok=4 |
| G_goal_probe | ok=5 |
| L_filename | ok=4 |
| N_no_result | false_positive:semantic_search=3 |
| P_personalization | ok=2 |
| S_semantic | ok=1, retrieval_miss:semantic=2, retrieval_ordering:semantic_search=3 |
| X_cross_user | false_positive:semantic_search=1, ok=3 |

## Per-query outcome

| id | cat | query | success | returned (top) | tools | attribution |
|---|---|---|---|---|---|---|
| L1 | L | quarterly budget review | True | alice/quarterly_budget_review.txt | metadata | ok |
| L2 | L | employee handbook | True | alice/employee_handbook.pdf | metadata | ok |
| L3 | L | server migration plan | True | alice/server_migration_plan.md | metadata | ok |
| L4 | L | chess_openings.md | True | alice/chess_openings.md | metadata | ok |
| C1 | C | sourdough levain hydration | True | alice/kitchen_notes.txt | metadata>bm25 | ok |
| C2 | C | glacier moraine sediment samples | True | alice/field_report_2023.docx | metadata>bm25 | ok |
| C3 | C | kubernetes autoscaling postmortem | True | alice/meeting_minutes_q2.txt | metadata>bm25 | ok |
| C4 | C | tetanus booster flu shot | True | alice/vaccination_records.txt | metadata>bm25 | ok |
| S1 | S | growing vegetables outdoors | False | alice/thai_curry_guide.txt | metadata>bm25>semantic>hybrid>rerank>get | retrieval_ordering:semantic_search |
| S2 | S | keeping my cardiovascular system in good shape | False | alice/hiking_trip_itinerary.txt | metadata>bm25>semantic>hybrid>rerank>get | retrieval_ordering:semantic_search |
| S3 | S | obligations when renting an apartment | False | alice/embedding_primer.md | metadata>bm25 | retrieval_miss:semantic |
| S4 | S | watching galaxies and constellations at night | True | alice/astronomy_observing_guide.txt | metadata>bm25>semantic>hybrid>rerank>get | ok |
| S5 | S | routine for learning a musical instrument | False | alice/solar_install_quote.txt | metadata>bm25>semantic>hybrid>rerank>get | retrieval_ordering:semantic_search |
| S6 | S | generating electricity from sunlight at home | False | alice/plant_biology_primer.md | metadata>bm25 | retrieval_miss:semantic |
| G1 | G | doctor appointment and vaccination schedule | True | alice/vaccination_records.txt | metadata>rerank>get | ok |
| G2 | G | how photosynthesis works in leaves | True | alice/plant_biology_primer.md | metadata>bm25 | ok |
| G3 | G | excellent seafood restaurants in Lisbon | True | alice/lisbon_dining_guide.pdf | metadata | ok |
| G4 | G | learned numeric representations of words | True | alice/embedding_primer.md | metadata>bm25 | ok |
| G5 | G | landslides on the northern slope | True | alice/erosion_survey.txt | metadata>bm25 | ok |
| A1 | A | the notes | True | alice/kitchen_notes.txt | metadata>metadata>rerank>get | ok |
| A2 | A | the plan | True | clarify | metadata>rerank>get | ok |
| A3 | A | something about money | False | alice/hiking_trip_itinerary.txt | metadata>bm25>semantic>hybrid>rerank>get | retrieval_ordering:semantic_search |
| P1 | P | dinner recipe ideas | True | alice/recipe_pasta.md | metadata | ok |
| P2 | P | trip plans | True | alice/hiking_trip_itinerary.txt | metadata | ok |
| N1 | N | quantum chromodynamics lattice gauge theory | False | alice/astronomy_observing_guide.txt | metadata>bm25>semantic>hybrid>rerank>get | false_positive:semantic_search |
| N2 | N | medieval siege warfare catapults and trebuchets | False | alice/hiking_trip_itinerary.txt | metadata>bm25>semantic>hybrid>rerank>get | false_positive:semantic_search |
| N3 | N | cryptocurrency mining hardware profitability | False | alice/car_maintenance_log.txt | metadata>bm25>semantic>hybrid>rerank>get | false_positive:semantic_search |
| X1 | X | merger acquisition negotiation valuation terms | False | alice/car_maintenance_log.txt | metadata>bm25>semantic>hybrid>rerank>get | false_positive:semantic_search |
| X2 | X | tomato seedlings raised beds compost | True | alice/garden_journal.md | metadata>bm25 | ok |
| X3 | X | growing vegetables outdoors | True | bob/bob_garden_diary.txt | metadata>bm25>semantic | ok |
| X4 | X | quarterly budget review | True | bob/quarterly_budget_review.txt | metadata | ok |
