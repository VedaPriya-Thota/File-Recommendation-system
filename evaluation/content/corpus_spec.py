"""
Phase 11 real-content evaluation corpus -- the single source of truth for every
document's text. Each document is ~30-50 words (one chunk). The committed files
under evaluation/content/corpus/<owner>/ are generated from this spec by
make_corpus.py; the evaluation reads the committed files, and ground truth in
queries.py refers to these files by (owner, relative path) only.

Design intent per group (verified statically in tests, NOT by running the system):
  * lexical-filename   : the filename itself states the topic
  * content-only       : distinctive words appear only in the body (BM25's job, not metadata's)
  * semantic           : bodies deliberately avoid the words later used in the paraphrase queries
  * distractors        : same-domain / unrelated documents that make ranking non-trivial
  * goal probes        : each body uses a word that a SUBSTRING keyword match would misread
                         as a file-type hint (doctor~doc, photosynthesis~photo, excellent~excel,
                         landslides~slides, representations~presentation)
  * bob                : a second user whose documents overlap alice's topics (isolation)
"""

ALICE = {
    # --- lexical / filename -------------------------------------------------
    "quarterly_budget_review.txt": "Budget review for the quarter. Departments overspent on travel while hiring costs stayed below plan. Finance recommends freezing discretionary spending until the next forecast is approved.",
    "employee_handbook.pdf": "Employee handbook. Annual leave is twenty days. Remote work requires manager approval. Passwords must be rotated every ninety days and laptops encrypted. Conduct complaints go to the people team.",
    "server_migration_plan.md": "# Server migration plan\nWe will migrate the production database to the new cloud region over a weekend maintenance window. A rollback script restores the old cluster if replication lag exceeds the agreed limit.",
    "chess_openings.md": "The Sicilian defence answers the king pawn with a flank pawn. Control the centre, develop knights early and castle.",
    # --- content-only lexical ------------------------------------------------
    "kitchen_notes.txt": "Sourdough starter log. Feed the levain twice daily and keep hydration near seventy percent. The loaf proofed overnight in the fridge and baked with steam for a crackling crust.",
    "field_report_2023.docx": "Fieldwork summary. We measured glacier retreat along the northern moraine and collected sediment samples from three meltwater streams for isotope analysis.",
    "meeting_minutes_q2.txt": "Incident review. The autoscaling policy on the kubernetes cluster failed during the traffic spike. Action items cover alert thresholds and a blameless postmortem document.",
    # --- semantic targets ------------------------------------------------------
    "garden_journal.md": "Planted tomato seedlings in raised beds. The soil needs compost and mulch to hold moisture. Water every morning and pull weeds before they spread to the lettuce rows.",
    "heart_health_leaflet.pdf": "Lowering cholesterol and blood pressure reduces the risk of cardiac arrest. Walk briskly most days, eat more fibre, and limit salt and fried food.",
    "lease_agreement_summary.docx": "The tenant pays rent on the first of each month and a refundable deposit. The landlord must give written notice before any eviction and repair the plumbing promptly.",
    "astronomy_observing_guide.txt": "Set up the telescope away from city lights and let the eyepiece cool. Jupiter and Saturn are bright in autumn, and a dark sky shows the nebula in Orion.",
    "piano_practice_plan.md": "Practice scales and arpeggios with a metronome for twenty minutes, then work on the left hand of the sonata. Mark fingerings on the sheet music.",
    "solar_install_quote.txt": "Quote for twelve panels and one inverter on the south roof, producing about five kilowatt of peak output. Installation takes two days and includes a ten year warranty.",
    # --- distractors (also personalization pairs: pasta/thai, hiking/flight) -------
    "stock_market_primer.txt": "Stocks represent ownership in a company. Index funds spread risk across many shares, and dividends are paid from profits.",
    "recipe_pasta.md": "Boil the pasta, simmer tomatoes with garlic and basil, then toss with parmesan. Serves four for a quick weeknight dinner.",
    "thai_curry_guide.txt": "Fry the curry paste in coconut milk, add chicken, bamboo shoots and basil, and serve with jasmine rice. A fragrant dinner that takes thirty minutes.",
    "hiking_trip_itinerary.txt": "Day one: drive to the trailhead and hike to the alpine lake. Day two: summit the ridge before noon and camp by the river.",
    "flight_booking_receipt.pdf": "Flight confirmation. Departure from Berlin at nine in the morning, arrival in Lisbon at noon. Seat 14C, one checked bag included.",
    "car_maintenance_log.txt": "Changed the engine oil and replaced the brake pads. Tyre pressure checked. Next service due at sixty thousand kilometres.",
    "marathon_training_schedule.docx": "Sixteen week plan. Long runs on Sunday increase by two kilometres weekly, with tempo intervals on Tuesday and rest on Friday.",
    "book_club_notes.md": "Notes from the book club. We discussed the unreliable narrator in the novel and voted to read a mystery next month.",
    "tax_filing_checklist.pdf": "Gather income statements, receipts for deductible expenses and last year's return. File before the deadline to avoid penalties.",
    "wifi_troubleshooting.txt": "Restart the router, check the cable, and move closer to the access point. If the signal drops again update the firmware.",
    "team_offsite_agenda.docx": "Agenda for the offsite: icebreaker, strategy session, lunch, workshop on priorities, and a closing retrospective.",
    # --- goal probes (substring false-positive class) ------------------------------
    "vaccination_records.txt": "Doctor appointment reminders: the tetanus booster is due in March and the flu shot each autumn. Keep the vaccination card with the passport.",
    "plant_biology_primer.md": "Photosynthesis converts sunlight, water and carbon dioxide into sugar and oxygen inside chloroplasts. Leaves contain the green pigment chlorophyll.",
    "lisbon_dining_guide.pdf": "Lisbon dining guide. Try grilled sardines near the harbour, custard tarts in Belem, and an excellent seafood rice at the family run tavern.",
    "embedding_primer.md": "Learned numeric representations of words place similar meanings close together, so search can match paraphrases even when the wording differs.",
    "erosion_survey.txt": "Survey of landslides on the northern slope. Heavy rain loosened the soil and eroded the road embankment.",
}

BOB = {
    "merger_negotiation_terms.txt": "Merger negotiation notes: valuation, earn-out terms and the acquisition timeline for the target company.",
    "bob_garden_diary.txt": "Planted tomato seedlings in raised beds. Compost, mulch and morning watering keep the lettuce rows healthy.",
    "quarterly_budget_review.txt": "Bob's budget review for the quarter. Marketing spend rose while the sales forecast was revised upward.",
}

CORPUS = {"alice": ALICE, "bob": BOB}
