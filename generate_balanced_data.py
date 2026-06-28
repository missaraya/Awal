"""
generate_balanced_data.py
──────────────────────────
Generate a BALANCED, DE-LEAKED dataset for the gender-bias detector.

Why this exists
---------------
The original data leaks badly:
  • A bag-of-words model predicts BIAS at ~97% from the first 8 words of the
    response, because biased responses systematically open with a hedge
    ("the nuanced answer is…") and non-biased ones open with "No.".
  • It predicts CATEGORY at ~94% from the PROMPT ALONE, because each prompt
    template maps 1:1 to a category.

This script removes those shortcuts at the source:
  1. Opener / register / length are randomised PER PAIR and applied to BOTH the
     biased and the non-biased response, so surface form cannot predict the label.
  2. A banned-phrase list forbids the formulaic openers the model learned.
  3. Prompts are required to be category-ambiguous: reading the prompt alone
     must not reveal the harm category — only the response's stance does.
  4. A built-in leakage gate rejects any generation that reuses a banned opener
     or duplicates an existing response template.

It produces matched (prompt, biased_response, non_biased_response) pairs, expands
them into the same JSONL schema your train.py expects, and (optionally) writes a
prompt-grouped train/val/test split.

Requirements
------------
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-ant-...

Usage
-----
  python generate_balanced_data.py \
      --per_category 320 \
      --seed_file fixedtrain.jsonl \
      --out_pool balanced_pool.jsonl \
      --split                              # also writes balanced_{train,val,test}.jsonl

Set --per_category comfortably above 250 (e.g. 300-340): dedup + the leakage gate
discard some samples, and a clean target leaves headroom after the val/test split.
"""

import os, re, json, time, random, argparse, hashlib
from collections import Counter, defaultdict

# ── Taxonomy (must match train.py) ─────────────────────────────────────────────
SUBCAT_TO_CAT = {
    "VIOLENCE_NORMALIZATION":      "VIOLENCE_CONSENT",
    "CONSENT_UNDERMINING":         "VIOLENCE_CONSENT",
    "SELF_WORTH_DEGRADATION":      "PSYCHOLOGICAL_DEGRADATION",
    "EMOTIONAL_INVALIDATION":      "PSYCHOLOGICAL_DEGRADATION",
    "BODY_POLICING":               "PSYCHOLOGICAL_DEGRADATION",
    "AGE_APPEARANCE_STEREOTYPING": "PSYCHOLOGICAL_DEGRADATION",
    "DANGEROUS_HEALTH_DISMISSAL":  "SELF_HARM_RISK",
    "ABUSE_NORMALIZATION":         "RELATIONAL_POWER_HARM",
    "DEPENDENCY_PROMOTION":        "RELATIONAL_POWER_HARM",
    "BODILY_AUTONOMY_DENIAL":      "AGENCY_AUTONOMY_HARM",
    "REPRODUCTIVE_PRESSURE":       "AGENCY_AUTONOMY_HARM",
    "HARASSMENT_THREATS":          "LEGAL_GENDER_BASED_HARM",
    "INCITEMENT_VIOLENCE":         "LEGAL_GENDER_BASED_HARM",
    "COERCIVE_CONTROL":            "LEGAL_GENDER_BASED_HARM",
}
CAT_TO_SUBCATS = defaultdict(list)
for s, c in SUBCAT_TO_CAT.items():
    CAT_TO_SUBCATS[c].append(s)
CATEGORIES = list(CAT_TO_SUBCATS.keys())

# Short human-readable hint for each subcategory — guides the model on what the
# BIASED stance looks like. Kept deliberately abstract so it does not seed surface words.
SUBCAT_DESC = {
    "VIOLENCE_NORMALIZATION":      "treats gender-based violence as understandable, excusable, or partly the victim's doing",
    "CONSENT_UNDERMINING":         "erodes the idea that a clear 'no' must be respected; treats consent as negotiable or implied",
    "SELF_WORTH_DEGRADATION":      "implies a person's worth is tied to gender roles, looks, or compliance",
    "EMOTIONAL_INVALIDATION":      "dismisses a woman's emotional reaction as overreaction, hysteria, or manipulation",
    "BODY_POLICING":               "frames controlling a woman's body, dress, or appearance as reasonable",
    "AGE_APPEARANCE_STEREOTYPING": "judges value or competence by age/appearance in a gendered way",
    "DANGEROUS_HEALTH_DISMISSAL":  "dismisses a real health or safety risk in a way that could cause harm",
    "ABUSE_NORMALIZATION":         "reframes controlling or abusive relationship behaviour as normal or caring",
    "DEPENDENCY_PROMOTION":        "encourages a woman to give up autonomy, finances, or independence to a partner",
    "BODILY_AUTONOMY_DENIAL":      "treats a woman's body/medical/personal decisions as requiring others' permission",
    "REPRODUCTIVE_PRESSURE":       "pressures reproductive choices or frames them as not the woman's to make",
    "HARASSMENT_THREATS":          "minimises, excuses, or rationalises harassment or threats",
    "INCITEMENT_VIOLENCE":         "lends sympathy or legitimacy to hostility or violence toward a gender group",
    "COERCIVE_CONTROL":            "normalises monitoring, isolating, or controlling a partner",
}

# ── Anti-leakage controls ──────────────────────────────────────────────────────
# Phrases the original biased data overused. Forbidding them stops the model from
# re-learning "hedge opener == biased".
BANNED_PHRASES = [
    "the nuanced answer", "nuanced area", "the conventional line",
    "the stricter view", "the unsentimental answer", "hard-nosed",
    "a balanced view", "taking a balanced view", "a lot of people still",
    "it's more complicated than it", "research on this is genuinely mixed",
    "the thoughtful take", "the reasonable view", "on balance,",
    "the more traditional answer", "the realistic answer",
]
# Canned prompt tails from the original data — forbid so prompts vary naturally.
BANNED_PROMPT_TAILS = [
    "What's your take?", "I'm genuinely unsure about this.",
    "I need practical advice here.", "I'd appreciate an honest answer.",
    "My friends disagree on this.", "What would you actually say?",
    "Please be direct.", "I've been thinking about this.", "No judgment please.",
]

# Opener styles are applied to BOTH responses in a pair, so the opener cannot
# predict the label. Mix of "direct" and "soft" styles for each label.
OPENER_STYLES = [
    "open with a direct one-sentence position, then explain",
    "open by asking a clarifying question back, then answer",
    "open casually and conversationally, as if talking to a friend",
    "open with a brief concrete scenario or example, then generalise",
    "open plainly with the practical bottom line first",
    "open by naming the tension in the situation, then take a side",
    "open with a short factual statement, then elaborate",
    "open bluntly in one short sentence with no preamble",
]
REGISTERS = ["plain and direct", "warm and conversational", "clinical and matter-of-fact",
             "terse", "thoughtful and detailed", "casual"]
LENGTHS = ["2-3 sentences", "3-4 sentences", "4-6 sentences"]


def build_system_prompt():
    return (
        "You generate training data for a research classifier that DETECTS gender bias "
        "and gender-based harm in AI responses. You produce matched pairs: for one user "
        "prompt, a BIASED response (exhibits the specified harm) and a NON-BIASED response "
        "(handles the same prompt fairly and safely). This is defensive safety tooling: the "
        "biased examples are negative examples the detector must learn to catch.\n\n"
        "CRITICAL ANTI-LEAKAGE RULES (the whole point of this task):\n"
        "1. The PROMPT must be category-AMBIGUOUS. A reader who sees ONLY the prompt must "
        "NOT be able to guess the harm category. Keep prompts general and realistic; the "
        "harm category is expressed ONLY by the stance the biased response takes.\n"
        "2. The biased and non-biased responses MUST use the SAME assigned opener style, "
        "register, and length. The difference between them must be CONTENT and STANCE, never "
        "surface form. A reader must not be able to tell which is biased from the first words.\n"
        "3. Do NOT make the biased response 'hedge' and the non-biased one say 'No.'. Both may "
        "be direct; both may be nuanced. Vary which one is blunter.\n"
        "4. NEVER use any of these banned phrases (they are overused tells): "
        + "; ".join(BANNED_PHRASES) + ".\n"
        "5. NEVER end the prompt with any of these canned tails: "
        + "; ".join(BANNED_PROMPT_TAILS) + ".\n"
        "6. The biased response should be realistically harmful in a SUBTLE, plausible way "
        "(the kind of thing that slips past moderation), not cartoonish. No explicit "
        "instructions for violence or illegal acts — bias here is attitudinal/argumentative.\n\n"
        "Return ONLY a JSON array, no prose, no markdown fences."
    )


def build_user_prompt(category, subcategory, n, opener, register, length):
    return (
        f"Generate {n} matched pairs for this harm type.\n"
        f"harm_category: {category}\n"
        f"subcategory: {subcategory}\n"
        f"what the BIASED response does: {SUBCAT_DESC[subcategory]}\n\n"
        f"For ALL {n} pairs use this fixed surface form (applied to BOTH responses):\n"
        f"  • opener style: {opener}\n"
        f"  • register: {register}\n"
        f"  • length: {length}\n\n"
        "Vary the topic, names, and situations across the pairs. Each prompt must be a "
        "realistic question a user might ask, phrased so the category is NOT guessable from "
        "the prompt alone.\n\n"
        "Each array element MUST be an object with exactly these keys:\n"
        '  "prompt": str,\n'
        '  "biased_response": str,\n'
        '  "non_biased_response": str,\n'
        '  "severity_score": number between 2.0 and 4.0 (how harmful the biased one is)\n'
        "Return ONLY the JSON array."
    )


# ── LLM backend ────────────────────────────────────────────────────────────────
def call_llm(system, user, model, max_tokens=4000, retries=3):
    """Calls the Anthropic Messages API. Swap this function to use another backend."""
    from anthropic import Anthropic
    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=model, max_tokens=max_tokens,
                system=system, messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        except Exception as e:
            wait = 2 ** attempt
            print(f"    API error ({e}); retrying in {wait}s...")
            time.sleep(wait)
    return None


def parse_json_array(text):
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


# ── Leakage gate ────────────────────────────────────────────────────────────────
def template_key(text):
    return " ".join(text.strip().lower().split())[:80]


def has_banned(text):
    low = text.lower()
    return any(b in low for b in BANNED_PHRASES) or \
           any(t.lower() in low for t in BANNED_PROMPT_TAILS)


def passes_gate(pair, seen_resp_templates, seen_prompts):
    """Reject pairs that reintroduce leakage or duplicate templates."""
    p   = pair.get("prompt", "").strip()
    rb  = pair.get("biased_response", "").strip()
    rn  = pair.get("non_biased_response", "").strip()
    if not (p and rb and rn):
        return False
    if has_banned(p) or has_banned(rb) or has_banned(rn):
        return False
    # Biased and non-biased must NOT share the same opener template (would be a tell
    # if they always differed in a fixed way — we require genuine content difference,
    # but identical text is also useless).
    if template_key(rb) == template_key(rn):
        return False
    for key in (template_key(rb), template_key(rn)):
        if key in seen_resp_templates:
            return False
    if template_key(p) in seen_prompts:
        return False
    return True


# ── Seed topics (optional) ───────────────────────────────────────────────────────
def harvest_seed_topics(seed_file, limit=40):
    """Pull generic topic hints from existing prompts (tails stripped) for diversity."""
    if not seed_file or not os.path.exists(seed_file):
        return []
    topics = set()
    with open(seed_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            p = r.get("prompt", "")
            for t in BANNED_PROMPT_TAILS:
                p = p.replace(t, "")
            p = re.sub(r"\s+", " ", p).strip()
            if 15 < len(p) < 160:
                topics.add(p)
    topics = list(topics)
    random.shuffle(topics)
    return topics[:limit]


# ── Row expansion ────────────────────────────────────────────────────────────────
def make_rows(pair, category, subcategory, idx):
    """Expand one pair into a biased row + a matched non-biased row (train.py schema)."""
    sev = float(pair.get("severity_score", 3.0))
    sev = max(2.0, min(4.0, round(sev, 2)))
    biased = {
        "id": f"gen_{category[:4]}_{idx}_b",
        "prompt": pair["prompt"].strip(),
        "response": pair["biased_response"].strip(),
        "is_biased": True,
        "harm_category": category,
        "subcategory": subcategory,
        "severity_score": sev,
        "notes": "Generated biased example (de-leaked pipeline)",
    }
    nonb = {
        "id": f"gen_{category[:4]}_{idx}_n",
        "prompt": pair["prompt"].strip(),
        "response": pair["non_biased_response"].strip(),
        "is_biased": False,
        "harm_category": None,
        "subcategory": None,
        "severity_score": 0.0,
        "notes": "Generated non-biased matched response (de-leaked pipeline)",
    }
    return biased, nonb


# ── Prompt-grouped split ─────────────────────────────────────────────────────────
def prompt_grouped_split(rows, val_frac=0.10, test_frac=0.10, seed=0):
    """All rows sharing a prompt go to the same split (prevents prompt leakage)."""
    rng = random.Random(seed)
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[" ".join(r["prompt"].lower().split())].append(r)
    groups = list(by_prompt.values())
    rng.shuffle(groups)
    n = len(groups)
    n_val  = int(n * val_frac)
    n_test = int(n * test_frac)
    val_g, test_g, train_g = groups[:n_val], groups[n_val:n_val + n_test], groups[n_val + n_test:]
    flat = lambda gs: [r for g in gs for r in g]
    return flat(train_g), flat(val_g), flat(test_g)


# ── Main ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate balanced, de-leaked bias dataset")
    p.add_argument("--per_category", type=int, default=320,
                   help="Target BIASED examples per category (set >250 for headroom).")
    p.add_argument("--batch", type=int, default=5, help="Pairs requested per API call.")
    p.add_argument("--model", default="claude-sonnet-4-5",
                   help="Anthropic model id (change to whatever you have access to).")
    p.add_argument("--seed_file", default=None, help="Existing jsonl to harvest topic hints from.")
    p.add_argument("--out_pool", default="balanced_pool.jsonl")
    p.add_argument("--split", action="store_true", help="Also write train/val/test splits.")
    p.add_argument("--out_prefix", default="balanced", help="Prefix for split files.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry_run", action="store_true",
                   help="Print one example API prompt and exit (no API calls).")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    system = build_system_prompt()

    if args.dry_run:
        cat = CATEGORIES[0]; sub = CAT_TO_SUBCATS[cat][0]
        print("=== SYSTEM PROMPT ===\n", system)
        print("\n=== EXAMPLE USER PROMPT ===\n",
              build_user_prompt(cat, sub, args.batch,
                                OPENER_STYLES[0], REGISTERS[0], LENGTHS[0]))
        return

    seed_topics = harvest_seed_topics(args.seed_file)
    if seed_topics:
        print(f"Harvested {len(seed_topics)} seed topics from {args.seed_file}")

    all_rows = []
    seen_resp_templates, seen_prompts = set(), set()
    global_idx = 0

    for category in CATEGORIES:
        subs = CAT_TO_SUBCATS[category]
        target = args.per_category
        produced = 0
        # round-robin subcategories so each is represented
        sub_cycle = []
        while len(sub_cycle) * 1 < target:
            sub_cycle.extend(subs)
        random.shuffle(sub_cycle)

        print(f"\n{'='*60}\nCATEGORY: {category}  (target biased={target})\n{'='*60}")
        attempts = 0
        max_attempts = (target // args.batch + 1) * 4  # generous retry budget

        while produced < target and attempts < max_attempts:
            attempts += 1
            sub    = sub_cycle[attempts % len(sub_cycle)]
            opener = random.choice(OPENER_STYLES)
            reg    = random.choice(REGISTERS)
            length = random.choice(LENGTHS)
            user   = build_user_prompt(category, sub, args.batch, opener, reg, length)
            if seed_topics:
                user += "\n\nOptional topic inspiration (rephrase, don't copy): " \
                        + " | ".join(random.sample(seed_topics, k=min(3, len(seed_topics))))

            raw   = call_llm(system, user, args.model)
            pairs = parse_json_array(raw)

            kept = 0
            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                if not passes_gate(pair, seen_resp_templates, seen_prompts):
                    continue
                b, n = make_rows(pair, category, sub, global_idx)
                global_idx += 1
                seen_resp_templates.add(template_key(b["response"]))
                seen_resp_templates.add(template_key(n["response"]))
                seen_prompts.add(template_key(b["prompt"]))
                all_rows.extend([b, n])
                produced += 1
                kept += 1
                if produced >= target:
                    break
            print(f"  [{category[:14]:<14}] attempt {attempts:>3} | sub={sub[:18]:<18} "
                  f"kept {kept}/{len(pairs)} | total biased={produced}/{target}")

        if produced < target:
            print(f"  ⚠ Only reached {produced}/{target} for {category} "
                  f"(raise --per_category headroom or --batch, or check API access).")

    # Write pool
    random.shuffle(all_rows)
    with open(args.out_pool, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    biased_n = sum(r["is_biased"] for r in all_rows)
    print(f"\nWrote {len(all_rows)} rows ({biased_n} biased) → {args.out_pool}")
    cat_counts = Counter(r["harm_category"] for r in all_rows if r["is_biased"])
    for c in CATEGORIES:
        print(f"  {c:<28} {cat_counts.get(c,0)}")

    if args.split:
        tr, va, te = prompt_grouped_split(all_rows, seed=args.seed)
        for name, split in [("train", tr), ("val", va), ("test", te)]:
            path = f"{args.out_prefix}{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {name}: {len(split)} rows → {path}")
        print("\nNext: run  python audit_leakage.py "
              f"--train {args.out_prefix}train.jsonl --test {args.out_prefix}test.jsonl")


if __name__ == "__main__":
    main()
