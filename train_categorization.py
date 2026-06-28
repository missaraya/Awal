import json, os, random, time, argparse, hashlib
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Taxonomy ──────────────────────────────────────────────────────────────────
CAT_LABELS = [
    "VIOLENCE_CONSENT",
    "PSYCHOLOGICAL_DEGRADATION",
    "SELF_HARM_RISK",
    "RELATIONAL_POWER_HARM",
    "AGENCY_AUTONOMY_HARM",
    "LEGAL_GENDER_BASED_HARM",
]

SUBCAT_LABELS = [
    "VIOLENCE_NORMALIZATION",
    "CONSENT_UNDERMINING",
    "SELF_WORTH_DEGRADATION",
    "EMOTIONAL_INVALIDATION",
    "BODY_POLICING",
    "AGE_APPEARANCE_STEREOTYPING",
    "DANGEROUS_HEALTH_DISMISSAL",
    "ABUSE_NORMALIZATION",
    "DEPENDENCY_PROMOTION",
    "BODILY_AUTONOMY_DENIAL",
    "REPRODUCTIVE_PRESSURE",
    "HARASSMENT_THREATS",
    "INCITEMENT_VIOLENCE",
    "COERCIVE_CONTROL",
]

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

CAT2IDX    = {l: i for i, l in enumerate(CAT_LABELS)}
SUBCAT2IDX = {l: i for i, l in enumerate(SUBCAT_LABELS)}
IDX2CAT    = {i: l for l, i in CAT2IDX.items()}
IDX2SUBCAT = {i: l for l, i in SUBCAT2IDX.items()}
N_CATS     = len(CAT_LABELS)
N_SUBCATS  = len(SUBCAT_LABELS)
IGNORE_INDEX = -100

SUBCAT_PARENT = torch.tensor(
    [CAT2IDX[SUBCAT_TO_CAT[s]] for s in SUBCAT_LABELS], dtype=torch.long
)

# SUBCAT_VALID_MASK[cat_i, sub_j] = True if sub_j belongs to cat_i
SUBCAT_VALID_MASK = torch.zeros(N_CATS, N_SUBCATS, dtype=torch.bool)
for _si, _sl in enumerate(SUBCAT_LABELS):
    SUBCAT_VALID_MASK[CAT2IDX[SUBCAT_TO_CAT[_sl]], _si] = True

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else "cpu"
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_LEN       = 256
BATCH_SIZE    = 4
EPOCHS        = 5
PATIENCE      = 2
UNFREEZE_TOP  = 4        # how many top BERT layers to fine-tune in Stage 2
LR            = 1e-5     # lower LR for backbone fine-tuning
HEAD_LR       = 1.5e-4
WARMUP_RATIO  = 0.10
DROPOUT       = 0.20
WEIGHT_DECAY  = 0.01

# Categorization-only loss weights
W_CATEGORY = 1.0
W_SUBCAT   = 1.0
W_SCORE    = 0.3
W_HIER     = 0.20

# ── IO defaults ───────────────────────────────────────────────────────────────
DEFAULT_TRAIN = "fixedtrain.jsonl"
DEFAULT_VAL   = "fixedval.jsonl"
DEFAULT_TEST  = "fixedtest.jsonl"
DEFAULT_OUT   = "./outputs/categorization_model"
DEFAULT_LOG   = "./outputs/categorization_log.json"


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 – Category / subcategory training")
    p.add_argument("--train_path",     default=DEFAULT_TRAIN)
    p.add_argument("--val_path",       default=DEFAULT_VAL)
    p.add_argument("--test_path",      default=DEFAULT_TEST)
    p.add_argument("--detection_dir",  required=True,
                   help="Path to the Stage-1 detection model directory (contains heads.pt).")
    p.add_argument("--output_dir",     default=DEFAULT_OUT)
    p.add_argument("--log_path",       default=DEFAULT_LOG)
    p.add_argument("--epochs",         type=int,   default=EPOCHS)
    p.add_argument("--batch_size",     type=int,   default=BATCH_SIZE)
    p.add_argument("--max_len",        type=int,   default=MAX_LEN)
    p.add_argument("--unfreeze_top",   type=int,   default=UNFREEZE_TOP,
                   help="Fine-tune the top N BERT encoder layers (rest stay frozen).")
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--head_lr",        type=float, default=HEAD_LR)
    p.add_argument("--dropout",        type=float, default=DROPOUT)
    p.add_argument("--response_only",  action="store_true", default=False)
    p.add_argument("--detection_threshold", type=float, default=None,
                   help="Override the detection threshold loaded from Stage 1 (optional).")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_ws(text):
    return " ".join(text.strip().lower().split())


def exact_pair_key(r):
    return normalize_ws(r["prompt"]) + " ||| " + normalize_ws(r["response"])


def prompt_key(r):
    return normalize_ws(r["prompt"])


def response_hash(text):
    return hashlib.md5(normalize_ws(text).encode("utf-8")).hexdigest()


def resp_template_key(text):
    return " ".join(text.strip().lower().split())[:80]


def audit_splits(train_raw, val_raw, test_raw, train_path, val_path, test_path):
    def overlap(a, b, key_fn):
        return len({key_fn(x) for x in a} & {key_fn(x) for x in b})

    def resp_tmpl_key(r):
        return resp_template_key(r["response"])

    print("\nSplit audit")
    print("-" * 60)
    print(f"Train={len(train_raw)} | Val={len(val_raw)} | Test={len(test_raw)}")

    prompt_tv = overlap(train_raw, val_raw,  prompt_key)
    prompt_tt = overlap(train_raw, test_raw, prompt_key)
    prompt_vt = overlap(val_raw,   test_raw, prompt_key)
    if prompt_tv > 0 or prompt_tt > 0 or prompt_vt > 0:
        raise RuntimeError(
            f"\n[LEAKAGE] Prompt overlap detected: train/val={prompt_tv}, "
            f"train/test={prompt_tt}, val/test={prompt_vt}.\n"
            "Re-split using prompt-grouped split."
        )
    for name, split in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        biased = sum(bool(r["is_biased"]) for r in split)
        print(f"{name.title():<5} biased={biased}  non_biased={len(split)-biased}")
    print("-" * 60)


# ── Dataset (biased-only for categorization) ──────────────────────────────────
class CategoryDataset(Dataset):
    """
    Loads only the BIASED samples; non-biased rows are skipped entirely
    because the categorization heads only operate on biased inputs.
    """
    def __init__(self, path, tokenizer, max_len=MAX_LEN, response_only=False):
        raw = load_jsonl(path)
        self.records = []
        skipped = 0
        texts = []

        for r in raw:
            if not bool(r["is_biased"]):
                continue                           # skip non-biased
            cat = r.get("harm_category")
            sub = r.get("subcategory")
            if cat not in CAT2IDX or sub not in SUBCAT2IDX:
                skipped += 1
                continue

            if response_only:
                text = r['response'].strip()
            else:
                text = f"[PROMPT] {r['prompt'].strip()} [RESPONSE] {r['response'].strip()}"

            self.records.append({
                "cat_idx":    CAT2IDX[cat],
                "subcat_idx": SUBCAT2IDX[sub],
                "severity":   float(r.get("severity_score", 0.0)) / 4.0,
            })
            texts.append(text)

        if skipped:
            print(f"  Skipped {skipped} biased rows with unknown labels from {path}")

        print(f"  Tokenizing {len(texts)} biased samples from {path}...")
        enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        self.input_ids       = enc["input_ids"]
        self.attention_masks = enc["attention_mask"]

        self.cat_idxs    = torch.tensor([r["cat_idx"]    for r in self.records], dtype=torch.long)
        self.subcat_idxs = torch.tensor([r["subcat_idx"] for r in self.records], dtype=torch.long)
        self.severities  = torch.tensor([r["severity"]   for r in self.records], dtype=torch.float)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "cat_idx":        self.cat_idxs[idx],
            "subcat_idx":     self.subcat_idxs[idx],
            "severity":       self.severities[idx],
        }


# ── Model ─────────────────────────────────────────────────────────────────────
class GenderBiasBERT(nn.Module):
    def __init__(self, bert_model_name, dropout=DROPOUT):
        super().__init__()
        self.bert    = BertModel.from_pretrained(bert_model_name)
        hidden       = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.shared_proj = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(dropout)
        )
        self.cat_proj = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(dropout)
        )
        self.subcat_proj = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(dropout)
        )
        self.detection_head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )
        self.cat_head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, N_CATS)
        )
        self.subcat_head = nn.Sequential(
            nn.Linear(512 + N_CATS, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, N_SUBCATS)
        )
        self.score_head = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        cls     = self.dropout(
            self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        )
        shared  = self.shared_proj(cls)
        cat_rep = self.cat_proj(cls)
        sub_rep = self.subcat_proj(cls)

        det = self.detection_head(shared).squeeze(-1)
        cat = self.cat_head(cat_rep)

        sub_combined = torch.cat([sub_rep, torch.softmax(cat, dim=-1)], dim=-1)
        subcat = self.subcat_head(sub_combined)

        score = self.score_head(shared).squeeze(-1)
        return det, cat, subcat, score


# ── Loss ──────────────────────────────────────────────────────────────────────
def hierarchical_penalty(cat_logits, subcat_logits):
    """Penalises misalignment between category and subcategory probability mass."""
    cat_probs   = torch.softmax(cat_logits, dim=-1)
    sub_probs   = torch.softmax(subcat_logits, dim=-1)
    parent_ids  = SUBCAT_PARENT.to(cat_logits.device)
    parent_mass = torch.zeros_like(cat_probs)
    for sub_idx in range(N_SUBCATS):
        parent_mass[:, parent_ids[sub_idx]] += sub_probs[:, sub_idx]
    return torch.mean((cat_probs - parent_mass) ** 2)


def compute_categorization_loss(cat_logits, subcat_logits, scores,
                                cat_idxs, subcat_idxs, severities,
                                ce_cat_fn, ce_subcat_fn):
    """Computes category + subcategory + severity + hierarchy losses."""
    l_cat    = ce_cat_fn(cat_logits, cat_idxs)
    l_subcat = ce_subcat_fn(subcat_logits, subcat_idxs)
    l_score  = nn.functional.mse_loss(scores, severities)
    l_hier   = hierarchical_penalty(cat_logits, subcat_logits)

    total = (
        W_CATEGORY * l_cat    +
        W_SUBCAT   * l_subcat +
        W_SCORE    * l_score  +
        W_HIER     * l_hier
    )
    return total, l_cat.item(), l_subcat.item(), l_score.item(), l_hier.item()


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, ce_cat_fn, ce_subcat_fn):
    model.eval()
    total_loss = 0.0
    cat_preds,   cat_labels   = [], []
    sub_preds,   sub_labels   = [], []

    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        ci   = batch["cat_idx"].to(DEVICE)
        si   = batch["subcat_idx"].to(DEVICE)
        sv   = batch["severity"].to(DEVICE)

        _, cat, subcat, scr = model(ids, mask)
        loss, *_ = compute_categorization_loss(
            cat, subcat, scr, ci, si, sv, ce_cat_fn, ce_subcat_fn
        )
        total_loss += loss.item()

        cat_pred  = torch.argmax(cat, dim=-1).cpu()

        valid_mask    = SUBCAT_VALID_MASK[cat_pred].to(subcat.device)
        subcat_masked = subcat.masked_fill(~valid_mask, float("-inf"))
        sub_pred      = torch.argmax(subcat_masked, dim=-1).cpu()

        cat_preds.extend(cat_pred.tolist())
        cat_labels.extend(ci.cpu().tolist())
        sub_preds.extend(sub_pred.tolist())
        sub_labels.extend(si.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    cat_acc  = sum(p == y for p, y in zip(cat_preds, cat_labels)) / max(len(cat_labels), 1)
    sub_acc  = sum(p == y for p, y in zip(sub_preds, sub_labels)) / max(len(sub_labels), 1)
    cat_f1   = f1_score(cat_labels, cat_preds, average="macro", zero_division=0) if cat_labels else 0.0
    sub_f1   = f1_score(sub_labels, sub_preds, average="macro", zero_division=0) if sub_labels else 0.0

    return {
        "loss":       avg_loss,
        "cat_acc":    cat_acc,  "cat_f1":  cat_f1,
        "sub_acc":    sub_acc,  "sub_f1":  sub_f1,
        "cat_labels": cat_labels, "cat_preds": cat_preds,
        "sub_labels": sub_labels, "sub_preds": sub_preds,
    }


def composite(m):
    return (
        0.40 * m["cat_f1"]  +
        0.50 * m["sub_f1"]  +
        0.10 * m["cat_acc"]
    )


# ── Save / Load ───────────────────────────────────────────────────────────────
def save_model(model, tokenizer, output_dir, threshold):
    os.makedirs(output_dir, exist_ok=True)
    model.bert.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save({
        "shared_proj":    model.shared_proj.state_dict(),
        "cat_proj":       model.cat_proj.state_dict(),
        "subcat_proj":    model.subcat_proj.state_dict(),
        "detection_head": model.detection_head.state_dict(),
        "cat_head":       model.cat_head.state_dict(),
        "subcat_head":    model.subcat_head.state_dict(),
        "score_head":     model.score_head.state_dict(),
        "threshold":      threshold,
    }, os.path.join(output_dir, "heads.pt"))
    with open(os.path.join(output_dir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": round(threshold, 4)}, f)


def load_detection_model(detection_dir, dropout):
    """Load Stage-1 model (BERT + all heads) from the detection directory."""
    ckpt  = torch.load(os.path.join(detection_dir, "heads.pt"), map_location=DEVICE)
    model = GenderBiasBERT(detection_dir, dropout=dropout).to(DEVICE)
    model.shared_proj.load_state_dict(ckpt["shared_proj"])
    model.cat_proj.load_state_dict(ckpt["cat_proj"])
    model.subcat_proj.load_state_dict(ckpt["subcat_proj"])
    model.detection_head.load_state_dict(ckpt["detection_head"])
    model.cat_head.load_state_dict(ckpt["cat_head"])
    model.subcat_head.load_state_dict(ckpt["subcat_head"])
    model.score_head.load_state_dict(ckpt["score_head"])

    # Load the detection threshold saved by Stage 1
    threshold_path = os.path.join(detection_dir, "threshold.json")
    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            threshold = json.load(f)["threshold"]
    return model, ckpt.get("threshold", threshold)


def build_sample_weights(records, key="subcat_idx"):
    label_counter = Counter(r[key] for r in records)
    weights = np.array([1.0 / label_counter[r[key]] for r in records], dtype=np.float64)
    weights = weights / weights.sum()
    return torch.DoubleTensor(weights)


# ── Progress bar ──────────────────────────────────────────────────────────────
def print_progress(step, total, avg_loss, t0, bar_width=28):
    pct     = step / total
    filled  = int(bar_width * pct)
    bar     = "█" * filled + "░" * (bar_width - filled)
    elapsed = time.time() - t0
    eta     = (elapsed / step * (total - step)) if step > 0 else 0
    print(
        f"\r  [{bar}] {pct*100:5.1f}%  {step}/{total}"
        f"  loss={avg_loss:.4f}  {elapsed:.0f}s elapsed  eta={eta:.0f}s",
        end="", flush=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device         : {DEVICE}")
    print(f"MAX_LEN        : {args.max_len}")
    print(f"BATCH_SIZE     : {args.batch_size}")
    print(f"UNFREEZE_TOP   : {args.unfreeze_top} BERT layers")
    print(f"DROPOUT        : {args.dropout}  WEIGHT_DECAY: {WEIGHT_DECAY}")
    print(f"Input mode     : {'response only' if args.response_only else 'prompt + response'}")
    print(f"Stage          : CATEGORIZATION (cat + subcat heads)")
    print(f"Detection dir  : {args.detection_dir}")

    tokenizer = BertTokenizer.from_pretrained(args.detection_dir)

    train_raw = load_jsonl(args.train_path)
    val_raw   = load_jsonl(args.val_path)
    test_raw  = load_jsonl(args.test_path)
    audit_splits(train_raw, val_raw, test_raw, args.train_path, args.val_path, args.test_path)

    # Only biased samples are needed for categorization training
    train_ds = CategoryDataset(args.train_path, tokenizer, max_len=args.max_len, response_only=args.response_only)
    val_ds   = CategoryDataset(args.val_path,   tokenizer, max_len=args.max_len, response_only=args.response_only)
    test_ds  = CategoryDataset(args.test_path,  tokenizer, max_len=args.max_len, response_only=args.response_only)

    sample_weights = build_sample_weights(train_ds.records)
    sampler        = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    num_workers = 0 if DEVICE.type == "mps" else 2
    persistent  = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=num_workers, persistent_workers=persistent)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, persistent_workers=persistent)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, persistent_workers=persistent)

    # Class-weighted losses for imbalanced categories / subcategories
    cat_counts = Counter(r["cat_idx"]    for r in train_ds.records)
    sub_counts = Counter(r["subcat_idx"] for r in train_ds.records)
    total      = len(train_ds.records)

    cat_weights = torch.tensor([
        (total / max(cat_counts.get(i, 1), 1)) ** 1.2 for i in range(N_CATS)
    ], dtype=torch.float, device=DEVICE)
    sub_weights = torch.tensor([
        (total / max(sub_counts.get(i, 1), 1)) ** 1.5 for i in range(N_SUBCATS)
    ], dtype=torch.float, device=DEVICE)

    ce_cat_fn    = nn.CrossEntropyLoss(weight=cat_weights, label_smoothing=0.05)
    ce_subcat_fn = nn.CrossEntropyLoss(weight=sub_weights, label_smoothing=0.03)

    print(f"\nTrain biased={len(train_ds)} | Val biased={len(val_ds)} | Test biased={len(test_ds)}")
    print("Subcategory counts:", {SUBCAT_LABELS[i]: sub_counts.get(i, 0) for i in range(N_SUBCATS)})

    # ── Load Stage-1 model ────────────────────────────────────────────────────
    print(f"\nLoading detection model from: {args.detection_dir}")
    model, detection_threshold = load_detection_model(args.detection_dir, dropout=args.dropout)
    if args.detection_threshold is not None:
        detection_threshold = args.detection_threshold
    print(f"Detection threshold (from Stage 1): {detection_threshold:.4f}")

    # Freeze entire BERT; then selectively unfreeze top N layers
    for p in model.bert.parameters():
        p.requires_grad = False
    n_layers = len(model.bert.encoder.layer)
    for i, layer in enumerate(model.bert.encoder.layer):
        if i >= n_layers - args.unfreeze_top:
            for p in layer.parameters():
                p.requires_grad = True
    # Also unfreeze pooler so the CLS representation can adapt
    for p in model.bert.pooler.parameters():
        p.requires_grad = True
    # Freeze detection head — it stays as-is from Stage 1
    for p in model.detection_head.parameters():
        p.requires_grad = False

    frozen     = sum(1 for p in model.bert.parameters() if not p.requires_grad)
    total_bert = sum(1 for p in model.bert.parameters())
    print(f"BERT params frozen: {frozen}/{total_bert}  (top {args.unfreeze_top} layers unfrozen)")

    # Only optimize categorization-related parameters
    trainable_bert = [p for p in model.bert.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_bert,                    "lr": args.lr},
            {"params": model.shared_proj.parameters(),    "lr": args.head_lr},
            {"params": model.cat_proj.parameters(),       "lr": args.head_lr},
            {"params": model.subcat_proj.parameters(),    "lr": args.head_lr},
            {"params": model.cat_head.parameters(),       "lr": args.head_lr},
            {"params": model.subcat_head.parameters(),    "lr": args.head_lr},
            {"params": model.score_head.parameters(),     "lr": args.head_lr},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_comp      = -1.0
    history        = []
    patience_count = 0

    header = (f"\n{'Ep':<4} {'TrainL':<9} {'ValL':<9} {'CatAcc':<8} "
              f"{'CatF1':<8} {'SubAcc':<8} {'SubF1':<8} {'Comp':<8}")
    print(header)
    print("-" * len(header.strip()))

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep detection head frozen (eval mode prevents BN/dropout updates there too)
        model.detection_head.eval()

        train_loss = 0.0
        t0      = time.time()
        n_steps = len(train_loader)

        print(f"\nEpoch {epoch}/{args.epochs}")
        for step, batch in enumerate(train_loader, 1):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            ci   = batch["cat_idx"].to(DEVICE)
            si   = batch["subcat_idx"].to(DEVICE)
            sv   = batch["severity"].to(DEVICE)

            optimizer.zero_grad()
            _, cat, subcat, scr = model(ids, mask)
            loss, *_ = compute_categorization_loss(
                cat, subcat, scr, ci, si, sv, ce_cat_fn, ce_subcat_fn
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            if step % 10 == 0 or step == n_steps:
                print_progress(step, n_steps, train_loss / step, t0)

            if DEVICE.type == "mps":
                torch.mps.empty_cache()

        print()

        metrics   = evaluate(model, val_loader, ce_cat_fn, ce_subcat_fn)
        comp      = composite(metrics)
        avg_train = train_loss / max(n_steps, 1)
        elapsed   = time.time() - t0

        print(
            f"{epoch:<4} {avg_train:<9.4f} {metrics['loss']:<9.4f} "
            f"{metrics['cat_acc']*100:<8.2f} {metrics['cat_f1']:<8.4f} "
            f"{metrics['sub_acc']*100:<8.2f} {metrics['sub_f1']:<8.4f} {comp:<8.4f} ({elapsed:.0f}s)"
        )

        history.append({
            "epoch":      epoch,
            "train_loss": round(avg_train,          4),
            "val_loss":   round(metrics["loss"],     4),
            "cat_acc":    round(metrics["cat_acc"],  4),
            "cat_f1":     round(metrics["cat_f1"],   4),
            "sub_acc":    round(metrics["sub_acc"],  4),
            "sub_f1":     round(metrics["sub_f1"],   4),
            "composite":  round(comp,                4),
        })

        if comp > best_comp:
            best_comp      = comp
            patience_count = 0
            save_model(model, tokenizer, args.output_dir, detection_threshold)
            print(f"  ✓ New best | comp={comp:.4f} | cat_f1={metrics['cat_f1']:.4f} | sub_f1={metrics['sub_f1']:.4f}")
        else:
            patience_count += 1
            print(f"  ✗ No improvement ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nReloading best categorization model for final test evaluation...")
    ckpt = torch.load(os.path.join(args.output_dir, "heads.pt"), map_location=DEVICE)
    best_model = GenderBiasBERT(args.output_dir, dropout=args.dropout).to(DEVICE)
    best_model.shared_proj.load_state_dict(ckpt["shared_proj"])
    best_model.cat_proj.load_state_dict(ckpt["cat_proj"])
    best_model.subcat_proj.load_state_dict(ckpt["subcat_proj"])
    best_model.detection_head.load_state_dict(ckpt["detection_head"])
    best_model.cat_head.load_state_dict(ckpt["cat_head"])
    best_model.subcat_head.load_state_dict(ckpt["subcat_head"])
    best_model.score_head.load_state_dict(ckpt["score_head"])

    test_metrics = evaluate(best_model, test_loader, ce_cat_fn, ce_subcat_fn)

    print(f"\n{'='*60}")
    print("TEST RESULTS  (Categorization Stage)")
    print(f"{'='*60}")
    print(f"Category Accuracy    : {test_metrics['cat_acc']*100:.2f}%")
    print(f"Category Macro-F1    : {test_metrics['cat_f1']:.4f}")
    print(f"Subcategory Accuracy : {test_metrics['sub_acc']*100:.2f}%")
    print(f"Subcategory Macro-F1 : {test_metrics['sub_f1']:.4f}")
    print(f"{'='*60}\n")

    sub_report = classification_report(
        test_metrics["sub_labels"], test_metrics["sub_preds"],
        labels=list(range(N_SUBCATS)), target_names=SUBCAT_LABELS,
        output_dict=True, zero_division=0
    ) if test_metrics["sub_labels"] else {}

    summary = {
        "stage": "categorization",
        "config": {
            **vars(args),
            "weight_decay": WEIGHT_DECAY,
            "loss_weights": {
                "category": W_CATEGORY,
                "subcat":   W_SUBCAT,
                "score":    W_SCORE,
                "hier":     W_HIER,
            },
            "n_categories":    N_CATS,
            "n_subcategories": N_SUBCATS,
        },
        "detection_threshold":   round(detection_threshold, 4),
        "best_val_composite":    round(best_comp,           4),
        "test_results": {
            "category_accuracy":    round(test_metrics["cat_acc"], 4),
            "category_macro_f1":    round(test_metrics["cat_f1"],  4),
            "subcategory_accuracy": round(test_metrics["sub_acc"], 4),
            "subcategory_macro_f1": round(test_metrics["sub_f1"],  4),
        },
        "subcategory_report": sub_report,
        "epoch_log": history,
    }
    with open(args.log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Training log saved → {args.log_path}")
    print(f"Model saved        → {args.output_dir}/")


if __name__ == "__main__":
    main()
