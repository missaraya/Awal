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
N_CATS     = len(CAT_LABELS)
N_SUBCATS  = len(SUBCAT_LABELS)
IGNORE_INDEX = -100

SUBCAT_PARENT = torch.tensor(
    [CAT2IDX[SUBCAT_TO_CAT[s]] for s in SUBCAT_LABELS], dtype=torch.long
)
SUBCAT_VALID_MASK = torch.zeros(N_CATS, N_SUBCATS, dtype=torch.bool)
for _si, _sl in enumerate(SUBCAT_LABELS):
    SUBCAT_VALID_MASK[CAT2IDX[SUBCAT_TO_CAT[_sl]], _si] = True

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else "cpu"
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
BERT_BASE     = "bert-base-uncased"
MAX_LEN       = 256
BATCH_SIZE    = 4
EPOCHS        = 5
PATIENCE      = 2
FREEZE_LAYERS = 2
LR            = 2e-5
HEAD_LR       = 1.5e-4
WARMUP_RATIO  = 0.10
DROPOUT       = 0.20
WEIGHT_DECAY  = 0.01

# Detection-only loss weights (cat/subcat/score/hier are 0 in this stage)
W_DETECTION = 1.0

FIXED_THRESHOLD = 0.50

# ── IO defaults ───────────────────────────────────────────────────────────────
DEFAULT_TRAIN = "fixedtrain.jsonl"
DEFAULT_VAL   = "fixedval.jsonl"
DEFAULT_TEST  = "fixedtest.jsonl"
DEFAULT_OUT   = "./outputs/detection_model"
DEFAULT_LOG   = "./outputs/detection_log.json"


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 – Bias detection training")
    p.add_argument("--train_path",    default=DEFAULT_TRAIN)
    p.add_argument("--val_path",      default=DEFAULT_VAL)
    p.add_argument("--test_path",     default=DEFAULT_TEST)
    p.add_argument("--output_dir",    default=DEFAULT_OUT)
    p.add_argument("--log_path",      default=DEFAULT_LOG)
    p.add_argument("--epochs",        type=int,   default=EPOCHS)
    p.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--max_len",       type=int,   default=MAX_LEN)
    p.add_argument("--freeze_layers", type=int,   default=FREEZE_LAYERS)
    p.add_argument("--lr",            type=float, default=LR)
    p.add_argument("--head_lr",       type=float, default=HEAD_LR)
    p.add_argument("--dropout",       type=float, default=DROPOUT)
    p.add_argument("--response_only", action="store_true", default=False,
                   help="Feed only the response text to BERT (no [PROMPT] prefix).")
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
    print(f"Train path: {os.path.abspath(train_path)}")
    print(f"Val path  : {os.path.abspath(val_path)}")
    print(f"Test path : {os.path.abspath(test_path)}")
    print(f"Train={len(train_raw)} | Val={len(val_raw)} | Test={len(test_raw)}")
    print(f"Exact pair overlap  train/val={overlap(train_raw, val_raw, exact_pair_key)}  "
          f"train/test={overlap(train_raw, test_raw, exact_pair_key)}  "
          f"val/test={overlap(val_raw, test_raw, exact_pair_key)}")
    print(f"Prompt overlap      train/val={overlap(train_raw, val_raw, prompt_key)}  "
          f"train/test={overlap(train_raw, test_raw, prompt_key)}  "
          f"val/test={overlap(val_raw, test_raw, prompt_key)}")
    tv = overlap(train_raw, val_raw,  resp_tmpl_key)
    tt = overlap(train_raw, test_raw, resp_tmpl_key)
    vt = overlap(val_raw,   test_raw, resp_tmpl_key)
    template_ok = tv == 0 and tt == 0 and vt == 0
    flag = "" if template_ok else "  ⚠ RUN resplit.py"
    print(f"Resp template overlap train/val={tv}  train/test={tt}  val/test={vt}{flag}")

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
    for name, split in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        rep_b = sum(c > 1 for c in Counter(
            response_hash(r["response"]) for r in split if r["is_biased"]
        ).values())
        print(f"{name.title():<5} repeated biased response templates: {rep_b}")
    print("-" * 60)


# ── Dataset ───────────────────────────────────────────────────────────────────
class BiasDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=MAX_LEN, response_only=False):
        raw = load_jsonl(path)
        self.records = []
        skipped = 0
        texts = []

        for r in raw:
            is_biased = bool(r["is_biased"])
            if response_only:
                text = r['response'].strip()
            else:
                text = f"[PROMPT] {r['prompt'].strip()} [RESPONSE] {r['response'].strip()}"

            if is_biased:
                cat = r.get("harm_category")
                sub = r.get("subcategory")
                if cat not in CAT2IDX or sub not in SUBCAT2IDX:
                    skipped += 1
                    continue
                cat_idx = CAT2IDX[cat]
                sub_idx = SUBCAT2IDX[sub]
            else:
                cat_idx = IGNORE_INDEX
                sub_idx = IGNORE_INDEX

            self.records.append({
                "bias_label": float(is_biased),
                "cat_idx":    cat_idx,
                "subcat_idx": sub_idx,
                "severity":   float(r.get("severity_score", 0.0)) / 4.0,
                "is_biased":  is_biased,
            })
            texts.append(text)

        if skipped:
            print(f"  Skipped {skipped} rows with unknown labels from {path}")

        print(f"  Tokenizing {len(texts)} samples from {path}...")
        enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        self.input_ids       = enc["input_ids"]
        self.attention_masks = enc["attention_mask"]

        self.bias_labels = torch.tensor([r["bias_label"] for r in self.records], dtype=torch.float)
        self.cat_idxs    = torch.tensor([r["cat_idx"]    for r in self.records], dtype=torch.long)
        self.subcat_idxs = torch.tensor([r["subcat_idx"] for r in self.records], dtype=torch.long)
        self.severities  = torch.tensor([r["severity"]   for r in self.records], dtype=torch.float)
        self.is_biased   = torch.tensor([r["is_biased"]  for r in self.records], dtype=torch.bool)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "bias_label":     self.bias_labels[idx],
            "cat_idx":        self.cat_idxs[idx],
            "subcat_idx":     self.subcat_idxs[idx],
            "severity":       self.severities[idx],
            "is_biased":      self.is_biased[idx],
        }


# ── Model ─────────────────────────────────────────────────────────────────────
class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, hard_labels=None):
        bce    = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs  = torch.sigmoid(logits)
        ref    = hard_labels if hard_labels is not None else targets
        pt     = torch.where(ref == 1, probs, 1 - probs)
        alpha_t = torch.where(ref == 1,
                              torch.full_like(probs, self.alpha),
                              torch.full_like(probs, 1 - self.alpha))
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


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


# ── Detection-only loss ───────────────────────────────────────────────────────
def compute_detection_loss(det_logits, bias_labels, det_loss_fn):
    """Only computes the detection (binary bias) loss. Category/subcat losses are skipped."""
    smooth = bias_labels * 0.95 + (1 - bias_labels) * 0.05
    return det_loss_fn(det_logits, smooth, hard_labels=bias_labels)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, threshold, det_loss_fn):
    model.eval()
    total_loss = 0.0
    bias_preds, bias_labels_all = [], []

    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        bl   = batch["bias_label"].to(DEVICE)

        det, _, _, _ = model(ids, mask)
        loss = compute_detection_loss(det, bl, det_loss_fn)
        total_loss += loss.item()

        probs = torch.sigmoid(det)
        bias_preds.extend((probs >= threshold).long().cpu().tolist())
        bias_labels_all.extend(bl.long().cpu().tolist())

    avg_loss  = total_loss / max(len(loader), 1)
    bias_acc  = sum(p == y for p, y in zip(bias_preds, bias_labels_all)) / max(len(bias_labels_all), 1)
    bias_f1   = f1_score(bias_labels_all, bias_preds, average="macro", zero_division=0)
    bias_prec = precision_score(bias_labels_all, bias_preds, average="macro", zero_division=0)
    bias_rec  = recall_score(bias_labels_all,   bias_preds, average="macro", zero_division=0)

    return {
        "loss":      avg_loss,
        "bias_acc":  bias_acc,
        "bias_f1":   bias_f1,
        "bias_prec": bias_prec,
        "bias_rec":  bias_rec,
    }


@torch.no_grad()
def tune_threshold(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        det, _, _, _ = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
        )
        all_probs.extend(torch.sigmoid(det).cpu().tolist())
        all_labels.extend(batch["bias_label"].long().tolist())

    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.20, 0.81, 0.01):
        preds = [1 if p >= t else 0 for p in all_probs]
        f1    = f1_score(all_labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


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


def load_best_model(output_dir, dropout):
    ckpt  = torch.load(os.path.join(output_dir, "heads.pt"), map_location=DEVICE)
    model = GenderBiasBERT(output_dir, dropout=dropout).to(DEVICE)
    model.shared_proj.load_state_dict(ckpt["shared_proj"])
    model.cat_proj.load_state_dict(ckpt["cat_proj"])
    model.subcat_proj.load_state_dict(ckpt["subcat_proj"])
    model.detection_head.load_state_dict(ckpt["detection_head"])
    model.cat_head.load_state_dict(ckpt["cat_head"])
    model.subcat_head.load_state_dict(ckpt["subcat_head"])
    model.score_head.load_state_dict(ckpt["score_head"])
    return model


def build_sample_weights(records):
    label_counter = Counter()
    for r in records:
        key = "biased" if r["is_biased"] else "non_biased"
        label_counter[key] += 1
    weights = []
    for r in records:
        key = "biased" if r["is_biased"] else "non_biased"
        weights.append(1.0 / label_counter[key])
    weights = np.array(weights, dtype=np.float64)
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

    print(f"Device        : {DEVICE}")
    print(f"MAX_LEN       : {args.max_len}")
    print(f"BATCH_SIZE    : {args.batch_size}")
    print(f"FREEZE_LAYERS : {args.freeze_layers}/12")
    print(f"DROPOUT       : {args.dropout}  WEIGHT_DECAY: {WEIGHT_DECAY}")
    print(f"Input mode    : {'response only' if args.response_only else 'prompt + response'}")
    print(f"Stage         : DETECTION ONLY (bias vs. non-bias)")

    print("\nLoading tokenizer and datasets...")
    tokenizer = BertTokenizer.from_pretrained(BERT_BASE)

    train_raw = load_jsonl(args.train_path)
    val_raw   = load_jsonl(args.val_path)
    test_raw  = load_jsonl(args.test_path)
    audit_splits(train_raw, val_raw, test_raw, args.train_path, args.val_path, args.test_path)

    train_ds = BiasDataset(args.train_path, tokenizer, max_len=args.max_len, response_only=args.response_only)
    val_ds   = BiasDataset(args.val_path,   tokenizer, max_len=args.max_len, response_only=args.response_only)
    test_ds  = BiasDataset(args.test_path,  tokenizer, max_len=args.max_len, response_only=args.response_only)

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

    det_loss_fn = FocalBCEWithLogits(alpha=0.5, gamma=2.0)

    print("\nBuilding model...")
    model = GenderBiasBERT(BERT_BASE, dropout=args.dropout).to(DEVICE)

    # Freeze early BERT layers; only train detection-related heads fully
    bert_layer_params = []
    for i, layer in enumerate(model.bert.encoder.layer):
        if i < args.freeze_layers:
            for p in layer.parameters():
                p.requires_grad = False
        else:
            decay = 0.9 ** (11 - i)
            bert_layer_params.append({"params": layer.parameters(), "lr": args.lr * decay})

    frozen     = sum(1 for p in model.bert.parameters() if not p.requires_grad)
    total_bert = sum(1 for p in model.bert.parameters())
    print(f"BERT params frozen: {frozen}/{total_bert}")

    optimizer = torch.optim.AdamW(
        bert_layer_params + [
            {"params": model.bert.embeddings.parameters(), "lr": args.lr * 0.15},
            {"params": model.bert.pooler.parameters(),     "lr": args.lr},
            {"params": model.shared_proj.parameters(),     "lr": args.head_lr},
            {"params": model.detection_head.parameters(),  "lr": args.head_lr},
            # cat/subcat/score heads are in the model but not optimised here
        ],
        weight_decay=WEIGHT_DECAY,
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1        = -1.0
    threshold      = FIXED_THRESHOLD
    history        = []
    patience_count = 0

    header = f"\n{'Ep':<4} {'TrainL':<9} {'ValL':<9} {'BiasAcc':<10} {'BiasF1':<8} {'Prec':<8} {'Rec':<8}"
    print(header)
    print("-" * len(header.strip()))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0      = time.time()
        n_steps = len(train_loader)

        print(f"\nEpoch {epoch}/{args.epochs}")
        for step, batch in enumerate(train_loader, 1):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            bl   = batch["bias_label"].to(DEVICE)

            optimizer.zero_grad()
            det, _, _, _ = model(ids, mask)
            loss = compute_detection_loss(det, bl, det_loss_fn)
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

        threshold, _ = tune_threshold(model, val_loader)
        metrics      = evaluate(model, val_loader, threshold, det_loss_fn)
        avg_train    = train_loss / max(n_steps, 1)
        elapsed      = time.time() - t0

        print(
            f"{epoch:<4} {avg_train:<9.4f} {metrics['loss']:<9.4f} "
            f"{metrics['bias_acc']*100:<10.2f} {metrics['bias_f1']:<8.4f} "
            f"{metrics['bias_prec']:<8.4f} {metrics['bias_rec']:<8.4f}  ({elapsed:.0f}s)"
        )

        history.append({
            "epoch":      epoch,
            "train_loss": round(avg_train,           4),
            "val_loss":   round(metrics["loss"],      4),
            "bias_acc":   round(metrics["bias_acc"],  4),
            "bias_f1":    round(metrics["bias_f1"],   4),
            "bias_prec":  round(metrics["bias_prec"], 4),
            "bias_rec":   round(metrics["bias_rec"],  4),
        })

        if metrics["bias_f1"] > best_f1:
            best_f1        = metrics["bias_f1"]
            patience_count = 0
            save_model(model, tokenizer, args.output_dir, threshold)
            print(f"  ✓ New best | bias_f1={metrics['bias_f1']:.4f}")
        else:
            patience_count += 1
            print(f"  ✗ No improvement ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print("\nReloading best model for final test evaluation...")
    best_model = load_best_model(args.output_dir, dropout=args.dropout)
    tuned_threshold, tuned_f1 = tune_threshold(best_model, val_loader)
    print(f"Best threshold on val: {tuned_threshold:.2f} | val bias macro-F1={tuned_f1:.4f}")

    with open(os.path.join(args.output_dir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": round(tuned_threshold, 4)}, f)

    test_metrics = evaluate(best_model, test_loader, tuned_threshold, det_loss_fn)

    print(f"\n{'='*60}")
    print("TEST RESULTS  (Detection Stage)")
    print(f"{'='*60}")
    print(f"Bias Accuracy   : {test_metrics['bias_acc']*100:.2f}%")
    print(f"Bias Macro-F1   : {test_metrics['bias_f1']:.4f}")
    print(f"Bias Precision  : {test_metrics['bias_prec']:.4f}")
    print(f"Bias Recall     : {test_metrics['bias_rec']:.4f}")
    print(f"{'='*60}\n")

    summary = {
        "stage": "detection",
        "config": {
            **vars(args),
            "weight_decay": WEIGHT_DECAY,
            "loss_weights": {"detection": W_DETECTION},
        },
        "best_val_bias_f1": round(best_f1,          4),
        "best_threshold":   round(tuned_threshold,   4),
        "test_results": {
            "bias_accuracy":        round(test_metrics["bias_acc"],  4),
            "bias_macro_f1":        round(test_metrics["bias_f1"],   4),
            "bias_macro_precision": round(test_metrics["bias_prec"], 4),
            "bias_macro_recall":    round(test_metrics["bias_rec"],  4),
        },
        "epoch_log": history,
    }
    with open(args.log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Training log saved → {args.log_path}")
    print(f"Model saved        → {args.output_dir}/")
    print(f"\nNext step: run train_categorization.py --detection_dir {args.output_dir}")


if __name__ == "__main__":
    main()
