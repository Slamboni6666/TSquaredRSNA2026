"""
llm_label_reports.py -- label all 4,407 reports with a local open-weights LLM.

No API key, no training, no fine-tuning. A pretrained model is prompted once per report and
its answer is written to disk. The model never appears in the submission -- it runs offline
at build time, so Kaggle's no-internet rule and the 9-hour limit do not apply to it, and it
can be as large as the GPU will hold.

WHY A GENERAL MULTILINGUAL MODEL RATHER THAN A MEDICAL ONE
Our measured bottleneck is recall across nine languages, not medical vocabulary: the rule
extractor reads a clause correctly once it finds it (MCL scores 0.983 on the studies where
it fires) but never sees the clause often enough. The findings are twelve standard
radiological entities; the difficulty is that they arrive in Greek, Bulgarian, Turkish and
Croatian, negated, hedged, and graded. Qwen covers those languages well. A medical model
trained mostly on English corpora would be strongest exactly where we are already fine.
Qwen is also Apache-2.0, which sits cleanly against the CC-BY-NC-4.0 winner requirement.

WHAT THE PROMPT ENCODES
Everything the 58 gold studies taught us, since that is the only place report language and
the true answer are both visible:
  * gold is IMAGE-derived and SEVERITY-thresholded; borderline cases were graded negative,
    so "small effusion" is a gold negative. Hence severity is asked for, not presence.
  * a report that NEGATES a finding is right 96% of the time -> trust denials.
  * a report that ASSERTS one is right only 65% of the time -> assertions need a grade.
  * silence is not absence (14% of unmentioned findings are gold-positive), so "not
    mentioned" is its own answer rather than a no.

WHAT IT WRITES
report_labels_llm.csv carries TWO columns per finding:
  <target>          a number in [0,1] -- the soft label the vision model learns to copy
  <target>__state   the raw severity word the LLM answered
The __state column is not decoration. Training MASKS OUT every cell whose state says the
report never mentioned the finding, so that cell contributes nothing to the loss. Score it
as a negative instead and the student simply learns each finding's base rate. Note that
build_report_labels.py spells this same state "silent" while this file spells it
"not_mentioned"; a reader that checks only one spelling silently disables masking for the
other labeller, which is a bug this project has already shipped once.

Resumable: one JSON line per study, appended. Kill and restart freely.

Setup:
    pip install transformers accelerate bitsandbytes

Usage:
    python llm_label_reports.py --limit 40            # smoke test
    python llm_label_reports.py --model Qwen/Qwen3-32B --load-4bit
    python llm_label_reports.py --score-only          # evaluate what is already written
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Severity ladder. Only the ORDER matters -- AUC reads nothing else -- but the gaps encode
# how much evidence separates the rungs, and where the gold threshold was observed to sit.
# Two placements are deliberate and look wrong at a glance:
#   not_mentioned (0.15) ranks ABOVE absent (0.03). A report that denies a finding is right
#   96% of the time; silence is right only 86%. Silence is therefore weaker evidence of
#   absence than a denial, and must not collapse onto it.
#   present_ungraded (0.62) ranks ABOVE mild (0.45). Gold was thresholded on severity and
#   graded borderline cases NEGATIVE, so "mild" is the rung most likely to be a gold
#   negative -- an unqualified assertion is better evidence than an explicitly small one.
SEVERITY_SCORE = {
    "severe": 0.95, "moderate": 0.75, "mild": 0.45,
    "present_ungraded": 0.62, "uncertain": 0.35, "not_mentioned": 0.15, "absent": 0.03,
}

SYSTEM = """You are a musculoskeletal radiologist reading knee MRI reports and extracting \
structured findings. Reports arrive in any of nine languages (English, Spanish, Turkish, \
Croatian, Greek, German, Bulgarian, Dutch, French). Read the report in its original \
language. Do not translate before deciding.

You extract twelve findings. For each, return a SEVERITY, not a yes/no:

  "severe"           unambiguous and marked: complete tear, large effusion, advanced OA,
                     grade 3-4 chondral loss, displaced fracture
  "moderate"         clearly present, intermediate grade
  "mild"             explicitly minimal / small / mild / trace / early / incipient / grade 1
  "present_ungraded" asserted with no severity qualifier anywhere
  "uncertain"        hedged: possible, suspected, cannot exclude, questionable
  "absent"           the report states it is normal, intact, preserved, or denies it
  "not_mentioned"    the report says nothing about this structure at all

The severity distinction is the whole task. The reference standard was read from the IMAGES
against explicit severity criteria and graded borderline cases NEGATIVE. A report can \
truthfully describe a small effusion that the reference standard calls absent. So "mild" and
"severe" are very different answers, and "not_mentioned" is NOT the same as "absent".

The twelve findings:
  ACL               anterior cruciate ligament tear or injury. NOT the posterior cruciate.
  MCL               medial collateral ligament tear or injury. NOT the lateral collateral.
  Medial Meniscus   medial meniscus TEAR. Grade 1-2 degeneration without a tear is "absent";
                    grade 3 signal reaching the articular surface is a tear.
  Lateral Meniscus  lateral meniscus tear, same rule.
  Medial OA         osteoarthritis / cartilage loss of the MEDIAL TIBIOFEMORAL compartment.
                    Not the medial meniscus, not the medial collateral ligament.
  Lateral OA        the same for the lateral tibiofemoral compartment.
  PF OA             patellofemoral osteoarthritis, retropatellar chondromalacia, trochlear
                    cartilage loss. NOT the patellar tendon, NOT the MPFL.
  Effusion          joint effusion / excess intra-articular fluid.
  Synovitis         synovial thickening, hypertrophy, proliferation, pannus.
  Baker's           Baker's cyst / popliteal cyst / gastrocnemio-semimembranosus bursa.
  Contusion         TRAUMATIC bone contusion, bone bruise, bone marrow oedema from injury.
                    Subchondral marrow oedema adjacent to cartilage loss is DEGENERATIVE,
                    not a contusion -- answer "absent" for that.
  Fracture          any fracture, including insufficiency and osteochondral fractures.

Return ONLY a JSON object, no prose, no code fence, mapping each of the twelve names
exactly as written above to one of the seven severity strings."""

USER_TMPL = """Report:
---
{report}
---
JSON:"""


def parse_answer(text):
    """Model output -> {target: severity}. Tolerant of fences and trailing prose.

    A key the model omitted, or a severity word outside the seven, becomes "not_mentioned"
    and never "absent". Anything we could not read has to land in the MASKED state so it
    drops out of the loss; mapping it to a confident negative would invent an answer for a
    finding nobody assessed. Returns None only when there is no JSON object at all.
    """
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for t in TARGETS:
        v = str(raw.get(t, "not_mentioned")).strip().lower().replace(" ", "_")
        out[t] = v if v in SEVERITY_SCORE else "not_mentioned"
    return out


def score_file(path):
    """Turn the jsonl into a labels CSV, then score it on the 58 gold studies.

    llm_labels.jsonl (Qwen, the default) -> report_labels_llm.csv, the file train.py and the
    analysis scripts read by default. Any other jsonl gets its own CSV named from its suffix
    (llm_labels_glm.jsonl -> report_labels_glm.csv). This used to write report_labels_llm.csv
    for every jsonl, so the 2026-09-04 Mistral and 2026-09-05 GLM runs each silently
    replaced the Qwen labels.

    It writes as well as scores, which is intended: the CSV is this file's actual product
    and this is the only place it is built, so a partial run can be inspected without a
    separate step. "Gold" means the 58 of 4,407 studies whose twelve labels were read from
    the IMAGES rather than from text -- the only trustworthy answers in the project. At
    n=58 a single target's AUC carries about +/-0.03 of noise, so read the MACRO (the plain
    unweighted mean of the twelve per-target AUCs, which is also the competition metric) and
    ignore small per-target moves. The rule extractor is printed in the next column as the
    bar to clear: an LLM pass that cannot beat regexes has not earned its GPU hours.
    """
    from sklearn.metrics import roc_auc_score
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        print("nothing written yet")
        return
    df = pd.DataFrame([{"StudyInstanceUID": r["study"],
                        **{t: SEVERITY_SCORE[r["sev"][t]] for t in TARGETS},
                        **{t + "__state": r["sev"][t] for t in TARGETS}} for r in rows])
    stem = path.stem
    name = ("report_labels_llm.csv" if stem == "llm_labels"
            else f"report_labels_{stem.removeprefix('llm_labels_')}.csv")
    df.to_csv(ROOT / name, index=False, encoding="utf-8")
    print(f"wrote {name}  ({len(df)} studies)")

    tr = pd.read_csv(ROOT / "train.csv")
    gold = tr[tr[TARGETS].notna().all(axis=1)]
    m = gold[["StudyInstanceUID"] + TARGETS].merge(df, on="StudyInstanceUID",
                                                   suffixes=("_g", "_p"))
    if len(m) < 5:
        print(f"only {len(m)} gold studies labelled so far — run more before judging")
        return
    # A target with one class present has no defined AUC. During a partial run that is
    # common, and averaging it in as NaN silently destroys the macro -- which is the one
    # number being watched to decide whether to let the run continue.
    def auc(y, p):
        return roc_auc_score(y, p) if len(set(y)) > 1 else np.nan

    per = {t: auc(m[t + "_g"], m[t + "_p"]) for t in TARGETS}
    rules = pd.read_csv(ROOT / "report_labels_ours.csv")
    mr = gold[["StudyInstanceUID"] + TARGETS].merge(rules, on="StudyInstanceUID",
                                                    suffixes=("_g", "_r"))
    per_r = {t: auc(mr[t + "_g"], mr[t + "_r"]) for t in TARGETS}
    both = [t for t in TARGETS if not np.isnan(per[t])]

    print(f"\n  {'target':<18} {'LLM':>7} {'rules':>7}")
    for t in TARGETS:
        a, b = per[t], per_r[t]
        print(f"  {t:<18} {'  n/a' if np.isnan(a) else f'{a:>7.3f}'} "
              f"{'  n/a' if np.isnan(b) else f'{b:>7.3f}'}")
    print(f"  {'MACRO':<18} {np.nanmean(list(per.values())):>7.3f} "
          f"{np.nanmean([per_r[t] for t in both]):>7.3f}"
          f"   (n={len(m)} gold, {len(both)}/12 targets scorable)")
    if len(both) < len(TARGETS):
        print(f"  NOTE: {len(TARGETS) - len(both)} targets have one class only at this "
              f"sample size; the rules column is restricted to the same targets so the "
              f"two macros are comparable.")
    cov = np.mean([(df[t + "__state"] != "not_mentioned").mean() for t in TARGETS])
    print(f"\n  mean coverage (non-'not_mentioned'): {100 * cov:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--out", default="llm_labels.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true", help="~20 GB for a 32B model")
    ap.add_argument("--max-new", type=int, default=260)
    ap.add_argument("--batch", type=int, default=8,
                    help="reports per generate() call; the single biggest speed lever")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--gold-first", action="store_true",
                    help="label the 58 gold studies first so quality is measurable early")
    a = ap.parse_args()

    out = ROOT / a.out
    if a.score_only:
        score_file(out)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tr = pd.read_csv(ROOT / "train.csv")
    done = set()
    if out.exists():
        done = {json.loads(l)["study"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"resuming: {len(done)} already labelled")

    gold_ids = set(tr[tr[TARGETS].notna().all(axis=1)].StudyInstanceUID)
    todo = tr[~tr.StudyInstanceUID.isin(done)].copy()
    if a.gold_first:
        todo["_g"] = ~todo.StudyInstanceUID.isin(gold_ids)
        todo = todo.sort_values("_g")
    if a.limit:
        todo = todo.head(a.limit)
    print(f"to label: {len(todo)}")

    print(f"loading {a.model} ...", flush=True)
    # Mistral tokenizers ship an incorrect pre-tokenizer regex; transformers warns that
    # loading WITHOUT fix_mistral_regex=True "will lead to incorrect tokenization" -- silent
    # label corruption on a multilingual corpus. Qwen tokenizers reject the kwarg, so guard it.
    try:
        tok = AutoTokenizer.from_pretrained(a.model, fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(a.model)
    kw = dict(dtype=torch.bfloat16, device_map="auto")
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    model.eval()

    # Decoder-only batching needs LEFT padding: with right padding the pad tokens sit
    # between the prompt and the first generated token, and the model attends to them.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def build(rep):
        # The 6,000-char cap is a guard, not a policy: the longest report in train.csv is
        # 4,743 chars, so it never fires today. It exists because truncation below cuts the
        # TAIL of the encoded prompt, which is the generation prompt, not the report -- one
        # pathological row could otherwise silently amputate the "answer in JSON" framing.
        # enable_thinking=False is load-bearing on Qwen3: left on, the model opens a <think>
        # block that consumes the 260-token budget before it ever emits the object, and
        # every answer comes back unparseable.
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(report=str(rep)[:6000])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    rows = list(todo.itertuples())
    t0, n_ok, n_bad, done = time.time(), 0, 0, 0
    with out.open("a", encoding="utf-8") as fh:
        for s in range(0, len(rows), a.batch):
            chunk = rows[s:s + a.batch]
            enc = tok([build(r.Report) for r in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=4096).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                     temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id)
            plen = enc["input_ids"].shape[1]
            for r, g in zip(chunk, gen):
                sev = parse_answer(tok.decode(g[plen:], skip_special_tokens=True))
                # A garbled generation is still WRITTEN, as all-"not_mentioned". The row
                # must exist or a resume retries it forever; every cell being masked means
                # the failure costs us one study's signal instead of poisoning twelve
                # labels. n_bad is the number to watch -- a few is normal, a spike means
                # the prompt or the chat template broke.
                if sev is None:
                    n_bad += 1
                    sev = {t: "not_mentioned" for t in TARGETS}
                else:
                    n_ok += 1
                fh.write(json.dumps({"study": r.StudyInstanceUID, "sev": sev},
                                    ensure_ascii=False) + "\n")
            fh.flush()
            done += len(chunk)
            if done % (a.batch * 4) < a.batch:
                el = time.time() - t0
                print(f"  {done}/{len(rows)}  ok={n_ok} unparsed={n_bad}  "
                      f"{el / done:.2f}s/report  "
                      f"eta {(len(rows) - done) * el / done / 60:.0f} min", flush=True)

    print(f"\ndone: {n_ok} parsed, {n_bad} unparsed")
    score_file(out)


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
