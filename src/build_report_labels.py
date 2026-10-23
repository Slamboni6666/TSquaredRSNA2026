"""
build_report_labels.py -- our own multilingual report labeler. Built from scratch.

Design follows three things measured on the 58 gold studies (see analyze_gold_disagreements.py):

  1. When a report NEGATES a finding it is right 96% of the time. Negation is reliable.
  2. When a report ASSERTS a finding it is right only 65% of the time. Bare assertion is
     nearly a coin flip, because gold was read off the images against a SEVERITY bar and
     graded borderline cases negative. "Small joint effusion" -> gold says no.
  3. Silence is not absence: findings never mentioned are still gold-positive 14% of the
     time, and for Synovitis far more, because radiologists see it and do not write it.

So the extractor emits a GRADED score, not a binary. Ordering is all the metric reads, and
severity is what separates a gold positive from a gold negative. A confident assertion of a
severe finding must outrank a confident assertion of a mild one, and both must outrank
silence, which must outrank an explicit denial.

Output: report_labels_ours.csv -- per target a score in [0,1] plus the evidence that
produced it: `__sev` (the severity word that fired, if any) and `__state` (which rung of
SCORE below the cell landed on). The cue clause itself is returned by label_report() and
printed by --show, but is not a CSV column: it is verbatim report text, which
check_repo_clean.py exists to keep off disk and out of git.

WHY `__state` MATTERS DOWNSTREAM
Training MASKS OUT every cell whose state is "silent" -- that cell contributes nothing to
the loss. That is the entire reason "the report never mentioned this" is a different state
from "the report says this is absent". Score silence as a negative and the student learns
each finding's base rate instead of its appearance. The LLM labeller spells this same state
"not_mentioned"; a reader that checks only one spelling silently disables masking for the
other labeller, which is a bug this project has already shipped.

WHY THIS FILE STILL EXISTS AFTER THE LLM LABELLER
It is no longer the primary teacher -- Qwen3-32B scores 0.861 on gold against these rules'
0.788, and wins 11 of 12 targets. What it still has is INDEPENDENT, DISCRETE evidence, and
that is exactly what teacher_consensus.py needs: the LLM only ever emits seven distinct
probabilities, so 12.9% of study pairs come out exactly tied and AUC scores a tie as 0.5.
These rules break those ties, worth a measured +0.0112 gold. Do not delete it.

Usage:
    python build_report_labels.py                  # write labels + evaluate on gold
    python build_report_labels.py --show Effusion  # print worked examples for one target
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# --------------------------------------------------------------------------- normalisation
# Turkish dotted/dotless i must fold before casefolding or "İZLENMEZ" and "izlenmez" diverge.
_PRE = str.maketrans({"ı": "i", "İ": "i", "I": "i", "ß": "ss", "đ": "d", "Đ": "d"})


def normalize(text) -> str:
    """Fold case, accents and separators. Keeps Greek and Cyrillic letters intact.

    NFKD strips Latin accents and Greek tonos alike (ά -> α), which is what we want since
    reports are inconsistent about accents. It also folds MICRO SIGN U+00B5 onto a real mu,
    which matters because many Greek reports here use the wrong codepoint.
    """
    if not isinstance(text, str):
        return ""
    t = text.translate(_PRE).lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("­", "")
    t = re.sub(r"[_/\\]+", " ", t)
    return t


def clauses(text: str) -> list[str]:
    """Split a report into clauses, repairing two layout habits that flip labels.

    Hard wrapping: many reports wrap at a fixed column, so a sentence breaks across lines
    with no punctuation. Splitting on newlines then severs a pathology word from the
    structure it belongs to, and the study comes out silent on a finding the report
    describes. A line not ending in sentence punctuation is a continuation.

    Headings: `Fractures :` followed by `Aucune.` is one statement across two lines. Split
    them and the negation is lost. A fragment ending in a colon is merged with the next,
    and does not also stand alone -- alone it carries the anatomy word with no negation in
    scope, which asserts the finding it was denying.
    """
    t = normalize(text)
    lines = [ln.strip() for ln in t.split("\n")]
    unwrapped, buf = [], ""
    for ln in lines:
        if not ln:
            if buf:
                unwrapped.append(buf)
                buf = ""
            continue
        buf = f"{buf} {ln}".strip() if buf else ln
        if re.search(r"[.;:!?]$", buf) or len(buf) > 400:
            unwrapped.append(buf)
            buf = ""
    if buf:
        unwrapped.append(buf)

    parts = []
    for blk in unwrapped:
        parts.extend(p.strip() for p in re.split(r"(?<=[.;!?])\s+", blk) if p.strip())

    merged, i = [], 0
    while i < len(parts):
        if parts[i].endswith(":") and i + 1 < len(parts):
            merged.append(f"{parts[i]} {parts[i + 1]}")
            i += 2
        else:
            merged.append(parts[i])
            i += 1
    return merged


def rx(*pats):
    return re.compile("|".join(pats))


# --------------------------------------------------------------------------- cue lexicons
# Every cue list carries all nine languages at once and each clause is tested against the
# union. Routing by detected language first would mean committing to a guess before reading
# any evidence, and the cheap guesses fail badly -- "la" is as common in Spanish as French.
# Greek and Cyrillic cues cannot collide with Latin-script ones, and the Latin-script
# medical vocabularies are close enough that a shared cue is usually right.

NEGATION = rx(
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\babsence\b", r"\babsent\b",
    r"\bno evidence\b", r"\bunremarkable\b", r"\bfree of\b", r"\bnone\b", r"\bnil\b",
    r"\bsin\b", r"\bno hay\b", r"\bausencia\b", r"\bausentes?\b", r"\bno se (observa|identifica|aprecia)",
    r"\bpas de\b", r"\bsans\b", r"\baucune?\b",
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b",
    r"\bkeine?\b", r"\bohne\b", r"\bnicht\b",
    r"\byok(tur)?\b", r"izlenme(di|mekte)", r"saptanmad", r"\bdegil\b", r"gozlenmemekte",
    r"mevcut degil", r"\bnema\b", r"\bbez\b", r"\bnisu\b", r"\bnije\b",
    r"\bδεν\b", r"\bχωρις\b", r"ουδεν",
    r"\bбез\b", r"\bне\b", r"липсва", r"\bняма\b",
)

NORMALITY = rx(
    r"\bnormal", r"\bintact\b", r"\bpreserved\b", r"\bwithin normal limits\b", r"\bcontinuous\b",
    r"limites normales", r"\bconservad", r"\bintegr", r"\bnormales\b",
    r"\bdoga(l|ll)\b", r"korunmus", r"\bnormaldir\b", r"olagan",
    r"\buredn", r"\bocuvan", r"\bodrzan", r"\bintakt",
    r"φυσιολογικ", r"ακεραι", r"unauffallig", r"regelrecht",
    r"нормал", r"запазен", r"съхранен", r"\bgaaf\b", r"\bnormaal\b",
)

HEDGE = rx(
    r"\bpossible\b", r"\bprobable\b", r"\bsuspicious\b", r"\bsuspected\b", r"\bcannot exclude\b",
    r"\bmay represent\b", r"\bquestionable\b", r"\bequivocal\b", r"\blikely\b", r"\bapparent\b",
    r"\bposible\b", r"\bsospech", r"\bno se puede excluir\b", r"\bdudos",
    r"\bmoglich", r"\bverdacht", r"\bfraglich",
    r"\bsuphe", r"\bolabilir\b", r"\bmuhtemel", r"\bekarte edilemez\b",
    r"\bmoguc", r"\bsumnj", r"\bmogelijk", r"\bverdacht",
    r"πιθαν", r"υποψ", r"възможн", r"съмнителн", r"\bprobablement\b", r"\bsuspect",
)

# Severity, ordered. The bar that separates a gold positive from a gold negative sits
# somewhere in here, and it is learned per target rather than assumed -- see fit_severity().
SEV_SEVERE = rx(
    r"\bsevere\b", r"\badvanced\b", r"\bmarked\b", r"\bgross\b", r"\blarge\b", r"\bextensive\b",
    r"\bcomplete\b", r"\bfull[- ]thickness\b", r"\bmassive\b", r"\bhigh[- ]grade\b",
    r"\bsever[ao]\b", r"\bavanzad", r"\bimportante\b", r"\bgrande\b", r"\bextens",
    r"\bschwer", r"\bausgepragt", r"\bgross", r"\bhochgradig",
    r"\bileri\b", r"\bbelirgin\b", r"\bbuyuk\b", r"\byaygin\b",
    r"\btesk", r"\bveliki\b", r"\bizrazit", r"σοβαρ", r"εκτεταμεν", r"μεγαλ",
    r"тежк", r"изразен", r"голям", r"\bernstig", r"\buitgebreid",
)
SEV_MODERATE = rx(
    r"\bmoderate\b", r"\bmodest\b", r"\bpartial[- ]thickness\b", r"\bintermediate\b",
    r"\bmoderad", r"\bmassig", r"\bmittelgradig", r"\borta\b", r"\bumjeren", r"\bosrednj",
    r"μετρι", r"умерен", r"\bmatig",
)
SEV_MILD = rx(
    r"\bmild\b", r"\bminimal\b", r"\bsmall\b", r"\bslight\b", r"\btrace\b", r"\bsome\b",
    r"\bmin[oi]r\b", r"\bearly\b", r"\bincipient\b", r"\bdiscrete\b", r"\bfocal tiny\b",
    r"\bleve\b", r"\bminim", r"\bpeque", r"\bdiscret", r"\bligero", r"\binicial",
    r"\bgering", r"\bleicht", r"\bdiskret", r"\bbeginnend",
    r"\bhafif\b", r"\bminimal\b", r"\baz\b", r"\bbaslangic",
    r"\bblag", r"\bmali\b", r"\bmanji\b",
    r"ηπι", r"μικρ", r"ελαφρ", r"лек", r"минимал", r"малк", r"\bgering", r"\blicht",
)

# Explicit numeric grade. Meniscal signal is graded 1-3 and only grade 3 -- signal reaching
# the articular surface -- is a tear. Cartilage uses Outerbridge 1-4. Reading the number is
# the difference between "degeneration" and "tear" on the same sentence.
GRADE_RX = re.compile(
    r"(?:grade|grado|grad|derece|evre|stupanj|βαθμ\w*|степен)\s*[:\-]?\s*"
    r"(iv|iii|ii|i|[0-4])\b", re.I)
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}


def grade_of(clause: str):
    """Numeric grade named in the clause (roman or arabic) as an int, else None.

    Callers use it as a threshold, not as a number: `grade_tear=3` makes grade-3 meniscal
    signal a tear and leaves grades 1-2 as degeneration, and for anything else 3-4 reads as
    severe, 2 as moderate, 1 as mild -- which is the Outerbridge cartilage scale.
    """
    m = GRADE_RX.search(clause)
    if not m:
        return None
    g = m.group(1).lower()
    return _ROMAN.get(g, None) if g in _ROMAN else int(g)


# --------------------------------------------------------------------------- pathology cues
TEAR = rx(
    r"\btear\b", r"\btears\b", r"\btorn\b", r"\brupture", r"\bdisrupt", r"\bdiscontinu",
    r"\bavuls", r"\bbucket[- ]handle\b", r"\bmacerat",
    r"\brotura\b", r"\bruptura", r"\bdesgarr", r"\brasgad",
    r"\bruptur", r"\briss\b", r"\bkontinuitatsunterbrechung",
    r"\byirtik", r"\brupturu", r"\bkopma",
    r"\bruptur", r"\bpuknuc", r"\brascjep",
    r"ρηξη", r"ρηγμα", r"διατομ", r"руптур", r"разкъс", r"\bscheur",
)
DEGEN = rx(
    r"\bdegener", r"\bmucoid\b", r"\bintrasubstance\b", r"\bsignal\b",
    r"\bdejeneras", r"\bdegeneraci", r"\bdegenerativ", r"εκφυλ", r"дегенерат",
)
OA = rx(
    r"\bosteoarthrit", r"\barthros", r"\bosteoarthros", r"\bgonarthros", r"\bdegenerative (change|disease)",
    r"\bjoint space narrowing\b", r"\bchondral loss\b", r"\bcartilage (loss|thinning|defect|fissur|damage)",
    r"\bchondropath", r"\bchondromalac", r"\bosteophyt",
    r"\bartros", r"\bcondropat", r"\bcondromalac", r"\bosteofit", r"\bpinzamiento\b",
    r"\bgonartros", r"\bkondropat", r"\bkondromalaz", r"\bkikirdak\b",
    r"\bknorpel", r"\barthrose", r"\bosteophyt",
    r"αρθριτ", r"αρθρωσ", r"χονδροπαθ", r"χονδρομαλακ", r"οστεοφυτ",
    r"артроз", r"хондропат", r"остеофит", r"\bkraakbeen", r"\bartroz",
)


def anat(*pats):
    return rx(*pats)


MEDIAL = anat(r"\bmedial\b", r"\binternal\b", r"\binner\b", r"\bmedial(is|en|er|e)\b",
              r"\bic\b", r"\bmedijaln", r"\bεσω\b", r"\bвътрешн", r"\bmediaal", r"\bintern")
LATERAL = anat(r"\blateral\b", r"\bexternal\b", r"\bouter\b", r"\blateral(is|en|er|e)\b",
               r"\bdis\b", r"\blateraln", r"\bεξω\b", r"\bвъншн", r"\blateraal", r"\bextern")

MENISCUS = anat(r"menisc", r"menisk", r"μηνισκ", r"мениск", r"menisque")
CRUCIATE_ANT = anat(r"\bacl\b", r"anterior cruciate", r"cruzado anterior", r"kreuzband",
                    r"on capraz", r"prednji krizni", r"προσθι\w* χιαστ", r"предна кръстна",
                    r"voorste kruisband", r"croise anterieur", r"\blca\b", r"\bvkb\b")
CRUCIATE_POST = anat(r"\bpcl\b", r"posterior cruciate", r"cruzado posterior", r"hinteres kreuzband",
                     r"arka capraz", r"straznji krizni", r"οπισθι\w* χιαστ", r"задна кръстна")
COLLAT_MED = anat(r"\bmcl\b", r"medial collateral", r"colateral (medial|interno)", r"innenband",
                  r"\bic yan\b", r"medijalni kolateraln", r"εσω πλαγι", r"вътрешна колатерал",
                  r"mediale collaterale", r"collateral (interne|medial)", r"\blli\b",
                  r"medial(es)? kollateral")

# A compartment is named by a compartment word, or by the bones that bound it. "Medial"
# on its own is not a compartment -- it also introduces the medial meniscus, the medial
# collateral ligament and the medial patellofemoral ligament, all of which live in clauses
# that may separately mention cartilage. Requiring this pattern is what stops
# "medial meniscus tear with patellar chondromalacia" asserting medial-compartment OA.
COMPARTMENT = anat(
    r"compartment", r"compartimento", r"compartiment", r"kompartiment", r"kompartman",
    r"odjeljak", r"διαμερισμ", r"компартимент", r"kompartment",
    r"femorotibial", r"tibiofemoral", r"femoro tibial",
    r"condyl", r"condilo", r"kondil", r"kondyl", r"κονδυλ", r"кондил",
    r"plateau", r"platillo", r"plato tibial", r"κνημιαι", r"тибиалн",
    r"joint space", r"espacio articular", r"gelenkspalt", r"eklem aralig",
    r"gonartros", r"gonarthros", r"γοναρθρ",
)

# Patellofemoral means the joint, not everything with "patella" in the name. The patellar
# tendon and the medial patellofemoral ligament are soft-tissue structures whose clauses
# routinely also mention cartilage.
PATELLOFEM = anat(
    r"patellofemor", r"patelofemor", r"femoropatel", r"retropatel", r"retro patell",
    r"patellar (cartilage|chondr|facet|articular)", r"patellofemoral joint",
    r"cartilage of the patella", r"chondromalac\w* patell", r"condromalac\w* rotulian",
    r"trochlea", r"troclea", r"trohlea", r"τροχιλ", r"τροχλ", r"трохле",
    r"condropat\w* rotulian", r"kondromalazi", r"επιγονατιδομηριαι", r"пателофемор",
    r"diz kapagi kikirdak", r"patella\w* kikirdak", r"rotulian",
    # Mined: Spanish reports write "condropatia ... de la rotula" -- the bare noun, not the
    # adjective. 335 clauses, 140 of which matched no anatomy pattern at all. NOT_PF still
    # removes "ligamento rotuliano" and the tendon.
    r"\brotula\b", r"\brotulas\b",
)
NOT_PF = anat(r"patellar tendon", r"tendon rotulian", r"patellarsehne", r"patella tendon",
              r"patellofemoral ligament", r"\bmpfl\b", r"ligamento (femoro)?rotulian",
              r"retinacul")

CFG = {
    "ACL":              dict(anat=[CRUCIATE_ANT], path=[TEAR]),
    "MCL":              dict(anat=[COLLAT_MED],   path=[TEAR]),
    "Medial Meniscus":  dict(anat=[MENISCUS, MEDIAL], path=[TEAR], grade_tear=3, degen=DEGEN),
    "Lateral Meniscus": dict(anat=[MENISCUS, LATERAL], path=[TEAR], grade_tear=3, degen=DEGEN),
    # Requiring COMPARTMENT unconditionally cost 1,005 studies of Lateral OA coverage and
    # 554 of Medial OA -- a third of the corpus went silent -- while gold AUC moved by less
    # than its own noise, because 58 studies cannot see a corpus-wide recall change. The
    # compartment word is only needed to disambiguate: "medial" is unambiguous in
    # "medial compartment cartilage loss", and ambiguous only when the clause also names a
    # medial structure that is not the compartment. So demand it only then.
    "Medial OA":        dict(anat=[MEDIAL],  path=[OA],
                             confusable=[MENISCUS, COLLAT_MED, NOT_PF], disambiguator=COMPARTMENT),
    "Lateral OA":       dict(anat=[LATERAL], path=[OA],
                             confusable=[MENISCUS, NOT_PF,
                                         anat(r"collateral", r"colateral", r"\blcl\b")],
                             disambiguator=COMPARTMENT),
    "PF OA":            dict(anat=[PATELLOFEM], path=[OA], exclude=[NOT_PF]),
    "Effusion":         dict(anat=[anat(r"effusion", r"derrame", r"epanchement", r"erguss",
                                        r"efuzyon", r"eklem sivi", r"hidrartroz", r"izliv",
                                        r"συλλογ", r"излив", r"\bhydrops", r"\bvocht\b",
                                        # Mined: Greek reports write "υδραρθρο"
                                        # (hydrarthrosis) where we only had "συλλογη".
                                        r"υδραρθρ", r"hidrartroz", r"hydrarthr",
                                        r"uitstorting", r"gelenkflussigkeit", r"suprapatellar bursa")],
                             path=[], self_pathology=True),
    "Synovitis":        dict(anat=[anat(r"synovit", r"sinovit", r"synovial (thicken|hypertroph|proliferat)",
                                        r"engrosamiento sinovial", r"hipertrofia sinovial",
                                        r"synovialverdickung", r"sinovijaln", r"υμενιτ",
                                        r"синовит", r"synoviale verdikking", r"synovite",
                                        r"pannus")],
                             path=[], self_pathology=True),
    "Baker's":          dict(anat=[anat(r"\bbaker", r"popliteal cyst", r"quiste poplite",
                                        r"poplitealzyste", r"kyste poplite", r"poplitealn\w* cist",
                                        r"popliteal kist", r"κυστη \w*οπισθι", r"бейкер",
                                        r"поплитеал", r"bakerova", r"gastrocnemio.?semimembranos",
                                        # Mined: Bulgarian spells it "бекерова киста" -- one
                                        # letter from the transliteration we had. 99 clauses,
                                        # 86 invisible to every anatomy pattern.
                                        r"бекеров", r"бекерова киста")],
                             path=[], self_pathology=True),
    "Contusion":        dict(anat=[anat(r"contusion", r"bone bruise", r"bone marrow (o|)edema",
                                        r"edema (oseo|de medula|medular)", r"knochenmark(s|)odem",
                                        r"kemik ili\w* odem", r"edem kostne", r"kostna kontuzi",
                                        r"οιδημα του οστικου μυελου", r"οστικ\w* μωλωπ",
                                        r"костномозъчен едем", r"контузи", r"beenmergoedeem",
                                        r"medullary o?edema")],
                             path=[], self_pathology=True,
                             # Subchondral marrow oedema adjacent to cartilage loss is
                             # degenerative, not a traumatic contusion, and gold does not
                             # count it. Measured on the 58: this pattern is a recurring
                             # false positive.
                             demote=rx(r"subchondral", r"subcondral", r"subchondrale")),
    "Fracture":         dict(anat=[anat(r"fracture", r"fractura", r"fraktur", r"kirik",
                                        r"prijelom", r"καταγμα", r"фрактур", r"счупв",
                                        r"\bbreuk\b", r"insufficiency fracture")],
                             path=[], self_pathology=True),
}

# Graded score per state. Only the ORDER matters -- AUC is invariant to any increasing map --
# but the gaps encode how much evidence separates the states. Three placements look wrong at
# a glance and are not:
#   "silent" (0.15) outranks "negated" (0.03). A denial is right 96% of the time, silence
#   only 86%, so silence is weaker evidence of absence and must not collapse onto a denial.
#   "silent" is also the one state that gets masked out of the loss entirely downstream.
#   "asserted" (0.62) outranks "mild" (0.45). Gold graded borderline findings NEGATIVE, so
#   an explicitly small finding is weaker evidence than an unqualified assertion.
#   "degen_only" (0.30) sits above silence and below a hedge: grade 1-2 meniscal signal is
#   a real observation and is definitely not a tear, so the clause is informative without
#   supporting the target.
SCORE = {
    "severe": 0.95, "moderate": 0.75, "asserted": 0.62, "mild": 0.45,
    "hedged": 0.35, "silent": 0.15, "degen_only": 0.30, "negated": 0.03,
}


def _matches(clause, pats):
    """Every pattern must hit the SAME clause. This AND is how a structure name is built
    from parts: anat=[MENISCUS, MEDIAL] means "medial meniscus". Matching across clauses
    instead would let "medial collateral ligament intact. lateral meniscus tear." satisfy
    both, which is how a laterality mix-up gets into the labels.
    """
    return all(p.search(clause) for p in pats)


def label_report(text: str) -> dict:
    """-> {target: (score, severity, state, cue_clause)}

    One pass per target over the report's clauses:
      1. keep the clauses that name the anatomy (all of the target's anatomy patterns must
         hit the same clause -- see _matches);
      2. drop clauses that name a look-alike structure instead (`exclude`), and for the two
         tibiofemoral compartments demand the compartment be named explicitly whenever a
         same-sided look-alike is also present (`confusable` / `disambiguator`);
      3. classify each surviving clause as asserted / negated / hedged / degeneration-only,
         and where it asserts, grade the severity;
      4. keep the HIGHEST-scoring clause of those.

    Step 4 is the decision worth knowing about. Reports routinely deny a finding in a
    boilerplate normality sentence and then describe it in the findings section, so taking
    the maximum lets one explicit description outrank any amount of template text. The cost
    is that a single false positive clause also wins, which is why the gates in steps 1-2
    are strict and run first. A target with no surviving clause comes out "silent", which
    downstream is MASKED OUT of the loss rather than scored as a negative.
    """
    cls = clauses(text)
    out = {}
    for tgt, cfg in CFG.items():
        anat_pats = cfg["anat"]
        hits = []
        for c in cls:
            if not _matches(c, anat_pats):
                continue
            # A clause naming a structure that merely shares a name with the target is
            # dropped outright -- the medial patellofemoral ligament is not the
            # patellofemoral joint. This loses the rare clause that names both, which is
            # the cheaper error: the confusable structures appear far more often than the
            # co-mentions, and every one of them was an outright false positive.
            if any(p.search(c) for p in cfg.get("exclude", [])):
                continue
            # A laterality word is ambiguous only when the clause also names a same-sided
            # structure that is not the compartment. Then, and only then, insist the
            # compartment be named explicitly.
            conf = cfg.get("confusable")
            if conf and any(p.search(c) for p in conf):
                if not cfg["disambiguator"].search(c):
                    continue
            hits.append(c)

        if not hits:
            out[tgt] = (SCORE["silent"], None, "silent", "")
            continue

        best = (SCORE["negated"], None, "negated", hits[0])
        for c in hits:
            # Does the clause assert a pathology, or only name the structure?
            if cfg.get("self_pathology"):
                asserts = True
            else:
                asserts = any(p.search(c) for p in cfg["path"])
                g = grade_of(c)
                if not asserts and cfg.get("grade_tear") and g is not None:
                    asserts = g >= cfg["grade_tear"]
                if not asserts and cfg.get("need_compartment"):
                    asserts = bool(OA.search(c))

            neg = NEGATION.search(c)
            norm_ = NORMALITY.search(c)
            hedge = HEDGE.search(c)

            if not asserts:
                if cfg.get("degen") and cfg["degen"].search(c):
                    state, sev = "degen_only", None
                else:
                    state, sev = ("negated" if (neg or norm_) else "silent"), None
            elif neg or norm_:
                # Directional negation: a cue governs what FOLLOWS it. "fracture ... without
                # collapse" asserts the fracture; "no fracture" denies it. Anchor on where
                # the pathology word sits relative to the cue.
                cue_pos = (neg or norm_).start()
                path_pos = min((p.search(c).start() for p in (cfg["path"] or [])
                                if p.search(c)), default=None)
                if path_pos is None and cfg.get("self_pathology"):
                    m = anat_pats[0].search(c)
                    path_pos = m.start() if m else None
                state = "asserted" if (path_pos is not None and path_pos < cue_pos) else "negated"
                sev = None
            elif hedge:
                state, sev = "hedged", None
            else:
                state = "asserted"
                sev = ("severe" if SEV_SEVERE.search(c) else
                       "moderate" if SEV_MODERATE.search(c) else
                       "mild" if SEV_MILD.search(c) else None)
                g = grade_of(c)
                if sev is None and g is not None:
                    sev = "severe" if g >= 3 else "moderate" if g == 2 else "mild"

            key = sev if (state == "asserted" and sev) else state
            sc = SCORE[key]
            if state == "asserted" and cfg.get("demote") and cfg["demote"].search(c):
                sc = SCORE["mild"]
                key = "mild(subchondral)"
            if sc > best[0]:
                best = (sc, sev, key, c[:200])
        out[tgt] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default=None, help="print worked examples for one target")
    ap.add_argument("--out", default="report_labels_ours.csv")
    a = ap.parse_args()

    tr = pd.read_csv(ROOT / "train.csv")
    recs = [label_report(r) for r in tr.Report]

    df = pd.DataFrame({"StudyInstanceUID": tr.StudyInstanceUID})
    for t in TARGETS:
        df[t] = [r[t][0] for r in recs]
        df[t + "__sev"] = [r[t][1] for r in recs]
        df[t + "__state"] = [r[t][2] for r in recs]
    out = ROOT / a.out
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out}  ({len(df)} studies)\n")

    print("state distribution per target (all 4,407):")
    print(f"{'target':<18} " + "".join(f"{s:>12}" for s in
          ["severe", "moderate", "asserted", "mild", "hedged", "silent", "negated"]))
    for t in TARGETS:
        st = pd.Series([r[t][2] for r in recs]).value_counts()
        row = "".join(f"{st.get(s, 0):>12}" for s in
                      ["severe", "moderate", "asserted", "mild", "hedged", "silent", "negated"])
        print(f"{t:<18} {row}")

    # ---- evaluate on the 58
    from sklearn.metrics import roc_auc_score
    LAB = [c for c in tr.columns if c in TARGETS]
    gold = tr[tr[LAB].notna().all(axis=1)]
    m = gold[["StudyInstanceUID"] + LAB].merge(df, on="StudyInstanceUID", suffixes=("_g", "_p"))
    print(f"\n{'=' * 70}\nEVALUATION ON {len(m)} GOLD STUDIES\n{'=' * 70}")
    print(f"{'target':<18} {'AUC':>7} {'n_pos':>6}")
    aucs = []
    for t in TARGETS:
        auc = roc_auc_score(m[t + "_g"], m[t + "_p"])
        aucs.append(auc)
        print(f"{t:<18} {auc:>7.3f} {int(m[t + '_g'].sum()):>6}")
    macro = float(np.mean(aucs))
    # Bootstrap confidence interval: draw 58 studies WITH replacement from the 58 we have,
    # 2,000 times, recomputing the macro each time. The middle 95% of those values is how
    # far the number would move had we been handed a different 58 studies. It comes out near
    # +/-0.04, which is the reason no single-target change in the table above is evidence of
    # anything. A draw in which ANY target lands all-positive or all-negative has no defined
    # AUC, and the whole draw is discarded rather than scored 0.5 -- that keeps the centre
    # honest, at the cost of a CI that is if anything slightly too narrow.
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(m), len(m))
        try:
            bs.append(np.mean([roc_auc_score(m[t + "_g"].values[i], m[t + "_p"].values[i])
                               for t in TARGETS]))
        except ValueError:
            pass
    print(f"\n  MACRO AUC = {macro:.3f}   95% CI [{np.percentile(bs, 2.5):.3f}, "
          f"{np.percentile(bs, 97.5):.3f}]")
    print(f"  reference: best public label set scores 0.900 (not used, benchmark only)")

    if a.show:
        t = a.show
        print(f"\n{'=' * 70}\nWORKED EXAMPLES — {t}\n{'=' * 70}")
        sub = m.sort_values(t + "_p")
        for r in list(sub.itertuples())[:6] + list(sub.itertuples())[-6:]:
            i = tr.index[tr.StudyInstanceUID == r.StudyInstanceUID][0]
            sc, sev, state, cue = recs[i][t]
            print(f"  gold={int(getattr(r, t + '_g'))} score={sc:.2f} state={state} :: {cue[:150]}")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
