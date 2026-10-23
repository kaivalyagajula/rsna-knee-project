"""Rule-based (weak) extractor for the 12 knee findings from radiology reports.

Handles English and Spanish reports only; other languages return all zeros
and are flagged by `supported()`.

Usage: python weak_labels.py [--csv data/train.csv]   (scores the labeled studies)
"""
import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# Negation cues. PRE cues negate what follows them in the clause; POST cues
# negate the term they follow ("ACL is intact").
PRE_NEG = re.compile(
    r"\b(no|not|without|negative for|absence of|absent|free of|ruled out|"
    r"no evidence of|nor|neither|sin|ausencia de|no hay|no se observa\w*|no se aprecia\w*|"
    r"no existe\w*|descarta\w*)\b"
)
POST_NEG = re.compile(
    r"\b(intact|normal|preserved|unremarkable|unchanged|within normal limits|"
    r"conservad[oa]s?|integr[oa]s?|indemne|sin alteraciones|sin cambios|"
    r"is not torn|are not torn|not seen|not identified|not present|negative)\b"
)
CONTRAST = re.compile(r"\b(but|however|whereas|while|although|though|except|pero|aunque|excepto|salvo)\b")

ABN_LIG = (
    r"tear|torn|rupture|ruptur|sprain|strain|partial|complete|disrupt|degenerat|"
    r"injur|avulsion|attenuat|laxity|edema|thicken|insufficien|incompeten|"
    r"rotura|rotur|desgarro|lesion|esguince|degeneracion|engrosamiento|laxitud|"
    r"sprained|elongat|discontinu|fray|grade i"
)
ABN_MEN = ABN_LIG + r"|signal|extrusion|maceration|flap|bucket|cleav|horizontal|radial|root|oblique|fissur|amputac"
ABN_OA = (
    r"osteoarthr|arthrosis|artrosis|osteophyt|osteofit|chondromalacia|condropat|"
    r"cartilage (loss|thinning|defect|fissur|erosion|wear|degenerat|ulcer|damage|irregular)|"
    r"(chondral|cartilage|cartilaginous) (loss|defect|thinning|fissur|erosion|wear|lesion|damage)|"
    r"degenerative|joint space narrowing|subchondral (sclerosis|cyst)|condral|"
    r"cartilago (adelgaz|defecto|perdida|lesion)|esclerosis subcondral|pinzamiento|"
    r"full[- ]thickness|grade (ii|iii|iv|2|3|4)|chondrosis|erosion|ulcer"
)

# structure regex, abnormality regex (None = the term itself implies disease)
SIMPLE = {
    "ACL": (r"\bacl\b|anterior cruciate|\blca\b|cruzado anterior|ligamento cruzado|cruciate", ABN_LIG),
    "MCL": (r"\bmcl\b|medial collateral|tibial collateral|\blcm\b|colateral medial|collateral", ABN_LIG),
    "Effusion": (r"effusion|joint fluid|derrame|hydrarthrosis|hidrartrosis", None),
    "Synovitis": (r"synovitis|synovial (thicken|hypertroph|enhanc|prolifer)|sinovitis|"
                  r"sinovial (engros|hipertrof)|hoffa\w* (fat pad )?(edema|inflam|impinge)|hoffitis|"
                  r"pinzamiento de la almohadilla", None),
    "Baker's": (r"baker|popliteal cyst|poplit\w* cyst|gastrocnemio[- ]semimembranoso|"
                r"quiste (de baker|poplit)|semimembranosus bursa", None),
    "Contusion": (r"contusion|bone bruis|bruis|marrow edema|bone marrow (edema|signal abnormal)|"
                  r"osseous edema|osteochondral|contusiones|edema oseo|edema de medula", None),
    "Fracture": (r"fracture|fractura|fracturar", None),
}
MED = r"\bmedial|\binterno|\binterna|\bmediales|tricompart|femorotibial\b(?! (medial|lateral))"
LAT = r"\blateral|\bexterno|\bexterna|tricompart|femorotibial\b(?! (medial|lateral))"
PF = r"patell|trochle|patellofemoral|femoropatelar|rotul|troclea|tricompart"
SIDED = {
    "Medial Meniscus": (r"menisc|menisci|menisco", ABN_MEN, MED),
    "Lateral Meniscus": (r"menisc|menisci|menisco", ABN_MEN, LAT),
    "Medial OA": (r"", ABN_OA, MED),
    "Lateral OA": (r"", ABN_OA, LAT),
    "PF OA": (r"", ABN_OA, PF),
}
# a side word that belongs to a different structure than the finding being scored
OTHER_SIDE_ONLY = {
    "Medial Meniscus": r"\blateral|\bexterno|\bexterna",
    "Lateral Meniscus": r"\bmedial|\binterno|\binterna",
}


def norm(text):
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def clauses(report):
    """Yield (clause, negated_from_previous_chunk) pieces of the report."""
    for sent in re.split(r"(?<=[a-z0-9\)])\.(?:\s|$)|[\n;]+|:\s", norm(report)):
        carry = False
        for part in CONTRAST.split(sent):
            if CONTRAST.fullmatch(part.strip() or "-"):
                carry = False
                continue
            for chunk in part.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                # "no fracture, effusion or cyst": a bare list item inherits the negation
                short = len(chunk.split()) <= 4 or re.match(r"(or|and|nor|o|y|ni)\b", chunk)
                inherited = carry and bool(short)
                yield chunk, inherited
                carry = inherited or bool(PRE_NEG.search(chunk))


def negated(chunk, inherited, start, end):
    if inherited and not PRE_NEG.search(chunk[:start]):
        return True
    if PRE_NEG.search(chunk[:start]):
        return True
    if POST_NEG.search(chunk[end:]):
        return True
    return False


def extract(report):
    text = report if isinstance(report, str) else ""
    out = dict.fromkeys(LABEL_COLS, 0)
    for chunk, inherited in clauses(text):
        for name, (struct, abn) in SIMPLE.items():
            for m in re.finditer(struct, chunk):
                if negated(chunk, inherited, m.start(), m.end()):
                    continue
                if abn is None or re.search(abn, chunk):
                    out[name] = 1
        for name, (struct, abn, side) in SIDED.items():
            if struct and not re.search(struct, chunk):
                continue
            a = re.search(abn, chunk)
            if not a or not re.search(side, chunk):
                continue
            if name in OTHER_SIDE_ONLY and re.search(OTHER_SIDE_ONLY[name], chunk) \
                    and not re.search(side, chunk):
                continue
            if negated(chunk, inherited, a.start(), a.end()):
                continue
            out[name] = 1
    return out


def supported(report):
    return bool(re.search(
        r"\b(the|and|of|is|no|with|knee|meniscus|ligament|rodilla|menisco|ligamento|derrame|rotura)\b",
        norm(report)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=Path("data/train.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df[LABEL_COLS].notna().any(axis=1)].reset_index(drop=True)
    pred = pd.DataFrame([extract(r) for r in df.Report])
    ok = df.Report.map(supported)
    print(f"{len(df)} labeled studies; {int(ok.sum())} look English/Spanish, "
          f"{int((~ok).sum())} do not (predicted all-zero)\n")

    rows = []
    for c in LABEL_COLS:
        y, p = df[c].astype(int), pred[c]
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        rows.append({
            "finding": c,
            "acc_all": (y == p).mean(),
            "acc_en_es": (y[ok] == p[ok]).mean(),
            "majority": max(y.mean(), 1 - y.mean()),
            "prec": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
        })
    res = pd.DataFrame(rows).set_index("finding")
    print(res.round(2).to_string())
    print(f"\nmean accuracy: all={res.acc_all.mean():.3f}  en/es only={res.acc_en_es.mean():.3f}  "
          f"majority-class baseline={res.majority.mean():.3f}")


if __name__ == "__main__":
    main()
