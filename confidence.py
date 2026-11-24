"""Part 2 (CPU only, reuses data/lens.jsonl): does the model KNOW when it misreads Roman Urdu?

A screening tool could hand uncertain cases to a human. That only protects Roman Urdu speakers if the model is less
confident exactly when it is wrong. For each answer we compute (on calibrated final-layer probabilities):
  conf      - probability of the chosen answer
  margin    - gap between the top two answers
  stability - share of the last third of layers whose (calibrated) answer equals the final answer (an internal signal)
and ask: (1) is confidence lower for wrong answers (AUROC), (2) is confidence honest (calibration error, ECE),
(3) with ONE deferral rule for everyone ("send the least confident 30% to a human"), who gets deferred and how many
errors remain, per language.
"""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CAL = json.load(open(os.path.join(HERE, "data", "calibration.json")))
OPTS = ["sometimes", "often", "daily"]
LANG = {"en": "English", "ur": "Urdu script", "ru": "Roman Urdu"}


def calp(model, layer, p):
    q = np.array(p) / np.array(CAL[model][layer])
    return q / q.sum()


def auroc(score, y):
    """P(score of a correct answer > score of a wrong one); 0.5 = no signal."""
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def ece(conf, y, bins=5):
    e, edges = 0.0, np.linspace(1 / 3, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi if hi == 1 else conf < hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - y[m].mean())
    return e


def main():
    rows = []
    for l in open(os.path.join(HERE, "data", "lens.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        n = len(r["layers"])
        probs = [calp(r["model"], i, r["layers"][i]["p"]) for i in range(n)]
        fin = probs[-1]
        pred = int(fin.argmax())
        s = sorted(fin, reverse=True)
        last = probs[n - n // 3:]
        rows.append({"model": r["model"].split("/")[1], "lang": r["lang"], "level": r["level"], "phrase": r["phrase"],
                     "correct": OPTS[pred] == r["answer"], "conf": float(s[0]), "margin": float(s[0] - s[1]),
                     "stability": float(np.mean([int(p.argmax()) == pred for p in last]))})
    df = pd.DataFrame(rows)
    out = ["# Part 2: does the model know when it misreads Roman Urdu? (calibrated final layer)", "",
           "AUROC: how well a signal separates right from wrong answers (0.5 = no signal, 1 = perfect). "
           "ECE: gap between stated confidence and actual accuracy (0 = honest).", ""]
    tab = []
    for (m, lg), g in df.groupby(["model", "lang"]):
        y = g.correct.values
        tab.append({"model": m, "language": LANG[lg], "accuracy": round(y.mean(), 2),
                    "conf if right": round(g[g.correct].conf.mean(), 2) if y.any() else np.nan,
                    "conf if wrong": round(g[~g.correct].conf.mean(), 2) if (~y).any() else np.nan,
                    "AUROC conf": round(auroc(g.conf.values, y), 2), "AUROC margin": round(auroc(g.margin.values, y), 2),
                    "AUROC stability": round(auroc(g.stability.values, y), 2), "ECE": round(ece(g.conf.values, y), 2)})
    out += [pd.DataFrame(tab).to_markdown(index=False), ""]
    # one deferral rule for everyone: defer the least confident 30% of ALL answers (per model)
    out += ["## One rule for everyone: send the least-confident 30% to a human", "",
            "Per language: share of that language's answers deferred, accuracy of the answers the model keeps, and the "
            "share of that language's errors that slip through (kept and wrong).", ""]
    tab2 = []
    for m, g in df.groupby("model"):
        thr = g.conf.quantile(0.30)
        for lg in ("en", "ur", "ru"):
            h = g[g.lang == lg]
            keep = h[h.conf > thr]
            tab2.append({"model": m, "language": LANG[lg], "deferred": round(1 - len(keep) / len(h), 2),
                         "accuracy before": round(h.correct.mean(), 2),
                         "accuracy of kept": round(keep.correct.mean(), 2) if len(keep) else np.nan,
                         "errors slipping through": round((~keep.correct).sum() / max(1, (~h.correct).sum()), 2)})
    out += [pd.DataFrame(tab2).to_markdown(index=False), ""]
    # the key phrase: "nearly every day" in Roman Urdu
    k = df[(df.lang == "ru") & (df.level == 3)]
    out += ["## 'Nearly every day' in Roman Urdu (taqreeban ...)", "",
            f"Accuracy {k.correct.mean():.2f}; mean confidence {k.conf.mean():.2f} "
            f"(English 'nearly every day': accuracy {df[(df.lang == 'en') & (df.level == 3)].correct.mean():.2f}, "
            f"confidence {df[(df.lang == 'en') & (df.level == 3)].conf.mean():.2f}).", ""]
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, "RESULTS_confidence.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
