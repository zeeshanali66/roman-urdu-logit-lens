"""Where does Roman Urdu lose the symptom? A logit-lens study of symptom-frequency understanding.

Our screening study found that LLMs misread how often a symptom happens when it is described in Urdu or Roman Urdu.
Here we look inside small open models to see WHERE (at which layer) the frequency is understood, per language.

Stimuli: 8 PHQ-9 symptoms x 3 frequencies x 3 languages (English, Urdu script, Roman Urdu) = 72 sentences, built
from one fixed frequency mapping (the same one used in the screening study), so only the language changes.
Task (instruction always in English): "How often...? Answer with one word: sometimes, often, or daily."
For every layer we apply the final norm + unembedding to the last position (logit lens) and read the probability
of the correct answer among the three options. We also decode the top token per layer to see whether the middle
layers "think" in Latin-script (English-like) tokens for Urdu input (cf. Wendler et al., ACL 2024).

Run with the F: environment:  F:/Masters/tools/pyenv/Scripts/python.exe lens_study.py selftest | stimuli | run | results
"""
import csv, json, os, sys, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "lens.jsonl")
RES = os.path.join(HERE, "results")
MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "bigscience/bloomz-1b7"]

SYMPTOMS = {  # item: (English, Roman Urdu, Urdu script). Reviewed for naturalness 2026-11-24 (not yet native-validated)
    "interest": ("I don't enjoy anything.", "mujhe kisi cheez mein maza nahi aata.", "مجھے کسی چیز میں مزہ نہیں آتا۔"),
    "mood": ("I feel sad and hopeless.", "main udaas aur na-umeed mehsoos karta hoon.", "میں اداس اور ناامید محسوس کرتا ہوں۔"),
    "sleep": ("I can't sleep at night.", "mujhe raat ko neend nahi aati.", "مجھے رات کو نیند نہیں آتی۔"),
    "energy": ("I feel tired and have no energy.", "main thaka hua mehsoos karta hoon aur mujh mein himmat nahi hoti.", "میں تھکا ہوا محسوس کرتا ہوں اور مجھ میں ہمت نہیں ہوتی۔"),
    "appetite": ("I don't feel like eating.", "mera khana khane ko dil nahi karta.", "میرا کھانا کھانے کو دل نہیں کرتا۔"),
    "self-worth": ("I feel like a failure.", "mujhe lagta hai main nakaam hoon.", "مجھے لگتا ہے میں ناکام ہوں۔"),
    "concentration": ("I can't concentrate on my work.", "main apne kaam par dhyan nahi de pata.", "میں اپنے کام پر دھیان نہیں دے پاتا۔"),
    "psychomotor": ("I feel restless and can't sit still.", "mujhe bechaini hoti hai aur main tik kar nahi baith sakta.", "مجھے بےچینی ہوتی ہے اور میں ٹک کر نہیں بیٹھ سکتا۔"),
}
FREQ = {  # several common ways to say each frequency (people phrase it differently).
    # Native-speaker guidance (Zeeshan, 2026-11-24): "kuch din" is less used, mostly "kabhi kabhi"; "zyada tar" without
    # "din"; nearly every day = "taqreeban har din" or "taqreeban roz" (plain "roz" = exactly every day).
    # The screening study's phrasing is included as one variant per level (marked * below).
    1: {"answer": "sometimes", "en": ["On a few days,", "Some days,", "Every now and then,"],
        "ru": ["Kabhi kabhi", "Kuch din*", "Chand din"], "ur": ["کبھی کبھی", "چند دن*", "کچھ دن"]},
    2: {"answer": "often", "en": ["On most days,", "Most days,", "Frequently,"],
        "ru": ["Zyada tar", "Aksar", "Zyada tar din*"], "ur": ["زیادہ تر", "اکثر", "زیادہ تر دن*"]},
    3: {"answer": "daily", "en": ["Nearly every day,", "Almost every day,", "Practically every day,"],
        "ru": ["Taqreeban roz", "Taqreeban har din", "Taqreeban har roz*"], "ur": ["تقریباً روز", "تقریباً ہر دن", "تقریباً ہر روز*"]},
}
LANGS = ["en", "ru", "ur"]
OPTIONS = ["sometimes", "often", "daily"]


def sentence(item, level, lang, v=0):
    en, ru, ur = SYMPTOMS[item]
    f = FREQ[level][lang][v].rstrip("*")
    if lang == "en":
        return f"{f} {en if en.startswith('I ') else en[0].lower() + en[1:]}"
    return f"{f} {ru if lang == 'ru' else ur}"


def prompt(text):
    return ("A patient says: \"" + text + "\"\n"
            "Question: How often does the patient have this problem? Answer with one word: sometimes, often, or daily.\n"
            "Answer:")


def stimuli():
    for item in SYMPTOMS:
        for level in FREQ:
            for lang in LANGS:
                for v, phrase in enumerate(FREQ[level][lang]):
                    yield {"item": item, "level": level, "lang": lang, "variant": v, "phrase": phrase.rstrip("*"),
                           "study_phrasing": phrase.endswith("*"), "text": sentence(item, level, lang, v),
                           "answer": FREQ[level]["answer"]}


def script_of(tok):
    """'latin', 'arabic' or 'other' for a decoded token."""
    letters = [c for c in tok if c.isalpha()]
    if not letters:
        return "other"
    names = [unicodedata.name(c, "") for c in letters]
    if all(n.startswith("LATIN") for n in names):
        return "latin"
    if any(n.startswith("ARABIC") for n in names):
        return "arabic"
    return "other"


def lens_parts(model):
    """Final norm and unembedding for the supported architectures."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):      # Qwen2 / Llama style
        return model.model.norm, model.lm_head
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):  # BLOOM / GPT style
        return model.transformer.ln_f, model.lm_head
    raise ValueError("unknown architecture")


def option_ids(tok):
    """First token id of ' sometimes', ' often', ' daily' (with the leading space the answer will have)."""
    ids = []
    for w in OPTIONS:
        t = tok.encode(" " + w, add_special_tokens=False)
        ids.append(t[0])
    assert len(set(ids)) == 3, f"option first-tokens collide: {ids}"
    return ids


CAL = os.path.join(HERE, "data", "calibration.json")
CONTENT_FREE = ["N/A", "...", "[blank]"]


def calibrate(models=MODELS):
    """Option prior per layer from content-free inputs (Zhao et al., 2021, "Calibrate Before Use")."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = {}
    for name in models:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda").eval()
        norm, head = lens_parts(model)
        opt = option_ids(tok)
        acc = None
        for cf in CONTENT_FREE:
            ids = tok(prompt(cf), return_tensors="pt").to("cuda")
            with torch.no_grad():
                hs = model(**ids, output_hidden_states=True).hidden_states
            rows = []
            for li in range(1, len(hs)):
                h = hs[li][0, -1]
                h = h if li == len(hs) - 1 else norm(h)
                rows.append(torch.softmax(head(h).float()[opt], dim=0).tolist())
            acc = rows if acc is None else [[a + b for a, b in zip(x, y)] for x, y in zip(acc, rows)]
        out[name] = [[v / len(CONTENT_FREE) for v in r] for r in acc]
        del model
        torch.cuda.empty_cache()
        print("calibrated", name, [round(v, 2) for v in out[name][-1]], flush=True)
    json.dump(out, open(CAL, "w"))


def run(models=MODELS):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["model"], r["item"], r["level"], r["lang"], r["variant"]))
    for name in models:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda").eval()
        norm, head = lens_parts(model)
        opt = option_ids(tok)
        for s in stimuli():
            if (name, s["item"], s["level"], s["lang"], s["variant"]) in done:
                continue
            ids = tok(prompt(s["text"]), return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            hs = out.hidden_states  # embeddings + one per layer
            per_layer = []
            for li in range(1, len(hs)):
                h = hs[li][0, -1]
                h = h if li == len(hs) - 1 else norm(h)   # the last hidden state is already normed in HF
                logits = head(h).float()
                p3 = torch.softmax(logits[opt], dim=0).tolist()
                top = tok.decode([int(logits.argmax())])
                per_layer.append({"p": [round(x, 4) for x in p3], "top": top, "script": script_of(top)})
            final = out.logits[0, -1].float()
            pred = OPTIONS[int(torch.argmax(final[opt]))]
            rec = {"model": name, **s, "n_layers": len(hs) - 1, "pred": pred, "correct": pred == s["answer"],
                   "n_tokens": int(ids["input_ids"].shape[1]), "layers": per_layer}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(name, s["item"], s["level"], s["lang"], s["variant"], pred, rec["correct"], flush=True)
        del model
        torch.cuda.empty_cache()
    print("done", flush=True)


def results():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(RES, exist_ok=True)
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    df = pd.DataFrame(rows)
    L = ["# Where does Roman Urdu lose the symptom? (logit lens)", "",
         f"{len(df)} stimuli x layers; models: {', '.join(df.model.unique())}.", "",
         "## Final-layer accuracy (choosing sometimes / often / daily)", "", "```",
         df.pivot_table(index="model", columns="lang", values="correct", aggfunc="mean").round(2)[LANGS].to_string(), "```", "",
         "## Accuracy by frequency level", "", "```",
         df.pivot_table(index=["model", "level"], columns="lang", values="correct", aggfunc="mean").round(2)[LANGS].to_string(), "```", ""]
    L += ["## Accuracy per phrasing (each row = one way of saying the frequency; * = the screening study's phrasing)", "", "```",
          df.assign(phrase=df.phrase + df.study_phrasing.map({True: "*", False: ""}))
            .pivot_table(index=["level", "lang", "phrase"], columns="model", values="correct", aggfunc="mean").round(2).to_string(),
          "```", ""]
    cal = json.load(open(CAL)) if os.path.exists(CAL) else None
    if cal:
        def cal_p(r, i):
            pr = cal[r["model"]][i]
            q = [a / b for a, b in zip(r["layers"][i]["p"], pr)]
            z = sum(q)
            return [x / z for x in q]
        df["cal_correct"] = [OPTIONS[int(np.argmax(cal_p(r, len(r["layers"]) - 1)))] == r["answer"] for r in rows]
        L += ["## CALIBRATED final-layer accuracy (option prior from content-free inputs divided out)", "",
              "Small models have strong answer habits (e.g. one never says 'often'); calibration removes them.", "", "```",
              df.pivot_table(index="model", columns="lang", values="cal_correct", aggfunc="mean").round(2)[LANGS].to_string(), "```", "",
              "## Calibrated accuracy by frequency level", "", "```",
              df.pivot_table(index=["model", "level"], columns="lang", values="cal_correct", aggfunc="mean").round(2)[LANGS].to_string(), "```", "",
              "## Calibrated accuracy per phrasing (* = screening study's phrasing)", "", "```",
              df.assign(phrase=df.phrase + df.study_phrasing.map({True: "*", False: ""}))
                .pivot_table(index=["level", "lang", "phrase"], columns="model", values="cal_correct", aggfunc="mean").round(2).to_string(),
              "```", ""]
    # layer curves: mean probability of the correct option at each relative depth
    curves, emerge, latin = [], [], []
    for r in rows:
        k = OPTIONS.index(r["answer"])
        ps = [(cal_p(r, i)[k] if cal else lay["p"][k]) for i, lay in enumerate(r["layers"])]
        n = len(ps)
        for i, p in enumerate(ps):
            curves.append({"model": r["model"], "lang": r["lang"], "depth": round((i + 1) / n, 2), "p": p})
            latin.append({"model": r["model"], "lang": r["lang"], "depth": round((i + 1) / n, 2),
                          "latin": r["layers"][i]["script"] == "latin"})
        # first layer from which the correct option stays the most likely to the end
        ok = [lay["p"][k] == max(lay["p"]) for lay in r["layers"]]
        first = next((i for i in range(n) if all(ok[i:])), None)
        emerge.append({"model": r["model"], "lang": r["lang"], "emerge_depth": None if first is None else (first + 1) / n})
    E = pd.DataFrame(emerge)
    L += ["## Where the right answer settles (relative depth 0-1; lower = earlier; NaN = never settles)", "", "```",
          E.groupby(["model", "lang"]).emerge_depth.agg(["mean", "count"]).round(2).to_string(), "```", ""]
    Lt = pd.DataFrame(latin)
    Lt["band"] = pd.cut(Lt.depth, [0, 0.33, 0.66, 1.0], labels=["early", "middle", "late"])
    L += ["## Share of layers whose top decoded token is Latin-script (English-like), by input language", "",
          "For Urdu-script input, a high middle-layer share suggests the model routes through English-like tokens.", "", "```",
          Lt.pivot_table(index=["model", "band"], columns="lang", values="latin", aggfunc="mean", observed=False).round(2)[LANGS].to_string(),
          "```", ""]
    C = pd.DataFrame(curves)
    fig, axs = plt.subplots(1, len(df.model.unique()), figsize=(6 * len(df.model.unique()), 3.6), squeeze=False)
    for ax, m in zip(axs[0], df.model.unique()):
        for lang, col in zip(LANGS, ["#5b8def", "#e07b39", "#3aa76d"]):
            g = C[(C.model == m) & (C.lang == lang)].groupby("depth").p.mean()
            ax.plot(g.index, g.values, label={"en": "English", "ru": "Roman Urdu", "ur": "Urdu script"}[lang], color=col)
        ax.axhline(1 / 3, ls=":", color="gray")
        ax.set_title(m.split("/")[-1]); ax.set_xlabel("relative layer depth"); ax.set_ylim(0, 1)
    axs[0][0].set_ylabel("P(correct frequency), calibrated"); axs[0][0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_lens.png"), dpi=150); plt.close(fig)
    open(os.path.join(RES, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


def write_validation_sheet():
    path = os.path.join(HERE, "data", "validate_urdu.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "level", "english", "roman_urdu", "urdu", "ok? (y / fix)"])
        for item in SYMPTOMS:
            for level in FREQ:
                w.writerow([item, level, sentence(item, level, "en"), sentence(item, level, "ru"), sentence(item, level, "ur"), ""])
    print("wrote", path)


def selftest():
    S = list(stimuli())
    assert len(S) == 8 * 3 * 3 * 3, len(S)
    assert sentence("sleep", 1, "ru", 0) == "Kabhi kabhi mujhe raat ko neend nahi aati."
    assert sentence("sleep", 2, "ru", 2) == "Zyada tar din mujhe raat ko neend nahi aati."
    assert sentence("sleep", 1, "en", 0) == "On a few days, I can't sleep at night."
    assert sum(x["study_phrasing"] for x in S) == 8 * 3 * 2
    assert not any(o in x["text"].lower() for x in S if x["lang"] == "en" for o in OPTIONS), "answer word leaks into English"
    assert script_of("the") == "latin" and script_of("نیند") == "arabic" and script_of("12") == "other"
    print("selftest ok:", len(S), "stimuli")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    {"selftest": selftest, "stimuli": write_validation_sheet, "run": run, "results": results, "calibrate": calibrate}[cmd]()
