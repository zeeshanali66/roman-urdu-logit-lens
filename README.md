# Where does Roman Urdu lose the symptom? A logit-lens pilot

Most people in Pakistan describe how they feel in Urdu, or in Roman Urdu (Urdu typed in the Latin alphabet). A
companion study ([urdu-phq9-llm-pilot](https://github.com/zeeshanali66/urdu-phq9-llm-pilot)) found that LLMs
screening for depression misread *how often* a symptom happens when it is described in Urdu or Roman Urdu: for
example, *"taqreeban har roz"* ("nearly every day") was read as "most days". This pilot looks **inside** small open
models to see where that understanding is lost.

## Design
- **216 sentences**: 8 PHQ-9 symptoms x 3 frequencies x 3 languages (English, Urdu script, Roman Urdu) x **3 common
  phrasings per frequency**, because people phrase things differently (for example *kabhi kabhi / kuch din / chand
  din* for "a few days"; *zyada tar / aksar / zyada tar din* for "most days"; *taqreeban roz / taqreeban har din /
  taqreeban har roz* for "nearly every day"). Phrasings follow native-speaker guidance; the companion study's
  phrasing is one of the three. Symptom sentences were reviewed for naturalness but are not yet formally validated.
- **Task** (instruction always in English): *"How often does the patient have this problem? Answer with one word:
  sometimes, often, or daily."*
- **Logit lens**: at every layer, the hidden state of the last token is passed through the final norm and the output
  layer, giving the probability of each answer if the model "stopped there".
- **Calibration**: small models have strong answer habits (Qwen 2.5 1.5B almost never says "often"; BLOOMZ almost
  never says "daily"). Following *Calibrate Before Use* (Zhao et al., 2021), the option prior measured on
  content-free inputs ("N/A", "...", "[blank]") is divided out at every layer.
- **Models**: Qwen 2.5 1.5B Instruct and BLOOMZ 1.7B (BLOOM's pre-training data includes Urdu). Run locally with
  PyTorch + Hugging Face Transformers on one 8 GB GPU.

## Results (calibrated; chance = 0.33)
| Model | English | Urdu script | Roman Urdu |
|---|---|---|---|
| Qwen 2.5 1.5B | 0.93 | 0.64 | **0.36** |
| BLOOMZ 1.7B | 0.76 | 0.64 | **0.33** |

- **Roman Urdu is at chance for both models**, Urdu script is in between, English is best: the same ordering as the
  screening study, now seen inside the models.
- **"Nearly every day" in Roman Urdu is not understood**: 0.00-0.25 for all three phrasings (taqreeban roz / har din
  / har roz) in both models. This matches the screening study's main error.
- Everyday phrasings matter: e.g. BLOOMZ understands *کبھی کبھی* (kabhi kabhi, Urdu script) perfectly but none of
  the Roman Urdu versions of "a few days".
- Intermediate layers mostly decode to Latin-script (English-like) tokens for every input language, which is
  consistent with "latent English" (Wendler et al., ACL 2024) but, with these two small models, not specific to
  Urdu input.

Full tables: [`results/RESULTS.md`](results/RESULTS.md). Layer curves: `results/fig_lens.png`. Problems and decisions:
[`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

## Limitations
Two small models (1.5-1.7B); 8 symptom templates; one answer format; sentences reviewed but not formally
native-validated. This is a pilot that points to *where* to look (romanized frequency words, especially
"taqreeban"), not a general claim about all LLMs.

## Run
```
python lens_study.py selftest     # checks the stimuli, no model needed
python lens_study.py run          # logit lens on all stimuli (needs a CUDA GPU)
python lens_study.py calibrate    # content-free option prior per layer
python lens_study.py results      # tables + figure -> results/
```

## Disclosure
Code and documentation were written with the help of an AI assistant; the research design, checking and
decisions are the author's.

## License
MIT
