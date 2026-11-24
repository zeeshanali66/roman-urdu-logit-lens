# Research log: Where does Roman Urdu lose the symptom?

### 1. Why this study
The screening study showed *that* LLMs misread symptom frequency in Urdu and Roman Urdu. A logit lens lets us ask
*where* in the network the right answer appears, per language. Prior work (Wendler et al., ACL 2024; RomanLens,
2025) studied latent languages and romanization in general; this pilot ties it to a concrete clinical error.

### 2. One phrasing per frequency would be unrealistic
The first design used the screening study's fixed phrases ("kuch din", "zyada tar din", "taqreeban har roz"). A native
speaker pointed out that people more often say "kabhi kabhi", say "zyada tar" without "din", and say "taqreeban roz"
or "taqreeban har din"; plain "roz" means exactly every day. Since "everyone has their own way of saying things", we
used three common phrasings per frequency in every language, including the original one, and report accuracy per
phrasing. (This also flags a limitation of the screening study: its dialogue phrasing is somewhat textbook.)

### 3. Avoiding answer leakage
English phrasings must not contain the answer words ("sometimes", "often", "daily"). A self-test enforces this
(e.g. "Sometimes," and "Almost daily," were rejected).

### 4. Raw accuracy was measuring answer habits, not understanding
Uncalibrated, Qwen never answered "often" (0% on that level, even in English) and BLOOMZ almost never answered
"daily". We applied contextual calibration (Zhao et al., 2021): the option probabilities on content-free inputs are
divided out at each layer. After calibration English accuracy rose to 0.93 / 0.76 and the language gap became clear.
We report both raw and calibrated numbers.

### 5. Disk and environment
The system drive had only ~4 GB free, so PyTorch, Transformers and the models were installed on a separate drive
(about 9 GB), and models load offline from a local cache.

### 6. What we do NOT claim
The "latent English" signal (Latin-script tokens in middle layers) appears for all input languages in these models,
so it does not by itself explain the Urdu gap. Settling depth differences between languages were small and noisy.

## Next steps
Larger models (7-8B in 4-bit), patching the frequency word's representation from English into Roman Urdu
(activation patching) to test whether the failure is in reading "taqreeban" or later, and a formally validated
stimulus set.
