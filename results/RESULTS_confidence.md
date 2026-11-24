# Part 2: does the model know when it misreads Roman Urdu? (calibrated final layer)

AUROC: how well a signal separates right from wrong answers (0.5 = no signal, 1 = perfect). ECE: gap between stated confidence and actual accuracy (0 = honest).

| model                 | language    |   accuracy |   conf if right |   conf if wrong |   AUROC conf |   AUROC margin |   AUROC stability |   ECE |
|:----------------------|:------------|-----------:|----------------:|----------------:|-------------:|---------------:|------------------:|------:|
| Qwen2.5-1.5B-Instruct | English     |       0.93 |            0.75 |            0.59 |         0.86 |           0.88 |              0.93 |  0.19 |
| Qwen2.5-1.5B-Instruct | Roman Urdu  |       0.36 |            0.62 |            0.51 |         0.64 |           0.69 |              0.65 |  0.22 |
| Qwen2.5-1.5B-Instruct | Urdu script |       0.64 |            0.7  |            0.61 |         0.7  |           0.72 |              0.51 |  0.08 |
| bloomz-1b7            | English     |       0.76 |            0.6  |            0.47 |         0.77 |           0.76 |              0.75 |  0.19 |
| bloomz-1b7            | Roman Urdu  |       0.33 |            0.48 |            0.48 |         0.45 |           0.46 |              0.56 |  0.15 |
| bloomz-1b7            | Urdu script |       0.64 |            0.61 |            0.48 |         0.82 |           0.78 |              0.63 |  0.16 |

## One rule for everyone: send the least-confident 30% to a human

Per language: share of that language's answers deferred, accuracy of the answers the model keeps, and the share of that language's errors that slip through (kept and wrong).

| model                 | language    |   deferred |   accuracy before |   accuracy of kept |   errors slipping through |
|:----------------------|:------------|-----------:|------------------:|-------------------:|--------------------------:|
| Qwen2.5-1.5B-Instruct | English     |       0.07 |              0.93 |               0.94 |                      0.8  |
| Qwen2.5-1.5B-Instruct | Urdu script |       0.22 |              0.64 |               0.7  |                      0.65 |
| Qwen2.5-1.5B-Instruct | Roman Urdu  |       0.61 |              0.36 |               0.5  |                      0.3  |
| bloomz-1b7            | English     |       0.33 |              0.76 |               0.85 |                      0.41 |
| bloomz-1b7            | Urdu script |       0.26 |              0.64 |               0.77 |                      0.46 |
| bloomz-1b7            | Roman Urdu  |       0.31 |              0.33 |               0.3  |                      0.73 |

## 'Nearly every day' in Roman Urdu (taqreeban ...)

Accuracy 0.04; mean confidence 0.48 (English 'nearly every day': accuracy 0.94, confidence 0.69).

