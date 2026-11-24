# Where does Roman Urdu lose the symptom? (logit lens)

432 stimuli x layers; models: Qwen/Qwen2.5-1.5B-Instruct, bigscience/bloomz-1b7.

## Final-layer accuracy (choosing sometimes / often / daily)

```
lang                          en    ru    ur
model                                       
Qwen/Qwen2.5-1.5B-Instruct  0.67  0.44  0.33
bigscience/bloomz-1b7       0.68  0.32  0.61
```

## Accuracy by frequency level

```
lang                                en    ru    ur
model                      level                  
Qwen/Qwen2.5-1.5B-Instruct 1      1.00  0.96  1.00
                           2      0.00  0.00  0.00
                           3      1.00  0.38  0.00
bigscience/bloomz-1b7      1      1.00  0.12  0.92
                           2      0.92  0.83  0.88
                           3      0.12  0.00  0.04
```

## Accuracy per phrasing (each row = one way of saying the frequency; * = the screening study's phrasing)

```
model                              Qwen/Qwen2.5-1.5B-Instruct  bigscience/bloomz-1b7
level lang phrase                                                                   
1     en   Every now and then,                           1.00                   1.00
           On a few days,                                1.00                   1.00
           Some days,                                    1.00                   1.00
      ru   Chand din                                     1.00                   0.12
           Kabhi kabhi                                   1.00                   0.12
           Kuch din*                                     0.88                   0.12
      ur   چند دن*                                       1.00                   0.88
           کبھی کبھی                                     1.00                   1.00
           کچھ دن                                        1.00                   0.88
2     en   Frequently,                                   0.00                   0.88
           Most days,                                    0.00                   0.88
           On most days,                                 0.00                   1.00
      ru   Aksar                                         0.00                   0.88
           Zyada tar                                     0.00                   0.75
           Zyada tar din*                                0.00                   0.88
      ur   اکثر                                          0.00                   0.88
           زیادہ تر                                      0.00                   0.75
           زیادہ تر دن*                                  0.00                   1.00
3     en   Almost every day,                             1.00                   0.00
           Nearly every day,                             1.00                   0.25
           Practically every day,                        1.00                   0.12
      ru   Taqreeban har din                             0.88                   0.00
           Taqreeban har roz*                            0.25                   0.00
           Taqreeban roz                                 0.00                   0.00
      ur   تقریباً روز                                   0.00                   0.00
           تقریباً ہر دن                                 0.00                   0.12
           تقریباً ہر روز*                               0.00                   0.00
```

## CALIBRATED final-layer accuracy (option prior from content-free inputs divided out)

Small models have strong answer habits (e.g. one never says 'often'); calibration removes them.

```
lang                          en    ru    ur
model                                       
Qwen/Qwen2.5-1.5B-Instruct  0.93  0.36  0.64
bigscience/bloomz-1b7       0.76  0.33  0.64
```

## Calibrated accuracy by frequency level

```
lang                                en    ru    ur
model                      level                  
Qwen/Qwen2.5-1.5B-Instruct 1      1.00  0.83  1.00
                           2      0.92  0.17  0.92
                           3      0.88  0.08  0.00
bigscience/bloomz-1b7      1      0.71  0.00  0.54
                           2      0.58  1.00  0.71
                           3      1.00  0.00  0.67
```

## Calibrated accuracy per phrasing (* = screening study's phrasing)

```
model                              Qwen/Qwen2.5-1.5B-Instruct  bigscience/bloomz-1b7
level lang phrase                                                                   
1     en   Every now and then,                           1.00                   0.75
           On a few days,                                1.00                   0.38
           Some days,                                    1.00                   1.00
      ru   Chand din                                     0.75                   0.00
           Kabhi kabhi                                   1.00                   0.00
           Kuch din*                                     0.75                   0.00
      ur   چند دن*                                       1.00                   0.25
           کبھی کبھی                                     1.00                   1.00
           کچھ دن                                        1.00                   0.38
2     en   Frequently,                                   1.00                   1.00
           Most days,                                    0.88                   0.50
           On most days,                                 0.88                   0.25
      ru   Aksar                                         0.00                   1.00
           Zyada tar                                     0.25                   1.00
           Zyada tar din*                                0.25                   1.00
      ur   اکثر                                          1.00                   1.00
           زیادہ تر                                      0.75                   1.00
           زیادہ تر دن*                                  1.00                   0.12
3     en   Almost every day,                             0.62                   1.00
           Nearly every day,                             1.00                   1.00
           Practically every day,                        1.00                   1.00
      ru   Taqreeban har din                             0.25                   0.00
           Taqreeban har roz*                            0.00                   0.00
           Taqreeban roz                                 0.00                   0.00
      ur   تقریباً روز                                   0.00                   0.00
           تقریباً ہر دن                                 0.00                   1.00
           تقریباً ہر روز*                               0.00                   1.00
```

## Where the right answer settles (relative depth 0-1; lower = earlier; NaN = never settles)

```
                                 mean  count
model                      lang             
Qwen/Qwen2.5-1.5B-Instruct en    0.83     48
                           ru    0.77     32
                           ur    0.68     24
bigscience/bloomz-1b7      en    0.92     51
                           ru    0.87     24
                           ur    0.89     47
```

## Share of layers whose top decoded token is Latin-script (English-like), by input language

For Urdu-script input, a high middle-layer share suggests the model routes through English-like tokens.

```
lang                                 en    ru    ur
model                      band                    
Qwen/Qwen2.5-1.5B-Instruct early   0.78  0.69  0.78
                           middle  0.47  0.50  0.43
                           late    0.90  0.90  0.90
bigscience/bloomz-1b7      early   0.00  0.00  0.00
                           middle  0.86  0.85  0.81
                           late    0.90  0.95  0.91
```
