# Audio samples

Same Piper voice (Mana, MIT); only the front-end differs. Sentences are SentenceBench homograph items.
Contested words: where the gold reading and espeak's reading differ; **LLR > 0** means the audio supports the gold reading (phoneme-recognizer forced choice).

GitHub does not play audio inline — click a file to open its player, or download.

### <span dir="rtl">من قدر تو را می‌دانم</span>

| System | Audio | Phonemes sent to the voice | Contested words (gold vs espeak, LLR) |
|---|---|---|---|
| espeak | [sb000_espeak.wav](sb000_espeak.wav) | `man q1ˈadr tˈo rɑ midˈɑːnam` | قدر: `q/dre1` vs `q/dr` (-3.6) |
| ours | [sb000_tagger.wav](sb000_tagger.wav) | `man q1ˈadre tˈo rɑ midˈɑːnam` | قدر: `q/dre1` vs `q/dr` (+5.2) |
| ours, k=1 | [sb000_tagger_k1.wav](sb000_tagger_k1.wav) | `man q1ˈadre tˈo rɑ midˈɑːnam` | قدر: `q/dre1` vs `q/dr` (+5.2) |

### <span dir="rtl">از قضای الهی به قدر الهی پناه می‌برم</span>

| System | Audio | Phonemes sent to the voice | Contested words (gold vs espeak, LLR) |
|---|---|---|---|
| espeak | [sb001_espeak.wav](sb001_espeak.wav) | `az q1azˈɑje elˈɑhi beː q1ˈadr elˈɑhi panˈɑh mibˈeram` | قضای: `q/zay` vs `q/zaye` (-4.1); قدر: `q/d/re1` vs `q/dr` (-11.0); می‌برم: `mib/r/m` vs `miber/m` (+2.5) |
| ours | [sb001_tagger.wav](sb001_tagger.wav) | `az q1azˈɑje alˈe beː q1ˈadr alˈeje panˈɑh mibˈaram` | قضای: `q/zay` vs `q/zaye` (+2.5); قدر: `q/d/re1` vs `q/dr` (-11.5); می‌برم: `mib/r/m` vs `miber/m` (+2.8) |
| ours, k=1 | [sb001_tagger_k1.wav](sb001_tagger_k1.wav) | `az q1azˈɑje alˈe beː q1ˈadr alˈeje panˈɑh mibˈaram` | قضای: `q/zay` vs `q/zaye` (+2.5); قدر: `q/d/re1` vs `q/dr` (-11.5); می‌برم: `mib/r/m` vs `miber/m` (+2.8) |

### <span dir="rtl">به دست و صورتم کرم زدم</span>

| System | Audio | Phonemes sent to the voice | Contested words (gold vs espeak, LLR) |
|---|---|---|---|
| espeak | [sb002_espeak.wav](sb002_espeak.wav) | `beː dˈast vˈa surˈatam kˈaram zˈadam` | و: `o` vs `v/` (+9.8); کرم: `kerem` vs `k/r/m` (-5.9) |
| ours | [sb002_tagger.wav](sb002_tagger.wav) | `beː dˈast ˈo surˈatam kˈaram zˈadam` | و: `o` vs `v/` (+10.4); کرم: `kerem` vs `k/r/m` (-4.5) |
| ours, k=1 | [sb002_tagger_k1.wav](sb002_tagger_k1.wav) | `beː dˈast ˈo surˈatam kˈaram zˈadam` | و: `o` vs `v/` (+10.4); کرم: `kerem` vs `k/r/m` (-4.5) |

### <span dir="rtl">از بزرگواری و کرم خداوند به ما نعمت زیادی رسیده</span>

| System | Audio | Phonemes sent to the voice | Contested words (gold vs espeak, LLR) |
|---|---|---|---|
| espeak | [sb003_espeak.wav](sb003_espeak.wav) | `az bozorɡvˈɑri vˈa kˈaram xodˈɑvand beː mɑ nˈeʔmat ziˈɑdi rˈesidˌeː` | کرم: `k/r/me1` vs `k/r/m` (-7.5); ما: `m/` vs `ma` (-2.9); نعمت: `ne@m/te1` vs `ne@m/t` (-4.6) |
| ours | [sb003_tagger.wav](sb003_tagger.wav) | `az bozorɡvˈɑri vˈa kˈarame xodˈɑvand beː mɑ nˈeʔmate zijˈɑdi rˈesidˌeː` | کرم: `k/r/me1` vs `k/r/m` (+5.5); ما: `m/` vs `ma` (-2.9); نعمت: `ne@m/te1` vs `ne@m/t` (+4.1) |
| ours, k=1 | [sb003_tagger_k1.wav](sb003_tagger_k1.wav) | `az bozorɡvˈɑri vˈa kˈarame xodˈɑvand beː mɑ nˈeʔmate zijˈɑdi rˈesidˌeː` | کرم: `k/r/me1` vs `k/r/m` (+5.5); ما: `m/` vs `ma` (-2.9); نعمت: `ne@m/te1` vs `ne@m/t` (+4.1) |

### <span dir="rtl">کرم خاکی باعث می‌شود گل‌ها بهتر رشد کنند</span>

| System | Audio | Phonemes sent to the voice | Contested words (gold vs espeak, LLR) |
|---|---|---|---|
| espeak | [sb004_espeak.wav](sb004_espeak.wav) | `kˈaram xˈɑki bˈɑʔes miʃavˈad ɡˈolhɑ bˈehtar rˈoʃd kˈonand` | کرم: `kerme1` vs `k/r/m` (-14.1) |
| ours | [sb004_tagger.wav](sb004_tagger.wav) | `kˈerme xˈɑki bˈɑʔes miʃavˈad ɡˈolhɑ bˈehtar rˈoʃd kˈonand` | کرم: `kerme1` vs `k/r/m` (+12.1) |
| ours, k=1 | [sb004_tagger_k1.wav](sb004_tagger_k1.wav) | `kˈerme xˈɑki bˈɑʔes miʃavˈad ɡˈolhɑ bˈehtare rˈoʃd kˈonand` | کرم: `kerme1` vs `k/r/m` (+11.1) |

### <span dir="rtl">نشست سران سیاسی کشورها به خوبی پیش رفت</span>

| System | Audio | Phonemes sent to the voice | Contested words (gold vs espeak, LLR) |
|---|---|---|---|
| espeak | [sb005_espeak.wav](sb005_espeak.wav) | `neʃˈast sarˈɑn siˈɑsi keʃvˈarhɑ beː xˈuːbi pˈiʃ rˈaft` | نشست: `ne$/ste1` vs `ne$/st` (-9.6); سران: `s/rane1` vs `s/ran` (-7.0); سیاسی: `siyasiye1` vs `siasi` (-5.2) |
| ours | [sb005_tagger.wav](sb005_tagger.wav) | `neʃˈast sarˈɑne sijˈɑsije keʃvˈarhɑ beː xˈuːbi pˈiʃ rˈaft` | نشست: `ne$/ste1` vs `ne$/st` (-9.2); سران: `s/rane1` vs `s/ran` (+3.7); سیاسی: `siyasiye1` vs `siasi` (-3.0) |
| ours, k=1 | [sb005_tagger_k1.wav](sb005_tagger_k1.wav) | `neʃˈast sarˈɑne sijˈɑsije keʃvˈarhɑ beː xˈuːbi pˈiʃ rˈaft` | نشست: `ne$/ste1` vs `ne$/st` (-8.8); سران: `s/rane1` vs `s/ran` (+3.5); سیاسی: `siyasiye1` vs `siasi` (-4.1) |

### <span dir="rtl">او میوه را نشست ولی آن را خورد</span>

| System | Audio | Phonemes sent to the voice | Contested words (gold vs espeak, LLR) |
|---|---|---|---|
| espeak | [sb006_espeak.wav](sb006_espeak.wav) | `ˈuː mˈiveː rɑ neʃˈast vˈali ɑn rɑ xˈord` | را: `r/` vs `ra` (-2.1); نشست: `n/$ost` vs `ne$/st` (-10.7); را: `r/` vs `ra` (-2.6) |
| ours | [sb006_tagger.wav](sb006_tagger.wav) | `ˈuː mˈiveː rɑ naʃˈost vˈali ɑn rɑ xˈord` | را: `r/` vs `ra` (-2.8); نشست: `n/$ost` vs `ne$/st` (-5.0); را: `r/` vs `ra` (-2.6) |
| ours, k=1 | [sb006_tagger_k1.wav](sb006_tagger_k1.wav) | `ˈuː mˈiveː rɑ naʃˈost vˈali ɑn rɑ xˈord` | را: `r/` vs `ra` (-2.2); نشست: `n/$ost` vs `ne$/st` (-4.5); را: `r/` vs `ra` (-2.4) |

### <span dir="rtl">دلم هری ریخت</span>

| System | Audio | Phonemes sent to the voice | Contested words (gold vs espeak, LLR) |
|---|---|---|---|
| espeak | [sb007_espeak.wav](sb007_espeak.wav) | `dˈelam hˌari rˈixt` | هری: `horri` vs `h/ri` (-11.0) |
| ours | [sb007_tagger.wav](sb007_tagger.wav) | `dˈelam hˌorri rˈixt` | هری: `horri` vs `h/ri` (-3.2) |
| ours, k=1 | [sb007_tagger_k1.wav](sb007_tagger_k1.wav) | `dˈelam hˌorri rˈixt` | هری: `horri` vs `h/ri` (+2.3) |
