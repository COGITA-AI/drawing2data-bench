# Drawing2Data — Fable 5.1 / Opus 5.5 / GPT-6 Astra vs protokół `benchmark_15`

**N=15**, COCO test AutoDraft, ten sam harness co Gemini / GPT-5.6. Trzy kompletne runy (15/15). Ranking obowiązuje **przy tym kontrakcie promptu**, nie jako porównanie prompt-neutralne.

---

## Setup

- Matching: Hungarian, IoU > 0,5.
- Feature F1: exact match na `text` + `category`; wariant z constraitem IoU i bez (unstructured).
- CER: znormalizowany edit distance; unstructured vs padded + lokalizacja.
- Prompt wspólny: JSON `FeatureList`; bbox z modelu `[ymin, xmin, ymax, xmax]` w `[0, 1000]`, potem swap osi i skala do pikseli obrazu źródłowego.
- Temperature 0, Instructor JSON.
- Konwencja współrzędnych jest zbliżona do tego, na czym trenowano Gemini — możliwy bias protokołu względem Claude / GPT.

Poza tym porównaniem: RF-DETR (ramki, puste `text`).

---

## Wyniki

| Metryka | Fable 5.1 | Opus 5.5 | Astra | Gemini 3.5 Flash* |
|---|---:|---:|---:|---:|
| Bbox F1 | 0,712 | 0,691 | **0,873** | 0,766 |
| Bbox precision | 0,708 | 0,690 | **0,854** | 0,776 |
| Bbox recall | 0,715 | 0,693 | **0,892** | 0,757 |
| Feature F1 EM, localization | 0,632 | 0,628 | **0,832** | 0,680 |
| Feature F1 EM, unstructured | 0,882 | **0,890** | 0,878 | 0,862 |
| CER unstructured | 0,937 | 0,938 | 0,938 | 0,940 |
| CER localization padded | 0,517 | 0,504 | **0,760** | 0,586 |
| USD / sample | 0,246 | **0,014** | 0,038 | 0,069 |

\*Kotwica harnessu z dokumentacji `benchmark_15` (inna sesja, ten sam typ protokołu).

Wariancja między modelami jest w **lokalizacji i joint loc+EM**, nie w unstructured EM/CER (~0,88–0,94). Astra: bbox F1 0,87 / F1 loc 0,83. Opus ≈ Fable (0,69 / 0,63 vs 0,71 / 0,63) przy 0,014 vs 0,246 USD/sample.

---

## Czego ten pomiar nie twierdzi

- Brak CI przy N=15.
- Domain = synthetic AutoDraft; transfer na skany niezmierzony.
- Brak ablations promptu / innego układu bbox.

---

## Koszt ewaluacji 15×3

~**4,5 USD** (Fable ~3,7, Opus ~0,2, Astra ~0,6). Inferencja: ~14 / 7 / 12 min.

---

## Astra vs Opus

Fable z dalszego porównania: brak zysku nad Opusem, najwyższy koszt.

| | Opus 5.5 | Astra |
|---|---:|---:|
| Bbox F1 | 0,691 | **0,873** |
| F1 EM loc (decyzyjna) | 0,628 | **0,832** |
| F1 EM unstructured | 0,890 | 0,878 |
| USD / sample | **0,014** | 0,038 |

Opus nie przegrywa OCR-u. Przegrywa **złożenie odczytu z geometrią ramki**.

- Quality path → **Astra**.
- Cost path przy podobnym unstructured → **Opus**.

---

## Rekomendacja

- Quality: **GPT-6 Astra**.
- Cost: **Claude Opus 5.5**; Fable poza kolejką.
- Nie ship na tej próbie.
- Następny eksperyment: **Astra vs Opus**, 10–20 real drawings, ten sam matching i prompt; ewentualnie bootstrap na F1 loc.

---

## Artefakty

`benchmarks/results/benchmark_fable_15_v2`, `benchmark_opus_15`, `benchmark_astra_15`  
Adaptery: `claude_fable_5_1`, `claude_opus_5_5`, `openai_gpt6_astra`  
Gałąź: `benchmarks-fable-opus-astra`
