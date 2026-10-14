# Notatka dla kierownika: benchmark Fable / Opus 5.5 / Astra

**Projekt:** Drawing2Data-Bench  
**Zadanie:** pomiar Claude Fable 5.1, Claude Opus 5.5 i GPT-6 Astra na odczycie rysunku technicznego (lokalizacja + tekst), w tym samym protokole co istniejący `benchmark_15`  
**Status:** zakończone — trzy kompletne runy po 15 próbek  
**Dane:** 15 syntetycznych rysunków (split test)

---

## Co się liczy

Na tym zadaniu **nie wystarczy, że model „w ogóle odczyta napis”**. Wszystkie trzy modele robią to podobnie dobrze (~0,88–0,94, gdy nie wymagamy trafienia w miejsce).

**Liczy się to, z czego korzysta proces biznesowy:**

1. **Czy znalazł adnotację we właściwym miejscu** (bbox F1) — bez tego nie da się wiarygodnie wyciąć regionu do dalszej obróbki ani zestawić wyniku z arkuszem.
2. **Czy odczyt i miejsce są zgodne naraz** (feature F1 z lokalizacją) — to jest metryka „czy można temu zaufać na rysunku”, nie tylko w izolowanym OCR.
3. **Koszt przy porównywalnej jakości** — przy Fable ≈ Opus nie ma sensu płacić za droższy model.

Pozostałe wskaźniki (sam tekst bez miejsca, CER) są **uzupełnieniem**, nie podstawą decyzji: pokazują, że problemem nie jest literowanie, tylko **celowanie w ramkę**.

**Wniosek z trzech ukończonych pomiarów:** na syntetycznych arkuszach **Astra wygrywa to, co się liczy (miejsce + odczyt razem)**. **Opus** jest wyborem, gdy ważniejszy jest koszt przy jakości zbliżonej do Fable. **Fable** nie wnosi przewagi i jest najdroższy.

To nie jest jeszcze decyzja o wdrożeniu na rysunkach z biura (N=15, dane wygenerowane).

---

## 1. Zakres

- Trzy modele w istniejącym runnerze OpenRouter (bez trenowania własnych wag).
- Te same metryki co `benchmark_15`: bbox F1/P/R, feature F1 exact-match (z lokalizacją i bez), CER, koszt.
- Porównanie orientacyjne z udokumentowanym wynikiem Gemini 3.5 Flash (ten sam typ protokołu, wcześniejsza sesja).

Poza zakresem: trening RF-DETR, skany z produkcji, zmiana promptu pod jednego dostawcę.

---

## 2. Wyniki (15/15, trzy ukończone runy)

| Metryka (wyżej = lepiej, koszt niżej = taniej) | Fable 5.1 | Opus 5.5 | Astra | Gemini 3.5 Flash* |
|---|---:|---:|---:|---:|
| Bbox F1 | 0,712 | 0,691 | **0,873** | 0,766 |
| Bbox precision | 0,708 | 0,690 | **0,854** | 0,776 |
| Bbox recall | 0,715 | 0,693 | **0,892** | 0,757 |
| Feature F1, exact, z lokalizacją | 0,632 | 0,628 | **0,832** | 0,680 |
| Feature F1, exact, bez lokalizacji | 0,882 | **0,890** | 0,878 | 0,862 |
| CER unstructured | 0,937 | 0,938 | **0,938** | 0,940 |
| CER localization padded | 0,517 | 0,504 | **0,760** | 0,586 |
| Średni koszt / rysunek (USD) | 0,246 | **0,014** | 0,038 | 0,069 |

\*Gemini: `benchmark_15` z dokumentacji repozytorium.

Astra przebija Gemini 3.5 na lokalizacji i na metryce tekst+miejsce. Fable i Opus są w okolicy Gemini albo nieco pod, przy czym Opus jest wielokrotnie tańszy od Fable.

---

## 3. Koszt i czas trzech ukończonych pomiarów

Średni koszt z metryki × 15 rysunków:

| Model | 15 rysunków |
|---|---:|
| Fable 5.1 | ~3,7 USD |
| Opus 5.5 | ~0,2 USD |
| Astra | ~0,6 USD |
| **Razem** | **~4,5 USD** |

Czas inferencji: Fable ~14 min, Opus ~7 min, Astra ~12 min.

Kwoty to użycie API z runnera, nie kopia faktury.

---

## 4. Rekomendacja

| Pytanie | Rekomendacja |
|---|---|
| Co jest metryką decyzyjną? | **Feature F1 z lokalizacją**, potem bbox F1, potem koszt |
| Model pod jakość odczytu | **GPT-6 Astra** |
| Model pod tani, powtarzalny odczyt | **Claude Opus 5.5** |
| Fable 5.1 na tym zadaniu | **Nie kontynuować** — brak zysku jakości, najwyższy koszt |
| Wdrożenie na podstawie tego pomiaru | **Nie** — za mały N, brak rysunków rzeczywistych |

**Proponowany następny krok:** 10–20 rzeczywistych rysunków (skan/PDF), head-to-head **Astra vs Opus**, ten sam protokół. Fable poza kolejką, dopóki nie zmieni się zadanie albo cena.

---

## 5. Ograniczenia

- N = 15, dane syntetyczne, nie skany produkcyjne.
- Jeden wspólny prompt (w tym konwencja współrzędnych); ranking obowiązuje **przy tym protokole**.
- Lokalny detektor projektu nie odczytuje tekstu — nie był w tej trójce.

---

## 6. Artefakty

- Adaptery: `benchmarks/models/claude_fable_5_1`, `claude_opus_5_5`, `openai_gpt6_astra`
- Wyniki: `benchmarks/results/benchmark_fable_15_v2`, `benchmark_opus_15`, `benchmark_astra_15`
- Wersja opisowa: `raport-fable-opus-astra.md`

Gałąź: `benchmarks-fable-opus-astra`.

---

## 7. Astra vs Opus

Fable odpada z dalszego porównania. Decyzja sprowadza się do **GPT-6 Astra** i **Claude Opus 5.5** na tym samym zestawie 15 rysunków.

| | Opus 5.5 | Astra | Różnica |
|---|---:|---:|---|
| Bbox F1 (miejsce) | 0,691 | **0,873** | Astra wyraźnie lepiej celuje w ramkę |
| Feature F1 z lokalizacją (tekst **i** miejsce) | 0,628 | **0,832** | to jest metryka decyzyjna — Astra wyżej |
| Feature F1 bez lokalizacji (sam tekst) | **0,890** | 0,878 | remis na praktykę — oba dobrze czytają napis |
| CER unstructured | 0,938 | 0,938 | bez różnicy |
| Średni koszt / rysunek | **0,014 USD** | 0,038 USD | Opus ~2,6× tańszy |
| Koszt 15 rysunków | **~0,2 USD** | ~0,6 USD | |

**Co z tego wynika.** Opus nie „przegrywa czytania”. Przegrywa **złożenie odczytu z miejscem na arkuszu**. Astra rzadziej wskazuje zły region albo dobry tekst w złej ramce. Przy masowym wyciąganiu danych z rysunku to jest ta różnica, która potem wchodzi do tabeli i do błędu procesu.

**Kiedy które.**

- Potrzebna **najwyższa wiarygodność** na arkuszu (RFQ, kontrola, mniej ręcznych poprawek) → **Astra**.
- Potrzebny **niski koszt przy dużym wolumenie**, a ktoś i tak sprawdzi wynik → **Opus**.
- Nie mieszać ich „po równo bez kryterium”: najpierw ustalić, czy bolą nas pomyłki miejsca, czy rachunek API.

**Rekomendacja.** Na dziś, na danych syntetycznych: **domyślnie Astra**, Opus jako wariant kosztowy. Nie wdrażać jeszcze na produkcji. Następny pomiar, który ma sens: **wyłącznie Astra vs Opus na 10–20 rzeczywistych rysunkach** — ten test rozstrzygnie, czy przewaga Astry na ramkach przenosi się na skany z biura, czy zostaje tylko na wygenerowanych arkuszach.
