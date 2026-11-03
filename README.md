# Mehrere Modelle: das billige Modell darf falsch sein – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-mehrere-modelle-demo.streamlit.app/)** -->

Viertes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [kontrollvariaten-demo](https://github.com/sebastian-hanisch/kontrollvariaten-demo) (Stück 3, Größe mit **bekanntem** Erwartungswert) und [gekoppelte-ziehungen-demo](https://github.com/sebastian-hanisch/gekoppelte-ziehungen-demo) (Stück 2, gemeinsame Zufallszahlen). Hier hebt das Stück genau eine Annahme auf: Der Erwartungswert der Hilfsgröße ist **unbekannt**, die Hilfsgröße ist ein **billiges Modell**, das das Ziel nur grob trifft und verzerrt sein darf. Vehikel ist wie in Stück 1 bis 3 die Zuverlässigkeit eines Gitternetzes, jetzt mit dem strengeren Ziel „**brückenlos** zusammenhängend“, und die Wahrheit ist exakt aufgezählt.

## Warum dieses Problem

Ein teures Verfahren lässt sich oft durch ein billiges Modell ergänzen. Hier: Das Ziel X2 fragt, ob das Netz zusammenhängt **und** es auch nach Ausfall einer beliebigen weiteren Kante bleibt. Das naive Verfahren prüft den Zusammenhang und entfernt dann jede haltende Kante einzeln. Billige Modelle sind „jeder Knoten hat mindestens zwei haltende Kanten“ (D2), „die Eckknoten haben beide Kanten“ (E) und der Zusammenhang (C), das Ziel der vorigen Stücke. Man rechnet das billige Modell auf vielen Eingaben, das teure auf wenigen, und korrigiert um die Differenz auf den gemeinsamen Eingaben. Die Kernfrage: **Wann lohnt das, wie teilt man das Budget auf, und was passiert, wenn das billige Modell falsch ist?**

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat:

- **Das Ziel wurde gewechselt.** Ein erster Ansatz mit „zusammenhängend“ als Ziel und auf wenige Durchläufe gekürzter Ausbreitung als billigem Modell war **nicht billiger** als die volle Ausbreitung; deshalb ist das Ziel jetzt „brückenlos“, und der Zusammenhang ist das billige Modell.
- **„Der Gewinn hängt an der Korrelation“ stimmt nur halb.** Er hängt an der Korrelation **und** am Kostenverhältnis; ein hoch korreliertes, aber nicht billiges Modell schadet.
- **Die naheliegende Wahl schadet.** Der Zusammenhang aus den Stücken 1 bis 3 ist als billiges Modell für „brückenlos“ schlechter als keines.
- **Eine Bedingung für mehrere Modelle, aus dem Gedächtnis hingeschrieben, war falsch** und wurde durch eine eigene Herleitung und eine unabhängige Rastersuche ersetzt.
- **Einzelmessungen an wenigen teuren Ziehungen rauschen stark** (Kurtosis um 20); die Messzellen mit fünf teuren Ziehungen sind keine belastbaren Belege, die exakte Formel schon.

## Modell

- **Netz und Ziel:** Gitter 2×4, 3×3, 2×5, 3×4 (bis 17 Kanten, alle Kantenmengen live aufgezählt), jede Kante hält unabhängig mit Wahrscheinlichkeit p. Ziel X2 = 1, wenn das Netz zusammenhängt und nach Entfernen jeder einzelnen Kante noch zusammenhängt.
- **Billige Modelle** (alle notwendig für X2, irren nur nach oben): **C** Zusammenhang; **D2** jeder Knoten hat Grad ≥ 2; **E** die Knoten vom Grad 2 (Ecken) haben beide Kanten.
- **Kosten in Kantenzugriffen** einer Tiefensuche (keine Rechenzeit): eine Zusammenhangsprüfung kostet die Summe der Knotengrade der Komponente von Knoten 0; das naive Ziel prüft den Zusammenhang und danach jede haltende Kante entfernt, bis zum ersten Fehlschlag; D2 liest jede Kante einmal, E die Eckkanten. Alternativ „Brückensuche in einem Lauf“ (Tarjan): dieselben Werte, Kosten einer Prüfung.
- **Schätzer** (ein billiges Modell f₁, Ziel f₀, N₁ ≥ N₀): s = Mittel(f₀ auf N₀) + α (Mittel(f₁ auf N₁) − Mittel(f₁ auf denselben N₀)), α = ρ σ₀/σ₁. Für **jedes** α und jedes (auch verzerrte) f₁ erwartungstreu.
- **Varianzfaktor** gegen den einfachen Mittelwert gleichen Budgets bei bester Aufteilung: (√(1−ρ²) + ρ√r)² mit r = Kosten billig / Kosten teuer; kleiner als 1 genau für r < r\* = ((1−√(1−ρ²))/ρ)². Mehrere Modelle: (Σ √(wᵢ(ρᵢ²−ρᵢ₊₁²)))² / w₀, anwendbar, wenn ρ und Kosten fallen und die Aufteilung nicht abnimmt. Mathematik im 📐-Expander der App.

## Methodik

Zufallszahlen aus festen SplitMix64-Strömen (nie ein einzelner Seed). Wahrheit, Kovarianzen und Kosten aus der vollständigen Aufzählung. Je Experiment 150 bis 400 Wiederholungen mit festen Strömen; Varianzverhältnisse rauschen um 8 bis 10 %, bei wenigen teuren Ziehungen deutlich mehr.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Standardfall: Gitter 3×4, p = 0,9, Budget 100 teure Auswertungen.

| Frage | Befund |
|---|---|
| Wahrheit und Kosten | E X2 = **0,4026**; ein teures Ziel kostet im Mittel **361,2** Kantenzugriffe, D2 **17**, E **8**, C **33,5** |
| Wie gut ist D2 als billiges Modell? | ρ = **0,982**, r = **0,047**, Faktor **0,163** (die Varianz sinkt auf rund 16 %); Schwelle r\* = **0,679**. Mit bekanntem Mittelwert wäre es 1 − ρ² = 0,037: der unbekannte Mittelwert kostet den Faktor 4,5 |
| Und der Zusammenhang C? | ρ = **0,199**, Faktor **1,083**: schlechter als gar kein billiges Modell, bei jedem p von 0,5 bis 0,99 |
| Wie hängt D2 an p? (exakt) | Faktor bei p = 0,5 / 0,7 / 0,8 / 0,9 / 0,95 / 0,99: **1,766** / **0,591** / **0,321** / **0,163** / **0,104** / **0,0535**. Bei p = 0,5 bricht die Prüfung des Ziels meist früh ab (mittlere Kosten 17,8), D2 ist nicht billiger und schadet |
| Lohnen zwei billige Modelle? | p = 0,9: D2 + E **0,152** gegen D2 allein 0,163 und E allein 0,221. p = 0,99: E allein (**0,0445**) schlägt D2 (0,0535); beide zusammen **0,0360**. p = 0,7: D2 allein **0,591** schlägt D2 + E (**0,698**): mehr Modelle schaden. p = 0,5: keine Kette lohnt |
| Wie stark hängt der Gewinn am teuren Verfahren? | Mit der Brückensuche in einem Lauf kostet das Ziel 33,5 statt 361,2: r(D2) = **0,508**, Faktor **0,794** statt 0,163; bestes Modell dann E allein (**0,626**) |
| Aufteilung N₁/N₀ (Formel, Budget 100, D2) | 1: **1,053** (nichts gewonnen), 3: 0,411, 10: 0,198, **24: 0,167** (Optimum 23,7), 60: 0,202, 150: 0,358, 400 (nur 5 teure Ziehungen): 0,778 |
| Kostenverhältnis r als Annahme (ρ = 0,982) | Faktor bei r = 0,01 / 0,05 / 0,1 / 0,25 / 0,5 / 0,75 / 1,0: **0,084** / **0,169** / **0,252** / **0,465** / **0,783** / **1,084** / **1,375**; über r\* = 0,679 ist der Schätzer schlechter als der einfache Mittelwert |
| Gemessen gegen Formel | Faktor 0,19 (exaktes α) und 0,18 (geschätztes α) im Standardfall gegen 0,163; über Budgets von 30 bis 1000 zwischen 0,15 und 0,22 |
| Das billige Modell ist verzerrt | E D2 − E X2 = **+0,0090**. Nur mit ihm gerechnet deckt sein 95-%-Intervall bei Budget 100 / 1000 / 5000 die Wahrheit in **88 / 51 / 1 %** der Wiederholungen; der Mehrmodell-Schätzer bleibt erwartungstreu (Fehler −0,0027 / −0,0001 / −0,0002) und deckt in **97 / 98 / 95 %** |
| Parameter schätzen | α aus den geteilten Eingaben kostet nichts (Faktor **0,153** gegen 0,153 mit exakten Parametern). Die Aufteilung aus einer Vorstudie ist heikel: ρ̂ liegt bei 20 und 50 Eingaben im Median bei **1,00**, das Verhältnis wird auf 326 statt 24 überschätzt; Faktor **0,51 / 0,59 / 0,49** bei Vorstudien von 20 / 50 / 100 Eingaben, auf 3 bis 100 begrenzt **0,29 / 0,26 / 0,22** |

## Ehrliche Grenzen

- **Kosten sind gezählte Kantenzugriffe, keine Rechenzeiten.** In der Vormessung ergab eine einmalige Zeitmessung ein anderes Verhältnis als die Zählung; das Kostenverhältnis gehört dem besten Verfahren für das Ziel und ist hier nur für die zwei Verfahren (naiv, Tarjan) belegt.
- **Das teure Ziel ist hier absichtlich naiv gerechnet.** Mit der Brückensuche in einem Lauf schrumpft der Gewinn auf 0,79; die 0,163 sind der günstigste Fall.
- **Einzelzellen mit fünf teuren Ziehungen sind Rauschen** (Aufteilung 400: gemessen 1,26 bei 400 Wiederholungen, mit 3000 Wiederholungen 0,76, Formel 0,78).
- **Netze bis 17 Kanten**, unabhängige gleiche Ausfälle, nur drei billige Modelle und Ketten aus höchstens zwei; kein gelerntes Ersatzmodell (in der Vormessung wäre seine Obergrenze ρ = 0,984 gegen 0,982 für D2 gewesen).
- **Plug-in-α und Intervalle:** die Überdeckung wird für ein naives Intervall aus der Streuung der Wiederholungen gemessen, nicht für ein Intervall aus einer einzelnen Stichprobe.
- **Literaturangaben im Mathematik-Expander** stammen aus dem Gedächtnis und sind nicht eingesehen.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): Zufallsstrom gegen Ganzzahl-Referenz, Bitpackung der Masken, Leiter 2×4 von Hand (4 brückenlose, 7 mit Grad ≥ 2, 105 zusammenhängende von 1024 Kantenmengen; ρ² mit Brüchen), Faktor, Schwelle, Aufteilung und Kettenformeln an kleinen Zahlen, verdrängte Modelle.
- **Orakel** (`tests/test_oracle_mmd.py`): X2 gegen `networkx` (Zusammenhang und Brücken) auf **allen** Kantenmengen der Gitter 2×3, 2×4 und 3×3 und auf Stichproben von 3×4; **Kosten gegen eine in reinem Python gezählte Tiefensuche** und eine eigene Tarjan-Brückensuche; E C und E X2 gegen eine gewichtete `networkx`-Summe; **Varianzformel des Schätzers gegen die vollständige Aufzählung aller Ausgänge** einer kleinen diskreten Verteilung mit Brüchen (ein und zwei billige Modelle, für verschiedene α erwartungstreu); Aufteilung und Teilmengenwahl gegen eine Rastersuche.
- **Eigenschaften** (`tests/test_properties.py`): Faktor unter 1 genau unter der Schwelle, nicht monoton in ρ (kleines ρ schadet zusätzlich), Erwartungstreue trotz verzerrtem Modell, N₁ = N₀ gibt den einfachen Mittelwert, die Brückensuche macht das Ziel so billig wie C, Modelle irren nur nach oben.
- **Randfälle** (`tests/test_edge_cases.py`): Eingabefehler der Spielwiese, Netz ohne Knoten vom Grad 2 (E fällt weg), Baum (keine Streuung), Dreieck (ρ = 1), jedes Gitter mit jeder Modellwahl und jedem Verfahren, p an den Grenzen.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten, deutsches Zahlenformat.
- **Presets, Layout, Oberfläche:** Presets vollständig und jedes mit dem erwarteten Verdikt; Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, jeden Regler an Minimum und Maximum, die sechs Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims.**

Zahl der Tests: siehe `pytest --collect-only -q tests`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `mmd_algorithm.py` | Kern: Strom, Modelle und Kosten, Aufzählung, Formeln, Ketten, Schätzer |
| `mmd_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `mmd_evaluation.py` | Kennzahlen, Verdikt-Logik, sechs Experimente über feste Ströme |
| `mmd_layout.py` | Fester Kopf (Stylesheet) |
| `mmd_pdf_export.py` | PDF der aktuellen Einstellung |
| `mmd_presets.py` | Permalink (Begrenzen), Presets, Zufallsstrom-Knopf |
| `mmd_scenario.py` | Gitter und eigene Netze |
| `mmd_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und networkx |
| `requirements.txt` | Laufzeit-Bibliotheken |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

| Annahme | Verweis |
|---|---|
| Der Mittelwert der Hilfsgröße ist bekannt | [kontrollvariaten-demo](https://github.com/sebastian-hanisch/kontrollvariaten-demo) (Stück 3) |
| Eine Folge verfeinerter Modelle mit gegen null gehender Verzerrung | Stück 12 der Linie (Multilevel) |
| Ein gelerntes Ersatzmodell, das die Simulation **ersetzt** | [surrogate-queue-demo](https://github.com/sebastian-hanisch/surrogate-queue-demo) |
| Nur die beste Einstellung finden statt den Mittelwert schätzen | [multi-fidelity-demo](https://github.com/sebastian-hanisch/multi-fidelity-demo) (Bayessche Optimierung) |
| Seltene Ereignisse und Gewichte | Stück 7 der Linie (Importance Sampling) |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
