"""Mehrere Modelle: das billige Modell darf falsch sein - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Ein billiges Modell darf falsch sein, solange es mit dem teuren mitschwankt. Man rechnet es auf vielen
Eingaben, das teure auf wenigen, und korrigiert um die Differenz auf den gemeinsamen Eingaben. Vehikel ist die Zuverlässigkeit eines Gitternetzes (Ziel: brückenlos
zusammenhängend), die Wahrheit ist exakt aufgezählt.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import mmd_constants as C
import mmd_evaluation as ev
from mmd_evaluation import Settings, de
from mmd_layout import dunkles_design, kopf_css
from mmd_pdf_export import build_pdf
from mmd_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mmd_scenario import named_netz, parse_edges
from mmd_visualization import build_allocation, build_bias, build_budget, build_choice, build_cost, build_netz, build_params, build_spread

st.set_page_config(page_title="Mehrere Modelle: das billige Modell darf falsch sein – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Alle Kantenmengen aufzählen …")
def _table(gitter):
    return ev.netz_table(named_netz(gitter))


@st.cache_data(show_spinner="Schätzungen wiederholen …")
def _analysis(settings, table):
    return ev.analyse(settings, table=table)


@st.cache_data(show_spinner="Gewinn gegen Budget messen …")
def _budget(settings, table):
    return ev.budget_sweep(settings, named_netz(settings.gitter), table)


@st.cache_data(show_spinner="Aufteilung messen …")
def _alloc(settings, table):
    return ev.allocation(settings, named_netz(settings.gitter), table)


@st.cache_data(show_spinner="Kostenverhältnis messen …")
def _cost(settings, table):
    return ev.cost_ratio(settings, named_netz(settings.gitter), table)


@st.cache_data(show_spinner=False)
def _choice(settings, table):
    return ev.model_choice(settings, named_netz(settings.gitter), table)


@st.cache_data(show_spinner="Verzerrung und Überdeckung messen …")
def _bias(settings, table):
    return ev.bias_exp(settings, named_netz(settings.gitter), table)


@st.cache_data(show_spinner="Parameter schätzen …")
def _params(settings, table):
    return ev.params_exp(settings, named_netz(settings.gitter), table)


@st.cache_data(show_spinner=False)
def _custom(text, settings):
    netz = parse_edges(text)
    return netz, ev.custom_result(settings, netz)


def _gitter_label(key):
    rows, cols = C.GRIDS[key]
    return f"{rows}×{cols} ({rows * (cols - 1) + cols * (rows - 1)} Kanten)"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mmd_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🪜 Mehrere Modelle: das billige Modell darf falsch sein")
st.markdown(
    """
Ein teures Verfahren (hier: ist das Netz **brückenlos** zusammenhängend, also auch nach Ausfall einer weiteren Kante?) lässt sich durch **billige Modelle** ergänzen, die dasselbe Ziel nur grob treffen.
Man rechnet das billige Modell auf **vielen** Eingaben, das teure auf **wenigen**, und korrigiert das billige Mittel um die Differenz auf den gemeinsamen Eingaben. Der Schätzer bleibt bei festem α für jedes billige Modell erwartungstreu, auch wenn es verzerrt ist; ein aus denselben Ziehungen geschätztes α macht ihn nur näherungsweise erwartungstreu.
Der Gewinn hängt an **zwei** Größen: der Korrelation ρ mit dem Ziel **und** dem Kostenverhältnis r. Der Varianzfaktor ist **(√(1−ρ²) + ρ√r)²**, und ein Modell, das nicht billig genug ist, schadet.

Weiter unten: **🔬 sechs Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Mehrmodell-Schätzer", expanded=True):
    st.markdown(
        """
1. **Ziel und Modelle:** Jede Kante hält unabhängig mit Wahrscheinlichkeit p. Das teure **Ziel** X2: das Netz hängt zusammen und bleibt es auch nach Ausfall jeder einzelnen weiteren Kante. Billige Modelle: **C** (das Netz hängt zusammen, das Ziel der vorigen Stücke), **D2** (jeder Knoten hat mindestens zwei haltende Kanten), **E** (die Eckknoten haben beide Kanten). Alle drei sind notwendig für X2.
2. **Kosten:** Gezählt werden Zugriffe auf den Zustand einer Kante. Das naive Ziel prüft den Zusammenhang und entfernt dann jede haltende Kante einzeln; eine Brückensuche in einem Lauf (Tarjan) kostet nur eine Prüfung.
3. **Schätzer:** s = Mittel(X2 auf N₀ Eingaben) + α · (Mittel(f auf N₁ Eingaben) − Mittel(f auf denselben N₀)), mit α = ρ·σ(X2)/σ(f). Die billige Größe f wird auf allen N₁ ≥ N₀ Eingaben gerechnet, das Ziel nur auf den ersten N₀.
4. **Aufteilung:** Bei festem Budget ist N₁/N₀ = √(w₀ρ² / (w₁(1−ρ²))) am besten; mit N₁ = N₀ gewinnt man nichts, der Gewinn kommt aus der **zusätzlichen billigen Stichprobe**.
5. **Wann es sich lohnt:** nur wenn r unter der Schwelle r* = ((1−√(1−ρ²))/ρ)² liegt. Sonst lässt man das Modell weg.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz und Modelle**")
    gitter = st.selectbox("Gitter", options("gitter_select"), format_func=_gitter_label, key="gitter_select", help="Knoten in Zeilen und Spalten, Kanten waagerecht und senkrecht. Die Wahrheit wird live aufgezählt.")
    modelle = st.selectbox("Billige Modelle", options("modelle_select"), format_func=C.MODEL_SET_LABELS.get, key="modelle_select",
                           help="Welche billigen Modelle benutzt werden: Grad ≥ 2 an jedem Knoten, die Ecken, der Zusammenhang, oder D2 und Ecken zusammen.")
    verfahren = st.selectbox("Ziel gerechnet durch", options("verfahren_select"), format_func=C.VERFAHREN.get, key="verfahren_select",
                             help="Dieselben Werte, verschiedene Kosten: jede Kante einzeln entfernen (teuer) oder eine Brückensuche in einem Lauf (billig).")
    p = st.slider("Haltewahrscheinlichkeit p je Kante", *bounds("p_slider"), key="p_slider", step=0.01, help="Jede Kante hält unabhängig mit dieser Wahrscheinlichkeit.")
    st.markdown("**Schätzung**")
    budget = st.select_slider("Budget (teure Auswertungen)", options("budget_select"), key="budget_select",
                              help="Das Rechenbudget in Einheiten einer teuren Auswertung. Der einfache Mittelwert zieht so viele Ziehungen, der Mehrmodell-Schätzer verteilt dasselbe Budget auf teure und billige.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"gitter_select": gitter, "modelle_select": modelle, "verfahren_select": verfahren, "p_slider": float(p), "budget_select": int(budget), "seed_input": int(seed)})

settings = Settings(str(gitter), float(p), int(budget), str(modelle), str(verfahren), int(seed))
table = _table(settings.gitter)
a = _analysis(settings, table)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Wahrheit E X2", de(a.truth, 4), help=f"Exakte Wahrscheinlichkeit, dass das {a.netz.name} bei p = {de(settings.p, 2)} brückenlos zusammenhängt (Aufzählung aller Kantenmengen).")
m2.metric("Faktor nach der Formel", de(a.factor_chain, 3) if a.names else "–", help="Varianz des Mehrmodell-Schätzers gegen den einfachen Mittelwert gleichen Budgets bei bester Aufteilung: (√(1−ρ²) + ρ√r)² für ein Modell.")
m3.metric("Gemessen (α exakt)", de(a.ratio_mf, 2) if a.names else "–", help=f"Varianzverhältnis aus {C.REPS} Wiederholungen mit festen Strömen und dem exakten α. Rauschen etwa 10 bis 16 %.")
m4.metric("Gemessen (α geschätzt)", de(a.ratio_plug, 2) if a.names else "–", help="Dasselbe mit α aus den ersten N₀ Eingaben der Stichprobe.")
if a.names:
    st.caption("Korrelation mit dem Ziel und Kostenverhältnis: " + "; ".join(f"{C.MODEL_LABELS[nm]}: ρ = {de(a.rho[nm], 3)}, r = {de(a.r[nm], 3)}, Schwelle r* = {de(a.r_star[nm], 3)}" for nm in a.names)
               + f". Kosten einer teuren Auswertung: {de(a.cost0, 1)} Kantenzugriffe. Stichprobengrößen: {', '.join(de(x, 0) for x in a.ns)} (einfacher Mittelwert: {de(a.n_plain, 0)}).")
st.plotly_chart(build_spread(a), width="stretch", key="main_chart")
st.caption("Kästen: Verteilung der Schätzungen über die Wiederholungen bei gleichem Budget (links einfacher Mittelwert, Mitte Mehrmodell mit exaktem α, rechts mit geschätztem α; Linie: Median, Raute und Balken: Mittel und Streuung). Gestrichelt: die Wahrheit.")

if st.button("📄 PDF der Schätzung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="mehrere-modelle.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wann lohnt ein billiges Modell, und wie teilt man das Budget auf?")
kind, text = ev.verdict_main(a, settings)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Gewinn gegen Budget")
st.caption(f"Varianzverhältnis (gegen den einfachen Mittelwert gleichen Budgets) bei Budgets von {', '.join(str(x) for x in C.SWEEP_BUDGETS)} teuren Auswertungen, je {C.SWEEP_REPS} Wiederholungen mit festen Strömen; gewählte Modelle, p und Ziel-Verfahren. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_budget"):
    st.session_state["budget_done"] = True
if st.session_state.get("budget_done"):
    ex1 = _budget(settings, table)
    if ex1["rows"]:
        st.plotly_chart(build_budget(ex1), width="stretch", key="budget_chart")
    kind, text = ev.verdict_budget(ex1)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Wie teilt man das Budget auf?")
st.caption(f"Bei festem Budget das Verhältnis N₁/N₀ (billige je teurer Ziehung) variiert, mit dem ersten gewählten Modell als einzigem billigen; je {C.ALLOC_REPS} Wiederholungen. Gestrichelt: die exakte Varianz der Formel.")
if st.button("▶ Messung starten", key="run_alloc"):
    st.session_state["alloc_done"] = True
if st.session_state.get("alloc_done"):
    ex2 = _alloc(settings, table)
    if ex2["rows"]:
        st.plotly_chart(build_allocation(ex2), width="stretch", key="alloc_chart")
    kind, text = ev.verdict_allocation(ex2)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Wie billig muss das billige Modell sein?")
st.caption(f"Das Kostenverhältnis r (Kosten des billigen Modells geteilt durch Kosten des Ziels) wird als **Annahme** gesetzt, die Korrelation bleibt die des ersten gewählten Modells; je {C.COST_REPS} Wiederholungen mit der besten Aufteilung nach der Formel.")
if st.button("▶ Messung starten", key="run_cost"):
    st.session_state["cost_done"] = True
if st.session_state.get("cost_done"):
    ex3 = _cost(settings, table)
    if ex3["rows"]:
        st.plotly_chart(build_cost(ex3), width="stretch", key="cost_chart")
    kind, text = ev.verdict_cost(ex3)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Welche Modelle?")
st.caption("Der **exakte** Varianzfaktor jedes billigen Modells allein und die beste gültige Wahl gegen p, aus der Aufzählung aller Kantenmengen (kein Zufall). Ein Faktor über 1 heißt: schlechter als gar kein billiges Modell.")
ex4 = _choice(settings, table)
if ex4["rows"]:
    st.plotly_chart(build_choice(ex4, settings.p), width="stretch", key="choice_chart")
    st.dataframe(
        {"p": [de(r["p"], 2) for r in ex4["rows"]], "Kosten Ziel": [de(r["cost0"], 0) for r in ex4["rows"]], "C": [de(r["C"], 2) for r in ex4["rows"]], "D2": [de(r["D2"], 3) for r in ex4["rows"]],
         "E": [de(r["E"], 3) for r in ex4["rows"]], "D2 und E": [de(r["pair"], 3) + ("" if r["pair_valid"] else " (verdrängt)") for r in ex4["rows"]],
         "beste Wahl": [(" + ".join(r["best_names"]) or "keine") + f" ({de(r['best'], 3)})" for r in ex4["rows"]]},
        hide_index=True, width="stretch", key="choice_table",
    )
kind, text = ev.verdict_choice(ex4)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Das billige Modell ist falsch, der Schätzer nicht")
st.caption(f"Nur das billige Modell (Mittel über alle seine Ziehungen) gegen den Mehrmodell-Schätzer bei Budgets von {', '.join(str(x) for x in C.BIAS_BUDGETS)} teuren Auswertungen: Überdeckung des 95-%-Intervalls, je {C.BIAS_REPS} Wiederholungen. Kostet etwa eine Viertelminute.")
if st.button("▶ Messung starten", key="run_bias"):
    st.session_state["bias_done"] = True
if st.session_state.get("bias_done"):
    ex5 = _bias(settings, table)
    if ex5["rows"]:
        st.plotly_chart(build_bias(ex5), width="stretch", key="bias_chart")
        st.dataframe(
            {"Budget": [r["budget"] for r in ex5["rows"]], "Fehler nur billig": [de(r["bias_lo"], 4) for r in ex5["rows"]], "Überdeckung nur billig": [de(r["cover_lo"], 2) for r in ex5["rows"]],
             "Fehler Mehrmodell": [de(r["bias_mf"], 4) for r in ex5["rows"]], "in Standardfehlern": [de(r["bias_mf_se"], 1) for r in ex5["rows"]], "Überdeckung Mehrmodell": [de(r["cover_mf"], 2) for r in ex5["rows"]]},
            hide_index=True, width="stretch", key="bias_table",
        )
    kind, text = ev.verdict_bias(ex5)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 6: Was kostet es, die Parameter zu schätzen?")
st.caption(f"Bei einem Budget von {C.PARAM_BUDGET} teuren Auswertungen: α aus den geteilten Eingaben gegen die Aufteilung aus einer Vorstudie (deren Kosten vom Budget abgehen), unbegrenzt und auf {C.PARAM_CAP[0]:g} bis {C.PARAM_CAP[1]:g} begrenzt; je {C.PARAM_REPS} Wiederholungen. Kostet etwa eine Viertelminute.".replace(".0", "").replace(".", ","))
if st.button("▶ Messung starten", key="run_params"):
    st.session_state["params_done"] = True
if st.session_state.get("params_done"):
    ex6 = _params(settings, table)
    if ex6["rows"]:
        st.plotly_chart(build_params(ex6), width="stretch", key="params_chart")
    kind, text = ev.verdict_params(ex6)
    getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein eigenes Netz"):
    st.caption(
        f"Eine Kante je Zeile, zwei ganze Zahlen (»0-1«). Höchstens {C.CUSTOM_EDGE_LIMIT} Kanten und {C.CUSTOM_NODE_LIMIT} Knoten; Modelle, p, Budget, Ziel-Verfahren und Strom stammen aus der Seitenleiste. "
        "Das Modell E gilt für Knoten vom Grad 2; hat das Netz keine, ist es konstant und fällt weg."
    )
    text_in = st.text_area("Kanten", "0-1\n1-2\n2-3\n3-0\n0-2\n2-4\n4-5\n5-2", height=140, key="eigene_kanten")
    try:
        cnetz, cres = _custom(text_in, settings)
    except ValueError as err:
        st.error(str(err))
    else:
        ca, cmo = cres["analysis"], cres["moments"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Wahrheit E X2", de(ca.truth, 4))
        s2.metric("Faktor nach der Formel", de(ca.factor_chain, 3) if ca.names else "–")
        s3.metric("Gemessen (α geschätzt)", de(ca.ratio_plug, 2) if ca.names else "–")
        st.plotly_chart(build_netz(cnetz), width="stretch", key="custom_net")
        st.markdown("**Exakte Kennzahlen bei diesem p:** " + "; ".join(
            f"{C.MODEL_LABELS[nm]}: ρ = {de(v['rho'], 3) if v['rho'] == v['rho'] else '–'}, r = {de(v['r'], 3) if v['r'] == v['r'] else '–'}, Faktor {de(v['factor'], 3) if v['factor'] == v['factor'] else '–'}"
            for nm, v in cmo["models"].items()) + ".")
        kind, text = ev.verdict_main(ca, settings)
        getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Kostenverhältnis ist fest** | Es gehört dem besten Verfahren für das Ziel: mit einer Brückensuche in einem Lauf statt dem Entfernen jeder Kante schrumpft der Gewinn (Preset „Schlaues Ziel, kaum Gewinn“). | – |
| **Das billige Modell hängt am Ziel** | Der Gewinn ist 1 − ρ² im besten Fall; ein billiges Modell mit kleinem ρ (der Zusammenhang) schadet (Experimente 3 und 4). | – |
| **Die Aufteilung ist bekannt** | Aus einer kleinen Vorstudie geschätzt, wird sie schlecht, weil ρ̂ nahe 1 liegt (Experiment 6). | – |
| **Der Mittelwert des billigen Modells ist unbekannt** | Ist er bekannt, genügt eine Kontrollvariate und der Faktor wird 1 − ρ² statt der Formel. | Stück 3 (Kontrollvariaten) |
| **Das billige Modell ist eine feste Näherung** | Eine ganze Folge verfeinerter Modelle, deren Verzerrung gegen null geht, ist ein eigenes Verfahren. | Stück 12 (Multilevel) |
| **Man will nur die beste Einstellung, nicht den Mittelwert** | Dann ist die Frage Optimierung und nicht Schätzung. | `multi-fidelity-demo` (Bayessche Optimierung) |
| **Ein gelerntes Ersatzmodell** | Als Ersatz der Simulation ist es verzerrt; als billiges Modell hier würde es genügen, aber seine Korrelation beschränkt den Gewinn. | `surrogate-queue-demo` |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Alle Netze und Ausfälle sind **erfunden**; die Wahrheit ist die exakte Aufzählung. Die Kanten fallen unabhängig und mit gleicher Wahrscheinlichkeit aus. Die Zufallszahlen kommen aus festen Strömen (SplitMix64), nie aus einem einzelnen Seed.

**Grenzen.** Aufgezählt wird live bis 17 Kanten. Kosten sind **gezählte Kantenzugriffe** einer Tiefensuche, keine Rechenzeiten; auf einem anderen Rechner und in einer anderen Sprache ändert sich das Verhältnis. Es gibt drei billige Modelle und Ketten aus höchstens zwei; ein Nutzer kann keine eigenen Modelle eintragen. Die Aufteilung der Formel ist für ganze Stichproben gerundet.

**Bekannte Probleme und Unsicherheiten.** Varianzverhältnisse aus 150 bis 400 Wiederholungen rauschen um rund 10 bis 16 %, bei seltenem Ausfall (p = 0,99) deutlich mehr. Bei kleinem p (etwa 0,5) bricht die Prüfung des Ziels meist früh ab und das billige Modell ist kaum billiger. Die Literaturangaben im Mathematik-Expander stammen aus dem Gedächtnis und sind nicht eingesehen. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gesucht ist $\mu_0 = \mathbb E\,f_0$ für das teure Ziel $f_0 = X_2 \in \{0,1\}$ („das Netz hängt brückenlos zusammen“). Ein billiges Modell $f_1$ auf denselben Eingaben (Kantenzuständen) hat unbekannten Erwartungswert, Korrelation $\rho$ mit $f_0$, Kosten $w_1 \ll w_0$.

**Schätzer.** Mit $N_1 \ge N_0$ Eingaben, $f_0$ auf den ersten $N_0$ und $f_1$ auf allen: $\hat\mu = \bar f_0^{(N_0)} + \alpha\,(\bar f_1^{(N_1)} - \bar f_1^{(N_0)})$. Für jedes $\alpha$ ist $\mathbb E\hat\mu = \mu_0$, auch bei verzerrtem $f_1$.

**Varianz.** $\operatorname{Var}\hat\mu = \tfrac{\sigma_0^2}{N_0} + \big(\tfrac1{N_0} - \tfrac1{N_1}\big)(\alpha^2\sigma_1^2 - 2\alpha\rho\sigma_0\sigma_1)$, kleinstes bei $\alpha = \rho\sigma_0/\sigma_1$: $\sigma_0^2/N_0 - (1/N_0 - 1/N_1)\rho^2\sigma_0^2$.

**Beste Aufteilung** bei Budget $B = N_0 w_0 + N_1 w_1$: $N_1/N_0 = \sqrt{w_0\rho^2 / (w_1(1-\rho^2))}$ und der Faktor gegen den einfachen Mittelwert gleichen Budgets
$$\big(\sqrt{1-\rho^2} + |\rho|\sqrt{r}\big)^2,\quad r = w_1/w_0.$$ Er ist kleiner als 1 genau für $r < r^\ast = \big((1-\sqrt{1-\rho^2})/|\rho|\big)^2$. Für $r \to 0$ ist er $1-\rho^2$ (Kontrollvariate mit bekanntem Mittelwert).

**Mehrere Modelle.** Mit $K$ billigen Modellen, absteigend nach $\rho_i$ und nach Kosten geordnet ($\rho_0 = 1$, $\rho_{K+1} = 0$): $\hat\mu = \bar f_0^{(N_0)} + \sum_i \alpha_i (\bar f_i^{(N_i)} - \bar f_i^{(N_{i-1})})$ und bei bester Aufteilung der Faktor $\big(\sum_{i=0}^K \sqrt{w_i(\rho_i^2 - \rho_{i+1}^2)}\big)^2 / w_0$. Die Aufteilung ist nur anwendbar, wenn $N_i/N_0 = \sqrt{w_0(\rho_i^2-\rho_{i+1}^2)/(w_i(1-\rho_1^2))}$ nicht abnimmt; sonst ist ein Modell verdrängt. Die beste Menge wird aus allen Ketten gewählt.

**Wahrheit.** Die Aufzählung aller $2^m$ Kantenmengen liefert $\mu_0$, alle Varianzen, Kovarianzen und mittleren Kosten exakt, mit den Gewichten $p^k(1-p)^{m-k}$.

**Literatur (aus dem Gedächtnis, nicht eingesehen).** Peherstorfer, Willcox und Gunzburger 2016, „Optimal model management for multifidelity Monte Carlo estimation“ (SIAM J. Sci. Comput.); Peherstorfer, Willcox und Gunzburger 2018, „Survey of multifidelity methods in uncertainty propagation, inference, and optimization“ (SIAM Review); Giles 2015, „Multilevel Monte Carlo methods“ (Acta Numerica).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
