"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

import numpy as np
from fpdf import FPDF

import mmd_constants as C
from mmd_evaluation import Analysis, Settings, de, verdict_main

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~",
            "σ": "Sigma", "ρ": "rho", "α": "alpha", "β": "beta", "θ": "theta", "∞": "unendlich", "Σ": "Summe", "δ": "delta", "Δ": "Delta", "½": "1/2", "̂": "", "ϱ": "rho"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Mehrere Modelle: das billige Modell darf falsch sein"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    modelle = ", ".join(C.MODEL_LABELS[nm] for nm in a.names) or "keines"
    pdf.multi_cell(0, 5, _s(
        f"{a.netz.name}, {a.netz.n} Knoten, {a.netz.m} Kanten; jede Kante hält mit Wahrscheinlichkeit {de(settings.p, 2)}; Ziel: das Netz hängt brückenlos zusammen (auch nach Ausfall einer weiteren Kante). "
        f"Billige Modelle: {modelle}. Ziel gerechnet durch: {C.VERFAHREN[settings.verfahren]}. Budget {de(settings.budget, 0)} teure Auswertungen, {C.REPS} Wiederholungen, Strom {settings.seed}. Alles simuliert, keine Messdaten."),
        new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    widths = (70, 40, 40)
    for h, w in zip(("Schätzer", "Streuung", "Varianz gegen einfach"), widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    sd_plug = float(np.std(a.plug, ddof=1))
    for label, sd, ratio in (("einfacher Mittelwert", a.sd_plain, 1.0), ("Mehrmodell, exaktes alpha", a.sd_mf, a.ratio_mf), ("Mehrmodell, alpha aus der Stichprobe", sd_plug, a.ratio_plug)):
        for text, w in zip((label, de(sd, 5), de(ratio, 3)), widths):
            pdf.cell(w, 6, _s(text), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(f"Wahrheit E X2 (Aufzählung aller Kantenmengen): {de(a.truth, 5)}. Kosten einer teuren Auswertung: {de(a.cost0, 1)} Kantenzugriffe. "
                            f"Stichprobengrößen (teuer, billig, ...): {', '.join(de(x, 0) for x in a.ns)}; einfacher Mittelwert: {de(a.n_plain, 0)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(verdict_main(a, settings)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: mehrere-modelle-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
