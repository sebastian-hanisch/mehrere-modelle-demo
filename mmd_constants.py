"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

GRIDS = {"2x4": (2, 4), "3x3": (3, 3), "2x5": (2, 5), "3x4": (3, 4)}
DEFAULT_GRID = "3x4"
LIVE_EDGE_LIMIT = 17          # Aufzählung aller Kantenmengen live bis hierher (2^17 = 131 072, rund 1 s); eigenes Netz ebenso
CUSTOM_EDGE_LIMIT = 17
CUSTOM_NODE_LIMIT = 14

# Billige Modelle des Ziels X2 (brückenlos zusammenhängend); alle irren nur nach oben (X2 <= Modell)
MODEL_NAMES = {"C": "Zusammenhang", "D2": "jeder Knoten hat mindestens zwei haltende Kanten", "E": "die Eckknoten (Grad 2) haben beide Kanten"}
MODEL_LABELS = {"C": "Zusammenhang C", "D2": "Grad ≥ 2 (D2)", "E": "Ecken (E)"}
MODEL_SETS = {"D2": ("D2",), "E": ("E",), "C": ("C",), "D2+E": ("D2", "E")}
MODEL_SET_LABELS = {"D2": "Grad ≥ 2 (D2)", "E": "Ecken (E)", "C": "Zusammenhang (C)", "D2+E": "D2 und Ecken"}
DEFAULT_MODELS = "D2"

# Verfahren für das teure Ziel: dieselben Werte, verschiedene Kosten (Kantenzugriffe einer Tiefensuche)
VERFAHREN = {"naiv": "jede Kante einzeln entfernen", "tarjan": "Brückensuche in einem Lauf (Tarjan)"}
DEFAULT_VERFAHREN = "naiv"

P_MIN, P_MAX, DEFAULT_P = 0.50, 0.99, 0.90
BUDGET_CHOICES = (30, 100, 300, 1000)     # Budget in Einheiten einer teuren Auswertung
BUDGET_MIN, BUDGET_MAX, DEFAULT_BUDGET = BUDGET_CHOICES[0], BUDGET_CHOICES[-1], 100
SEED_MAX, DEFAULT_SEED = 999999, 35
REPS = 300                                # Wiederholungen je Schätzung (feste Ströme); Rauschen der Varianzverhältnisse rund 2/sqrt(REPS − 1) ≈ 12 %
RATIO_MAX = 1000.0                        # Obergrenze für N_i / N_0 (bei Korrelation nahe 1 wächst das Optimum über alle Grenzen)

# Experimente
SWEEP_BUDGETS = (30, 100, 300, 1000)
SWEEP_REPS = 300
ALLOC_RATIOS = (1, 3, 10, 24, 60, 150, 400)
ALLOC_REPS = 400
COST_RS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
COST_REPS = 300
CHOICE_PS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
BIAS_BUDGETS = (100, 1000, 5000)
BIAS_REPS = 150
PARAM_PILOTS = (20, 50, 100)
PARAM_BUDGET = 1000
PARAM_REPS = 150
PARAM_CAP = (3.0, 100.0)

Z95 = 1.959963984540054

COLOR_PLAIN = "#4c78a8"
COLOR_MF = "#2F6B65"
COLOR_PLUG = "#E08E0B"
COLOR_TRUTH = "#222222"
COLOR_BAD = "#C0392B"
COLOR_OTHER = "rgba(150,150,150,0.55)"

PRESETS = {
    "Standardfall (Voreinstellung)": {"gitter": DEFAULT_GRID, "p": DEFAULT_P, "budget": DEFAULT_BUDGET, "modelle": DEFAULT_MODELS, "verfahren": DEFAULT_VERFAHREN, "seed": DEFAULT_SEED},
    "Die naheliegende Wahl": {"gitter": DEFAULT_GRID, "p": DEFAULT_P, "budget": DEFAULT_BUDGET, "modelle": "C", "verfahren": DEFAULT_VERFAHREN, "seed": DEFAULT_SEED},
    "Schlaues Ziel, kaum Gewinn": {"gitter": DEFAULT_GRID, "p": DEFAULT_P, "budget": DEFAULT_BUDGET, "modelle": DEFAULT_MODELS, "verfahren": "tarjan", "seed": DEFAULT_SEED},
    "Zwei billige Modelle": {"gitter": DEFAULT_GRID, "p": 0.99, "budget": 300, "modelle": "D2+E", "verfahren": DEFAULT_VERFAHREN, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Gitter 3×4, jede Kante hält mit 90 %, Budget von 100 teuren Auswertungen, billiges Modell „Grad ≥ 2“: ein deutlicher Gewinn.",
    "Die naheliegende Wahl": "Als billiges Modell der Zusammenhang, das Ziel der vorigen Stücke: er hängt kaum mit dem Ziel zusammen und schadet.",
    "Schlaues Ziel, kaum Gewinn": "Das teure Ziel wird mit einer Brückensuche in einem Lauf gerechnet: das billige Modell kostet nun etwa halb so viel wie das Ziel (r = 0,51 statt 0,05), der Gewinn schrumpft.",
    "Zwei billige Modelle": "Bei seltenem Ausfall (p = 0,99) hilft ein zweites, noch billigeres Modell zusätzlich.",
}
