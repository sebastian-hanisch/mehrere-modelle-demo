"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Ziel X2 = 1, wenn das Netz brückenlos zusammenhängt; gesucht ist R2 = E X2. Das teure Ziel kostet w0 Kantenzugriffe je Auswertung, die billigen Modelle (C, D2, E) weniger.
Der Mehrmodell-Schätzer rechnet die billigen Modelle auf vielen Eingaben, das teure auf wenigen, und korrigiert um die Differenz auf den gemeinsamen Eingaben.
Seine Varianz sinkt (bei bester Aufteilung eines Budgets) auf den Faktor (sqrt(1 - rho^2) + rho sqrt(r))^2 gegen den einfachen Mittelwert gleichen Budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import mmd_algorithm as A
import mmd_constants as C
from mmd_scenario import Netz, named_netz


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def de_e(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 4,2e-10."""
    return f"{x:.1e}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    gitter: str = C.DEFAULT_GRID
    p: float = C.DEFAULT_P
    budget: int = C.DEFAULT_BUDGET
    modelle: str = C.DEFAULT_MODELS
    verfahren: str = C.DEFAULT_VERFAHREN
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    netz: Netz
    names: tuple[str, ...]           # gewählte Modelle nach absteigendem rho (nur solche mit endlichem rho)
    truth: float                     # exaktes E X2
    var0: float
    cost0: float                     # mittlere Kosten einer teuren Auswertung (Kantenzugriffe)
    rho: dict
    r: dict
    r_star: dict
    bias_model: dict                 # E f - E X2 je Modell
    alone: dict                      # Faktor je Modell allein bei bester Aufteilung (Formel)
    factor_chain: float              # Formel-Faktor der gewählten Kette bei bester Aufteilung
    valid: bool                      # ist die gewählte Kette anwendbar (kein Modell verdrängt)
    best_names: tuple[str, ...]
    best_factor: float
    ns: tuple[int, ...]
    n_plain: int
    ratio_mf: float                  # gemessen: Varianz Mehrmodell (exaktes alpha) / einfach
    ratio_plug: float
    bias_mf: float
    sd_plain: float
    sd_mf: float
    plain: tuple[float, ...]
    mf: tuple[float, ...]
    plug: tuple[float, ...]


# --- Wahrheit -----------------------------------------------------------------------------------------------------------------------------

def netz_table(netz: Netz) -> dict:
    """Aufzählung aller Kantenmengen (höchstens C.LIVE_EDGE_LIMIT Kanten)."""
    if netz.m > C.LIVE_EDGE_LIMIT:
        raise ValueError(f"{netz.m} Kanten: aufgezählt wird nur bis {C.LIVE_EDGE_LIMIT}")
    return A.model_table(netz.n, netz.edges, netz.corner_edges())


def infos_of(mo: dict) -> dict:
    infos = {nm: (v["rho"], v["cost"]) for nm, v in mo["models"].items()}
    infos["_w0"] = mo["cost0"]
    return infos


def chosen_names(modelle: str, mo: dict) -> tuple[str, ...]:
    names = [nm for nm in C.MODEL_SETS[modelle] if np.isfinite(mo["models"][nm]["rho"])]
    return tuple(sorted(names, key=lambda nm: -abs(mo["models"][nm]["rho"])))


# --- Hauptanalyse -------------------------------------------------------------------------------------------------------------------------

def _var_ratio(a: np.ndarray, b: np.ndarray) -> float:
    vb = float(b.var(ddof=1))
    return float(a.var(ddof=1)) / vb if vb > 0 else float("nan")


def analyse(settings: Settings, netz: Netz | None = None, table: dict | None = None) -> Analysis:
    """REPS Schätzungen: einfacher Mittelwert (budget Ziehungen) gegen Mehrmodell-Schätzer (exaktes alpha und Plug-in-alpha) bei gleichem Budget."""
    netz = netz if netz is not None else named_netz(settings.gitter)
    table = table if table is not None else netz_table(netz)
    ci = netz.corner_edges()
    mo = A.exact_moments(table, netz.m, ci, settings.p, settings.verfahren)
    infos = infos_of(mo)
    names = chosen_names(settings.modelle, mo)
    chain = [infos[nm] for nm in names]
    w0 = mo["cost0"]
    best_names, best_factor = A.best_chain(infos) if mo["VarX"] > 1e-15 else ((), 1.0)
    n_plain = int(settings.budget)
    ns = A.chain_ns(settings.budget, w0, chain) if chain else [n_plain]
    key = (1, int(settings.p * 1000), int(settings.budget), settings.seed)
    plain = A.plain_estimates(table, netz.m, settings.p, (2, *key), C.REPS, n_plain)
    if chain:
        alphas = A.exact_alphas(mo, names)
        mf = A.mfmc_estimates(table, netz.m, settings.p, (3, *key), C.REPS, ns, names, alphas)
        plug = A.mfmc_estimates(table, netz.m, settings.p, (3, *key), C.REPS, ns, names, None)
    else:
        mf = plug = plain
    return Analysis(
        netz, names, mo["EX"], mo["VarX"], w0, {nm: v["rho"] for nm, v in mo["models"].items()}, {nm: v["r"] for nm, v in mo["models"].items()},
        {nm: v["r_star"] for nm, v in mo["models"].items()}, {nm: v["E"] - mo["EX"] for nm, v in mo["models"].items()}, {nm: v["factor"] for nm, v in mo["models"].items()},
        A.chain_factor(w0, chain), A.chain_valid(w0, chain), best_names, best_factor, tuple(ns), n_plain,
        _var_ratio(mf, plain), _var_ratio(plug, plain), float(mf.mean() - mo["EX"]), float(plain.std(ddof=1)), float(mf.std(ddof=1)),
        tuple(float(x) for x in plain), tuple(float(x) for x in mf), tuple(float(x) for x in plug),
    )


def verdict_main(a: Analysis, settings: Settings) -> tuple[str, str]:
    """Lohnt das gewählte billige Modell, und ist die Kette anwendbar?"""
    if a.var0 <= 1e-15:
        return "info", "Das Ziel hat bei diesen Einstellungen keine Streuung (das Netz ist nie oder immer brückenlos): es gibt nichts zu senken."
    if not a.names:
        return "info", "Für das gewählte Modell ist die Korrelation mit dem Ziel nicht definiert (das Modell ist konstant): es wird nicht benutzt."
    label = " und ".join(C.MODEL_LABELS[nm] for nm in a.names)
    kopf = (f"Mit {label} (Ziel gerechnet durch: {C.VERFAHREN[settings.verfahren]}) ist ρ = {', '.join(de(a.rho[nm], 3) for nm in a.names)} und das Kostenverhältnis r = {', '.join(de(a.r[nm], 3) for nm in a.names)}; "
            f"die Formel gibt den Varianzfaktor {de(a.factor_chain, 3)}, gemessen wurde {de(a.ratio_mf, 2)} (Plug-in-α: {de(a.ratio_plug, 2)}).")
    if a.factor_chain >= 1.0:
        return "warning", kopf + " Das Modell ist zu teuer oder zu wenig mit dem Ziel verbunden: gleiches Budget, mehr Streuung als der einfache Mittelwert. Es sollte weggelassen werden."
    if not a.valid:
        return "info", kopf + f" Ein Modell der gewählten Menge wird von den anderen verdrängt; die beste Wahl wäre {' und '.join(C.MODEL_LABELS[nm] for nm in a.best_names) or 'keines'} (Faktor {de(a.best_factor, 3)})."
    if a.best_names and tuple(a.best_names) != tuple(a.names) and a.best_factor < a.factor_chain - 0.005:
        return "info", kopf + f" Besser wäre {' und '.join(C.MODEL_LABELS[nm] for nm in a.best_names)} (Faktor {de(a.best_factor, 3)})."
    return "success", kopf + f" Die Varianz sinkt auf rund {de(a.factor_chain * 100, 0)} %."


# --- Experiment 1: Gewinn gegen Budget -----------------------------------------------------------------------------------------------------

def _first_chain(settings: Settings, table: dict, netz: Netz):
    """Exakte Momente und das erste (stärkste) gewählte Modell als Kette aus einem Modell; (mo, name, (rho, Kosten)) oder (mo, None, None)."""
    mo = A.exact_moments(table, netz.m, netz.corner_edges(), settings.p, settings.verfahren)
    names = chosen_names(settings.modelle, mo)
    if not names or mo["VarX"] <= 1e-15:
        return mo, None, None
    nm = names[0]
    return mo, nm, (mo["models"][nm]["rho"], mo["models"][nm]["cost"])


def budget_sweep(settings: Settings, netz: Netz, table: dict) -> dict:
    mo = A.exact_moments(table, netz.m, netz.corner_edges(), settings.p, settings.verfahren)
    names = chosen_names(settings.modelle, mo)
    rows = []
    if not names or mo["VarX"] <= 1e-15:
        return {"rows": rows, "factor_chain": float("nan"), "names": list(names)}
    chain = [(mo["models"][nm]["rho"], mo["models"][nm]["cost"]) for nm in names]
    alphas = A.exact_alphas(mo, names)
    w0 = mo["cost0"]
    for b in C.SWEEP_BUDGETS:
        ns = A.chain_ns(b, w0, chain)
        key = (4, int(settings.p * 1000), b, settings.seed)
        plain = A.plain_estimates(table, netz.m, settings.p, (5, *key), C.SWEEP_REPS, b)
        mf = A.mfmc_estimates(table, netz.m, settings.p, (6, *key), C.SWEEP_REPS, ns, names, alphas)
        plug = A.mfmc_estimates(table, netz.m, settings.p, (6, *key), C.SWEEP_REPS, ns, names, None)
        rows.append({"budget": b, "ns": ns, "ratio_mf": _var_ratio(mf, plain), "ratio_plug": _var_ratio(plug, plain),
                     "formula": A.chain_variance(mo["VarX"], [abs(c[0]) for c in chain], ns) / (mo["VarX"] / b), "bias": float(mf.mean() - mo["EX"])})
    return {"rows": rows, "factor_chain": A.chain_factor(w0, chain), "names": list(names)}


def verdict_budget(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    if not rows:
        return "info", "Mit dieser Modellwahl gibt es nichts zu messen."
    noise = 100 / np.sqrt(C.SWEEP_REPS / 2)
    return "info", (f"Die Formel gibt bei bester Aufteilung den Faktor {de(exp['factor_chain'], 3)}. Gemessen (exaktes α) liegt das Varianzverhältnis zwischen {de(min(r['ratio_mf'] for r in rows), 2)} und "
                    f"{de(max(r['ratio_mf'] for r in rows), 2)}, mit aus der Stichprobe geschätztem α zwischen {de(min(r['ratio_plug'] for r in rows), 2)} und {de(max(r['ratio_plug'] for r in rows), 2)}. "
                    f"Das Rauschen der Verhältnisse beträgt bei {C.SWEEP_REPS} Wiederholungen rund {de(noise, 0)} %.")


# --- Experiment 2: Aufteilung --------------------------------------------------------------------------------------------------------------

def allocation(settings: Settings, netz: Netz, table: dict) -> dict:
    """Varianz gegen das Verhältnis N1/N0 bei festem Budget, erstes gewähltes Modell als einziges billiges Modell."""
    mo, nm, chain1 = _first_chain(settings, table, netz)
    if nm is None:
        return {"rows": [], "name": None}
    rho, w1 = chain1
    w0 = mo["cost0"]
    alpha = A.exact_alphas(mo, (nm,))
    b = settings.budget
    plain = A.plain_estimates(table, netz.m, settings.p, (8, int(settings.p * 1000), b, settings.seed), C.ALLOC_REPS, b)
    rows = []
    for ratio in C.ALLOC_RATIOS:
        n0 = max(int(b * w0 / (w0 + ratio * w1)), 2)
        n1 = max(int(round(ratio * n0)), n0)
        est = A.mfmc_estimates(table, netz.m, settings.p, (9, int(settings.p * 1000), b, ratio, settings.seed), C.ALLOC_REPS, [n0, n1], (nm,), alpha)
        rows.append({"ratio": ratio, "n0": n0, "n1": n1, "measured": _var_ratio(est, plain),
                     "formula": A.chain_variance(mo["VarX"], [abs(rho)], [n0, n1]) / (mo["VarX"] / b), "bias": float(est.mean() - mo["EX"])})
    return {"rows": rows, "name": nm, "optimum": A.optimal_ratio(rho, w0, w1), "factor": mo["models"][nm]["factor"]}


def verdict_allocation(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    if not rows:
        return "info", "Mit dieser Modellwahl gibt es nichts zu messen."
    first, best = rows[0], min(rows, key=lambda r: r["measured"])
    worst = rows[-1]
    return "info", (f"Gleich viele Ziehungen für beide Modelle (N1/N0 = 1) bringen {de(first['measured'], 2)}: nichts gewonnen, der Schätzer ist nur der einfache Mittelwert mit weniger Ziehungen. "
                    f"Das Optimum liegt nach der Formel bei N1/N0 = {de(exp['optimum'], 0)}; gemessen am besten bei {de(best['ratio'], 0)} mit {de(best['measured'], 2)}. "
                    f"Zu viel Budget in das billige Modell ({de(worst['ratio'], 0)} billige je teurer Auswertung, nur {de(worst['n0'], 0)} teure Ziehungen) gibt nach der Formel {de(worst['formula'], 2)}, gemessen {de(worst['measured'], 2)}; "
                    f"bei so wenigen teuren Ziehungen rauscht die Messung stark.")


# --- Experiment 3: Kostenverhältnis ----------------------------------------------------------------------------------------------------------

def cost_ratio(settings: Settings, netz: Netz, table: dict) -> dict:
    """Faktor gegen ein als Annahme gesetztes Kostenverhältnis r (Kosten des billigen Modells = r mal Kosten des Ziels), erstes gewähltes Modell."""
    mo, nm, chain1 = _first_chain(settings, table, netz)
    if nm is None:
        return {"rows": [], "name": None}
    rho, _ = chain1
    w0 = mo["cost0"]
    alpha = A.exact_alphas(mo, (nm,))
    b = settings.budget
    plain = A.plain_estimates(table, netz.m, settings.p, (10, int(settings.p * 1000), b, settings.seed), C.COST_REPS, b)
    rows = []
    for r in C.COST_RS:
        w1 = r * w0
        ratio = A.optimal_ratio(rho, w0, w1)
        ns = A.chain_ns(b, w0, [(rho, w1)])
        est = A.mfmc_estimates(table, netz.m, settings.p, (11, int(settings.p * 1000), b, int(r * 1000), settings.seed), C.COST_REPS, ns, (nm,), alpha)
        rows.append({"r": r, "ratio": ratio, "formula": A.factor_two(rho, r), "measured": _var_ratio(est, plain)})
    return {"rows": rows, "name": nm, "rho": rho, "r_star": A.breakeven(rho), "r_real": mo["models"][nm]["r"]}


def verdict_cost(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    if not rows:
        return "info", "Mit dieser Modellwahl gibt es nichts zu messen."
    over = [r for r in rows if r["formula"] >= 1.0]
    text = (f"Bei ρ = {de(exp['rho'], 3)} liegt die Schwelle bei r* = {de(exp['r_star'], 2)}: das billige Modell darf höchstens {de(exp['r_star'] * 100, 0)} % der Kosten des Ziels kosten. "
            f"Das wirkliche Kostenverhältnis dieses Modells ist r = {de(exp['r_real'], 3)}. ")
    if over:
        text += f"Ab r = {de(over[0]['r'], 2)} (Formel {de(over[0]['formula'], 2)}, gemessen {de(over[0]['measured'], 2)}) ist der Mehrmodell-Schätzer schlechter als gar kein billiges Modell."
    else:
        text += "In dieser Tabelle bleibt das Modell überall nützlich."
    return "info", text


# --- Experiment 4: welche Modelle (exakt) --------------------------------------------------------------------------------------------------------

def model_choice(settings: Settings, netz: Netz, table: dict) -> dict:
    """Für jedes p die Faktoren der Modelle allein, der Kette aus zwei und die beste gültige Wahl (exakt, ohne Zufall)."""
    rows = []
    for p in C.CHOICE_PS:
        mo = A.exact_moments(table, netz.m, netz.corner_edges(), p, settings.verfahren)
        if mo["VarX"] <= 1e-15:
            continue
        infos = infos_of(mo)
        names, fac = A.best_chain(infos)
        pair = {o: (f, ok) for o, f, ok in A.chains(infos) if len(o) == 2}
        d2e = pair.get(("D2", "E"), pair.get(("E", "D2"), (float("nan"), False)))
        rows.append({"p": p, "R": mo["EX"], "cost0": mo["cost0"], **{nm: mo["models"][nm]["factor"] for nm in C.MODEL_NAMES},
                     "rho": {nm: mo["models"][nm]["rho"] for nm in C.MODEL_NAMES}, "pair": d2e[0], "pair_valid": d2e[1], "best_names": names, "best": fac})
    return {"rows": rows}


def verdict_choice(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    if not rows:
        return "info", "Das Ziel hat bei keinem der Werte von p eine Streuung."
    row = next((r for r in rows if r["p"] == 0.9), rows[len(rows) // 2])
    hi, lo = rows[-1], rows[0]
    return "info", (f"Bei p = {de(row['p'], 2)} gewinnt {' und '.join(C.MODEL_LABELS[nm] for nm in row['best_names']) or 'kein Modell'} (Faktor {de(row['best'], 3)}); "
                    f"der Zusammenhang (C), das Ziel der vorigen Stücke, kommt auf {de(row['C'], 2)}: er ist als billiges Modell schlechter als keines. "
                    f"Bei p = {de(hi['p'], 2)} ist {C.MODEL_LABELS[min(('D2', 'E'), key=lambda nm: hi[nm])]} allein am besten ({de(min(hi['D2'], hi['E']), 3)}), bei p = {de(lo['p'], 2)} gewinnt kein Modell: "
                    f"{de(min(lo['D2'], lo['E']), 2)} für das beste.")


# --- Experiment 5: Verzerrung des billigen Modells ---------------------------------------------------------------------------------------------

def bias_exp(settings: Settings, netz: Netz, table: dict) -> dict:
    """Nur das billige Modell (Mittel über N1 Ziehungen) gegen den Mehrmodell-Schätzer bei wachsendem Budget; Überdeckung des 95-%-Intervalls."""
    mo, nm, chain1 = _first_chain(settings, table, netz)
    if nm is None:
        return {"rows": [], "name": None}
    w0 = mo["cost0"]
    alpha = A.exact_alphas(mo, (nm,))
    f_mean = mo["models"][nm]["E"]
    rows = []
    for b in C.BIAS_BUDGETS:
        ns = A.chain_ns(b, w0, [chain1])
        key = (12, int(settings.p * 1000), b, settings.seed)
        mf = A.mfmc_estimates(table, netz.m, settings.p, key, C.BIAS_REPS, ns, (nm,), alpha)
        lo = A.cheap_estimates(table, netz.m, settings.p, (13, *key[1:]), C.BIAS_REPS, ns[-1], nm)
        sd_lo = float(np.sqrt(f_mean * (1 - f_mean) / ns[-1]))
        sd_mf = float(mf.std(ddof=1))
        rows.append({"budget": b, "ns": ns, "bias_lo": float(lo.mean() - mo["EX"]), "cover_lo": float(np.mean(np.abs(lo - mo["EX"]) <= C.Z95 * sd_lo)),
                     "bias_mf": float(mf.mean() - mo["EX"]), "bias_mf_se": float((mf.mean() - mo["EX"]) / (sd_mf / np.sqrt(C.BIAS_REPS))) if sd_mf > 0 else float("nan"),
                     "cover_mf": float(np.mean(np.abs(mf - mo["EX"]) <= C.Z95 * sd_mf)), "sd_mf": sd_mf})
    return {"rows": rows, "name": nm, "model_bias": f_mean - mo["EX"]}


def verdict_bias(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    if not rows:
        return "info", "Mit dieser Modellwahl gibt es nichts zu messen."
    last = rows[-1]
    return "warning", (f"Das billige Modell ist um {de(exp['model_bias'], 4)} verzerrt. Rechnet man nur mit ihm, deckt sein 95-%-Intervall die Wahrheit bei Budget {de(rows[0]['budget'], 0)} noch in "
                       f"{de(rows[0]['cover_lo'] * 100, 0)} % der Wiederholungen ab, bei Budget {de(last['budget'], 0)} nur noch in {de(last['cover_lo'] * 100, 0)} %: mehr Ziehungen machen den falschen Wert nur sicherer. "
                       f"Der Mehrmodell-Schätzer hat bei Budget {de(last['budget'], 0)} den Fehler des Mittels {de(last['bias_mf'], 4)} ({de(last['bias_mf_se'], 1)} Standardfehler) und streut mit {de(last['sd_mf'], 4)}.")


# --- Experiment 6: Parameter schätzen ----------------------------------------------------------------------------------------------------------

def params_exp(settings: Settings, netz: Netz, table: dict) -> dict:
    """α aus den geteilten Eingaben gegen Aufteilung aus einer Vorstudie (Kosten abgezogen), unbegrenzt und auf PARAM_CAP begrenzt; erstes gewähltes Modell."""
    mo, nm, chain1 = _first_chain(settings, table, netz)
    if nm is None:
        return {"rows": [], "name": None}
    rho, w1 = chain1
    w0 = mo["cost0"]
    b = C.PARAM_BUDGET
    p, m = settings.p, netz.m
    alpha = A.exact_alphas(mo, (nm,))
    plain = A.plain_estimates(table, m, p, (14, int(p * 1000), settings.seed), C.PARAM_REPS, b)
    ideal_ns = A.chain_ns(b, w0, [chain1])
    ideal = A.mfmc_estimates(table, m, p, (15, int(p * 1000), settings.seed), C.PARAM_REPS, ideal_ns, (nm,), alpha)
    plug = A.mfmc_estimates(table, m, p, (15, int(p * 1000), settings.seed), C.PARAM_REPS, ideal_ns, (nm,), None)
    rows = [{"label": "exakte Parameter", "ratio_med": ideal_ns[1] / ideal_ns[0], "factor": _var_ratio(ideal, plain)},
            {"label": "α aus den geteilten Eingaben", "ratio_med": ideal_ns[1] / ideal_ns[0], "factor": _var_ratio(plug, plain)}]
    for npilot in C.PARAM_PILOTS:
        est, est_cap, ratios, rhos = [], [], [], []
        for rep in range(C.PARAM_REPS):
            pm = A.draw_masks(A.stream_seed(16, int(p * 1000), npilot, rep, settings.seed), npilot, p, m)
            a, bb = table["X2"][pm], table[nm][pm]
            alpha_h = A.plugin_alpha(a, bb)
            rho_h = float(np.corrcoef(a, bb)[0, 1]) if a.std() > 0 and bb.std() > 0 else 0.0
            ratio = A.optimal_ratio(min(abs(rho_h), 0.9999), w0, w1)
            rhos.append(rho_h)
            ratios.append(ratio)
            left = b - npilot * (w0 + w1) / w0
            for out, rt in ((est, ratio), (est_cap, float(np.clip(ratio, *C.PARAM_CAP)))):
                n0 = max(int(left * w0 / (w0 + rt * w1)), 2)
                n1 = max(int(round(rt * n0)), n0)
                mk = A.draw_masks(A.stream_seed(17, int(p * 1000), npilot, rep, settings.seed), n1, p, m)
                out.append(A.mfmc_value(table["X2"][mk], [table[nm][mk]], [n0, n1], [alpha_h]))
        rows.append({"label": f"Vorstudie {npilot}", "pilot": npilot, "ratio_med": float(np.median(ratios)), "rho_med": float(np.median(rhos)),
                     "factor": _var_ratio(np.array(est), plain), "factor_cap": _var_ratio(np.array(est_cap), plain)})
    return {"rows": rows, "name": nm, "ideal_factor": A.factor_two(rho, mo["models"][nm]["r"]), "optimum": A.optimal_ratio(rho, w0, w1)}


def verdict_params(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    if not rows:
        return "info", "Mit dieser Modellwahl gibt es nichts zu messen."
    plug = rows[1]
    pil = rows[2:]
    worst = max(pil, key=lambda r: r["factor"])
    return "warning", (f"α aus den geteilten Eingaben kostet fast nichts ({de(plug['factor'], 2)} gegen {de(rows[0]['factor'], 2)} mit exakten Parametern). Die Aufteilung aus einer Vorstudie ist heikel: "
                       f"die geschätzte Korrelation liegt oft bei 1 und das Verhältnis N1/N0 wird überschätzt (Median {de(worst['ratio_med'], 0)} statt {de(exp['optimum'], 0)}); "
                       f"Faktor {', '.join(de(r['factor'], 2) for r in pil)} bei Vorstudien von {', '.join(str(r['pilot']) for r in pil)} Eingaben, auf {de(C.PARAM_CAP[0], 0)} bis {de(C.PARAM_CAP[1], 0)} begrenzt "
                       f"{', '.join(de(r['factor_cap'], 2) for r in pil)}.")


# --- Spielwiese ---------------------------------------------------------------------------------------------------------------------------

def custom_result(settings: Settings, netz: Netz) -> dict:
    """Ein eigenes Netz: Analyse und die exakten Kennzahlen aller Modelle bei p."""
    table = netz_table(netz)
    a = analyse(settings, netz=netz, table=table)
    mo = A.exact_moments(table, netz.m, netz.corner_edges(), settings.p, settings.verfahren)
    return {"analysis": a, "moments": mo}
