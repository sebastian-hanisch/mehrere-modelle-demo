"""Kern der Demo (nur numpy).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom       stream_seed, uniforms, draw_masks        SplitMix64, zählerbasiert, ein fester Strom je (Einstellung, Wiederholung)
  Modelle und Kosten propagate, model_table                   Ziel X2 (brückenlos zusammenhängend) und die billigen Modelle C, D2, E; Kosten in Kantenzugriffen
  Wahrheit           exact_moments                            Aufzählung aller 2^m Kantenmengen: Erwartungswerte, Korrelationen, mittlere Kosten
  Formeln            factor_two, breakeven, optimal_ratio,    Varianzfaktor (sqrt(1-rho^2) + rho sqrt(r))^2, Schwelle, beste Aufteilung,
                     chain_factor, chain_ns, best_chain       Ketten aus mehreren Modellen und die beste Teilmenge
  Schätzer           plain_estimates, mfmc_estimates, ...     einfacher Mittelwert, Mehrmodell-Schätzer (MFMC), nur das billige Modell
Zufall kommt nur über einen übergebenen Strom, nie aus einem globalen Generator.

Schätzer (ein billiges Modell f1, Ziel f0; N1 >= N0; f1 auf allen N1 Eingaben, f0 auf den ersten N0):
    s = mean_{N0} f0 + alpha (mean_{N1} f1 - mean_{N0} f1),   alpha = rho sigma0 / sigma1.
Er ist für jedes feste alpha erwartungstreu, auch wenn f1 verzerrt ist.
"""

from __future__ import annotations

import itertools

import numpy as np

import mmd_constants as C

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1


# --- Zufallsstrom -------------------------------------------------------------------------------------------------------------------------

def mix_int(z: int) -> int:
    """SplitMix64-Mischfunktion mit Python-Ganzzahlen."""
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys: int) -> int:
    """Seed eines Stroms aus einem Schlüsseltupel (Einstellung, Wiederholung, ...)."""
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed: int, count: int) -> np.ndarray:
    """`count` gleichverteilte Zahlen in [0, 1) (53 Bit) aus dem Strom `seed`; Ausgabe i hängt nur von (seed, i) ab."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def draw_masks(seed: int, count: int, p: float, m: int) -> np.ndarray:
    """`count` Kantenmengen als Bitmaske (Kante e = Bit e): Kante e der Ziehung i hält genau dann, wenn ihre Zufallszahl (Nr. i*m + e) unter p liegt."""
    bits = (uniforms(seed, count * m).reshape(count, m) < p).astype(np.uint32)
    return (bits << np.arange(m, dtype=np.uint32)).sum(axis=1).astype(np.uint32)


# --- Ziel, billige Modelle und Kosten -------------------------------------------------------------------------------------------------------

def propagate(present: np.ndarray, n: int, edges) -> tuple[np.ndarray, np.ndarray]:
    """Kleinste-Marke-Ausbreitung je Zeile (`present` ist (B, m) bool). Rückgabe: (zusammenhängend (B,), Marken (B, n)).

    Zusammenhängend genau dann, wenn alle Marken 0 sind; die Knoten mit Marke 0 bilden die Komponente von Knoten 0."""
    B = present.shape[0]
    if n <= 1:
        return np.ones(B, dtype=bool), np.zeros((B, max(n, 1)), dtype=np.int16)
    lab = np.tile(np.arange(n, dtype=np.int16), (B, 1))
    while True:
        before = int(lab.sum())
        for e, (u, v) in enumerate(edges):
            low = np.minimum(lab[:, u], lab[:, v])
            pe = present[:, e]
            lab[:, u] = np.where(pe, low, lab[:, u])
            lab[:, v] = np.where(pe, low, lab[:, v])
        if int(lab.sum()) == before:
            break
    return lab.max(axis=1) == 0, lab


def _degrees(n: int, edges) -> np.ndarray:
    deg = np.zeros(n)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def model_values(present: np.ndarray, n: int, edges, corner_idx) -> dict:
    """X2, C, D2, E und die Kosten (Kantenzugriffe) des Ziels und von C je Zeile.

    Kosten einer Zusammenhangsprüfung durch Tiefensuche von Knoten 0 aus = Summe der Knotengrade im vollen Netz über die Komponente von Knoten 0
    (jeder erreichte Knoten liest die Zustände aller seiner Kanten). Das naive Ziel prüft den Zusammenhang und danach jede haltende Kante einzeln
    entfernt, bis zum ersten Fehlschlag; die Brückensuche in einem Lauf (Tarjan) kostet dasselbe wie eine Prüfung (`cost_C`).
    D2 liest jede Kante einmal (m), E die Kanten an den Knoten vom Grad 2."""
    m = len(edges)
    deg_full = _degrees(n, edges)
    conn, lab = propagate(present, n, edges)
    cost_c = ((lab == 0) * deg_full[None, :]).sum(axis=1)
    alive = conn.copy()
    x2 = conn.copy()
    cost_x2 = cost_c.astype(np.int64).copy()
    for e in range(m):
        todo = alive & present[:, e]
        if not todo.any():
            continue
        sub = present[todo].copy()
        sub[:, e] = False
        ok, lab_e = propagate(sub, n, edges)
        cost_x2[todo] += ((lab_e == 0) * deg_full[None, :]).sum(axis=1).astype(np.int64)
        fail = np.zeros(present.shape[0], dtype=bool)
        fail[todo] = ~ok
        x2 &= ~fail
        alive &= ~fail
    up_deg = np.zeros((present.shape[0], n), dtype=np.int32)
    for e, (u, v) in enumerate(edges):
        up_deg[:, u] += present[:, e]
        up_deg[:, v] += present[:, e]
    corner = present[:, list(corner_idx)].all(axis=1) if len(corner_idx) else np.ones(present.shape[0], dtype=bool)
    return {
        "X2": x2.astype(np.float64), "C": conn.astype(np.float64), "D2": (up_deg >= 2).all(axis=1).astype(np.float64), "E": corner.astype(np.float64),
        "cost_X2": cost_x2.astype(np.float64), "cost_C": cost_c.astype(np.float64),
    }


def model_table(n: int, edges, corner_idx, chunk: int = 1 << 15) -> dict:
    """Alle 2^m Kantenmengen: k (haltende Kanten), Werte der Modelle und die Kosten (Grundlage der exakten Momente und des Nachschlagens beim Ziehen)."""
    m = len(edges)
    shifts = np.arange(m, dtype=np.uint32)
    cols: dict[str, list] = {}
    for start in range(0, 1 << m, chunk):
        masks = np.arange(start, min(start + chunk, 1 << m), dtype=np.uint32)
        present = ((masks[:, None] >> shifts) & 1).astype(bool)
        v = model_values(present, n, edges, corner_idx)
        v["k"] = present.sum(axis=1)
        for key, arr in v.items():
            cols.setdefault(key, []).append(arr)
    return {k: np.concatenate(v) for k, v in cols.items()}


# --- Wahrheit durch Aufzählung ---------------------------------------------------------------------------------------------------------

def weights(table: dict, m: int, p: float) -> np.ndarray:
    k = table["k"]
    return p**k * (1.0 - p) ** (m - k)


def model_cost(table: dict, m: int, corner_idx, name: str, verfahren: str, p: float) -> float:
    """Mittlere Kosten eines Modells (name in C, D2, E) oder des Ziels (name = "X2") in Kantenzugriffen."""
    if name == "X2":
        return float(np.sum(weights(table, m, p) * table["cost_X2" if verfahren == "naiv" else "cost_C"]))
    if name == "C":
        return float(np.sum(weights(table, m, p) * table["cost_C"]))
    return float(m if name == "D2" else len(corner_idx))


def _nan_if_flat(cov: float, va: float, vb: float) -> float:
    return cov / np.sqrt(va * vb) if va > 1e-15 and vb > 1e-15 else float("nan")


def factor_two(rho: float, r: float) -> float:
    """Varianzfaktor eines Ziels mit einem billigen Modell bei bester Aufteilung: (sqrt(1 - rho^2) + |rho| sqrt(r))^2, r = Kosten billig / Kosten teuer."""
    return float((np.sqrt(1.0 - rho**2) + abs(rho) * np.sqrt(r)) ** 2)


def breakeven(rho: float) -> float:
    """Kostenverhältnis, ab dem das Modell nichts mehr bringt: factor_two = 1 bei r* = ((1 - sqrt(1 - rho^2)) / |rho|)^2."""
    rho = abs(rho)
    return float(((1.0 - np.sqrt(1.0 - rho**2)) / rho) ** 2) if rho > 0 else 0.0


def exact_moments(table: dict, netz_m: int, corner_idx, p: float, verfahren: str) -> dict:
    """Exakte Momente aus der Aufzählung: E X2, Var X2, mittlere Kosten des Ziels und je Modell E, Var, Kovarianz mit X2, Korrelation rho, Kosten, r, Faktor, Schwelle."""
    w = weights(table, netz_m, p)
    X = table["X2"]
    ex = float(np.sum(w * X))
    vx = float(np.sum(w * X * X) - ex**2)
    w0 = model_cost(table, netz_m, corner_idx, "X2", verfahren, p)
    out = {"EX": ex, "VarX": vx, "cost0": w0, "models": {}}
    for nm in C.MODEL_NAMES:
        f = table[nm]
        ef = float(np.sum(w * f))
        vf = float(np.sum(w * f * f) - ef**2)
        cov = float(np.sum(w * f * X) - ef * ex)
        rho = _nan_if_flat(cov, vx, vf)
        cost = model_cost(table, netz_m, corner_idx, nm, verfahren, p)
        r = cost / w0 if w0 > 0 else float("nan")
        ok = np.isfinite(rho) and np.isfinite(r) and r > 0
        out["models"][nm] = {"E": ef, "Var": vf, "cov": cov, "rho": rho, "cost": cost, "r": r,
                             "factor": factor_two(rho, r) if ok else float("nan"), "r_star": breakeven(rho) if np.isfinite(rho) else float("nan")}
    f1, f2 = table["D2"], table["E"]
    out["cov_D2_E"] = float(np.sum(w * f1 * f2) - np.sum(w * f1) * np.sum(w * f2))
    return out


# --- Ketten aus mehreren Modellen ---------------------------------------------------------------------------------------------------------

def chain_factor(w0: float, chain) -> float:
    """Varianzfaktor einer Kette [(rho, Kosten), ...] mit absteigendem rho: (Summe_{i=0..K} sqrt(w_i (rho_i^2 - rho_{i+1}^2)))^2 / w0, rho_0 = 1, rho_{K+1} = 0."""
    if not chain:
        return 1.0
    rhos = [1.0] + [abs(c[0]) for c in chain] + [0.0]
    costs = [w0] + [c[1] for c in chain]
    return float(sum(np.sqrt(costs[i] * (rhos[i] ** 2 - rhos[i + 1] ** 2)) for i in range(len(costs))) ** 2 / w0)


def chain_ratios(w0: float, chain) -> list[float]:
    """N_i / N_0 der besten Aufteilung (Index 0 ist 1): sqrt(w0 (rho_i^2 - rho_{i+1}^2) / (w_i (1 - rho_1^2)))."""
    rhos = [1.0] + [abs(c[0]) for c in chain] + [0.0]
    costs = [w0] + [c[1] for c in chain]
    if rhos[1] >= 1.0:
        return [1.0] + [C.RATIO_MAX] * len(chain)
    out = [np.sqrt(w0 * (rhos[i] ** 2 - rhos[i + 1] ** 2) / (costs[i] * (1.0 - rhos[1] ** 2))) for i in range(len(costs))]
    return [float(min(x, C.RATIO_MAX)) for x in out]


def chain_valid(w0: float, chain) -> bool:
    """Eine Kette ist anwendbar, wenn rho und Kosten strikt fallen und die Aufteilung N_i nicht abnimmt (sonst ist das Modell verdrängt)."""
    if not chain:
        return True
    rhos = [1.0] + [abs(c[0]) for c in chain] + [0.0]
    costs = [w0] + [c[1] for c in chain]
    if any(rhos[i] <= rhos[i + 1] for i in range(len(rhos) - 1)) or any(costs[i] <= costs[i + 1] for i in range(len(costs) - 1)):
        return False
    ratio = chain_ratios(w0, chain)
    return all(ratio[i] >= ratio[i - 1] - 1e-12 for i in range(1, len(ratio)))


def chains(infos: dict) -> list[tuple[tuple[str, ...], float, bool]]:
    """Alle Ketten aus höchstens zwei Modellen (infos: Name -> (rho, Kosten), w0 unter "_w0"): (Namen nach absteigendem rho, Faktor, gültig). Modelle ohne endliches rho werden übergangen."""
    w0 = infos["_w0"]
    names = [nm for nm, v in infos.items() if nm != "_w0" and np.isfinite(v[0]) and v[1] > 0]
    out = []
    for k in (1, 2):
        for sub in itertools.combinations(names, k):
            order = tuple(sorted(sub, key=lambda nm: -abs(infos[nm][0])))
            chain = [infos[nm] for nm in order]
            out.append((order, chain_factor(w0, chain), chain_valid(w0, chain)))
    return out


def best_chain(infos: dict) -> tuple[tuple[str, ...], float]:
    """Gültige Kette mit dem kleinsten Faktor; die leere Kette (nur das Ziel, Faktor 1) gilt, wenn keine Kette unter 1 liegt."""
    best: tuple[tuple[str, ...], float] = ((), 1.0)
    for order, fac, ok in chains(infos):
        if ok and fac < best[1] - 1e-12:
            best = (order, fac)
    return best


def optimal_ratio(rho: float, w0: float, w1: float) -> float:
    """N1 / N0 mit der kleinsten Varianz bei festem Budget für ein billiges Modell (mindestens 1, höchstens RATIO_MAX)."""
    if abs(rho) >= 1.0:
        return C.RATIO_MAX
    return float(min(max(1.0, np.sqrt(w0 * rho**2 / (w1 * (1.0 - rho**2)))), C.RATIO_MAX))


def chain_ns(budget_units: float, w0: float, chain) -> list[int]:
    """Stichprobengrößen [N_0, N_1, ...] der besten Aufteilung bei einem Budget von `budget_units` teuren Auswertungen (Kosten w0 je Auswertung)."""
    ratios = [1.0]
    for r in chain_ratios(w0, chain)[1:]:
        ratios.append(max(r, ratios[-1]))          # bei einer verdrängten Kette nie weniger billige als teure Ziehungen
    costs = [w0] + [c[1] for c in chain]
    n0 = max(int(budget_units * w0 / sum(r * c for r, c in zip(ratios, costs))), 2)
    ns = [n0]
    for r in ratios[1:]:
        ns.append(max(int(round(r * n0)), ns[-1]))
    return ns


def chain_variance(v0: float, rhos, ns) -> float:
    """Exakte Varianz des Mehrmodell-Schätzers bei optimalem alpha und den Größen ns: v0 (1/N0 - Summe_i (1/N_{i-1} - 1/N_i) rho_i^2)."""
    var = 1.0 / ns[0]
    for i in range(1, len(ns)):
        var -= (1.0 / ns[i - 1] - 1.0 / ns[i]) * rhos[i - 1] ** 2
    return float(v0 * var)


# --- Schätzer -----------------------------------------------------------------------------------------------------------------------------

def mfmc_value(f0: np.ndarray, fs, ns, alphas) -> float:
    """f0 und fs[i] sind Werte auf denselben Eingaben (f0 wird nur auf den ersten ns[0] gelesen, fs[i] auf den ersten ns[i+1] und ns[i])."""
    s = float(f0[: ns[0]].mean())
    for i, (f, a) in enumerate(zip(fs, alphas)):
        s += a * float(f[: ns[i + 1]].mean() - f[: ns[i]].mean())
    return s


def plugin_alpha(f0: np.ndarray, f1: np.ndarray) -> float:
    """alpha = Kov(f0, f1) / Var(f1) aus der Stichprobe; null, wenn eine der beiden Reihen konstant ist."""
    if len(f0) < 3 or f0.std() == 0 or f1.std() == 0:
        return 0.0
    return float(np.cov(f0, f1)[0, 1] / f1.var(ddof=1))


def plain_estimates(table: dict, m: int, p: float, key: tuple, reps: int, n: int) -> np.ndarray:
    """Einfache Mittelwerte des Ziels aus je n Ziehungen (fester Strom je Wiederholung)."""
    return np.array([table["X2"][draw_masks(stream_seed(*key, rep), n, p, m)].mean() for rep in range(reps)])


def cheap_estimates(table: dict, m: int, p: float, key: tuple, reps: int, n: int, name: str) -> np.ndarray:
    """Mittelwerte nur des billigen Modells aus je n Ziehungen: der Schätzer, der das billige Modell für die Wahrheit hält."""
    return np.array([table[name][draw_masks(stream_seed(*key, rep), n, p, m)].mean() for rep in range(reps)])


def mfmc_estimates(table: dict, m: int, p: float, key: tuple, reps: int, ns, names, alphas=None) -> np.ndarray:
    """Mehrmodell-Schätzer über feste Ströme. `alphas` fest (exakt) oder None: dann je Wiederholung aus den ersten N_0 Ziehungen geschätzt (Plug-in)."""
    out = np.empty(reps)
    for rep in range(reps):
        mk = draw_masks(stream_seed(*key, rep), ns[-1], p, m)
        f0 = table["X2"][mk]
        fs = [table[nm][mk] for nm in names]
        a = alphas if alphas is not None else [plugin_alpha(f0[: ns[0]], f[: ns[0]]) for f in fs]
        out[rep] = mfmc_value(f0, fs, ns, a)
    return out


def exact_alphas(mo: dict, names) -> list[float]:
    """alpha_i = rho_i sigma_0 / sigma_i aus den exakten Momenten (Modelle in der Reihenfolge `names`)."""
    return [mo["models"][nm]["rho"] * float(np.sqrt(mo["VarX"] / mo["models"][nm]["Var"])) for nm in names]
