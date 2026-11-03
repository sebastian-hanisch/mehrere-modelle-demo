"""Orakel-Tests: andere Rechenwege als die der Demo.

- Ziel X2 (brückenlos zusammenhängend) gegen networkx (Zusammenhang und Brücken) auf allen Kantenmengen kleiner Gitter und Stichproben größerer.
- Kosten: eine in reinem Python gezählte Tiefensuche (naives Verfahren) gegen die Tabellenkosten; eine eigene Tarjan-Brückensuche gegen X2 und `cost_C`.
- Mehrmodell-Schätzer: die Varianzformel gegen die vollständige Aufzählung aller Ausgänge einer kleinen diskreten Verteilung (mit Brüchen), für jedes alpha erwartungstreu.
- Aufteilung und Teilmengenwahl: gegen eine Rastersuche über ganzzahlige Aufteilungen.
"""

import itertools
from fractions import Fraction

import numpy as np
import pytest

import mmd_algorithm as A
import mmd_evaluation as ev
from mmd_scenario import grid_netz

nx = pytest.importorskip("networkx")


def _adj(n, edges):
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    return adj


def x2_networkx(n, edges, mask):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(e for i, e in enumerate(edges) if (mask >> i) & 1)
    return nx.is_connected(g) and not any(True for _ in nx.bridges(g))


def dfs_naiv(n, edges, mask):
    """Naives X2 in reinem Python mit gezählten Kantenzugriffen (jeder gelesene Adjazenzeintrag zählt 1)."""
    adj = _adj(n, edges)
    up = [(mask >> i) & 1 == 1 for i in range(len(edges))]

    def conn(skip=-1):
        seen = [False] * n
        seen[0] = True
        stack, cnt, visits = [0], 1, 0
        while stack:
            x = stack.pop()
            for y, i in adj[x]:
                visits += 1
                if up[i] and i != skip and not seen[y]:
                    seen[y] = True
                    cnt += 1
                    stack.append(y)
        return cnt == n, visits

    ok, cost = conn()
    if not ok:
        return False, cost, cost
    base = cost
    for i in range(len(edges)):
        if up[i]:
            ok, v = conn(skip=i)
            cost += v
            if not ok:
                return False, cost, base
    return True, cost, base


def tarjan_bruecken(n, edges, mask):
    """Zusammenhang und keine Brücke in einem Tiefensuchlauf (iterativ); Rückgabe (brückenlos zusammenhängend, gezählte Zugriffe)."""
    adj = _adj(n, edges)
    up = [(mask >> i) & 1 == 1 for i in range(len(edges))]
    disc, low = [-1] * n, [0] * n
    disc[0] = low[0] = 0
    t, visits, bridge = 1, 0, False
    stack = [(0, -1, 0)]
    while stack:
        x, pe, k = stack.pop()
        if k < len(adj[x]):
            stack.append((x, pe, k + 1))
            y, i = adj[x][k]
            visits += 1
            if not up[i] or i == pe:
                continue
            if disc[y] == -1:
                disc[y] = low[y] = t
                t += 1
                stack.append((y, i, 0))
            else:
                low[x] = min(low[x], disc[y])
        elif stack:
            px = stack[-1][0]
            low[px] = min(low[px], low[x])
            bridge |= low[x] > disc[px]
    return t == n and not bridge, visits


@pytest.mark.parametrize("rows,cols", [(2, 3), (2, 4), (3, 3)])
def test_x2_gegen_networkx_auf_allen_kantenmengen(rows, cols):
    netz = grid_netz(rows, cols)
    t = ev.netz_table(netz)
    for mask in range(1 << netz.m):
        assert bool(t["X2"][mask]) == x2_networkx(netz.n, netz.edges, mask), mask


def test_x2_gegen_networkx_auf_stichproben_des_gitters_3x4(gitter34):
    netz, t = gitter34
    rng = np.random.default_rng(3)
    for mask in rng.integers(0, 1 << netz.m, 1500).tolist() + [0, (1 << netz.m) - 1]:
        assert bool(t["X2"][mask]) == x2_networkx(netz.n, netz.edges, mask), mask


def test_kosten_und_werte_gegen_die_gezaehlte_tiefensuche(gitter34):
    netz, t = gitter34
    rng = np.random.default_rng(5)
    for mask in rng.integers(0, 1 << netz.m, 1500).tolist() + [0, (1 << netz.m) - 1]:
        ok, cost, base = dfs_naiv(netz.n, netz.edges, mask)
        assert ok == bool(t["X2"][mask])
        assert cost == t["cost_X2"][mask], mask
        assert base == t["cost_C"][mask], mask                      # eine Prüfung = Kosten des Modells C


def test_tarjan_gleich_x2_und_kostet_eine_pruefung(gitter34):
    netz, t = gitter34
    rng = np.random.default_rng(9)
    for mask in rng.integers(0, 1 << netz.m, 1500).tolist() + [0, (1 << netz.m) - 1]:
        ok, visits = tarjan_bruecken(netz.n, netz.edges, mask)
        assert ok == bool(t["X2"][mask]), mask
        if t["C"][mask]:
            assert visits == t["cost_C"][mask], mask                 # zusammenhängend: jeder Knoten liest alle seine Kanten, 2m Zugriffe
        else:
            assert visits <= 2 * netz.m


def test_modelle_d2_und_e_gegen_direkte_schleifen(gitter34):
    netz, t = gitter34
    deg = netz.degrees()
    rng = np.random.default_rng(11)
    for mask in rng.integers(0, 1 << netz.m, 800).tolist():
        up = [(mask >> i) & 1 for i in range(netz.m)]
        d = [0] * netz.n
        for i, (u, v) in enumerate(netz.edges):
            d[u] += up[i]
            d[v] += up[i]
        assert bool(t["D2"][mask]) == all(x >= 2 for x in d)
        ecken = all(d[v] == 2 for v in range(netz.n) if deg[v] == 2)
        assert bool(t["E"][mask]) == ecken


def test_zuverlaessigkeit_von_c_und_x2_gegen_networkx_summe():
    """E C und E X2 bei p = 0,7 auf dem Gitter 3×3: gewichtete Summe über alle 4096 Kantenmengen mit networkx gegen die Tabelle."""
    netz = grid_netz(3, 3)
    t = ev.netz_table(netz)
    p = 0.7
    rc = rx = 0.0
    for mask in range(1 << netz.m):
        k = bin(mask).count("1")
        w = p**k * (1 - p) ** (netz.m - k)
        g = nx.Graph()
        g.add_nodes_from(range(netz.n))
        g.add_edges_from(e for i, e in enumerate(netz.edges) if (mask >> i) & 1)
        if nx.is_connected(g):
            rc += w
            rx += w * (not any(True for _ in nx.bridges(g)))
    mo = A.exact_moments(t, netz.m, netz.corner_edges(), p, "naiv")
    assert mo["models"]["C"]["E"] == pytest.approx(rc, abs=1e-12)
    assert mo["EX"] == pytest.approx(rx, abs=1e-12)


# --- Schätzervarianz durch vollständige Aufzählung aller Ausgänge ---------------------------------------------------------------------------------

ATOME = [  # (f0, f1, f2) mit Wahrscheinlichkeiten; f1 verzerrt (Mittel 0.5 gegen 0.4)
    ((0, 0, 0), Fraction(3, 10)), ((0, 1, 1), Fraction(2, 10)), ((1, 1, 0), Fraction(1, 10)), ((1, 1, 1), Fraction(3, 10)), ((1, 0, 1), Fraction(1, 10))]


def _momente(idx_a, idx_b=None):
    ea = sum(p * a[idx_a] for a, p in ATOME)
    eb = ea if idx_b is None else sum(p * a[idx_b] for a, p in ATOME)
    eab = sum(p * a[idx_a] * a[idx_b if idx_b is not None else idx_a] for a, p in ATOME)
    return ea, eb, eab - ea * eb


def _exakt_schaetzer(ns, alphas):
    """Erwartungswert und Varianz des Mehrmodell-Schätzers über alle len(ATOME)^N Ausgänge (Brüche)."""
    N = ns[-1]
    ew = Fraction(0)
    ew2 = Fraction(0)
    for combo in itertools.product(range(len(ATOME)), repeat=N):
        pr = Fraction(1)
        for c in combo:
            pr *= ATOME[c][1]
        f = [[ATOME[c][0][j] for c in combo] for j in range(3)]
        s = Fraction(sum(f[0][: ns[0]]), ns[0])
        for i, a in enumerate(alphas):
            s += a * (Fraction(sum(f[i + 1][: ns[i + 1]]), ns[i + 1]) - Fraction(sum(f[i + 1][: ns[i]]), ns[i]))
        ew += pr * s
        ew2 += pr * s * s
    return ew, ew2 - ew * ew


@pytest.mark.parametrize("alpha", [Fraction(0), Fraction(1, 2), Fraction(3, 1), Fraction(-2, 3)])
def test_einmodellschaetzer_fuer_jedes_alpha_erwartungstreu_und_varianzformel(alpha):
    ew, var = _exakt_schaetzer([2, 4], [alpha])
    e0, _, v0 = _momente(0)
    e1, _, v1 = _momente(1)
    _, _, c01 = _momente(0, 1)
    assert ew == e0                                                   # unabhängig von alpha und vom Mittelwert des Modells (0,5 gegen 0,4)
    formel = v0 / 2 + (Fraction(1, 2) - Fraction(1, 4)) * (alpha**2 * v1 - 2 * alpha * c01)
    assert var == formel


def test_zweimodellschaetzer_erwartungstreu_und_varianzformel():
    e0, _, v0 = _momente(0)
    e1, _, v1 = _momente(1)
    e2, _, v2 = _momente(2)
    _, _, c01 = _momente(0, 1)
    _, _, c02 = _momente(0, 2)
    a1, a2 = c01 / v1, c02 / v2                                       # alpha = Kov / Var (= rho sigma0 / sigma_i)
    ew, var = _exakt_schaetzer([2, 3, 4], [a1, a2])
    assert ew == e0
    formel = v0 / 2 + (Fraction(1, 2) - Fraction(1, 3)) * (a1**2 * v1 - 2 * a1 * c01) + (Fraction(1, 3) - Fraction(1, 4)) * (a2**2 * v2 - 2 * a2 * c02)
    assert var == formel                                              # die Kreuzterme der Modelle verschwinden
    rho1sq, rho2sq = c01**2 / (v0 * v1), c02**2 / (v0 * v2)
    assert var == v0 * (Fraction(1, 2) - (Fraction(1, 2) - Fraction(1, 3)) * rho1sq - (Fraction(1, 3) - Fraction(1, 4)) * rho2sq)


def test_chain_variance_gleich_der_aufzaehlung():
    e0, _, v0 = _momente(0)
    _, _, c01 = _momente(0, 1)
    _, _, v1 = _momente(1)
    rho = float(c01) / np.sqrt(float(v0) * float(v1))
    _, var = _exakt_schaetzer([2, 4], [c01 / v1])
    assert A.chain_variance(float(v0), [rho], [2, 4]) == pytest.approx(float(var))


# --- Aufteilung und Teilmengenwahl gegen Rastersuche ---------------------------------------------------------------------------------------------

def _raster_faktor(w0, chain, einheiten=120):
    rhos2 = [c[0] ** 2 for c in chain]
    costs = [c[1] for c in chain]
    B = einheiten * w0
    best = np.inf
    for n0 in range(2, einheiten + 1):
        rest0 = B - n0 * w0
        if rest0 < 0:
            break
        if len(chain) == 1:
            for n1 in range(n0, int(rest0 / costs[0]) + 1, max(1, n0 // 10)):
                best = min(best, 1 / n0 - (1 / n0 - 1 / n1) * rhos2[0])
        else:
            for n1 in range(n0, int(rest0 / costs[0]) + 1, max(1, n0 // 6)):
                rest1 = rest0 - n1 * costs[0]
                n2 = max(int(rest1 / costs[1]), n1)
                if n0 * w0 + n1 * costs[0] + n2 * costs[1] > B + 1e-9:
                    continue
                best = min(best, 1 / n0 - (1 / n0 - 1 / n1) * rhos2[0] - (1 / n1 - 1 / n2) * rhos2[1])
    return float(best * B / w0)


@pytest.mark.parametrize("rho,r", [(0.98, 0.047), (0.9, 0.1), (0.7, 0.05)])
def test_einmodellfaktor_gegen_rastersuche(rho, r):
    assert A.chain_factor(100.0, [(rho, 100.0 * r)]) == pytest.approx(_raster_faktor(100.0, [(rho, 100.0 * r)]), rel=0.03)


@pytest.mark.parametrize("chain", [[(0.98, 5.0), (0.94, 2.0)], [(0.99, 4.0), (0.9, 1.0)]])
def test_zweimodellfaktor_gegen_rastersuche(chain):
    w0 = 100.0
    assert A.chain_valid(w0, chain)
    assert A.chain_factor(w0, chain) == pytest.approx(_raster_faktor(w0, chain), rel=0.05)


def test_beste_kette_gegen_vollstaendige_suche(gitter34):
    netz, t = gitter34
    for p in (0.7, 0.9, 0.99):
        mo = A.exact_moments(t, netz.m, netz.corner_edges(), p, "naiv")
        infos = ev.infos_of(mo)
        names, fac = A.best_chain(infos)
        alle = [(o, f) for o, f, ok in A.chains(infos) if ok] + [((), 1.0)]
        assert fac == min(f for _, f in alle)
        assert fac <= A.chain_factor(infos["_w0"], [infos[nm] for nm in names]) + 1e-12
