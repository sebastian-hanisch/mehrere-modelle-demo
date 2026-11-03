"""Jede Zahl der README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Exakte Zahlen (Aufzählung, Formeln) mit engen Bändern. Gemessene Zahlen stammen aus festen Strömen und sind reproduzierbar; sie stehen hier mit Bändern, die das
Rauschen (150 bis 400 Wiederholungen: 8 bis 10 %, bei wenigen teuren Ziehungen mehr) und kleine Plattformunterschiede der Gleitkommasummen abdecken."""

import numpy as np
import pytest

import mmd_algorithm as A
import mmd_evaluation as ev
from mmd_scenario import named_netz

S = ev.Settings()


@pytest.fixture(scope="module")
def netz():
    return named_netz("3x4")


@pytest.fixture(scope="module")
def tab(netz):
    return ev.netz_table(netz)


@pytest.fixture(scope="module")
def mo(netz, tab):
    return A.exact_moments(tab, netz.m, netz.corner_edges(), 0.9, "naiv")


@pytest.fixture(scope="module")
def main_a(netz, tab):
    return ev.analyse(S, netz, tab)


def _mo(netz, tab, p, verfahren="naiv"):
    return A.exact_moments(tab, netz.m, netz.corner_edges(), p, verfahren)


def test_wahrheit_und_kosten(mo):
    assert mo["EX"] == pytest.approx(0.4026, abs=5e-5)
    assert mo["cost0"] == pytest.approx(361.2, abs=0.05)
    assert mo["models"]["D2"]["cost"] == 17 and mo["models"]["E"]["cost"] == 8
    assert mo["models"]["C"]["cost"] == pytest.approx(33.5, abs=0.05)


def test_d2_als_billiges_modell(mo):
    d2 = mo["models"]["D2"]
    assert d2["rho"] == pytest.approx(0.982, abs=5e-4) and d2["r"] == pytest.approx(0.047, abs=5e-4)
    assert d2["factor"] == pytest.approx(0.163, abs=5e-4) and d2["r_star"] == pytest.approx(0.679, abs=5e-4)
    assert 1 - d2["rho"] ** 2 == pytest.approx(0.037, abs=5e-4)
    assert d2["factor"] / (1 - d2["rho"] ** 2) == pytest.approx(4.5, abs=0.05)         # der unbekannte Mittelwert kostet diesen Faktor


def test_zusammenhang_schadet_bei_jedem_p(netz, tab):
    c = _mo(netz, tab, 0.9)["models"]["C"]
    assert c["rho"] == pytest.approx(0.199, abs=5e-4) and c["factor"] == pytest.approx(1.083, abs=5e-4)
    for p in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        assert _mo(netz, tab, p)["models"]["C"]["factor"] > 1.0


def test_d2_faktor_gegen_p(netz, tab):
    soll = {0.5: 1.766, 0.7: 0.591, 0.8: 0.321, 0.9: 0.163, 0.95: 0.104, 0.99: 0.0535}
    for p, f in soll.items():
        assert _mo(netz, tab, p)["models"]["D2"]["factor"] == pytest.approx(f, abs=6e-4), p
    assert _mo(netz, tab, 0.5)["cost0"] == pytest.approx(17.8, abs=0.05)


def test_zwei_billige_modelle(netz, tab):
    ch = {r["p"]: r for r in ev.model_choice(S, netz, tab)["rows"]}
    assert ch[0.9]["pair"] == pytest.approx(0.152, abs=5e-4) and ch[0.9]["D2"] == pytest.approx(0.163, abs=5e-4) and ch[0.9]["E"] == pytest.approx(0.221, abs=5e-4)
    assert ch[0.99]["E"] == pytest.approx(0.0445, abs=5e-5) and ch[0.99]["D2"] == pytest.approx(0.0535, abs=5e-5) and ch[0.99]["pair"] == pytest.approx(0.0360, abs=5e-5)
    assert ch[0.99]["E"] < ch[0.99]["D2"]
    assert ch[0.7]["D2"] == pytest.approx(0.591, abs=5e-4) and ch[0.7]["pair"] == pytest.approx(0.698, abs=5e-4) and ch[0.7]["D2"] < ch[0.7]["pair"]
    assert ch[0.7]["best_names"] == ("D2",) and ch[0.9]["best_names"] == ("D2", "E")
    assert ch[0.5]["best_names"] == ()                                                  # bei p = 0,5 lohnt keine Kette


def test_schlaues_ziel(netz, tab):
    t = _mo(netz, tab, 0.9, "tarjan")
    assert t["cost0"] == pytest.approx(33.5, abs=0.05)
    assert t["models"]["D2"]["r"] == pytest.approx(0.508, abs=1e-3) and t["models"]["D2"]["factor"] == pytest.approx(0.794, abs=1e-3)
    names, fac = A.best_chain(ev.infos_of(t))
    assert names == ("E",) and fac == pytest.approx(0.626, abs=1e-3)


def test_aufteilung_exakte_formel_und_gemessen(netz, tab, mo):
    rho, w0, w1 = mo["models"]["D2"]["rho"], mo["cost0"], 17.0
    soll = {1: 1.053, 3: 0.411, 10: 0.198, 24: 0.167, 60: 0.202, 150: 0.358, 400: 0.778}
    for ratio, f in soll.items():
        n0 = max(int(100 * w0 / (w0 + ratio * w1)), 2)
        n1 = max(int(round(ratio * n0)), n0)
        assert A.chain_variance(mo["VarX"], [rho], [n0, n1]) / (mo["VarX"] / 100) == pytest.approx(f, abs=1.5e-3), ratio
    assert A.optimal_ratio(rho, w0, w1) == pytest.approx(23.7, abs=0.05)
    ex = ev.allocation(S, netz, tab)
    gem = {r["ratio"]: r for r in ex["rows"]}
    assert gem[1]["measured"] == pytest.approx(1.0, abs=0.25)                          # nichts gewonnen
    assert gem[24]["measured"] == pytest.approx(0.167, abs=0.06) and min(ex["rows"], key=lambda r: r["measured"])["ratio"] in (10, 24, 60)
    assert gem[400]["n0"] == 5 and gem[400]["measured"] > 0.78 * 0.7                  # Rauschzelle: gemessen 1,26


def test_ausreisser_zelle_mit_vielen_wiederholungen_naehert_sich_der_formel(netz, tab, mo):
    rho = mo["models"]["D2"]["rho"]
    est = A.mfmc_estimates(tab, netz.m, 0.9, (99, 1, 400), 3000, [5, 2000], ("D2",), A.exact_alphas(mo, ("D2",)))
    assert est.var(ddof=1) / (mo["VarX"] / 100) == pytest.approx(0.76, abs=0.3)
    assert A.chain_variance(mo["VarX"], [rho], [5, 2000]) / (mo["VarX"] / 100) == pytest.approx(0.778, abs=2e-3)


def test_kostenverhaeltnis_formel_und_gemessen(netz, tab, mo):
    rho = mo["models"]["D2"]["rho"]
    soll = {0.01: 0.084, 0.05: 0.169, 0.1: 0.252, 0.25: 0.465, 0.5: 0.783, 0.75: 1.084, 1.0: 1.375}
    for r, f in soll.items():
        assert A.factor_two(rho, r) == pytest.approx(f, abs=1.5e-3), r
    assert A.breakeven(rho) == pytest.approx(0.679, abs=5e-4)
    ex = ev.cost_ratio(S, netz, tab)
    for row in ex["rows"]:
        assert row["measured"] == pytest.approx(row["formula"], rel=0.25), row["r"]
    assert next(r for r in ex["rows"] if r["r"] == 0.75)["measured"] > 1.0              # über der Schwelle schlechter als der einfache Mittelwert


def test_gemessen_gegen_formel_im_standardfall(main_a):
    assert main_a.factor_chain == pytest.approx(0.163, abs=5e-4)
    assert main_a.ratio_mf == pytest.approx(0.19, abs=0.04) and main_a.ratio_plug == pytest.approx(0.18, abs=0.04)
    assert main_a.ns == (47, 1113) and main_a.n_plain == 100 and main_a.best_names == ("D2", "E")


def test_budget_sweep_zwischen_015_und_022(netz, tab):
    rows = ev.budget_sweep(S, netz, tab)["rows"]
    gem = [r["ratio_mf"] for r in rows] + [r["ratio_plug"] for r in rows]
    assert min(gem) == pytest.approx(0.15, abs=0.03) and max(gem) == pytest.approx(0.22, abs=0.04)
    assert all(r["formula"] == pytest.approx(0.163, abs=0.015) for r in rows)


def test_verzerrung_des_billigen_modells(netz, tab, mo):
    assert mo["models"]["D2"]["E"] - mo["EX"] == pytest.approx(0.0090, abs=5e-5)
    ex = ev.bias_exp(S, netz, tab)
    cov_lo = [r["cover_lo"] for r in ex["rows"]]
    assert cov_lo[0] == pytest.approx(0.88, abs=0.08) and cov_lo[1] == pytest.approx(0.51, abs=0.12) and cov_lo[2] < 0.05
    assert all(abs(r["bias_mf_se"]) < 3.0 for r in ex["rows"])                         # Mehrmodell: Fehler im Rahmen des Standardfehlers
    assert all(abs(r["bias_mf"]) < 0.005 for r in ex["rows"]) and all(r["bias_lo"] > 0.007 for r in ex["rows"])
    assert [r["cover_mf"] for r in ex["rows"]] == pytest.approx([0.97, 0.98, 0.95], abs=0.05)


def test_parameter_schaetzen(netz, tab):
    ex = ev.params_exp(S, netz, tab)
    rows = {r["label"]: r for r in ex["rows"]}
    assert rows["α aus den geteilten Eingaben"]["factor"] == pytest.approx(rows["exakte Parameter"]["factor"], abs=0.04)
    assert rows["exakte Parameter"]["factor"] == pytest.approx(0.153, abs=0.04)
    assert rows["Vorstudie 20"]["rho_med"] == pytest.approx(1.0, abs=1e-6) and rows["Vorstudie 50"]["rho_med"] == pytest.approx(1.0, abs=1e-6)
    assert rows["Vorstudie 20"]["ratio_med"] == pytest.approx(326, abs=1) and ex["optimum"] == pytest.approx(23.7, abs=0.05)
    for name, f, cap in (("Vorstudie 20", 0.51, 0.29), ("Vorstudie 50", 0.59, 0.26), ("Vorstudie 100", 0.49, 0.22)):
        assert rows[name]["factor"] == pytest.approx(f, abs=0.12), name
        assert rows[name]["factor_cap"] == pytest.approx(cap, abs=0.08), name
        assert rows[name]["factor_cap"] < rows[name]["factor"]


def test_verdikt_des_standardfalls_nennt_die_bessere_kette(main_a):
    kind, text = ev.verdict_main(main_a, S)
    assert kind == "info" and "rund" not in text and "Besser wäre" in text
