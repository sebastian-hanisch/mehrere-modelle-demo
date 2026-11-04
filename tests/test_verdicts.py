"""Verdikte (reine Funktionen in mmd_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text."""

import dataclasses

import mmd_evaluation as ev
from mmd_scenario import grid_netz

S = ev.Settings()


def _a(**kw) -> ev.Analysis:
    basis = ev.Analysis(
        netz=grid_netz(3, 4), names=("D2",), truth=0.4, var0=0.24, cost0=360.0, rho={"D2": 0.98, "E": 0.94, "C": 0.2}, r={"D2": 0.047, "E": 0.022, "C": 0.09},
        r_star={"D2": 0.68, "E": 0.5, "C": 0.01}, bias_model={"D2": 0.009, "E": 0.05, "C": 0.5}, alone={"D2": 0.163, "E": 0.22, "C": 1.08}, factor_chain=0.163, valid=True,
        best_names=("D2",), best_factor=0.163, ns=(47, 1113), n_plain=100, ratio_mf=0.19, ratio_plug=0.18, bias_mf=0.0, sd_plain=0.05, sd_mf=0.02,
        plain=(0.4,) * 3, mf=(0.4,) * 3, plug=(0.4,) * 3)
    return dataclasses.replace(basis, **kw)


def test_erfolg_wenn_die_gewaehlte_kette_die_beste_ist():
    kind, text = ev.verdict_main(_a(), S)
    assert kind == "success"
    assert "ρ = 0,980" in text and "r = 0,047" in text and "0,163" in text and "0,19" in text and "0,18" in text and "rund 16 %" in text


def test_hinweis_wenn_eine_bessere_wahl_existiert():
    kind, text = ev.verdict_main(_a(best_names=("D2", "E"), best_factor=0.152), S)
    assert kind == "info" and "Besser wäre Grad ≥ 2 (D2) und Ecken (E)" in text and "0,152" in text


def test_kein_hinweis_bei_gleichem_faktor_innerhalb_der_toleranz():
    kind, _ = ev.verdict_main(_a(best_names=("E",), best_factor=0.160), S)
    assert kind == "success"


def test_warnung_wenn_der_faktor_ueber_eins_liegt():
    kind, text = ev.verdict_main(_a(names=("C",), factor_chain=1.08, valid=False, best_names=("D2",), best_factor=0.163, rho={"C": 0.2}, r={"C": 0.09}), S)
    assert kind == "warning" and "weggelassen" in text and "1,08" in text


def test_verdraengte_kette_mit_faktor_unter_eins():
    kind, text = ev.verdict_main(_a(valid=False, factor_chain=0.9), S)
    assert kind == "info" and "verdrängt" in text


def test_keine_streuung_und_kein_modell():
    assert "keine Streuung" in ev.verdict_main(_a(var0=0.0), S)[1]
    assert "nicht definiert" in ev.verdict_main(_a(names=()), S)[1]


def test_verdikt_nennt_das_verfahren_des_ziels():
    _, text = ev.verdict_main(_a(), ev.Settings(verfahren="tarjan"))
    assert "Brückensuche in einem Lauf (Tarjan)" in text


def test_budget_verdikt_zahlen():
    exp = {"rows": [{"budget": 30, "ratio_mf": 0.2, "ratio_plug": 0.25}, {"budget": 100, "ratio_mf": 0.15, "ratio_plug": 0.17}], "factor_chain": 0.163}
    kind, text = ev.verdict_budget(exp)
    assert kind == "info" and "0,163" in text and "zwischen 0,15 und 0,20" in text and "zwischen 0,17 und 0,25" in text and "rund 12 %" in text


def test_aufteilungs_verdikt_zahlen():
    rows = [{"ratio": 1, "measured": 1.05, "formula": 1.05, "n0": 95}, {"ratio": 24, "measured": 0.18, "formula": 0.17, "n0": 46}, {"ratio": 400, "measured": 0.64, "formula": 0.78, "n0": 5}]
    kind, text = ev.verdict_allocation({"rows": rows, "optimum": 23.7})
    assert kind == "info" and "1,05" in text and "N1/N0 = 24" in text and "gemessen am besten bei 24 mit 0,18" in text and "400 billige" in text and "nur 5 teure Ziehungen" in text and "nach der Formel 0,78, gemessen 0,64" in text and "rauscht" in text


def test_kosten_verdikt_nennt_die_erste_stelle_ueber_eins():
    rows = [{"r": 0.5, "formula": 0.78, "measured": 0.76}, {"r": 0.75, "formula": 1.08, "measured": 1.06}]
    _, text = ev.verdict_cost({"rows": rows, "rho": 0.98, "r_star": 0.68, "r_real": 0.047})
    assert "r* = 0,68" in text and "68 %" in text and "r = 0,047" in text and "Ab r = 0,75" in text and "1,08" in text and "1,06" in text
    _, text = ev.verdict_cost({"rows": rows[:1], "rho": 0.98, "r_star": 0.68, "r_real": 0.047})
    assert "überall nützlich" in text


def test_auswahl_verdikt_zahlen():
    base = {"cost0": 360.0, "C": 1.08, "pair": 0.15, "pair_valid": True, "rho": {}}
    rows = [{"p": 0.5, "D2": 1.77, "E": 1.42, "best": 1.0, "best_names": (), **base}, {"p": 0.9, "D2": 0.163, "E": 0.22, "best": 0.152, "best_names": ("D2", "E"), **base},
            {"p": 0.99, "D2": 0.0535, "E": 0.0452, "best": 0.036, "best_names": ("D2", "E"), **base}]
    _, text = ev.verdict_choice({"rows": rows})
    assert "Grad ≥ 2 (D2) und Ecken (E) (Faktor 0,152)" in text and "1,08" in text and "Ecken (E) allein am besten (0,045)" in text and "1,42 für das beste" in text


def test_verzerrungs_verdikt_zahlen():
    rows = [{"budget": 100, "cover_lo": 0.88, "bias_mf": 0.0, "bias_mf_se": 0.0, "sd_mf": 0.02}, {"budget": 5000, "cover_lo": 0.01, "bias_mf": -0.0002, "bias_mf_se": -0.7, "sd_mf": 0.0029}]
    kind, text = ev.verdict_bias({"rows": rows, "model_bias": 0.009})
    assert kind == "warning" and "0,0090" in text and "88 %" in text and "1 %" in text and "−0,0002" in text and "−0,7 Standardfehler" in text and "0,0029" in text


def test_parameter_verdikt_zahlen():
    rows = [{"label": "exakt", "factor": 0.15}, {"label": "plug", "factor": 0.16},
            {"label": "v20", "pilot": 20, "ratio_med": 326.0, "factor": 0.5, "factor_cap": 0.25}, {"label": "v50", "pilot": 50, "ratio_med": 326.0, "factor": 0.6, "factor_cap": 0.26}]
    kind, text = ev.verdict_params({"rows": rows, "optimum": 23.7})
    assert kind == "warning" and "0,16 gegen 0,15" in text and "Median 326 statt 24" in text and "0,50, 0,60" in text and "0,25, 0,26" in text


def test_leere_experimente_sind_hinweise_und_keine_fehler():
    for fn in (ev.verdict_budget, ev.verdict_allocation, ev.verdict_cost, ev.verdict_bias, ev.verdict_params):
        assert fn({"rows": []})[0] == "info"
    assert ev.verdict_choice({"rows": []})[0] == "info"


def test_deutsche_zahlen():
    assert ev.de(1234.567, 1) == "1.234,6" and ev.de(-0.5, 2) == "−0,50" and ev.de_e(0.000042) == "4,2e-05"
