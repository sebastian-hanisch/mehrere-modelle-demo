"""Kern von Hand geprüft: Zufallsstrom, Aufzählung der Leiter 2×4, Formeln an kleinen Zahlen, Aufteilung."""

from fractions import Fraction

import numpy as np
import pytest

import mmd_algorithm as A
import mmd_constants as C


def test_uniforms_gegen_die_ganzzahl_referenz():
    seed = A.stream_seed(3, 7)
    soll = [(A.mix_int((seed + i * A.GAMMA) & A.MASK) >> 11) * (1.0 / (1 << 53)) for i in range(1, 6)]
    assert A.uniforms(seed, 5).tolist() == soll


def test_ausgabe_i_haengt_nur_von_seed_und_i_ab():
    assert A.uniforms(5, 100)[:10].tolist() == A.uniforms(5, 10).tolist()


def test_masken_bitpackung_von_hand():
    # Kante e hält genau dann, wenn Zufallszahl Nr. i*m + e unter p liegt; Bit e der Maske
    m, p = 4, 0.5
    u = A.uniforms(11, 3 * m).reshape(3, m)
    soll = [sum(int(u[i, e] < p) << e for e in range(m)) for i in range(3)]
    assert A.draw_masks(11, 3, p, m).tolist() == soll


def test_leiter_zaehlungen_von_hand(leiter):
    netz, t = leiter
    assert (int(t["X2"].sum()), int(t["D2"].sum()), int(t["C"].sum())) == (4, 7, 105)
    assert np.all(t["X2"] <= t["C"]) and np.all(t["X2"] <= t["D2"]) and np.all(t["X2"] <= t["E"])


def test_leiter_exakte_momente_bei_p_ein_halb_mit_bruechen(leiter):
    netz, t = leiter
    mo = A.exact_moments(t, netz.m, netz.corner_edges(), 0.5, "naiv")
    a, b = Fraction(4, 1024), Fraction(7, 1024)
    assert mo["EX"] == pytest.approx(float(a))
    assert mo["VarX"] == pytest.approx(float(a * (1 - a)))
    # X2 ⊂ D2: die Kovarianz ist P(X2) - P(X2) P(D2)
    rho2 = (a - a * b) ** 2 / (a * (1 - a) * b * (1 - b))
    assert mo["models"]["D2"]["rho"] ** 2 == pytest.approx(float(rho2))
    assert mo["models"]["D2"]["E"] == pytest.approx(float(b))
    assert mo["models"]["C"]["E"] == pytest.approx(105 / 1024)


def test_kosten_von_d2_und_e_von_hand(leiter):
    netz, t = leiter
    ci = netz.corner_edges()
    assert A.model_cost(t, netz.m, ci, "D2", "naiv", 0.5) == 10                       # jede Kante einmal
    assert len(ci) == 6 and A.model_cost(t, netz.m, ci, "E", "naiv", 0.5) == 6        # Leiter: vier Ecken mit Grad 2, Kanten t0t1, t0b0, b0b1, t3t2, t3b3, b3b2


def test_faktor_und_schwelle_von_hand():
    assert A.factor_two(0.8, 0.04) == pytest.approx((0.6 + 0.8 * 0.2) ** 2)
    assert A.factor_two(0.8, 0.0) == pytest.approx(0.36)                              # r -> 0: 1 - rho^2 (Kontrollvariate mit bekanntem Mittelwert)
    assert A.breakeven(0.6) == pytest.approx(((1 - 0.8) / 0.6) ** 2)
    assert A.factor_two(0.6, A.breakeven(0.6)) == pytest.approx(1.0)
    assert A.factor_two(0.6, 0.2) > 1.0 > A.factor_two(0.6, 0.05)


def test_optimale_aufteilung_von_hand():
    assert A.optimal_ratio(0.8, 100.0, 1.0) == pytest.approx(np.sqrt(100 * 0.64 / 0.36))
    assert A.optimal_ratio(0.1, 100.0, 90.0) == 1.0                                   # nie weniger billige als teure Ziehungen
    assert A.optimal_ratio(1.0, 100.0, 1.0) == C.RATIO_MAX


def test_stichprobengroessen_verbrauchen_das_budget():
    # w0 = 100, w1 = 1, rho = 0.8: Verhältnis 13,333; Budget 50 teure Auswertungen = 5000 Einheiten: N0 (100 + 13,33 * 1) = 5000 -> N0 = 44
    ns = A.chain_ns(50, 100.0, [(0.8, 1.0)])
    assert ns == [44, 587]
    assert ns[0] * 100 + ns[1] * 1 <= 5000


def test_kettenvarianz_gegen_die_einmodellformel():
    v = A.chain_variance(1.0, [0.8], [40, 400])
    assert v == pytest.approx(1 / 40 - (1 / 40 - 1 / 400) * 0.64)
    assert A.chain_variance(1.0, [0.8], [40, 40]) == pytest.approx(1 / 40)            # N1 = N0: nichts gewonnen


def test_kettenfaktor_einmodell_gleich_zweimodellformel():
    for rho, r in ((0.9, 0.05), (0.5, 0.01), (0.99, 0.2)):
        assert A.chain_factor(100.0, [(rho, 100.0 * r)]) == pytest.approx(A.factor_two(rho, r))
    assert A.chain_factor(100.0, []) == 1.0


def test_verdraengtes_modell_ist_ungueltig():
    w0 = 100.0
    assert not A.chain_valid(w0, [(0.2, 30.0)])                                       # kaum korreliert und nicht billig genug: Aufteilung < 1
    assert not A.chain_valid(w0, [(0.9, 20.0), (0.95, 10.0)])                         # rho steigt statt zu fallen
    assert not A.chain_valid(w0, [(0.9, 20.0), (0.5, 30.0)])                          # zweites Modell teurer als das erste
    assert A.chain_valid(w0, [(0.95, 20.0), (0.8, 5.0)])


def test_beste_kette_waehlt_den_kleinsten_gueltigen_faktor():
    infos = {"D2": (0.98, 17.0), "E": (0.94, 8.0), "C": (0.2, 33.0), "_w0": 361.0}
    names, fac = A.best_chain(infos)
    gueltig = [(o, f) for o, f, ok in A.chains(infos) if ok]
    assert fac == min(f for _, f in gueltig) and names == min(gueltig, key=lambda x: x[1])[0]
    assert A.best_chain({"C": (0.2, 33.0), "_w0": 361.0}) == ((), 1.0)                # nichts lohnt: nur das Ziel
