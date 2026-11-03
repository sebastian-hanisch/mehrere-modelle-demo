"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für jede zulässige Eingabe gelten müssen."""

import numpy as np
import pytest

import mmd_algorithm as A
import mmd_evaluation as ev
from mmd_scenario import grid_netz


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9, 0.99])
def test_faktor_unter_eins_genau_unter_der_schwelle(rho):
    rstar = A.breakeven(rho)
    assert A.factor_two(rho, rstar * 0.9) < 1.0 < A.factor_two(rho, rstar * 1.1)
    assert A.factor_two(rho, 0.0) == pytest.approx(1 - rho**2)


def test_faktor_waechst_mit_dem_kostenverhaeltnis_und_faellt_mit_rho():
    rs = np.linspace(0.0, 1.0, 21)
    assert all(A.factor_two(0.9, a) < A.factor_two(0.9, b) for a, b in zip(rs, rs[1:]))
    # In rho ist der Faktor nicht monoton: ein kaum korreliertes Modell schadet noch mehr (Ableitung bei rho -> 0 ist +sqrt(r)); er fällt erst ab rho* = sqrt(r / (1 + r))
    r = 0.1
    rho_star = np.sqrt(r / (1 + r))
    unten = np.linspace(0.01, rho_star * 0.95, 10)
    oben = np.linspace(rho_star * 1.05, 0.99, 20)
    assert all(A.factor_two(a, r) < A.factor_two(b, r) for a, b in zip(unten, unten[1:]))
    assert all(A.factor_two(a, r) > A.factor_two(b, r) for a, b in zip(oben, oben[1:]))
    assert A.factor_two(0.05, r) > 1.0


@pytest.mark.parametrize("budget", [30, 100, 1000])
def test_aufteilung_nie_ueber_dem_budget_und_nie_mit_weniger_billigen(budget):
    for chain in ([(0.98, 17.0)], [(0.98, 17.0), (0.94, 8.0)], [(0.2, 33.0)]):
        ns = A.chain_ns(budget, 361.0, chain)
        assert all(a <= b for a, b in zip(ns, ns[1:]))
        kosten = ns[0] * 361.0 + sum(n * c[1] for n, c in zip(ns[1:], chain))
        assert kosten <= budget * 361.0 * 1.01 + 2 * 361.0                 # Rundung und die Untergrenze N0 >= 2


def test_gleich_viele_ziehungen_geben_den_einfachen_mittelwert():
    f0 = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    f1 = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    assert A.mfmc_value(f0, [f1], [4, 4], [0.7]) == pytest.approx(f0[:4].mean())


def test_gleiche_staerke_gleiche_schaetzung_und_anderer_strom_andere():
    netz = grid_netz(2, 4)
    t = ev.netz_table(netz)
    e1 = A.plain_estimates(t, netz.m, 0.8, (1, 2), 20, 50)
    assert e1.tolist() == A.plain_estimates(t, netz.m, 0.8, (1, 2), 20, 50).tolist()
    assert e1.tolist() != A.plain_estimates(t, netz.m, 0.8, (1, 3), 20, 50).tolist()


def test_mehrmodellschaetzer_ist_erwartungstreu_obwohl_das_billige_modell_verzerrt_ist(gitter34):
    netz, t = gitter34
    mo = A.exact_moments(t, netz.m, netz.corner_edges(), 0.9, "naiv")
    assert abs(mo["models"]["D2"]["E"] - mo["EX"]) > 0.008                    # verzerrt um rund 0,009
    ns = A.chain_ns(60, mo["cost0"], [(mo["models"]["D2"]["rho"], mo["models"]["D2"]["cost"])])
    est = A.mfmc_estimates(t, netz.m, 0.9, (7, 1), 800, ns, ("D2",), A.exact_alphas(mo, ("D2",)))
    z = (est.mean() - mo["EX"]) / (est.std(ddof=1) / np.sqrt(len(est)))
    assert abs(z) < 4.0
    assert abs(A.cheap_estimates(t, netz.m, 0.9, (7, 2), 800, ns[-1], "D2").mean() - mo["EX"]) > 0.005   # nur das billige Modell: verzerrt


@pytest.mark.parametrize("p", [0.8, 0.9, 0.95])
def test_gewaehltes_modell_d2_senkt_die_streuung_bei_gleichem_budget(p):
    s = ev.Settings(p=p, budget=100)
    a = ev.analyse(s)
    assert a.valid and a.factor_chain < 0.5
    assert a.ratio_mf < 0.5 and a.ratio_plug < 0.5


def test_d2_faktor_faellt_mit_p_bei_festem_netz(gitter34):
    netz, t = gitter34
    faktoren = [A.exact_moments(t, netz.m, netz.corner_edges(), p, "naiv")["models"]["D2"]["factor"] for p in (0.7, 0.8, 0.9, 0.95, 0.99)]
    assert all(a > b for a, b in zip(faktoren, faktoren[1:]))


def test_zusammenhang_als_billiges_modell_schadet_bei_jedem_p(gitter34):
    netz, t = gitter34
    for p in (0.7, 0.8, 0.9, 0.95, 0.99):
        mo = A.exact_moments(t, netz.m, netz.corner_edges(), p, "naiv")
        assert mo["models"]["C"]["factor"] > 1.0
        names, fac = A.best_chain(ev.infos_of(mo))
        assert "C" not in names


def test_brueckensuche_in_einem_lauf_kostet_so_viel_wie_das_modell_c(gitter34):
    netz, t = gitter34
    mo = A.exact_moments(t, netz.m, netz.corner_edges(), 0.9, "tarjan")
    assert mo["cost0"] == pytest.approx(mo["models"]["C"]["cost"])
    assert mo["models"]["C"]["r"] == pytest.approx(1.0)
    naiv = A.exact_moments(t, netz.m, netz.corner_edges(), 0.9, "naiv")
    assert naiv["cost0"] > 5 * mo["cost0"]
    assert mo["models"]["D2"]["factor"] > 3 * naiv["models"]["D2"]["factor"]   # das schlauere Ziel nimmt dem billigen Modell den Gewinn


def test_modelle_irren_nur_nach_oben_auf_mehreren_gittern():
    for rows, cols in ((2, 4), (3, 3), (2, 5)):
        t = ev.netz_table(grid_netz(rows, cols))
        assert np.all(t["X2"] <= t["C"]) and np.all(t["X2"] <= t["D2"]) and np.all(t["X2"] <= t["E"])
