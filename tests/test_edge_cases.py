"""Randfälle: eigene Netze, entartete Ziele, Modelle ohne Streuung, Eingabefehler."""

import math

import pytest

import mmd_algorithm as A
import mmd_constants as C
import mmd_evaluation as ev
from mmd_scenario import grid_netz, parse_edges


def _s(**kw):
    return ev.Settings(**kw)


def test_eingabefehler_der_spielwiese():
    with pytest.raises(ValueError, match="Schleife"):
        parse_edges("0-0")
    with pytest.raises(ValueError, match="Keine Kanten"):
        parse_edges("\n# nur Kommentar\n")
    with pytest.raises(ValueError, match="Doppelte"):
        parse_edges("0-1\n1-0")
    with pytest.raises(ValueError, match="erwartet zwei ganze Zahlen"):
        parse_edges("0-a")
    with pytest.raises(ValueError, match="höchstens"):
        parse_edges("\n".join(f"{i}-{i + 1}" for i in range(C.CUSTOM_EDGE_LIMIT + 1)))
    with pytest.raises(ValueError, match="Knoten"):
        parse_edges("\n".join(f"{2 * i}-{2 * i + 1}" for i in range(8)))


def test_netz_ohne_knoten_vom_grad_zwei_laesst_das_eckenmodell_weg():
    netz = parse_edges("0-1\n0-2\n0-3\n1-2\n1-3\n2-3")                      # K4: jeder Knoten hat Grad 3
    assert netz.corner_edges() == ()
    a = ev.analyse(_s(modelle="D2+E"), netz=netz)
    assert a.names == ("D2",)                                              # E ist konstant und fällt weg (ρ nicht definiert)
    assert math.isnan(a.rho["E"])
    only_e = ev.analyse(_s(modelle="E"), netz=netz)
    assert only_e.names == () and ev.verdict_main(only_e, _s(modelle="E"))[0] == "info"


def test_baum_hat_kein_brueckenloses_ziel_und_keine_streuung():
    netz = parse_edges("0-1\n1-2\n2-3")                                     # jede Kante ist eine Brücke: X2 ist immer 0
    a = ev.analyse(_s(), netz=netz)
    assert a.truth == 0.0 and a.var0 == pytest.approx(0.0)
    kind, text = ev.verdict_main(a, _s())
    assert kind == "info" and "keine Streuung" in text
    assert a.best_names == () and a.best_factor == 1.0


def test_dreieck_ist_gleich_d2_und_rho_eins_bricht_nichts():
    netz = parse_edges("0-1\n1-2\n2-0")                                     # X2 = D2 = alle drei Kanten halten
    a = ev.analyse(_s(modelle="D2", budget=30), netz=netz)
    assert a.rho["D2"] == pytest.approx(1.0)
    assert all(math.isfinite(x) for x in (a.ratio_mf, a.ratio_plug, a.bias_mf))
    assert a.ns[1] <= C.RATIO_MAX * a.ns[0] + 1


def test_alle_modelle_laufen_auf_jedem_gitter_ohne_fehler():
    for gitter in C.GRIDS:
        for modelle in C.MODEL_SETS:
            for verfahren in C.VERFAHREN:
                a = ev.analyse(_s(gitter=gitter, modelle=modelle, verfahren=verfahren, budget=30, p=0.7))
                assert math.isfinite(a.truth) and len(a.plain) == C.REPS


@pytest.mark.parametrize("p", [C.P_MIN, C.P_MAX])
def test_reglergrenzen_fuer_p(p):
    a = ev.analyse(_s(p=p))
    assert 0.0 <= a.truth <= 1.0 and a.cost0 > 0


def test_bei_p_ein_halb_ist_das_ziel_fast_nie_wahr_und_d2_kaum_billiger():
    a = ev.analyse(_s(p=0.5))
    assert a.truth < 0.002
    assert a.r["D2"] > 0.9                                                  # das Ziel bricht meist früh ab
    assert a.alone["D2"] > 1.0


def test_kleinstes_budget_und_mindestgroesse():
    ns = A.chain_ns(1, 361.0, [(0.98, 17.0)])
    assert ns[0] >= 2 and ns[1] >= ns[0]


def test_aufzaehlung_ueber_dem_limit_wird_abgelehnt():
    with pytest.raises(ValueError, match="aufgezählt"):
        ev.netz_table(grid_netz(3, 5))
