"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mmd_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_budget", "Die Formel gibt bei bester Aufteilung den Faktor 0,163"),
    ("run_alloc", "Gleich viele Ziehungen für beide Modelle (N1/N0 = 1) bringen"),
    ("run_cost", "das billige Modell darf höchstens 68 % der Kosten des Ziels kosten"),
    ("run_bias", "mehr Ziehungen machen den falschen Wert nur sicherer"),
    ("run_params", "Die Aufteilung aus einer Vorstudie ist heikel"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_wahl_experiment_steht_ohne_klick_da():
    assert "der Zusammenhang (C), das Ziel der vorigen Stücke" in _texte(_app())


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_alloc").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="budget_select").set_value(300).run(timeout=TIMEOUT)
    assert not at.exception
    assert "Gleich viele Ziehungen für beide Modelle" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    assert "Besser wäre Grad ≥ 2 (D2) und Ecken (E)" in _texte(at)
    at.sidebar.button(key="preset_Die naheliegende Wahl").click().run(timeout=TIMEOUT)
    assert "weggelassen" in _texte(at)
    at.sidebar.button(key="preset_Zwei billige Modelle").click().run(timeout=TIMEOUT)
    assert "Die Varianz sinkt auf rund" in _texte(at)
    at.sidebar.button(key="preset_Schlaues Ziel, kaum Gewinn").click().run(timeout=TIMEOUT)
    assert "Brückensuche in einem Lauf (Tarjan)" in _texte(at)


def test_jede_modellwahl_und_jedes_verfahren_laeuft_durch():
    at = _app()
    for modelle in C.MODEL_SETS:
        at.sidebar.selectbox(key="modelle_select").set_value(modelle).run(timeout=TIMEOUT)
        assert not at.exception, (modelle, [str(e) for e in at.exception])
    for verfahren in C.VERFAHREN:
        at.sidebar.selectbox(key="verfahren_select").set_value(verfahren).run(timeout=TIMEOUT)
        assert not at.exception, (verfahren, [str(e) for e in at.exception])


def test_spielwiese_gueltiges_netz_und_fehlermeldungen():
    at = _app()
    assert not at.error
    assert "Exakte Kennzahlen bei diesem p" in " ".join(m.value for m in at.markdown)
    at.text_area(key="eigene_kanten").set_value("0-0").run(timeout=TIMEOUT)
    assert not at.exception and "Schleife" in " ".join(e.value for e in at.error)
    at.text_area(key="eigene_kanten").set_value("").run(timeout=TIMEOUT)
    assert not at.exception and "Keine Kanten" in " ".join(e.value for e in at.error)


def test_spielwiese_baum_hat_keine_streuung():
    at = _app()
    at.text_area(key="eigene_kanten").set_value("0-1\n1-2\n2-3").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "keine Streuung" in _texte(at)


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="budget_select").set_value(300).run(timeout=TIMEOUT)     # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["p"] = "5"
    at.query_params["b"] = "2400"
    at.query_params["g"] = "9x9"            # unbekanntes Gitter: verworfen
    at.query_params["m"] = "quatsch"        # unbekannte Modellwahl: verworfen
    at.query_params["v"] = "tarjan"
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="p_slider").value == C.P_MAX
    assert at.sidebar.select_slider(key="budget_select").value == 1000                   # 2400 liegt näher an 1000 als an 300
    assert at.sidebar.selectbox(key="gitter_select").value == C.DEFAULT_GRID
    assert at.sidebar.selectbox(key="modelle_select").value == C.DEFAULT_MODELS
    assert at.sidebar.selectbox(key="verfahren_select").value == "tarjan"
    assert at.sidebar.number_input(key="seed_input").value == 0
