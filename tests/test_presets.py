"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink; jedes Preset läuft durch."""

import pytest

import mmd_constants as C
import mmd_evaluation as ev
import mmd_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert C.P_MIN <= preset["p"] <= C.P_MAX and 0 <= preset["seed"] <= C.SEED_MAX, name
        assert preset["budget"] in C.BUDGET_CHOICES and preset["gitter"] in C.GRIDS and preset["modelle"] in C.MODEL_SETS and preset["verfahren"] in C.VERFAHREN, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_standard_in_mindestens_einem_wert():
    standard = C.PRESETS["Standardfall (Voreinstellung)"]
    for name, preset in C.PRESETS.items():
        if name != "Standardfall (Voreinstellung)":
            assert preset != standard, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_rechnet_und_hat_das_erwartete_verdikt(name):
    s = ev.Settings(**C.PRESETS[name])
    a = ev.analyse(s)
    kind, _ = ev.verdict_main(a, s)
    erwartet = {"Standardfall (Voreinstellung)": "info", "Die naheliegende Wahl": "warning", "Schlaues Ziel, kaum Gewinn": {"info", "success"}, "Zwei billige Modelle": "success"}[name]
    assert kind in (erwartet if isinstance(erwartet, set) else {erwartet})


def test_preset_naheliegende_wahl_ist_der_zusammenhang_mit_faktor_ueber_eins():
    a = ev.analyse(ev.Settings(**C.PRESETS["Die naheliegende Wahl"]))
    assert a.names == ("C",) and a.factor_chain > 1.0 and a.ratio_mf > 1.0


def test_preset_schlaues_ziel_hat_einen_viel_kleineren_gewinn():
    a = ev.analyse(ev.Settings(**C.PRESETS["Schlaues Ziel, kaum Gewinn"]))
    b = ev.analyse(ev.Settings())
    assert a.factor_chain > 0.6 > b.factor_chain and a.factor_chain < 1.0


def test_preset_zwei_modelle_wird_von_der_kette_tatsaechlich_genutzt():
    a = ev.analyse(ev.Settings(**C.PRESETS["Zwei billige Modelle"]))
    assert a.names == ("D2", "E") and a.valid and len(a.ns) == 3
    einzeln = ev.analyse(ev.Settings(**{**C.PRESETS["Zwei billige Modelle"], "modelle": "D2"}))
    assert a.factor_chain < einzeln.factor_chain
