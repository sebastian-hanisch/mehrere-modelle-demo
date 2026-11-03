"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import mmd_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_spread(a):
    """Verteilung der Schätzungen aus REPS Wiederholungen bei gleichem Budget: einfach, Mehrmodell mit exaktem α, Mehrmodell mit geschätztem α; gestrichelt die Wahrheit."""
    fig = go.Figure()
    for name, series, color in (("einfach", a.plain, C.COLOR_PLAIN), ("Mehrmodell, exaktes α", a.mf, C.COLOR_MF), ("Mehrmodell, geschätztes α", a.plug, C.COLOR_PLUG)):
        fig.add_trace(go.Box(y=list(series), name=name, marker_color=color, boxmean="sd", boxpoints="outliers"))
    fig.add_hline(y=a.truth, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Wahrheit", annotation_position="top right")
    fig.update_yaxes(title_text="Schätzung von E X2")
    fig = _base(fig, 340)
    fig.update_layout(showlegend=False)
    return fig


def build_budget(exp):
    rows = exp["rows"]
    fig = go.Figure()
    bs = [str(r["budget"]) for r in rows]
    fig.add_trace(go.Scatter(x=bs, y=[r["ratio_mf"] for r in rows], mode="markers+lines", name="exaktes α", line=dict(color=C.COLOR_MF)))
    fig.add_trace(go.Scatter(x=bs, y=[r["ratio_plug"] for r in rows], mode="markers+lines", name="geschätztes α", line=dict(color=C.COLOR_PLUG)))
    fig.add_hline(y=exp["factor_chain"], line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Faktor der Formel", annotation_position="top right")
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_OTHER, dash="dot"), annotation_text="einfacher Mittelwert", annotation_position="top left")
    fig.update_xaxes(title_text="Budget (teure Auswertungen)", type="category")
    fig.update_yaxes(title_text="Varianz Mehrmodell / einfach")
    return _base(fig, 330)


def build_allocation(exp):
    rows = exp["rows"]
    fig = go.Figure()
    xs = [r["ratio"] for r in rows]
    fig.add_trace(go.Scatter(x=xs, y=[r["measured"] for r in rows], mode="markers+lines", name="gemessen", line=dict(color=C.COLOR_MF)))
    fig.add_trace(go.Scatter(x=xs, y=[r["formula"] for r in rows], mode="lines", name="Formel", line=dict(color=C.COLOR_TRUTH, dash="dash")))
    fig.add_vline(x=exp["optimum"], line=dict(color=C.COLOR_OTHER, dash="dot"), annotation_text="Optimum", annotation_position="top right")
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_OTHER, dash="dot"))
    fig.update_xaxes(title_text="billige je teurer Ziehung N1/N0", type="log")
    fig.update_yaxes(title_text="Varianz Mehrmodell / einfach")
    return _base(fig, 330)


def build_cost(exp):
    rows = exp["rows"]
    fig = go.Figure()
    xs = [r["r"] for r in rows]
    fig.add_trace(go.Scatter(x=xs, y=[r["measured"] for r in rows], mode="markers+lines", name="gemessen", line=dict(color=C.COLOR_MF)))
    fig.add_trace(go.Scatter(x=xs, y=[r["formula"] for r in rows], mode="lines", name="Formel", line=dict(color=C.COLOR_TRUTH, dash="dash")))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_BAD, dash="dot"), annotation_text="einfacher Mittelwert", annotation_position="bottom right")
    fig.add_vline(x=exp["r_star"], line=dict(color=C.COLOR_OTHER, dash="dot"), annotation_text="Schwelle r*", annotation_position="top left")
    fig.update_xaxes(title_text="Kostenverhältnis r (billig / teuer, Annahme)")
    fig.update_yaxes(title_text="Varianz Mehrmodell / einfach")
    return _base(fig, 330)


def build_choice(exp, p_now):
    rows = exp["rows"]
    fig = go.Figure()
    ps = [r["p"] for r in rows]
    for key, color, dash, label in (("C", C.COLOR_BAD, "solid", C.MODEL_LABELS["C"]), ("D2", C.COLOR_MF, "solid", C.MODEL_LABELS["D2"]), ("E", C.COLOR_PLAIN, "solid", C.MODEL_LABELS["E"]),
                                    ("best", C.COLOR_PLUG, "dash", "beste Wahl")):
        fig.add_trace(go.Scatter(x=ps, y=[r[key] for r in rows], mode="markers+lines", name=label, line=dict(color=color, dash=dash)))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_OTHER, dash="dot"))
    fig.add_vline(x=p_now, line=dict(color=C.COLOR_OTHER, dash="dot"), annotation_text="Regler p", annotation_position="bottom right")
    fig.update_xaxes(title_text="Haltewahrscheinlichkeit p")
    fig.update_yaxes(title_text="exakter Varianzfaktor", type="log")
    return _base(fig, 340)


def build_bias(exp):
    rows = exp["rows"]
    fig = go.Figure()
    bs = [str(r["budget"]) for r in rows]
    fig.add_trace(go.Scatter(x=bs, y=[r["cover_lo"] for r in rows], mode="markers+lines", name="nur das billige Modell", line=dict(color=C.COLOR_BAD)))
    fig.add_trace(go.Scatter(x=bs, y=[r["cover_mf"] for r in rows], mode="markers+lines", name="Mehrmodell", line=dict(color=C.COLOR_MF)))
    fig.add_hline(y=0.95, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Soll 0,95", annotation_position="bottom right")
    fig.update_xaxes(title_text="Budget (teure Auswertungen)", type="category")
    fig.update_yaxes(title_text="Überdeckung des 95-%-Intervalls", range=[0, 1.02])
    return _base(fig, 320)


def build_params(exp):
    rows = exp["rows"]
    fig = go.Figure()
    labels = [r["label"] for r in rows]
    fig.add_trace(go.Bar(x=labels, y=[r["factor"] for r in rows], name="Aufteilung wie geschätzt", marker_color=C.COLOR_PLUG))
    fig.add_trace(go.Bar(x=labels[2:], y=[r["factor_cap"] for r in rows[2:]], name="Verhältnis auf 3 bis 100 begrenzt", marker_color=C.COLOR_MF))
    fig.add_hline(y=exp["ideal_factor"], line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Faktor der Formel", annotation_position="top right")
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_OTHER, dash="dot"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Varianz Mehrmodell / einfach")
    return _base(fig, 330)


def build_netz(netz):
    fig = go.Figure()
    xs, ys = [c[0] for c in netz.coords], [c[1] for c in netz.coords]
    ex, ey = [], []
    for u, v in netz.edges:
        ex += [xs[u], xs[v], None]
        ey += [ys[u], ys[v], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#666", width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=22, color="white", line=dict(color="#222", width=2)), showlegend=False, hoverinfo="skip"))
    for i, (x, y) in enumerate(zip(xs, ys)):
        fig.add_annotation(x=x, y=y, text=str(i), showarrow=False, font=dict(size=10, color="#222"))
    pad = 0.5
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    fig = _base(fig, 280)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig
