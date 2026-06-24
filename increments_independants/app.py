# -*- coding: utf-8 -*-
"""
app.py — Interface Streamlit du diagnostic d'INCRÉMENTS INDÉPENDANTS (iid).

Teste l'hypothèse « log(prix) est un processus de Lévy » — socle du pricer CPPI
par opérateur de Markov (Paulot–Lacroze). Cadre : validation de modèle SR 11-7.

Lancer :  streamlit run app.py
Déploiement Streamlit Community Cloud : placer les .xlsx à côté de ce fichier
(chemins RELATIFS) ou utiliser l'upload. Aucun chemin absolu.
"""
import os
import io
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import increments as inc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Diagnostic iid — MASI / CPPI Markov",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Style : badges de verdict, encadrés H0/H1, seuil, limites
# ----------------------------------------------------------------------------
VERDICT_COLORS = {
    inc.V_OK:     ("#0b6b3a", "#e3f5ea"),
    inc.V_MAT:    ("#a01313", "#fbe4e4"),
    inc.V_NONMAT: ("#9a5b00", "#fdeede"),
    inc.V_EXT:    ("#1f4e9c", "#e5edfb"),
    inc.V_IND:    ("#5a4a00", "#f7f1d8"),
    inc.V_NT:     ("#555555", "#ededed"),
    inc.V_CHAR:   ("#5c2e91", "#efe6f8"),
}

st.markdown("""
<style>
.badge {display:inline-block; padding:4px 12px; border-radius:14px;
        font-weight:700; font-size:0.86rem; letter-spacing:.2px;}
.box {border-left:4px solid #ccc; padding:.5rem .9rem; margin:.4rem 0;
      background:#fafafa; border-radius:4px; font-size:.92rem;}
.box-h {border-left-color:#5a6b8c; background:#f3f6fb;}
.box-seuil {border-left-color:#9a5b00; background:#fdf6ec;}
.box-mean {border-left-color:#1f4e9c; background:#f1f5fc;}
.box-lim {border-left-color:#999; background:#f4f4f4; color:#444; font-size:.86rem;}
.metricline {font-size:.95rem; color:#222;}
h3 {margin-top:.3rem;}
small.dim {color:#777;}
</style>
""", unsafe_allow_html=True)


def badge(verdict: str) -> str:
    fg, bg = VERDICT_COLORS.get(verdict, ("#333", "#eee"))
    return f'<span class="badge" style="color:{fg};background:{bg};">{verdict}</span>'


# ----------------------------------------------------------------------------
# Chargement + calculs (mis en cache)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_path(path: str) -> pd.Series:
    return inc.load_series(path)


@st.cache_data(show_spinner=False)
def load_bytes(name: str, data: bytes) -> pd.Series:
    bio = io.BytesIO(data)
    bio.name = name
    return inc.load_series(bio)


@st.cache_data(show_spinner=False)
def run_cached(series_id: str, _series: pd.Series, kind: str, freq: str, p_dict: dict):
    p = inc.Params(**p_dict)
    return inc.run_all_tests(_series, kind, freq, p)


@st.cache_data(show_spinner=False)
def scan_cached(series_id: str, _series: pd.Series, kind: str, p_dict: dict):
    return inc.frequency_scan(_series, kind, inc.Params(**p_dict))


# ----------------------------------------------------------------------------
# Graphiques (plotly) — couche interface
# ----------------------------------------------------------------------------
PAL = dict(line="#1f4e9c", band="rgba(31,78,156,.15)", ref="#a01313",
           pos="#0b6b3a", neg="#a01313", grid="#e8e8e8")


def _layout(fig, title="", h=320, xt="", yt=""):
    fig.update_layout(title=title, height=h, margin=dict(l=10, r=10, t=40, b=10),
                      template="simple_white", xaxis_title=xt, yaxis_title=yt,
                      font=dict(size=12))
    fig.update_xaxes(gridcolor=PAL["grid"]); fig.update_yaxes(gridcolor=PAL["grid"])
    return fig


def plot_acf(acf_vals, n, title="ACF des rendements"):
    lags = list(range(len(acf_vals)))
    ci = 1.96 / np.sqrt(n)
    fig = go.Figure()
    fig.add_bar(x=lags[1:], y=acf_vals[1:], marker_color=PAL["line"], name="ACF")
    fig.add_hline(y=ci, line_dash="dot", line_color=PAL["ref"])
    fig.add_hline(y=-ci, line_dash="dot", line_color=PAL["ref"])
    fig.add_hline(y=0, line_color="#888")
    return _layout(fig, title, xt="retard k", yt="autocorrélation")


def plot_vr(per_q):
    qs = [d["q"] for d in per_q]
    vr = [d["vr"] for d in per_q]
    lo = [d["surr_band"][0] for d in per_q]
    hi = [d["surr_band"][1] for d in per_q]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qs + qs[::-1], y=hi + lo[::-1], fill="toself",
                             fillcolor=PAL["band"], line=dict(width=0),
                             name="bande iid (surrogate 95%)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=qs, y=vr, mode="lines+markers", line=dict(color=PAL["line"]),
                             marker=dict(size=9), name="VR(q) observé"))
    fig.add_hline(y=1.0, line_dash="dash", line_color=PAL["ref"],
                  annotation_text="VR=1 (marche aléatoire)")
    return _layout(fig, "Variance Ratio Lo–MacKinlay vs horizon", xt="horizon q", yt="VR(q)")


def plot_bds(dims, stat, pval, alpha):
    colors = [PAL["neg"] if p < alpha else PAL["line"] for p in pval]
    fig = go.Figure()
    fig.add_bar(x=[f"m={m}" for m in dims], y=stat, marker_color=colors,
                text=[f"p={p:.3f}" for p in pval], textposition="outside")
    fig.add_hline(y=1.96, line_dash="dot", line_color=PAL["ref"])
    fig.add_hline(y=-1.96, line_dash="dot", line_color=PAL["ref"])
    return _layout(fig, "BDS par dimension de plongement (rouge = rejet iid)",
                   yt="statistique BDS (σ)")


def plot_runs(d):
    fig = go.Figure()
    fig.add_bar(x=["observé", "attendu (H0)"], y=[d["runs"], d["expected"]],
                marker_color=[PAL["line"], "#9aa7bd"],
                text=[f"{d['runs']:.0f}", f"{d['expected']:.1f}"], textposition="outside")
    return _layout(fig, "Nombre de runs : observé vs attendu sous indépendance",
                   yt="nombre de runs", h=300)


def plot_abs_acf(acf_abs, n):
    return plot_acf(acf_abs, n, title="ACF des |rendements| (mémoire de volatilité)")


def plot_price(prices, logp=True, title="log-prix"):
    y = np.log(prices.values) if logp else prices.values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=y, mode="lines", line=dict(color=PAL["line"])))
    return _layout(fig, title, xt="date", yt=("log(prix)" if logp else "prix"), h=300)


def plot_loglog(x, y, slope, intercept, xlab, ylab, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=PAL["line"], size=7),
                             name="points"))
    xf = np.array([min(x), max(x)])
    fig.add_trace(go.Scatter(x=xf, y=slope * xf + intercept, mode="lines",
                             line=dict(color=PAL["ref"], dash="dash"),
                             name=f"pente={slope:.3f}"))
    return _layout(fig, title, xt=xlab, yt=ylab, h=320)


def plot_hurst_bars(H_dict, band):
    keys = list(H_dict.keys())
    vals = [H_dict[k] if H_dict[k] is not None else np.nan for k in keys]
    fig = go.Figure()
    if band[0] is not None:
        fig.add_hrect(y0=band[0], y1=band[1], fillcolor="rgba(31,78,156,.10)",
                      line_width=0, annotation_text="bande iid (surrogate)")
    fig.add_bar(x=keys, y=vals, marker_color=PAL["line"],
                text=[f"{v:.3f}" for v in vals], textposition="outside")
    fig.add_hline(y=0.5, line_dash="dash", line_color=PAL["ref"],
                  annotation_text="H=0.5 (pas de mémoire)")
    return _layout(fig, "Exposant de Hurst par méthode", yt="H", h=320)


def plot_qq(sample_q, theo_q):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theo_q, y=sample_q, mode="markers",
                             marker=dict(color=PAL["line"], size=5, opacity=.6), name="rendements"))
    lo, hi = min(theo_q), max(theo_q)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color=PAL["ref"], dash="dash"), name="normale"))
    return _layout(fig, "QQ-plot vs loi normale", xt="quantiles théoriques",
                   yt="quantiles empiriques", h=340)


def plot_hist(returns):
    r = returns.values
    fig = go.Figure()
    fig.add_histogram(x=r, nbinsx=60, histnorm="probability density",
                      marker_color=PAL["line"], opacity=.6, name="rendements")
    xs = np.linspace(r.min(), r.max(), 200)
    from scipy import stats as sps
    fig.add_trace(go.Scatter(x=xs, y=sps.norm.pdf(xs, r.mean(), r.std(ddof=1)),
                             mode="lines", line=dict(color=PAL["ref"]), name="normale ajustée"))
    return _layout(fig, "Distribution des rendements vs normale", xt="rendement",
                   yt="densité", h=340)


# ----------------------------------------------------------------------------
# Rendu d'un panneau de test (droiture : stat + p + n + H0/H1 + seuil + limites)
# ----------------------------------------------------------------------------
def render_panel(res: inc.TestResult, fig=None, extra=None):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"### {res.name}")
    with c2:
        st.markdown(badge(res.verdict), unsafe_allow_html=True)

    st.markdown(f'<div class="box box-h"><b>H0 / H1.</b> {res.h0}<br>{res.h1}</div>',
                unsafe_allow_html=True)

    # ligne de chiffres : statistique, p/IC, taille d'effet, n
    def fmt(v):
        if isinstance(v, dict):
            return ", ".join(f"{k}={vv}" for k, vv in v.items())
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    pieces = []
    if res.statistic is not None:
        pieces.append(f"**Statistique** : {fmt(res.statistic)}")
    if res.p_value is not None:
        pieces.append(f"**p-value** : {res.p_value:.4g}")
    if res.ci is not None:
        pieces.append(f"**IC / bande** : {res.ci}")
    if res.effect_size is not None:
        pieces.append(f"**Taille d'effet** ({res.effect_label}) : {fmt(res.effect_size)}")
    pieces.append(f"**n** = {res.n}")
    st.markdown('<div class="metricline">' + "  ·  ".join(pieces) + "</div>",
                unsafe_allow_html=True)

    if fig is not None:
        st.plotly_chart(fig, width='stretch')
    if extra is not None:
        extra()

    st.markdown(f'<div class="box box-seuil"><b>Seuil de matérialité.</b> {res.threshold}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="box box-mean"><b>Ce que ça veut dire.</b> {res.interpretation}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="box box-lim"><b>Limite du test.</b> {res.limits}</div>',
                unsafe_allow_html=True)
    st.divider()


def _adf_kpss_table(sta: inc.TestResult):
    d = sta.details
    rows = [
        {"Test": "ADF (H0: racine unitaire)", "Statistique": round(d["adf_stat"], 3),
         "p-value": round(d["adf_p"], 4),
         "Seuils crit. (1/5/10%)": ", ".join(f"{k}:{v:.2f}" for k, v in d["adf_crit"].items())},
        {"Test": "KPSS (H0: stationnaire)", "Statistique": round(d["kpss_stat"], 3),
         "p-value": round(d["kpss_p"], 4),
         "Seuils crit. (1/5/10%)": ", ".join(f"{k}:{v}" for k, v in d["kpss_crit"].items())},
    ]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    hl = d["half_life_years"]
    st.caption(f"Demi-vie estimée : "
               f"{('%.2f ans' % hl) if np.isfinite(hl) else '∞ (pas de retour à la moyenne)'} "
               f"(b = {d['b']:.5f} ; Δy_t = a + b·y_(t-1)).")


# ============================================================================
# SIDEBAR — réglages
# ============================================================================
st.sidebar.title("⚙️ Réglages")

avail = inc.available_series(BASE_DIR)
default_masi = inc.find_default_masi(BASE_DIR)
default_name = None
if default_masi:
    default_name = os.path.splitext(os.path.basename(default_masi))[0]

st.sidebar.markdown("**Série(s)**")
uploads = st.sidebar.file_uploader("Charger des séries (.xlsx / .xls / .csv)",
                                   type=["xlsx", "xls", "csv"], accept_multiple_files=True)

# Pool de séries disponibles : fichiers locaux + uploads
pool_names = list(avail.keys())
upload_map = {}
for up in uploads or []:
    upload_map[up.name] = up.getvalue()
    if up.name not in pool_names:
        pool_names.append(up.name)

if not pool_names:
    st.sidebar.error("Aucune série trouvée. Placez un fichier MASI à côté de app.py "
                     "ou utilisez l'upload.")
    st.stop()

# sélection des séries à comparer (synthèse multi-actifs)
default_sel = [default_name] if default_name in pool_names else [pool_names[0]]
selected = st.sidebar.multiselect("Séries à inclure (comparaison)", pool_names,
                                   default=default_sel)
if not selected:
    selected = default_sel

primary = st.sidebar.selectbox("Série analysée en détail", selected,
                               index=0)

st.sidebar.markdown("---")
kind = st.sidebar.radio("Type de rendement", ["log", "simple"], index=0,
                        help="log par défaut, cohérent avec « log(prix) Lévy ».")
freq_label = st.sidebar.radio("Fréquence (rebalancement)",
                              ["mensuel", "hebdomadaire", "quotidien"], index=0,
                              help="Mensuel par défaut : c'est l'indépendance à la "
                                   "fréquence de rebalancement qui fait tenir la chaîne "
                                   "de Markov.")
FREQ = {"quotidien": "D", "hebdomadaire": "W", "mensuel": "M"}[freq_label]


def get_series(name) -> pd.Series:
    if name in upload_map:
        return load_bytes(name, upload_map[name])
    return load_path(avail[name])


# fenêtre de dates (basée sur la série primaire)
s_primary_full = get_series(primary)
dmin, dmax = s_primary_full.index.min().date(), s_primary_full.index.max().date()
st.sidebar.markdown("---")
dr = st.sidebar.date_input("Fenêtre de dates", value=(dmin, dmax),
                           min_value=dmin, max_value=dmax)
if isinstance(dr, tuple) and len(dr) == 2:
    d0, d1 = dr
else:
    d0, d1 = dmin, dmax

st.sidebar.markdown("---")
st.sidebar.markdown("**Paramètres par test**")
lb_lags = st.sidebar.multiselect("Retards Ljung-Box", [1, 2, 3, 5, 10, 15, 20, 30],
                                 default=[1, 5, 10, 20])
vr_qs = st.sidebar.multiselect("Horizons q (Variance Ratio)", [2, 3, 4, 6, 8, 12, 16, 24],
                               default=[2, 4, 8, 16])
bds_dim = st.sidebar.slider("Dimension max BDS (m)", 2, 6, 5)
arch_lags = st.sidebar.slider("Retards ARCH-LM", 1, 24, 12)
rs_min = st.sidebar.slider("Plus petite fenêtre Hurst (R/S, DFA)", 4, 32, 10)
maturity = st.sidebar.slider("Maturité produit (ans) — matérialité demi-vie", 1, 20, 10)
n_boot = st.sidebar.slider("Taille bootstrap / surrogate", 100, 2000, 500, step=100)
seed = st.sidebar.number_input("Graine aléatoire (reproductibilité)", value=inc.SEED_DEFAUT,
                               step=1)

st.sidebar.markdown("**Seuils de matérialité**")
acf_mat = st.sidebar.slider("|rho(1)| matériel", 0.02, 0.30, 0.10, 0.01)
vr_mat = st.sidebar.slider("|VR(q)-1| matériel", 0.05, 0.60, 0.20, 0.05)
hurst_mat = st.sidebar.slider("|H-0.5| matériel", 0.02, 0.30, 0.10, 0.01)
alpha = st.sidebar.slider("Niveau α", 0.01, 0.10, 0.05, 0.01)

st.sidebar.caption(f"🎲 Graine fixe affichée : **{seed}** — bootstrap/surrogate reproductibles.")

# construit les paramètres
params = inc.Params(
    alpha=alpha, lb_lags=tuple(sorted(lb_lags)) or (1, 5, 10, 20),
    vr_qs=tuple(sorted(vr_qs)) or (2, 4, 8, 16), bds_max_dim=bds_dim,
    arch_lags=arch_lags, acf_material=acf_mat, vr_material=vr_mat,
    hurst_material=hurst_mat, maturity_years=float(maturity), rs_min_window=rs_min,
    n_boot=n_boot, seed=int(seed),
)
p_dict = asdict(params)


def slice_series(s):
    return s.loc[str(d0):str(d1)]


# ============================================================================
# CALCULS sur la série primaire
# ============================================================================
s_primary = slice_series(s_primary_full)
sid = f"{primary}|{d0}|{d1}"
if s_primary.shape[0] < 40:
    st.error("Série trop courte après filtrage (≥40 points requis). Élargissez la fenêtre.")
    st.stop()

with st.spinner("Calcul de la batterie de tests…"):
    R = run_cached(sid, s_primary, kind, FREQ, p_dict)
    SCAN = scan_cached(sid, s_primary, kind, p_dict)

# ============================================================================
# EN-TÊTE
# ============================================================================
st.title("Diagnostic de l'hypothèse d'incréments indépendants (iid)")
st.caption("« log(prix) est un processus de Lévy » — socle du pricer CPPI par opérateur de "
           "Markov (Paulot–Lacroze). Exercice de validation de modèle au sens **SR 11-7** "
           "(Conceptual Soundness + Outcomes Analysis).")

tabs = st.tabs(["1 · Synthèse / Verdict", "2 · Indépendance", "3 · Volatilité",
                "4 · Stationnarité", "5 · Mémoire longue", "6 · Distribution",
                "7 · Méthode"])

# ----------------------------------------------------------------------------
# ONGLET 1 — SYNTHÈSE
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Verdict de synthèse")
    st.info(inc.plain_language_verdict(R, primary))

    st.markdown("**Verdicts par axe** (lecture holistique — on ne cueille pas une "
                "p-value isolée parmi dix) :")
    av = R["_axis_verdicts"]
    cols = st.columns(len(av))
    for col, (ax, v) in zip(cols, av.items()):
        with col:
            st.markdown(f'<small class="dim">{ax}</small><br>{badge(v)}',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"#### Table livrable — {primary}")
    st.caption(f"Rendement **{kind}**, fréquence **{freq_label}**, "
               f"n_rendements = **{R['_n_returns']}**, fenêtre {d0} → {d1}.")
    df_syn = inc.synthesis_table(R)
    df_syn["Impact pricing (Phase 2)"] = ""   # colonne réservée, vide en v1
    st.dataframe(df_syn, width='stretch', hide_index=True)

    st.download_button("⬇️ Exporter la table (CSV)",
                       df_syn.to_csv(index=False).encode("utf-8"),
                       file_name=f"synthese_iid_{primary}.csv", mime="text/csv")

    # comparaison multi-actifs
    if len(selected) > 1:
        st.markdown("---")
        st.markdown("#### Comparaison multi-actifs — verdicts par axe")
        rows = []
        with st.spinner("Calcul de la comparaison multi-actifs…"):
            for nm in selected:
                s_nm = slice_series(get_series(nm))
                if s_nm.shape[0] < 40:
                    continue
                r_nm = run_cached(f"{nm}|{d0}|{d1}", s_nm, kind, FREQ, p_dict)
                row = {"Actif": nm, "n": r_nm["_n_returns"]}
                row.update(r_nm["_axis_verdicts"])
                rows.append(row)
        comp = pd.DataFrame(rows)
        st.dataframe(comp, width='stretch', hide_index=True)
        st.download_button("⬇️ Exporter la comparaison (CSV)",
                           comp.to_csv(index=False).encode("utf-8"),
                           file_name="comparaison_multiactifs.csv", mime="text/csv")

    with st.expander("Légende des verdicts"):
        for v in [inc.V_OK, inc.V_MAT, inc.V_NONMAT, inc.V_EXT, inc.V_IND,
                  inc.V_CHAR, inc.V_NT]:
            gloss = {
                inc.V_OK: "hypothèse iid tenue sur cet axe.",
                inc.V_MAT: "violation dont l'ampleur dépasse le seuil de matérialité.",
                inc.V_NONMAT: "écart statistiquement détecté mais d'ampleur négligeable.",
                inc.V_EXT: "violation réelle mais absorbée par une extension du modèle "
                           "(régimes de volatilité, Paulot §7.1).",
                inc.V_IND: "tests contradictoires ou borderline — à confirmer (Phase 2).",
                inc.V_CHAR: "axe descriptif (loi marginale) : n'est PAS un test de l'iid.",
                inc.V_NT: "non testé (hors périmètre v1).",
            }[v]
            st.markdown(f"- {badge(v)} — {gloss}", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ONGLET 2 — INDÉPENDANCE
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Indépendance sérielle directe — le cœur du test")
    st.markdown(f"**Verdict d'axe : {badge(av[inc.AX_IND])}**", unsafe_allow_html=True)

    # sensibilité à la fréquence (point 6)
    st.markdown("##### Sensibilité à la fréquence de rebalancement")
    st.caption("Une dépendance quotidienne qui disparaît en mensuel est immatérielle pour "
               "le pricer (la chaîne de Markov tient à la fréquence de rebalancement). "
               "Comparez rho(1) et VR(2) selon la fréquence :")
    st.dataframe(SCAN, width='stretch', hide_index=True)
    st.divider()

    lb = R["ljung_box"]
    render_panel(lb, fig=plot_acf(np.array(lb.details["acf"]), lb.n))

    vr = R["variance_ratio"]
    render_panel(vr, fig=plot_vr(vr.details["per_q"]),
                 extra=lambda: st.caption(
                     "Triple contrôle par horizon : p asymptotique robuste, ampleur |VR-1|, "
                     "et position vs bande iid par permutation (surrogate)."))

    bd = R["bds"]
    render_panel(bd, fig=plot_bds(bd.details["dims"], bd.details["stat"],
                                  bd.details["pval"], params.alpha))

    rn = R["runs"]
    render_panel(rn, fig=plot_runs(rn.details))

# ----------------------------------------------------------------------------
# ONGLET 3 — VOLATILITÉ
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Hétéroscédasticité conditionnelle (clustering de volatilité)")
    st.markdown(f"**Verdict d'axe : {badge(av[inc.AX_VOL])}**", unsafe_allow_html=True)
    st.caption("Rappel de droiture : le clustering de volatilité est quasi certain sur "
               "actions. Ce n'est PAS une raison d'abandonner le modèle — il est absorbé "
               "par l'extension à régimes de volatilité (Paulot–Lacroze §7.1).")
    ar = R["arch"]
    render_panel(ar, fig=plot_abs_acf(np.array(ar.details["acf_abs"]), ar.n))

    st.markdown("##### Mémoire de volatilité (|rendements|)")
    mv = R["memory_vol"]
    render_panel(mv, fig=plot_hurst_bars(mv.statistic, mv.ci))

# ----------------------------------------------------------------------------
# ONGLET 4 — STATIONNARITÉ
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Stationnarité / retour à la moyenne")
    st.markdown(f"**Verdict d'axe : {badge(av[inc.AX_STA])}**", unsafe_allow_html=True)
    st.caption("L'hypothèse iid = log(prix) marche aléatoire = RACINE UNITAIRE. "
               "Un retour à la moyenne (série stationnaire, demi-vie < maturité) "
               "violerait l'iid. Indices actions : racine unitaire en général ; "
               "matières premières : souvent du MR.")
    sta = R["stationarity"]
    render_panel(sta, fig=plot_price(R["_prices_freq"], logp=True,
                                     title="log-prix à la fréquence choisie"),
                 extra=lambda: _adf_kpss_table(sta))

# ----------------------------------------------------------------------------
# ONGLET 5 — MÉMOIRE LONGUE
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Mémoire longue (exposant de Hurst)")
    st.markdown(f"**Verdict d'axe : {badge(av[inc.AX_MEM])}**", unsafe_allow_html=True)
    st.caption("Trois méthodes (R/S, DFA, GPH) calculées à la main. On reporte "
               "honnêtement leur désaccord et une bande iid par données de substitution "
               "(surrogate). Sur les RENDEMENTS : H≈0.5 ⇒ pas de mémoire ⇒ iid OK.")
    mr = R["memory_returns"]

    def _hurst_extra():
        rs = mr.details["rs"]; dfa = mr.details["dfa"]
        c1, c2 = st.columns(2)
        with c1:
            if rs["logn"]:
                st.plotly_chart(plot_loglog(rs["logn"], rs["logrs"], rs["H"],
                                rs["intercept"], "log(taille fenêtre)", "log(R/S)",
                                "R/S — régression log-log"), width='stretch')
        with c2:
            if dfa["logn"]:
                st.plotly_chart(plot_loglog(dfa["logn"], dfa["logF"], dfa["H"],
                                dfa["intercept"], "log(taille boîte)", "log F(n)",
                                "DFA — régression log-log"), width='stretch')
        g = mr.details["gph"]
        st.caption(f"GPH : d = {g['d']:.3f} (±{g['se']:.3f}), H = {g['H']:.3f}, "
                   f"bande m = {g['m']} fréquences. "
                   f"Désaccord inter-méthodes = preuve de fragilité, reporté tel quel.")

    render_panel(mr, fig=plot_hurst_bars(mr.statistic, mr.ci), extra=_hurst_extra)

# ----------------------------------------------------------------------------
# ONGLET 6 — DISTRIBUTION
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Forme de la distribution (caractérisation, PAS une violation iid)")
    st.markdown(f"**Verdict d'axe : {badge(av[inc.AX_DIS])}**", unsafe_allow_html=True)
    st.warning("⚠️ Des queues épaisses (kurtosis, sauts) ne violent PAS l'hypothèse iid : "
               "elles restent dans la famille de Lévy à sauts (Merton, Kou). Ce panneau "
               "décrit la loi marginale ; il indique qu'une loi à sauts (Kou) est plus "
               "adaptée que Black-Scholes, sans remettre en cause l'indépendance.")
    di = R["distribution"]

    def _dist_extra():
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_qq(di.details["sample_q"], di.details["theo_q"]),
                            width='stretch')
        with c2:
            st.plotly_chart(plot_hist(R["_returns"]), width='stretch')

    render_panel(di, extra=_dist_extra)

    st.markdown("##### Volatilité locale σ(S,t) — hors scope")
    lv = R["local_vol"]
    render_panel(lv)

# ----------------------------------------------------------------------------
# ONGLET 7 — MÉTHODE
# ----------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Méthode — honnêteté épistémique")
    st.markdown("""
**Pourquoi ce test ?** Sous incréments iid, la valeur renormalisée X = coussin/seuil est
une **chaîne de Markov en dimension 1**, et tout le pricing rapide du CPPI en découle
(Paulot–Lacroze). Si l'hypothèse casse, la réduction s'effondre et le pricing redevient
un problème en haute dimension. Cet outil dit *où* l'hypothèse casse et *avec quelle sévérité*.
""")
    st.markdown("##### Quatre principes de lecture (non négociables)")
    st.markdown("""
- **4 — Significativité ≠ matérialité.** Sur ~6000 points quotidiens, une déviation
  minuscule devient « significative ». On reporte donc TOUJOURS une **taille d'effet**
  (ampleur d'autocorrélation, écart de VR à 1, demi-vie en années) et on fonde le verdict
  sur la **matérialité**, pas sur p<0,05 seul.
- **5 — Non-normalité ≠ non-indépendance.** Les queues épaisses restent dans la famille
  de Lévy (Merton, Kou) que le pricer gère. Elles ne sont **jamais** étiquetées comme une
  violation des incréments indépendants.
- **6 — Fréquence de rebalancement.** C'est l'indépendance à la fréquence de rebalancement
  (mensuelle par défaut) qui fait tenir la chaîne de Markov. Une dépendance quotidienne qui
  disparaît en mensuel est **immatérielle** pour le pricer — et l'outil le montre (onglet 2).
- **9 — Test multiple, lecture holistique.** On n'isole pas une p-value significative parmi
  dix. La synthèse reste **conservatrice** : si les tests se contredisent ou sont borderline,
  le verdict est « indéterminé / à confirmer », jamais un binaire forcé.
""")
    st.markdown("##### Détail des tests (H0 / H1 · seuil · limite)")
    order = ["ljung_box", "variance_ratio", "bds", "runs", "arch", "stationarity",
             "memory_returns", "memory_vol", "distribution", "local_vol"]
    for k in order:
        r = R[k]
        with st.expander(f"{r.name}  —  {r.axis}"):
            st.markdown(f"**H0/H1.** {r.h0}  {r.h1}")
            st.markdown(f"**Seuil de matérialité.** {r.threshold}")
            st.markdown(f"**Ce que ça veut dire.** {r.interpretation}")
            st.markdown(f"**Limite.** {r.limits}")

    st.markdown("##### Reproductibilité")
    st.markdown(f"- Graine aléatoire fixe et affichée : **{params.seed}**.")
    st.markdown(f"- Bootstrap / surrogate : **{params.n_boot}** tirages.")
    st.markdown("- Chemins de données **relatifs** au script ; déployable sur Streamlit "
                "Community Cloud.")
    st.markdown("##### Phase 2 (à prévoir, non bloquante)")
    st.markdown("Pour toute violation jugée matérielle, **chiffrer l'impact prix** : "
                "comparer le put de garantie sous hypothèse iid (pricer Markov actuel) à un "
                "Monte Carlo sous un modèle portant la caractéristique détectée "
                "(AR(1)/Ornstein-Uhlenbeck pour le MR, GARCH pour le clustering, mouvement "
                "brownien fractionnaire pour la mémoire longue). Le verdict « matériel » ne "
                "devient définitif que si l'écart de prix dépasse un seuil (ex. 5 %). "
                "La colonne « Impact pricing » de la table de synthèse est réservée à cet effet.")
