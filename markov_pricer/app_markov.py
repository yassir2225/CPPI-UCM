"""
app_markov.py
=============
Interface Streamlit du PRICER CPPI PAR OPERATEURS DE MARKOV.

Deux dynamiques pour l'actif risqué :
  - LOGNORMALE : modèle paramétrique GBM (volatilité σ).
  - EMPIRIQUE  : la distribution réelle des rendements d'un indice (ex. MASI)
                 pilote directement la loi de transition (queues, asymétrie).
                 Mesure risque-neutre (prix) ou historique réelle (risque).

Monte Carlo intégré (même loi) comme contrôle. Projet indépendant du backtest.

    streamlit run app_markov.py
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import markov_cppi as mk
import empirical as emp


st.set_page_config(page_title="Pricer CPPI — Opérateurs de Markov", layout="wide")

st.sidebar.title("Paramètres")

st.sidebar.subheader("Dynamique de l'actif risqué")
mode = st.sidebar.radio("Source de la loi de rendement",
                        ["Lognormale (paramétrique)", "Empirique (indice réel)"])

series = None
emp_method = "overlap"
emp_measure = "Risque-neutre (prix)"
sigma = 0.15

if mode.startswith("Lognormale"):
    sigma = st.sidebar.slider("Volatilité annualisée σ", 0.01, 1.50, 0.15, 0.01)
else:
    up = st.sidebar.file_uploader("Fichier indice (ex. MASI_HISTO.xlsx)", type=["xls", "xlsx"])
    emp_method = st.sidebar.selectbox(
        "Construction des rendements de période", ["overlap", "bootstrap"],
        format_func=lambda x: {"overlap": "Fenêtres glissantes observées",
                               "bootstrap": "Bootstrap journalier composé"}[x])
    emp_measure = st.sidebar.radio("Mesure",
                                   ["Risque-neutre (prix)", "Historique réelle (risque)"])
    if up is not None:
        try:
            series = emp.load_series(io.BytesIO(up.getvalue()), name=up.name)
            st.sidebar.success(f"{up.name} : {len(series)} points "
                               f"({series.index.min():%Y-%m} → {series.index.max():%Y-%m})")
        except Exception as e:
            st.sidebar.error(f"Lecture impossible : {e}")

st.sidebar.subheader("Modèle")
rate = st.sidebar.number_input(
    "Taux sans risque r", min_value=0.0, max_value=0.20,
    value=0.03, step=0.0005, format="%.4f")
maturity = st.sidebar.slider("Maturité T (années)", 0.5, 15.0, 5.0, 0.5)
n_rebal = st.sidebar.slider("Nombre de rebalancements", 4, 500, 60, 1)
spot = st.sidebar.number_input("Spot de l'actif risqué", value=100.0, min_value=0.01)

st.sidebar.subheader("Produit (CPPI)")
initial = st.sidebar.number_input("Capital investi C₀", value=100.0, min_value=0.01)
guarantee = st.sidebar.number_input("Nominal garanti G", value=100.0, min_value=0.0)
multiplier = st.sidebar.slider("Multiplicateur m", 0.0, 12.0, 4.0, 0.1)
c1, c2 = st.sidebar.columns(2)
w_min = c1.number_input("Poids min", value=0.0, min_value=0.0, max_value=10.0, step=0.1)
w_max = c2.number_input("Poids max", value=1.5, min_value=0.0, max_value=10.0, step=0.1)

st.sidebar.subheader("Numérique")
n_grid = int(st.sidebar.number_input("Taille de grille N",
            min_value=100, max_value=6000, value=600, step=50))
if n_grid > 1500:
    st.sidebar.warning("Calcul en O(N²) : au-delà de 1500, le pricing ralentit.")
n_paths = st.sidebar.select_slider("Simulations Monte Carlo",
                                   options=[50_000, 100_000, 200_000, 500_000, 1_000_000], value=200_000)


def make_params(**over) -> mk.MarkovParams:
    base = dict(spot=spot, sigma=sigma, rate=rate, maturity=maturity, n_rebal=n_rebal,
                initial=initial, guarantee=guarantee, multiplier=multiplier,
                w_min=w_min, w_max=w_max, n_grid=n_grid)
    base.update(over)
    return mk.MarkovParams(**base)


def make_law(q: mk.MarkovParams):
    if mode.startswith("Lognormale"):
        return mk.LognormalLaw(q.sigma, q.maturity / q.n_rebal)
    if series is None:
        return None
    return emp.make_empirical_law(series, q, recenter=emp_measure.startswith("Risque"),
                                  method=emp_method)


p = make_params()
law = make_law(p)
empirical_active = not mode.startswith("Lognormale")
x0 = (initial / guarantee) * np.exp(rate * maturity) if guarantee > 0 else np.inf

st.title("Pricer CPPI — Opérateurs de Markov")
st.caption("Pricing du gap risk par propagation de probabilités sur grille, avec "
           "comptabilité exacte de la défaisance. Dynamique lognormale ou empirique.")

if empirical_active and series is None:
    st.info("⬅️ Charge un fichier indice (ex. MASI_HISTO.xlsx) dans la barre latérale "
            "pour activer la dynamique empirique.")
    st.stop()

tab_price, tab_data, tab_ctrl, tab_sens, tab_help = st.tabs(
    ["Pricing", "Dynamique empirique", "Contrôle Markov vs MC", "Sensibilités", "Méthode"])

res = mk.price_markov(p, law=law)
hist_mode = empirical_active and emp_measure.startswith("Historique")


with tab_price:
    st.markdown(f"**Dynamique active :** {law.label}"
                + (f" — vol implicite {law.sigma_ann:.1%}" if empirical_active else ""))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prix put (gap risk)" if not hist_mode else "Perte attendue actualisée",
              f"{res['put_price']:.4f}", help="e^{-rT}·G·E[max(1−X_T,0)]")
    k2.metric("Probabilité de gap", f"{res['p_gap']*100:.2f} %", help="P(X_T < 1)")
    k3.metric("Prix de la stratégie", f"{res['strategy_price']:.4f}",
              help=f"≈ C₀={initial:.2f} en risque-neutre ; >C₀ en historique (normal)")
    k4.metric("Poids risqué initial w₀", f"{float(mk.weight(x0, multiplier, w_min, w_max)):.3f}",
              help=f"X₀ = {x0:.4f}")

    if hist_mode:
        st.warning("Mesure **historique réelle** : proba et perte telles qu'observées — "
                   "ce ne sont **pas** des prix d'arbitrage. Pour un prix, choisis "
                   "la mesure risque-neutre.")

    if empirical_active:
        rLN = mk.price_markov(make_params(sigma=law.sigma_ann),
                              law=mk.LognormalLaw(law.sigma_ann, maturity / n_rebal))
        d1, d2 = st.columns(2)
        d1.metric("Put lognormal à MÊME vol", f"{rLN['put_price']:.4f}")
        d2.metric("Surcoût dû aux queues empiriques", f"{res['put_price']-rLN['put_price']:+.4f}",
                  help="Écart entre la loi empirique et une gaussienne de même volatilité")

    if not hist_mode:
        st.markdown("##### Sensibilités (greeks)")
        gk = mk.greeks_markov(p, law=law)
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("∂Put/∂C₀", f"{gk['sens_C0']:+.5f}")
        g2.metric("Γ", f"{gk['gamma_C0']:+.6f}")
        g3.metric("Vega", "—" if np.isnan(gk['vega']) else f"{gk['vega']:+.4f}",
                  help="Non défini en empirique")
        g4.metric("Delta sous-jacent", f"{gk['delta_underlying']:+.5f}")

    st.markdown("##### Distribution de X_T")
    nodes, dist = res["nodes"], res["dist"]
    fig = go.Figure()
    fig.add_bar(x=nodes, y=dist, name="X_T vivant (>1)", marker_color="#2c7fb8")
    fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="barrière X=1")
    fig.add_vline(x=x0, line_dash="dot", line_color="green", annotation_text=f"X₀={x0:.3f}")
    fig.add_bar(x=[0.97], y=[res["p_gap"]], name=f"défaisé X<1 ({res['p_gap']*100:.1f} %)",
                marker_color="#d95f0e", width=0.04)
    fig.update_layout(height=420, bargap=0.0, xaxis_title="X_T = C_T / G",
                      yaxis_title="probabilité", legend=dict(orientation="h"))
    fig.update_xaxes(range=[0.9, min(nodes.max(), x0 * 4)])
    st.plotly_chart(fig, use_container_width=True)

    df_out = pd.DataFrame({
        "indicateur": ["dynamique", "prix put / perte", "proba gap", "prix stratégie", "E[X_T]", "X_0", "w_0"],
        "valeur": [law.label, res["put_price"], res["p_gap"], res["strategy_price"], res["EX_T"],
                   x0, float(mk.weight(x0, multiplier, w_min, w_max))]})
    st.download_button("Exporter (CSV)", df_out.to_csv(index=False).encode(),
                       "markov_cppi_resultats.csv", "text/csv")


with tab_data:
    if not empirical_active:
        st.info("Active la dynamique empirique dans la barre latérale et charge un indice.")
    else:
        st.markdown(f"#### Distribution empirique des rendements — {series.name}")
        stt = emp.empirical_stats(series, p, method=emp_method)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Jours / période", stt["k_jours_periode"])
        s2.metric("Vol annualisée", f"{stt['vol_annualisee']:.1%}")
        s3.metric("Skewness", f"{stt['skewness']:+.2f}", help="<0 : queue de pertes plus lourde")
        s4.metric("Kurtosis (excès)", f"{stt['kurtosis_exces']:+.2f}", help="0 = normal ; >0 = queues épaisses")

        rho = emp.build_rho_sample(series, p, method=emp_method)
        rets = (rho - 1) * 100
        fig_d = go.Figure()
        fig_d.add_histogram(x=rets, nbinsx=80, name="empirique", histnorm="probability density",
                            marker_color="#2c7fb8", opacity=0.7)
        mu_, sd_ = rets.mean(), rets.std()
        xs = np.linspace(rets.min(), rets.max(), 200)
        gauss = np.exp(-0.5 * ((xs - mu_) / sd_) ** 2) / (sd_ * np.sqrt(2 * np.pi))
        fig_d.add_scatter(x=xs, y=gauss, mode="lines", name="gaussienne équivalente",
                          line=dict(color="red", dash="dash"))
        fig_d.update_layout(height=420, xaxis_title="rendement de période (%)", yaxis_title="densité",
                            legend=dict(orientation="h"),
                            title="Queues épaisses / asymétrie vs gaussienne de même volatilité")
        st.plotly_chart(fig_d, use_container_width=True)
        st.caption(f"Échantillon : {stt['n_echantillon']} rendements de {stt['k_jours_periode']} jours. "
                   f"Pire période observée : {stt['pire_rendement_periode']*100:.1f} %. "
                   "Le pricing utilise CETTE distribution, pas une gaussienne.")


with tab_ctrl:
    st.markdown("#### Markov vs Monte Carlo (même loi)")
    st.caption("Même loi de rendement des deux côtés. Le prix Markov doit tomber dans l'IC95 du MC.")
    if st.button("Lancer le Monte Carlo", type="primary"):
        with st.spinner(f"{n_paths:,} trajectoires…"):
            mc = mk.price_monte_carlo(p, law=law, n_paths=n_paths, seed=1)
        lo, hi = mc["put_ci95"]
        inside = lo - 2e-3 <= res["put_price"] <= hi + 2e-3
        comp = pd.DataFrame({
            "indicateur": ["prix put / perte", "proba gap", "prix stratégie", "E[X_T]"],
            "Markov": [res["put_price"], res["p_gap"], res["strategy_price"], res["EX_T"]],
            "Monte Carlo": [mc["put_price"], mc["p_gap"], mc["strategy_price"], mc["EX_T"]],
            "IC95 MC (put)": [f"[{lo:.4f}, {hi:.4f}]", "", "", ""]})
        st.dataframe(comp, use_container_width=True, hide_index=True)
        if inside:
            st.success("✅ Markov dans l'IC95 du MC")
        else:
            st.warning("⚠️ Hors IC95 — augmente N ou le nb de simulations.")

    st.markdown("#### Convergence en N")
    if st.button("Lancer l'étude de convergence"):
        rows = []
        with st.spinner("Plusieurs grilles…"):
            for N in [150, 300, 600, 1200, 2400]:
                q = make_params(n_grid=N)
                r = mk.price_markov(q, law=make_law(q))
                rows.append({"N": N, "prix put": r["put_price"], "proba gap": r["p_gap"], "E[X_T]": r["EX_T"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with tab_sens:
    st.markdown("#### Profil de sensibilité")
    opts = ["Multiplicateur m", "Nb de rebalancements", "Maturité T"]
    if not empirical_active:
        opts = ["Volatilité σ"] + opts
    param = st.selectbox("Paramètre à faire varier", opts)
    cfg = {"Volatilité σ": ("sigma", np.linspace(0.05, 0.80, 16)),
           "Multiplicateur m": ("multiplier", np.arange(1, 11, 1.0)),
           "Nb de rebalancements": ("n_rebal", np.array([4, 6, 12, 24, 52, 120, 250])),
           "Maturité T": ("maturity", np.linspace(1, 12, 12))}
    key, grid = cfg[param]
    puts, gaps = [], []
    with st.spinner("Calcul du profil…"):
        for val in grid:
            q = make_params(**{key: (int(val) if key == "n_rebal" else float(val))})
            if key == "sigma":
                lw = mk.LognormalLaw(float(val), q.maturity / q.n_rebal)
            elif key in ("n_rebal", "maturity"):
                lw = make_law(q)          # période change -> reconstruire la loi
            else:
                lw = law
            r = mk.price_markov(q, law=lw)
            puts.append(r["put_price"]); gaps.append(r["p_gap"] * 100)
    fig3 = go.Figure()
    fig3.add_scatter(x=grid, y=puts, mode="lines+markers", name="prix put", yaxis="y1")
    fig3.add_scatter(x=grid, y=gaps, mode="lines+markers", name="proba gap (%)", yaxis="y2")
    fig3.update_layout(height=440, xaxis_title=param, yaxis=dict(title="prix put"),
                       yaxis2=dict(title="proba gap (%)", overlaying="y", side="right"),
                       legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)


with tab_help:
    st.markdown(r"""
#### Principe

CPPI renormalisé en processus de Markov 1D sur $X_i = C_i/H_i$, $H_i = G\,e^{-r(T-t_i)}$ :

$$X_{i+1} = X_i\big[1 + w(X_i)(\rho-1)\big],\quad w(X)=\mathrm{clip}(m(1-\tfrac1X),w_{\min},w_{\max})$$

Région vivante $X>1$ sur grille log (Voronoï) ; la masse franchissant $X=1$ et son
*shortfall* sont comptabilisés **exactement** à chaque pas (proba de gap précise).

#### Deux dynamiques pour $\rho$

- **Lognormale** : $\rho = e^{-\frac12\sigma^2\Delta t+\sigma\sqrt{\Delta t}Z}$.
- **Empirique** : $\rho$ tiré de la **distribution réelle** de l'indice (fenêtres
  glissantes ou bootstrap). Les queues épaisses et l'asymétrie pilotent la dynamique.

#### Deux mesures (empirique)

- **Risque-neutre** : rendements recentrés à moyenne 1 → un **prix**.
- **Historique réelle** : dérive conservée → **proba de gap / perte observées**.

#### Contrôle

Monte Carlo avec la **même loi**. Validé en lognormal **et** en empirique
(Markov ≈ MC, convergence en $N$). Une gaussienne de même volatilité peut
**fortement sous-estimer** le gap : d'où l'intérêt de la loi empirique.
""")
