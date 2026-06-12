"""
app.py — Laboratoire interactif CPPI (Streamlit)
================================================
Application de backtest historique d'une stratégie CPPI (Constant Proportion
Portfolio Insurance) hautement paramétrable.

Lancement :
    streamlit run app.py

Toute la logique de calcul vit dans cppi.py ; ce fichier ne gère que
l'interface, la mise en forme et les exports.
"""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cppi

# ---------------------------------------------------------------------------
# Configuration générale de la page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Laboratoire CPPI", page_icon="📈", layout="wide")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("CPPI_DATA_DIR", _HERE)

# ---------------------------------------------------------------------------
# Chargement des données (mis en cache pour la rapidité)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_load(path: str) -> pd.DataFrame:
    return cppi.load_index_file(path)


def load_series(path: str) -> pd.Series:
    return cppi.to_price_series(cached_load(path))


# ---------------------------------------------------------------------------
# Sidebar — découverte des fichiers + tous les paramètres
# ---------------------------------------------------------------------------
def sidebar_data_selection():
    st.sidebar.header("1 · Données")

    folder = st.sidebar.text_input(
        "Dossier des fichiers Excel",
        value=DATA_FOLDER_DEFAULT,
        help="Dossier contenant MASI_HISTO.xlsx, MBI_*_HISTO.xlsx, ...",
    )

    files = {}
    if os.path.isdir(folder):
        files = cppi.discover_files(folder)

    # Permettre aussi l'upload manuel
    uploaded = st.sidebar.file_uploader(
        "...ou importer des fichiers", type=["xlsx", "xls"],
        accept_multiple_files=True,
    )
    if uploaded:
        tmp_dir = os.path.join(folder, "_uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        for uf in uploaded:
            p = os.path.join(tmp_dir, uf.name)
            with open(p, "wb") as f:
                f.write(uf.getbuffer())
            files[os.path.splitext(uf.name)[0]] = p

    if not files:
        st.sidebar.error("Aucun fichier Excel trouvé. Vérifiez le dossier.")
        st.stop()

    labels = list(files.keys())
    # Valeurs par défaut intelligentes : MASI en risqué, MBI CT en sûr.
    masi_default = next((l for l in labels if "MASI" in l.upper()), labels[0])
    ct_default = next((l for l in labels if "CT" in l.upper()), labels[0])

    risky_label = st.sidebar.selectbox(
        "Actif risqué (à insurer)", labels, index=labels.index(masi_default),
    )
    bench_label = st.sidebar.selectbox(
        "Benchmark de comparaison", labels, index=labels.index(masi_default),
        help="Indice auquel comparer la performance du CPPI.",
    )

    return files, risky_label, bench_label, ct_default, labels


def sidebar_parameters(files, labels, ct_default):
    p = cppi.default_params()

    st.sidebar.header("2 · Paramètres CPPI")

    if st.sidebar.button("↺ Réinitialiser les paramètres"):
        for k in list(st.session_state.keys()):
            if k.startswith("p_"):
                del st.session_state[k]
        st.rerun()

    p["capital"] = st.sidebar.number_input(
        "Capital initial", 1_000.0, 1e9, 100_000.0, step=1_000.0, key="p_capital")

    p["floor_mode"] = st.sidebar.selectbox(
        "Mode de floor",
        ["% du capital initial (fixe)", "Montant fixe absolu",
         "Croissant au taux sans risque"], key="p_floor_mode")

    if p["floor_mode"] == "Montant fixe absolu":
        p["floor_abs"] = st.sidebar.number_input(
            "Floor (montant absolu)", 0.0, 1e9, 0.80 * p["capital"],
            step=1_000.0, key="p_floor_abs")
    else:
        p["floor_pct"] = st.sidebar.slider(
            "Niveau de floor (% du capital)", 0.0, 1.0, 0.80, 0.01,
            key="p_floor_pct")

    p["multiplier"] = st.sidebar.slider(
        "Multiplicateur m", 1.0, 10.0, 4.0, 0.5, key="p_mult")
    p["rf"] = st.sidebar.slider(
        "Taux sans risque annuel", 0.0, 0.10, 0.03, 0.005,
        format="%.3f", key="p_rf")
    p["frequency"] = st.sidebar.selectbox(
        "Fréquence de rebalancement", cppi.FREQUENCIES, index=2, key="p_freq")

    c1, c2 = st.sidebar.columns(2)
    p["exp_min"] = c1.slider("Expo min", 0.0, 2.0, 0.0, 0.05, key="p_emin")
    p["exp_max"] = c2.slider("Expo max", 0.0, 3.0, 1.0, 0.05, key="p_emax")

    p["return_mode"] = st.sidebar.radio(
        "Mode de rendement (stats)", ["Simple", "Log"], horizontal=True,
        key="p_retmode")

    # --- Actif sûr ---
    st.sidebar.header("3 · Poche sûre")
    p["use_safe_index"] = st.sidebar.checkbox(
        "Poche sûre = indice obligataire", value=False, key="p_use_safe",
        help="Si décoché, la poche sûre capitalise au taux sans risque.")
    safe_label = None
    if p["use_safe_index"]:
        safe_label = st.sidebar.selectbox(
            "Indice de la poche sûre", labels,
            index=labels.index(ct_default), key="p_safe_label")

    # --- Avancé ---
    with st.sidebar.expander("4 · Paramètres avancés"):
        p["use_cushion_limit"] = st.checkbox("Activer cushion limit", key="p_use_cl")
        p["cushion_limit"] = st.slider(
            "Cushion limit (% du portefeuille)", 0.0, 1.0, 0.50, 0.05,
            key="p_cl", disabled=not p["use_cushion_limit"])

        p["use_lockin"] = st.checkbox("Activer profit lock-in (cliquet)", key="p_use_lock")
        p["lockin_level"] = st.slider(
            "Niveau de lock-in", 0.0, 1.0, 0.90, 0.01,
            key="p_lock_lvl", disabled=not p["use_lockin"])
        p["lockin_frequency"] = st.selectbox(
            "Fréquence de lock-in", cppi.FREQUENCIES, index=2,
            key="p_lock_freq", disabled=not p["use_lockin"])

        p["fee_mode"] = st.selectbox(
            "Mode de frais", ["Aucun", "Fixes", "Proportionnels"], key="p_fee_mode")
        if p["fee_mode"] == "Fixes":
            p["fee_fixed"] = st.number_input(
                "Frais fixes / rebalancement", 0.0, 1e6, 0.0, 10.0, key="p_fee_fix")
        elif p["fee_mode"] == "Proportionnels":
            p["fee_prop"] = st.number_input(
                "Frais proportionnels (sur turnover)", 0.0, 0.05, 0.0010,
                0.0005, format="%.4f", key="p_fee_prop")

    return p, safe_label


# ---------------------------------------------------------------------------
# Sélection de la fenêtre de dates
# ---------------------------------------------------------------------------
def date_window(series: pd.Series, key_prefix=""):
    dmin, dmax = series.index.min().date(), series.index.max().date()
    c1, c2 = st.columns(2)
    start = c1.date_input("Date de début", dmin, min_value=dmin, max_value=dmax,
                          key=f"{key_prefix}start")
    end = c2.date_input("Date de fin", dmax, min_value=dmin, max_value=dmax,
                        key=f"{key_prefix}end")
    return pd.Timestamp(start), pd.Timestamp(end)


def slice_series(series, start, end):
    return series[(series.index >= start) & (series.index <= end)]


# ---------------------------------------------------------------------------
# Graphiques (Plotly)
# ---------------------------------------------------------------------------
def plot_main(hist: pd.DataFrame, bench: pd.Series | None, capital: float):
    """Portefeuille CPPI vs benchmark + floor, sur 2 sous-graphes empilés."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
        subplot_titles=("Portefeuille, floor & benchmark", "Exposition risquée"))

    fig.add_trace(go.Scatter(x=hist.index, y=hist["value"], name="CPPI",
                             line=dict(color="#2563eb", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["floor"], name="Floor",
                             line=dict(color="#dc2626", width=1.5, dash="dash")),
                  row=1, col=1)

    if bench is not None:
        b = bench.reindex(hist.index).ffill()
        b = b / b.iloc[0] * capital  # rebasé sur le capital initial
        fig.add_trace(go.Scatter(x=b.index, y=b, name="Benchmark (rebasé)",
                                 line=dict(color="#6b7280", width=1.3)), row=1, col=1)

    fig.add_trace(go.Scatter(x=hist.index, y=hist["risky_weight"] * 100,
                             name="% risqué", fill="tozeroy",
                             line=dict(color="#16a34a", width=1)), row=2, col=1)

    fig.update_yaxes(title_text="Valeur", row=1, col=1)
    fig.update_yaxes(title_text="% risqué", row=2, col=1, range=[0, None])
    fig.update_layout(height=560, hovermode="x unified",
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_drawdown(hist: pd.DataFrame):
    V = hist["value"]
    dd = (V / V.cummax() - 1.0) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown",
                             fill="tozeroy", line=dict(color="#dc2626", width=1)))
    fig.update_layout(height=260, yaxis_title="Drawdown (%)",
                      margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------
# Onglet 1 — Backtest unique
# ---------------------------------------------------------------------------
def tab_single(files, risky_label, bench_label, p, safe_label):
    risky = load_series(files[risky_label])
    bench = load_series(files[bench_label])
    safe = load_series(files[safe_label]) if (p["use_safe_index"] and safe_label) else None

    start, end = date_window(risky, "single_")
    risky_s = slice_series(risky, start, end)
    bench_s = slice_series(bench, start, end)
    safe_s = slice_series(safe, start, end) if safe is not None else None

    if len(risky_s) < 2:
        st.warning("Fenêtre de dates trop courte.")
        return

    hist = cppi.run_cppi(risky_s, p, safe_prices=safe_s)
    metrics = cppi.compute_metrics(hist, p, benchmark_prices=bench_s)

    # --- KPI principaux ---
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Valeur finale", f"{metrics['Valeur finale']:,.0f}")
    k2.metric("Rendement total", f"{metrics['Rendement total']*100:.1f} %")
    k3.metric("CAGR", f"{metrics['Rendement annualisé (CAGR)']*100:.2f} %")
    k4.metric("Drawdown max", f"{metrics['Drawdown maximal']*100:.1f} %")
    over = metrics.get("Sur/sous-performance (total)")
    k5.metric("vs benchmark", f"{over*100:+.1f} %" if over is not None else "—")

    # --- Graphiques ---
    st.plotly_chart(plot_main(hist, bench_s, p["capital"]), use_container_width=True)
    with st.expander("Drawdown", expanded=False):
        st.plotly_chart(plot_drawdown(hist), use_container_width=True)

    # --- Tableau de synthèse + export ---
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Indicateurs synthétiques")
        synth = cppi.metrics_to_frame(metrics)
        st.dataframe(synth, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Synthèse (CSV)", synth.to_csv(index=False).encode("utf-8"),
            file_name="cppi_synthese.csv", mime="text/csv")

    with right:
        st.subheader("Historique détaillé")
        detail = hist.copy()
        detail.index = detail.index.date
        show_cols = ["index", "index_return", "value", "floor", "cushion",
                     "risky_value", "safe_value", "risky_weight", "safe_weight"]
        st.dataframe(detail[show_cols].round(4), use_container_width=True, height=320)
        st.download_button(
            "⬇ Historique détaillé (CSV)",
            hist.to_csv().encode("utf-8"),
            file_name="cppi_historique.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Onglet 2 — Comparaison de scénarios
# ---------------------------------------------------------------------------
def tab_compare(files, risky_label, bench_label, base_p, safe_label):
    st.markdown("Comparez plusieurs paramétrages sur la **même fenêtre** "
                "que l'onglet *Backtest*. Faites varier un paramètre clé :")

    risky = load_series(files[risky_label])
    bench = load_series(files[bench_label])
    safe = load_series(files[safe_label]) if (base_p["use_safe_index"] and safe_label) else None
    start, end = date_window(risky, "cmp_")
    risky_s = slice_series(risky, start, end)
    bench_s = slice_series(bench, start, end)
    safe_s = slice_series(safe, start, end) if safe is not None else None

    c1, c2 = st.columns(2)
    varying = c1.selectbox(
        "Paramètre à faire varier",
        ["Multiplicateur", "Niveau de floor", "Exposition max", "Lock-in on/off"])
    values_txt = c2.text_input(
        "Valeurs à tester (séparées par des virgules)",
        value={"Multiplicateur": "3, 4, 5",
               "Niveau de floor": "0.80, 0.90",
               "Exposition max": "1.0, 1.5",
               "Lock-in on/off": "off, on"}[varying])

    if not st.button("▶ Lancer la comparaison", type="primary"):
        return

    fig = go.Figure()
    rows = []
    for raw in [v.strip() for v in values_txt.split(",") if v.strip()]:
        p = dict(base_p)  # copie du paramétrage courant
        try:
            if varying == "Multiplicateur":
                p["multiplier"] = float(raw); label = f"m={raw}"
            elif varying == "Niveau de floor":
                p["floor_mode"] = "% du capital initial (fixe)"
                p["floor_pct"] = float(raw); label = f"floor={float(raw)*100:.0f}%"
            elif varying == "Exposition max":
                p["exp_max"] = float(raw); label = f"expo_max={raw}"
            else:  # Lock-in
                p["use_lockin"] = (raw.lower() in ("on", "1", "true", "oui"))
                label = f"lockin={'on' if p['use_lockin'] else 'off'}"
        except ValueError:
            st.warning(f"Valeur ignorée : {raw}")
            continue

        hist = cppi.run_cppi(risky_s, p, safe_prices=safe_s)
        m = cppi.compute_metrics(hist, p, benchmark_prices=bench_s)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["value"], name=label))
        rows.append({
            "Scénario": label,
            "Valeur finale": round(m["Valeur finale"], 0),
            "Rendement total": f"{m['Rendement total']*100:.1f} %",
            "CAGR": f"{m['Rendement annualisé (CAGR)']*100:.2f} %",
            "Volatilité": f"{m['Volatilité annualisée']*100:.1f} %",
            "Drawdown max": f"{m['Drawdown maximal']*100:.1f} %",
            "Sharpe": round(m["Sharpe (simplifié)"], 2),
            "Breaches": m["Nb de breaches du floor"],
        })

    # Benchmark rebasé en référence
    if bench_s is not None and len(bench_s) >= 2:
        b = bench_s / bench_s.iloc[0] * base_p["capital"]
        fig.add_trace(go.Scatter(x=b.index, y=b, name="Benchmark",
                                 line=dict(color="#9ca3af", dash="dot")))

    fig.update_layout(height=480, hovermode="x unified",
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(l=10, r=10, t=40, b=10),
                      yaxis_title="Valeur du portefeuille")
    st.plotly_chart(fig, use_container_width=True)

    comp = pd.DataFrame(rows)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.download_button("⬇ Comparaison (CSV)",
                       comp.to_csv(index=False).encode("utf-8"),
                       file_name="cppi_comparaison.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Onglet 3 — Aperçu des données
# ---------------------------------------------------------------------------
def tab_data(files, labels):
    sel = st.selectbox("Indice à inspecter", labels)
    s = load_series(files[sel])
    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", f"{len(s):,}")
    c2.metric("Début", str(s.index.min().date()))
    c3.metric("Fin", str(s.index.max().date()))
    fig = go.Figure(go.Scatter(x=s.index, y=s, line=dict(color="#2563eb")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Niveau")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(s.reset_index().rename(columns={"date": "Date", 0: "Valeur"}),
                 use_container_width=True, height=300)


# ---------------------------------------------------------------------------
# Application principale
# ---------------------------------------------------------------------------
def main():
    st.title("📈 Laboratoire interactif CPPI")
    st.caption("Backtest historique paramétrable d'une stratégie Constant "
               "Proportion Portfolio Insurance.")

    files, risky_label, bench_label, ct_default, labels = sidebar_data_selection()
    p, safe_label = sidebar_parameters(files, labels, ct_default)

    tab1, tab2, tab3 = st.tabs(["🎯 Backtest", "⚖️ Comparaison", "🗂️ Données"])
    with tab1:
        tab_single(files, risky_label, bench_label, p, safe_label)
    with tab2:
        tab_compare(files, risky_label, bench_label, p, safe_label)
    with tab3:
        tab_data(files, labels)


if __name__ == "__main__":
    main()
