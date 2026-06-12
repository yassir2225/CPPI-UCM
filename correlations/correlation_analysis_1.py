#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Etude de correlation MASI x MBI (par maturite)
 Lecture critique CPPI - Acte III.C : correlation actions / obligataire
 Upline Capital Management
================================================================================

Ce script est ADAPTATIF :
  - il decouvre automatiquement les fichiers MASI_*.xls(x) et MBI_*_*.xls(x)
    dans le dossier --data ;
  - il accepte .xls et .xlsx ;
  - il s'aligne sur le calendrier de bourse du MASI (les indices MBI sont
    calendaires : on ne garde que les jours ou MASI cote) ;
  - il calcule les correlations a 3 frequences (quotidien / hebdo / mensuel) ;
  - il produit : un tableau de resultats (CSV), des figures PNG, et le fichier
    processed_data.json qui alimente le dashboard interactif HTML.

Pourquoi la maturite compte
  Les indices MBI sont des indices de PERFORMANCE (total return) : ils integrent
  le portage (carry) quotidien. Au court terme (CT) la serie est quasi pure
  carry -> correlation mecaniquement proche de zero. Le signal "prix" (donc le
  co-mouvement avec les actions) est porte par la DURATION -> il croit avec la
  maturite. C'est pourquoi on traite chaque maturite separement.

Usage
  python correlation_analysis.py --data ./donnees --out ./resultats
  python correlation_analysis.py --data . --freq M --window 24 --method pearson

Dependances : pandas, numpy, scipy, matplotlib, openpyxl (et xlrd pour .xls)
  pip install pandas numpy scipy matplotlib openpyxl xlrd
================================================================================
"""

import argparse
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config ---
# Ordre canonique des maturites (du plus court au plus long).
MAT_ORDER = ["CT", "MT", "MLT", "LT", "GLOBAL"]
EQUITY_KEY = "MASI"            # actif risque
REGIME_SPLIT = "2022-01-01"    # bascule de regime (remontee inflation/taux)


# -------------------------------------------------------------- decouverte ---
def discover_files(folder):
    """Retourne {cle: chemin}. Cle = MASI ou maturite (CT/MT/MLT/LT/GLOBAL)."""
    found = {}
    for path in glob.glob(os.path.join(folder, "*.xls*")):
        name = os.path.basename(path).upper()
        if "MASI" in name:
            found[EQUITY_KEY] = path
            continue
        if "MBI" in name:
            # extrait la maturite : MBI_CT_HISTO -> CT, MBI_GLOBAL -> GLOBAL
            m = re.search(r"MBI[_\-]?([A-Z]+)", name)
            if m:
                tag = m.group(1)
                # GLOBAL peut s'ecrire GLOBAL/GLOB
                if tag.startswith("GLOB"):
                    tag = "GLOBAL"
                if tag in MAT_ORDER:
                    found[tag] = path
    return found


def load_series(path):
    """Charge une serie temporelle (DATE, CODE, VALEUR REF) -> pd.Series."""
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip().upper() for c in df.columns]
    # trouve la colonne date et la colonne valeur de maniere souple
    date_col = next((c for c in df.columns if "DATE" in c), df.columns[0])
    val_col = next((c for c in df.columns if "VAL" in c), df.columns[-1])
    df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="coerce")
    if df[date_col].isna().mean() > 0.5:                 # format date alternatif
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    s = (df.dropna(subset=[date_col])
           .set_index(date_col)[val_col]
           .astype(float)
           .sort_index())
    return s[~s.index.duplicated(keep="last")]


# ---------------------------------------------------------------- rendements -
def build_panel(files):
    """Charge tout, aligne sur le calendrier MASI, renvoie les niveaux."""
    series = {k: load_series(p) for k, p in files.items()}
    levels = pd.DataFrame(series)
    if EQUITY_KEY not in levels:
        sys.exit("ERREUR : fichier MASI introuvable dans le dossier --data.")
    cal = levels[EQUITY_KEY].dropna().index            # jours de bourse
    levels = levels.reindex(cal).dropna(how="any")
    return levels


def log_returns(levels, freq):
    """freq in {'D','W','M'} -> rendements log."""
    if freq == "D":
        px = levels
    elif freq == "W":
        px = levels.resample("W-FRI").last()
    else:
        px = levels.resample("ME").last()
    return np.log(px).diff().dropna()


# ------------------------------------------------------------------- stats ---
def fisher_ci(r, n, alpha=0.05):
    if n < 5:
        return (np.nan, np.nan)
    from scipy import stats
    z = np.arctanh(np.clip(r, -0.999, 0.999))
    se = 1.0 / np.sqrt(n - 3)
    q = stats.norm.ppf(1 - alpha / 2)
    return (np.tanh(z - q * se), np.tanh(z + q * se))


def static_table(rets, mats):
    """Pearson + Spearman + IC + p-value, par maturite."""
    from scipy import stats
    rows = []
    x = rets[EQUITY_KEY].values
    for m in mats:
        y = rets[m].values
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        lo, hi = fisher_ci(pr, len(x))
        rows.append(dict(maturite=m, pearson=pr, p_pearson=pp,
                         spearman=sr, p_spearman=sp,
                         ci_low=lo, ci_high=hi, n=len(x)))
    return pd.DataFrame(rows)


def rolling_corr(rets, mat, window, method="pearson"):
    x = rets[EQUITY_KEY]
    y = rets[mat]
    if method == "spearman":
        return x.rolling(window).corr(y, method="spearman") \
            if False else _roll_spearman(x, y, window)
    return x.rolling(window).corr(y)


def _roll_spearman(x, y, window):
    out = pd.Series(index=x.index, dtype=float)
    xv, yv = x.values, y.values
    from scipy import stats
    for i in range(window - 1, len(xv)):
        out.iloc[i] = stats.spearmanr(xv[i - window + 1:i + 1],
                                      yv[i - window + 1:i + 1]).statistic
    return out


def cross_correlation(rets, mat, max_lag=6):
    """k>0 : MASI precede l'obligataire de k pas."""
    x = (rets[EQUITY_KEY] - rets[EQUITY_KEY].mean()) / rets[EQUITY_KEY].std()
    y = (rets[mat] - rets[mat].mean()) / rets[mat].std()
    xv, yv, n = x.values, y.values, len(x)
    lags = list(range(-max_lag, max_lag + 1))
    vals = []
    for k in lags:
        if k == 0:
            c = np.corrcoef(xv, yv)[0, 1]
        elif k > 0:
            c = np.corrcoef(xv[:n - k], yv[k:])[0, 1]
        else:
            c = np.corrcoef(xv[-k:], yv[:n + k])[0, 1]
        vals.append(c)
    return lags, vals


def regime_split(rets, mats, split=REGIME_SPLIT):
    pre = rets.loc[:split]
    post = rets.loc[split:]
    rows = []
    from scipy import stats
    for m in mats:
        rows.append(dict(maturite=m,
                         pre=stats.pearsonr(pre[EQUITY_KEY], pre[m])[0],
                         post=stats.pearsonr(post[EQUITY_KEY], post[m])[0],
                         n_pre=len(pre), n_post=len(post)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- figures ---
def make_figures(levels, mats, outdir, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.grid": True, "grid.alpha": .25})
    col = {"CT": "#6fb1ff", "MT": "#e0a800", "MLT": "#f7795b",
           "LT": "#d63864", "GLOBAL": "#7a5cff"}
    rm = log_returns(levels, "M")

    # 1) correlation glissante (mensuelle)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for m in mats:
        rc = rolling_corr(rm, m, window)
        ax.plot(rc.index, rc.values, label=f"MASI-{m}", color=col.get(m), lw=1.8)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.axvspan(pd.Timestamp(REGIME_SPLIT), rm.index.max(),
               color="red", alpha=.07, label="régime post-2022")
    ax.set_title(f"Corrélation glissante MASI x MBI (mensuel, fenêtre {window} mois)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_rolling.png")); plt.close(fig)

    # 2) correlation par maturite x regime
    rs = regime_split(rm, mats)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(mats)); w = .38
    ax.bar(x - w/2, rs["pre"], w, label="2002–2021", color="#3ad29f")
    ax.bar(x + w/2, rs["post"], w, label="2022 →", color="#ff6b6b")
    ax.axhline(0, color="grey", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(mats)
    ax.set_title("Corrélation par maturité et par régime (mensuel)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_regime.png")); plt.close(fig)

    # 3) niveaux normalises
    base = levels / levels.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(base.index, base[EQUITY_KEY], color="#1f6feb", lw=2, label="MASI")
    for m in mats:
        ax.plot(base.index, base[m], color=col.get(m), lw=1.2, label=m)
    ax.axvline(pd.Timestamp(REGIME_SPLIT), color="red", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_title("Niveaux normalisés base 100")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_levels.png")); plt.close(fig)


# ------------------------------------------------- export pour le dashboard --
def export_json(levels, mats, outdir):
    R = {f: log_returns(levels, f) for f in ["D", "W", "M"]}
    exp = {"meta": {"start": str(levels.index.min().date()),
                    "end": str(levels.index.max().date()),
                    "maturities": mats, "n_levels": len(levels)},
           "series": {}, "levels": {}, "leadlag": {}}
    for f, r in R.items():
        blk = {"dates": [d.strftime("%Y-%m-%d") for d in r.index],
               EQUITY_KEY: [round(v, 8) for v in r[EQUITY_KEY]]}
        for m in mats:
            blk[m] = [round(v, 8) for v in r[m]]
        exp["series"][f] = blk
    base = levels / levels.iloc[0] * 100
    exp["levels"]["dates"] = [d.strftime("%Y-%m-%d") for d in base.index]
    for c in [EQUITY_KEY] + mats:
        exp["levels"][c] = [round(v, 4) for v in base[c]]
    lags = list(range(-6, 7))
    exp["leadlag"] = {"lags": lags,
                      "data": {m: [round(v, 4) for v in cross_correlation(R["M"], m)[1]]
                               for m in mats}}
    with open(os.path.join(outdir, "processed_data.json"), "w") as fh:
        json.dump(exp, fh)


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description="Etude correlation MASI x MBI")
    ap.add_argument("--data", default=".", help="dossier des fichiers Excel")
    ap.add_argument("--out", default="./resultats", help="dossier de sortie")
    ap.add_argument("--window", type=int, default=24, help="fenetre glissante (obs)")
    ap.add_argument("--freq", default="M", choices=["D", "W", "M"],
                    help="frequence pour le tableau statique imprime")
    ap.add_argument("--method", default="pearson",
                    choices=["pearson", "spearman"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = discover_files(args.data)
    if EQUITY_KEY not in files:
        sys.exit(f"Aucun fichier MASI trouve dans {args.data}")
    mats = [m for m in MAT_ORDER if m in files and m != EQUITY_KEY]
    print(f"Fichiers detectes : MASI + {mats}")

    levels = build_panel(files)
    print(f"Periode alignee   : {levels.index.min().date()} -> "
          f"{levels.index.max().date()}  ({len(levels)} obs.)")

    # tableau statique a la frequence demandee
    rets = log_returns(levels, args.freq)
    tbl = static_table(rets, mats)
    print(f"\nCorrelations statiques ({args.freq}) :")
    print(tbl.to_string(index=False,
          formatters={c: "{:+.3f}".format for c in
                      ["pearson", "spearman", "ci_low", "ci_high"]}))

    # split de regime (toujours mensuel, plus lisible)
    rm = log_returns(levels, "M")
    rs = regime_split(rm, mats)
    print("\nBascule de regime (mensuel) :")
    print(rs.to_string(index=False,
          formatters={"pre": "{:+.3f}".format, "post": "{:+.3f}".format}))

    # exports
    tbl.to_csv(os.path.join(args.out, "correlations_statiques.csv"), index=False)
    rs.to_csv(os.path.join(args.out, "correlations_regime.csv"), index=False)
    make_figures(levels, mats, args.out, args.window)
    export_json(levels, mats, args.out)
    print(f"\nSorties ecrites dans : {os.path.abspath(args.out)}")
    print("  - correlations_statiques.csv / correlations_regime.csv")
    print("  - fig_rolling.png / fig_regime.png / fig_levels.png")
    print("  - processed_data.json  (alimente le dashboard HTML)")


if __name__ == "__main__":
    main()
