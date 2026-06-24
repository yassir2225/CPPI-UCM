# -*- coding: utf-8 -*-
"""
increments.py — Moteur de diagnostic de l'hypothèse d'INCRÉMENTS INDÉPENDANTS
(hypothèse iid ; « log(prix) est un processus de Lévy »).

Fonctions PURES, aucune dépendance d'interface. Chaque test renvoie un objet
``TestResult`` structuré : { statistique, p_value (ou IC), taille_effet, n,
verdict, details, limites, H0, H1, seuil de matérialité }.

Cadre : validation de modèle au sens SR 11-7 (Conceptual Soundness +
Outcomes Analysis). On challenge l'hypothèse implicite du pricer CPPI par
opérateur de Markov (Paulot–Lacroze) : sous incréments iid, X = coussin/seuil
est une chaîne de Markov 1-D et tout le pricing rapide en découle.

Principe de droiture (cf. cahier des charges) :
- significativité statistique ≠ matérialité : on reporte TOUJOURS une taille
  d'effet à côté de la p-value, et le verdict se fonde sur la matérialité.
- non-normalité ≠ non-indépendance : queues épaisses = famille de Lévy à sauts
  (Merton/Kou), PAS une violation des incréments indépendants.
- clustering de volatilité = traité par l'extension à régimes (Paulot §7.1),
  pas une invalidation du modèle.
"""

from __future__ import annotations

import os
import glob
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd

from scipy import stats as sps
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss, bds, acf

# ----------------------------------------------------------------------------
# Constantes : verdicts (vocabulaire en clair) et reproductibilité
# ----------------------------------------------------------------------------
V_OK = "OK (hypothèse tenue)"
V_MAT = "Violation matérielle"
V_NONMAT = "Violation non matérielle"
V_EXT = "Traitée par extension"
V_IND = "Indéterminé"
V_NT = "Non testé"
V_CHAR = "Caractérisation"  # axe descriptif (loi marginale) : n'invalide pas iid

SEED_DEFAUT = 12345  # graine fixe, affichée dans l'interface

# Périodes par an selon la fréquence de rééchantillonnage
_PPA = {"D": 252.0, "W": 52.0, "M": 12.0}
_RESAMPLE_RULE = {"W": "W-FRI", "M": "ME"}
_FREQ_LABEL = {"D": "quotidien", "W": "hebdomadaire", "M": "mensuel"}


def periods_per_year(freq: str) -> float:
    return _PPA.get(freq, 252.0)


# ----------------------------------------------------------------------------
# Paramètres / seuils de matérialité (tunables ; valeurs par défaut documentées)
# ----------------------------------------------------------------------------
@dataclass
class Params:
    """Réglages et seuils de matérialité. Chaque seuil est jugemental et son
    origine est annoncée à l'écran (cf. esprit du livrable, point 3)."""
    alpha: float = 0.05                 # niveau de significativité
    lb_lags: tuple = (1, 5, 10, 20)     # retards Ljung-Box
    vr_qs: tuple = (2, 4, 8, 16)        # horizons du variance ratio
    bds_max_dim: int = 5                # dimensions de plongement BDS 2..m
    arch_lags: int = 12                 # retards ARCH-LM
    acf_material: float = 0.10          # |rho(1)| matériel (auto-corr. directe)
    vr_material: float = 0.20           # |VR(q)-1| matériel
    hurst_material: float = 0.10        # |H-0.5| matériel
    hurst_min_n: int = 2000             # n minimal pour fiabiliser Hurst
    maturity_years: float = 10.0        # maturité produit (matérialité demi-vie)
    rs_min_window: int = 10             # plus petite fenêtre R/S et DFA
    gph_power: float = 0.5              # bande GPH : m = N**power
    n_boot: int = 500                   # tirages bootstrap / surrogate
    seed: int = SEED_DEFAUT


# ----------------------------------------------------------------------------
# Objet résultat structuré
# ----------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    axis: str                                   # axe (Indépendance, Volatilité, …)
    h0: str
    h1: str
    statistic: Union[float, dict, None] = None
    p_value: Optional[float] = None
    ci: Optional[tuple] = None                  # intervalle de confiance / bande
    effect_size: Union[float, dict, None] = None
    effect_label: str = ""
    n: int = 0
    threshold: str = ""                         # seuil de matérialité (énoncé)
    verdict: str = V_NT
    interpretation: str = ""
    limits: str = ""
    details: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """Ligne synthétique pour la table livrable (cellules en chaînes lisibles,
        compatibles Arrow/Streamlit)."""
        def _num(x):
            return f"{x:.4f}" if isinstance(x, float) else str(x)

        def _cell(x):
            if x is None:
                return "—"
            if isinstance(x, dict):
                return "; ".join(f"{k}={_num(v)}" for k, v in x.items())
            if isinstance(x, (tuple, list)):
                return "; ".join(_num(v) for v in x)
            return _num(x)

        if self.p_value is not None:
            p_ic = f"p={self.p_value:.4g}"
        elif self.ci is not None:
            p_ic = _cell(self.ci)
        else:
            p_ic = "—"
        eff = _cell(self.effect_size)
        if self.effect_label and eff != "—":
            eff = f"{self.effect_label}: {eff}"
        return {
            "Axe": self.axis,
            "Test": self.name,
            "Statistique": _cell(self.statistic),
            "p-value / IC": p_ic,
            "Taille d'effet": eff,
            "n": self.n,
            "Seuil": self.threshold,
            "VERDICT": self.verdict,
        }


# ============================================================================
# 1. CHARGEMENT DES DONNÉES (robuste, auto-détection des colonnes)
# ============================================================================
_DATE_HINTS = ("date", "séance", "seance", "jour", "day", "time", "période", "periode")
_VALUE_HINTS = ("valeur", "value", "cours", "close", "prix", "price", "vl",
                "indice", "index", "ref", "réf", "clôture", "cloture", "last")


def _try_datetime(col: pd.Series) -> Optional[pd.Series]:
    """Tente de parser une colonne en dates (jj/mm/aaaa prioritaire)."""
    if pd.api.types.is_numeric_dtype(col):
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d"):
        try:
            d = pd.to_datetime(col, format=fmt, errors="coerce")
            if d.notna().mean() > 0.9:
                return d
        except Exception:
            pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(col, dayfirst=True, errors="coerce")
    if d.notna().mean() > 0.9:
        return d
    return None


def _detect_columns(df: pd.DataFrame):
    """Renvoie (col_date_parsée, nom_col_valeur).

    Stratégie : nom évocateur d'abord, repli sur « première colonne datable »
    + « dernière colonne numérique » (motif du projet)."""
    # --- colonne date ---
    date_series, date_name = None, None
    for c in df.columns:                                    # priorité au nom
        if any(h in str(c).lower() for h in _DATE_HINTS):
            d = _try_datetime(df[c])
            if d is not None:
                date_series, date_name = d, c
                break
    if date_series is None:                                 # première datable
        for c in df.columns:
            d = _try_datetime(df[c])
            if d is not None:
                date_series, date_name = d, c
                break
    if date_series is None:
        raise ValueError("Aucune colonne datable détectée.")

    # --- colonne valeur ---
    candidates = [c for c in df.columns if c != date_name]
    num_cols = [c for c in candidates
                if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
    if not num_cols:
        raise ValueError("Aucune colonne numérique détectée pour la valeur.")
    val_name = None
    for c in num_cols:                                      # priorité au nom
        if any(h in str(c).lower() for h in _VALUE_HINTS):
            val_name = c
            break
    if val_name is None:                                   # dernière numérique
        val_name = num_cols[-1]
    return date_series, val_name


def load_series(source, sheet=0) -> pd.Series:
    """Charge une série de prix depuis un .xlsx/.xls/.csv.

    ``source`` : chemin OU objet fichier (upload Streamlit).
    Renvoie une pd.Series indexée par DatetimeIndex, triée, dédupliquée.
    Gère openpyxl (.xlsx) et xlrd (.xls)."""
    name = getattr(source, "name", str(source)).lower()
    if name.endswith(".csv"):
        df = pd.read_csv(source, sep=None, engine="python")
    elif name.endswith(".xls"):
        df = pd.read_excel(source, sheet_name=sheet, engine="xlrd")
    else:  # .xlsx par défaut
        df = pd.read_excel(source, sheet_name=sheet, engine="openpyxl")

    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    dates, val_name = _detect_columns(df)
    vals = pd.to_numeric(df[val_name], errors="coerce")

    s = pd.Series(vals.values, index=dates, name=val_name)
    s = s[s.index.notna()].dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s[s > 0]                                            # prix strictement positifs
    return s


def find_default_masi(base_dir: str) -> Optional[str]:
    """Cherche un fichier MASI à côté du script (chemin RELATIF)."""
    for pat in ("*MASI*.xlsx", "*MASI*.xls", "*masi*.xlsx", "*masi*.xls"):
        hits = sorted(glob.glob(os.path.join(base_dir, pat)))
        if hits:
            return hits[0]
    return None


def available_series(base_dir: str) -> dict:
    """Liste {nom_affichage: chemin} des séries .xlsx/.xls présentes."""
    out = {}
    for pat in ("*.xlsx", "*.xls"):
        for p in sorted(glob.glob(os.path.join(base_dir, pat))):
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


# ============================================================================
# 2. RENDEMENTS (rééchantillonnage à la fréquence de rebalancement)
# ============================================================================
def resample_prices(s: pd.Series, freq: str) -> pd.Series:
    """Rééchantillonne les prix. D = série brute (évite des NaN de week-end)."""
    if freq == "D":
        return s.dropna()
    rule = _RESAMPLE_RULE[freq]
    return s.resample(rule).last().dropna()


def compute_returns(s: pd.Series, kind: str = "log", freq: str = "M") -> pd.Series:
    """Rendements log (défaut, cohérent avec « log(prix) Lévy ») ou simples."""
    p = resample_prices(s, freq)
    if kind == "log":
        r = np.log(p).diff()
    else:
        r = p.pct_change()
    return r.dropna()


# ============================================================================
# 3. ESTIMATEURS IMPLÉMENTÉS À LA MAIN (transparence + robustesse déploiement)
# ============================================================================
def variance_ratio_lomackinlay(returns: np.ndarray, q: int):
    """Variance Ratio de Lo–MacKinlay (1988), statistique ROBUSTE à
    l'hétéroscédasticité.

    Pour des rendements r_t (t=1..T), de moyenne mu :
        VR(q) = sigma2_c(q) / sigma2_a
        sigma2_a   = (1/(T-1)) * Σ (r_t - mu)^2                         [1 période]
        sigma2_c(q)= (1/m) * Σ_{k=q}^{T} ( Σ_{i=0}^{q-1} r_{k-i} - q*mu )^2
                     m = q*(T-q+1)*(1 - q/T)                            [q périodes]
    Écart-type robuste (hétéroscédasticité-consistant) :
        delta_j = [ Σ_{t=j+1}^{T} (r_t-mu)^2 (r_{t-j}-mu)^2 ]
                  / [ Σ_{t=1}^{T} (r_t-mu)^2 ]^2
        theta(q)= Σ_{j=1}^{q-1} ( 2*(q-j)/q )^2 * delta_j
        z(q)    = (VR(q) - 1) / sqrt(theta(q))   ~ N(0,1)
    (Sous homoscédasticité, theta(q) -> 2*(2q-1)*(q-1)/(3qT), variance classique.)
    """
    r = np.asarray(returns, dtype=float)
    T = r.size
    if T < q + 2:
        return dict(q=q, vr=np.nan, z=np.nan, p_value=np.nan, n=T)
    mu = r.mean()
    dev = r - mu
    sigma2_a = np.sum(dev**2) / (T - 1)

    # somme glissante de q rendements consécutifs
    csum = np.concatenate(([0.0], np.cumsum(r)))            # csum[k]=Σ_{i<=k} r_i
    q_sums = csum[q:] - csum[:-q]                           # longueur T-q+1
    m = q * (T - q + 1) * (1.0 - q / T)
    sigma2_c = np.sum((q_sums - q * mu) ** 2) / m
    vr = sigma2_c / sigma2_a

    denom = np.sum(dev**2) ** 2
    theta = 0.0
    for j in range(1, q):
        num = np.sum((dev[j:] ** 2) * (dev[:-j] ** 2))
        delta_j = num / denom
        theta += (2.0 * (q - j) / q) ** 2 * delta_j
    z = (vr - 1.0) / np.sqrt(theta) if theta > 0 else np.nan
    p = 2.0 * (1.0 - sps.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return dict(q=q, vr=float(vr), z=float(z), p_value=float(p), n=T)


def runs_test_ww(returns: np.ndarray):
    """Test des runs de Wald–Wolfowitz sur le SIGNE des rendements (zéros exclus).

        n1 = #(+), n2 = #(-), n = n1+n2, R = nombre de runs
        E[R]   = 2*n1*n2/n + 1
        Var[R] = 2*n1*n2*(2*n1*n2 - n) / ( n^2 * (n-1) )
        z = (R - E[R]) / sqrt(Var[R])    ~ N(0,1)   (sans correction de continuité)
    """
    r = np.asarray(returns, dtype=float)
    signs = np.sign(r)
    signs = signs[signs != 0]
    n = signs.size
    n1 = int(np.sum(signs > 0))
    n2 = int(np.sum(signs < 0))
    if n1 == 0 or n2 == 0 or n < 3:
        return dict(runs=np.nan, expected=np.nan, z=np.nan, p_value=np.nan, n=n)
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    er = 2.0 * n1 * n2 / n + 1.0
    vr = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n**2 * (n - 1.0))
    z = (runs - er) / np.sqrt(vr) if vr > 0 else np.nan
    p = 2.0 * (1.0 - sps.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return dict(runs=runs, expected=float(er), z=float(z), p_value=float(p), n=n)


def _log_window_sizes(N: int, min_w: int, n_sizes: int = 18):
    """Fenêtres espacées géométriquement entre min_w et N//2 (entiers uniques)."""
    hi = max(min_w + 1, N // 2)
    sizes = np.unique(np.floor(np.geomspace(min_w, hi, n_sizes)).astype(int))
    return sizes[sizes >= min_w]


def hurst_rs(x: np.ndarray, min_w: int = 10):
    """Exposant de Hurst par analyse R/S (rescaled range).

    Pour chaque fenêtre n : on découpe la série en blocs non chevauchants ;
    par bloc, déviations cumulées Z_k = Σ_{i<=k}(x_i - moyenne_bloc),
    étendue R = max Z - min Z, écart-type S du bloc, ratio R/S.
    On moyenne R/S sur les blocs, puis régresse log(R/S) sur log(n) : pente = H.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    sizes = _log_window_sizes(N, min_w)
    logn, logrs = [], []
    for n in sizes:
        nb = N // n
        if nb < 1:
            continue
        rs_vals = []
        for b in range(nb):
            seg = x[b * n:(b + 1) * n]
            z = np.cumsum(seg - seg.mean())
            rng = z.max() - z.min()
            s = seg.std(ddof=0)
            if s > 0 and rng > 0:
                rs_vals.append(rng / s)
        if rs_vals:
            logn.append(np.log(n))
            logrs.append(np.log(np.mean(rs_vals)))
    if len(logn) < 3:
        return dict(H=np.nan, n=N, logn=[], logrs=[])
    H, c = np.polyfit(logn, logrs, 1)
    return dict(H=float(H), intercept=float(c), n=N,
                logn=list(map(float, logn)), logrs=list(map(float, logrs)))


def hurst_dfa(x: np.ndarray, min_w: int = 10):
    """Exposant de Hurst par DFA (Detrended Fluctuation Analysis, ordre 1).

    Profil intégré Y_k = Σ_{i<=k}(x_i - moyenne). Pour chaque taille de boîte n,
    on ajuste une tendance linéaire par boîte et calcule la RMS des résidus :
        F(n) = sqrt( moyenne sur boîtes des résidus^2 ).
    Régression log F(n) sur log n : pente = alpha.
    DFA inclut l'intégration : appliqué aux RENDEMENTS (bruit), alpha = H,
    alpha = 0.5 = incréments non corrélés.

    Détendance vectorisée (OLS en forme close, sans boucle sur les boîtes) :
    pour idx=0..n-1 identique à toutes les boîtes,
        pente   = (n·Σ(idx·y) - Σidx·Σy) / (n·Σidx² - (Σidx)²)
        ordonnée= (Σy - pente·Σidx) / n
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    Y = np.cumsum(x - x.mean())
    sizes = _log_window_sizes(N, max(min_w, 4))
    logn, logF = [], []
    for n in sizes:
        nb = N // n
        if nb < 1 or n < 4:
            continue
        B = Y[:nb * n].reshape(nb, n)
        idx = np.arange(n, dtype=float)
        Sx = idx.sum()
        Sxx = (idx * idx).sum()
        denom = n * Sxx - Sx * Sx
        Sy = B.sum(axis=1)
        Sxy = B @ idx
        slope = (n * Sxy - Sx * Sy) / denom
        intercept = (Sy - slope * Sx) / n
        resid = B - (intercept[:, None] + slope[:, None] * idx[None, :])
        F = np.sqrt(np.mean(resid * resid))
        if F > 0:
            logn.append(np.log(n))
            logF.append(np.log(F))
    if len(logn) < 3:
        return dict(H=np.nan, n=N, logn=[], logF=[])
    alpha, c = np.polyfit(logn, logF, 1)
    return dict(H=float(alpha), intercept=float(c), n=N,
                logn=list(map(float, logn)), logF=list(map(float, logF)))


def gph_estimator(x: np.ndarray, power: float = 0.5):
    """Estimateur GPH (Geweke–Porter-Hudak, 1983) du paramètre d ; H = d + 0.5.

    Périodogramme I(w_k) aux fréquences de Fourier w_k = 2*pi*k/N, k=1..m,
    avec bande m = floor(N**power). Régression :
        log I(w_k) = c - d * log( 4 sin^2(w_k/2) ) + erreur
    => d = -pente. Erreur-type asymptotique : sqrt( (pi^2/6) / Σ(X_k - Xbar)^2 ),
    X_k = log(4 sin^2(w_k/2)).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    N = x.size
    m = int(np.floor(N ** power))
    m = max(4, min(m, N // 2 - 1))
    fft = np.fft.fft(x)
    I = (np.abs(fft) ** 2) / (2.0 * np.pi * N)              # périodogramme
    k = np.arange(1, m + 1)
    w = 2.0 * np.pi * k / N
    X = np.log(4.0 * np.sin(w / 2.0) ** 2)
    Y = np.log(I[1:m + 1])
    Xc = X - X.mean()
    Sxx = np.sum(Xc**2)
    slope = np.sum(Xc * (Y - Y.mean())) / Sxx
    d = -slope
    se = np.sqrt((np.pi**2 / 6.0) / Sxx)
    return dict(d=float(d), H=float(d + 0.5), se=float(se), m=m, n=N)


def _surrogate_hurst_band(x: np.ndarray, estimator, params: Params):
    """Bande iid par données de substitution (surrogate) : on PERMUTE les
    rendements (mêmes valeurs marginales, dépendance détruite) et on recalcule H.
    Renvoie l'IC [2.5%, 97.5%] de H sous l'hypothèse iid. (Reproductible : graine.)"""
    rng = np.random.default_rng(params.seed)
    x = np.asarray(x, dtype=float)
    vals = []
    for _ in range(params.n_boot):
        xp = rng.permutation(x)
        res = estimator(xp, params.rs_min_window) if estimator is not hurst_gph_wrap \
            else estimator(xp, params.gph_power)
        h = res["H"]
        if np.isfinite(h):
            vals.append(h)
    if len(vals) < 10:
        return (np.nan, np.nan), vals
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi)), vals


def hurst_gph_wrap(x, power):           # adaptateur de signature pour le surrogate
    return gph_estimator(x, power)


# ============================================================================
# 4. TESTS -> TestResult (verdict individuel + seuil de matérialité énoncé)
# ============================================================================
AX_IND = "Indépendance sérielle"
AX_VOL = "Hétéroscédasticité"
AX_STA = "Stationnarité / retour à la moyenne"
AX_MEM = "Mémoire longue"
AX_DIS = "Forme de la distribution"
AX_LOC = "Volatilité locale"


# ---- A. Indépendance sérielle ----------------------------------------------
def test_ljung_box(returns: pd.Series, p: Params) -> TestResult:
    r = returns.values
    n = r.size
    lags = [L for L in p.lb_lags if L < n - 1]
    lb = acorr_ljungbox(r, lags=lags, return_df=True)
    pmin = float(lb["lb_pvalue"].min())
    ac = acf(r, nlags=max(lags), fft=False)
    rho1 = float(ac[1])
    max_abs = float(np.max(np.abs(ac[1:max(lags) + 1])))
    sig = pmin < p.alpha
    if not sig:
        verdict = V_OK
    elif max_abs >= p.acf_material:
        verdict = V_MAT
    else:
        verdict = V_NONMAT
    return TestResult(
        name="Ljung-Box (ACF rendements)", axis=AX_IND,
        h0="H0 : pas d'autocorrélation des rendements (rho(k)=0 ∀k).",
        h1="H1 : au moins une autocorrélation non nulle.",
        statistic={f"LB({L})": float(v) for L, v in zip(lags, lb["lb_stat"])},
        p_value=pmin,
        effect_size={"rho(1)": round(rho1, 4), "max|rho|": round(max_abs, 4)},
        effect_label="autocorrélation",
        n=n,
        threshold=(f"Matériel si p<{p.alpha} ET max|rho|≥{p.acf_material} "
                   f"(ampleur jugée économiquement pertinente à la fréquence de rebal.)."),
        verdict=verdict,
        interpretation=("Détecte la dépendance LINÉAIRE. Une autocorrélation faible "
                        "mais significative sur série longue n'est pas forcément matérielle "
                        "pour la chaîne de Markov."),
        limits=("Ne capte que la dépendance linéaire ; insensible aux dépendances non "
                "linéaires (ex. clustering de volatilité) — voir BDS et ARCH."),
        details=dict(lags=lags, lb_stat=list(map(float, lb["lb_stat"])),
                     lb_p=list(map(float, lb["lb_pvalue"])),
                     acf=list(map(float, ac))),
    )


def surrogate_vr_band(returns: np.ndarray, q: int, p: Params):
    """Bande iid de VR(q) par données de substitution : on PERMUTE les rendements
    (mêmes valeurs marginales, dépendance temporelle détruite) et on recalcule VR(q).
    Renvoie (IC[2.5%,97.5%], p_surrogate = fraction des VR permutés ≥ VR observé).
    Croisé avec l'asymptotique robuste, cela immunise le verdict contre une
    défaillance des approximations asymptotiques en petit échantillon. (Graine fixe.)"""
    rng = np.random.default_rng(p.seed + q)   # graine déterministe par q
    obs = variance_ratio_lomackinlay(returns, q)["vr"]
    sur = np.array([variance_ratio_lomackinlay(rng.permutation(returns), q)["vr"]
                    for _ in range(p.n_boot)])
    lo, hi = np.percentile(sur, [2.5, 97.5])
    # p bilatéral par surrogate (écart à 1 dans les deux sens)
    p_sur = 2.0 * min(np.mean(sur >= obs), np.mean(sur <= obs))
    outside = (obs < lo) or (obs > hi)
    return (float(lo), float(hi)), float(min(p_sur, 1.0)), bool(outside)


def test_variance_ratio(returns: pd.Series, p: Params) -> TestResult:
    r = returns.values
    res = [variance_ratio_lomackinlay(r, q) for q in p.vr_qs]
    res = [d for d in res if np.isfinite(d["vr"])]
    # bandes iid par surrogate pour chaque horizon
    bands = {}
    for d in res:
        band, p_sur, outside = surrogate_vr_band(r, d["q"], p)
        d["surr_band"] = band
        d["p_surrogate"] = p_sur
        d["outside_band"] = outside
        bands[d["q"]] = band
    dev = {f"VR({d['q']})": round(d["vr"], 4) for d in res}
    pvals = {d["q"]: d["p_value"] for d in res}
    max_dev = max(abs(d["vr"] - 1.0) for d in res) if res else np.nan
    any_sig = any((d["p_value"] < p.alpha) for d in res)
    # MATÉRIEL : un horizon doit cumuler p asympt.<alpha, effet≥seuil, ET hors bande iid
    material = any((d["p_value"] < p.alpha and abs(d["vr"] - 1.0) >= p.vr_material
                    and d["outside_band"]) for d in res)
    if not any_sig:
        verdict = V_OK
    elif material:
        verdict = V_MAT
    else:
        verdict = V_NONMAT
    pmin = min(pvals.values()) if pvals else np.nan
    return TestResult(
        name="Variance Ratio (Lo–MacKinlay, robuste)", axis=AX_IND,
        h0="H0 : marche aléatoire, VR(q)=1 ∀q.",
        h1="H1 : VR(q)≠1 (autocorrélation cumulée des rendements).",
        statistic=dev, p_value=float(pmin),
        effect_size=round(float(max_dev), 4), effect_label="max|VR(q)-1|",
        n=r.size,
        threshold=(f"Matériel si un horizon q cumule : p_asympt<{p.alpha}, "
                   f"|VR(q)-1|≥{p.vr_material}, ET VR(q) HORS bande iid surrogate "
                   f"(triple condition — immunise contre les défaillances asymptotiques "
                   f"en petit échantillon)."),
        verdict=verdict,
        interpretation=("VR>1 : persistance (sur-réaction lente) ; VR<1 : retour à la "
                        "moyenne (anti-corrélation). Statistique robuste à "
                        "l'hétéroscédasticité (la valeur n'est pas faussée par le "
                        "clustering de volatilité). La bande iid par permutation (surrogate) "
                        "calibre le bruit d'échantillonnage propre à la série."),
        limits=("Choix des horizons q ; recouvrement des fenêtres ; l'asymptotique de "
                "Lo–MacKinlay est moins fiable pour q/T grand — d'où le contrôle surrogate."),
        details=dict(per_q=res, bands=bands),
    )


def test_runs(returns: pd.Series, p: Params) -> TestResult:
    d = runs_test_ww(returns.values)
    sig = np.isfinite(d["p_value"]) and d["p_value"] < p.alpha
    rel = (d["runs"] - d["expected"]) / d["expected"] if np.isfinite(d["expected"]) else np.nan
    verdict = V_NONMAT if sig else V_OK   # test peu puissant : jamais "matériel" seul
    return TestResult(
        name="Runs (Wald–Wolfowitz, signe)", axis=AX_IND,
        h0="H0 : les signes des rendements sont en ordre aléatoire (indépendants).",
        h1="H1 : trop peu / trop de runs (dépendance dans les signes).",
        statistic=d["runs"], p_value=d["p_value"],
        effect_size=round(float(rel), 4) if np.isfinite(rel) else np.nan,
        effect_label="écart relatif runs",
        n=d["n"],
        threshold=(f"Significatif si p<{p.alpha} ; jamais classé « matériel » seul "
                   f"(test à faible puissance, indicatif)."),
        verdict=verdict,
        interpretation="Test non paramétrique d'indépendance, fondé uniquement sur les signes.",
        limits="Faible puissance ; ignore l'amplitude des rendements ; sensible aux ex aequo (zéros).",
        details=d,
    )


def test_bds(returns: pd.Series, p: Params) -> TestResult:
    r = returns.values
    n = r.size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval = bds(r, max_dim=p.bds_max_dim)
    stat = np.atleast_1d(stat).astype(float)
    pval = np.atleast_1d(pval).astype(float)
    dims = list(range(2, p.bds_max_dim + 1))
    pmin = float(np.nanmin(pval))
    n_rej = int(np.sum(pval < p.alpha))
    # BDS rejette aussi sous clustering de volatilité -> on flague, la synthèse réconcilie
    if pmin >= p.alpha:
        verdict = V_OK
    elif n_rej >= 2:
        verdict = V_IND     # rejet net mais cause ambiguë (à réconcilier avec ARCH)
    else:
        verdict = V_NONMAT
    return TestResult(
        name="BDS (iid, dépendance non linéaire incluse)", axis=AX_IND,
        h0="H0 : les rendements sont i.i.d.",
        h1="H1 : dépendance (linéaire OU non linéaire) — structure résiduelle.",
        statistic={f"m={m}": round(float(s), 3) for m, s in zip(dims, stat)},
        p_value=pmin,
        effect_size={f"m={m}": round(float(s), 3) for m, s in zip(dims, stat)},
        effect_label="stat BDS (standardisée)",
        n=n,
        threshold=(f"Rejet si p<{p.alpha} sur ≥2 dimensions. ATTENTION : un rejet peut "
                   f"venir du clustering de volatilité (variance), pas de la prévisibilité "
                   f"des rendements — réconcilié dans la synthèse via ARCH."),
        verdict=verdict,
        interpretation=("Test d'i.i.d. le plus direct : sensible à toute structure. "
                        "C'est aussi sa limite — il ne dit pas SI la dépendance est dans la "
                        "moyenne (dommageable) ou dans la variance (traitée par extension régimes)."),
        limits="Dépend du choix de epsilon et de m ; rejette sous hétéroscédasticité (non spécifique).",
        details=dict(dims=dims, stat=list(map(float, stat)), pval=list(map(float, pval))),
    )


# ---- B. Hétéroscédasticité conditionnelle ----------------------------------
def test_arch(returns: pd.Series, p: Params) -> TestResult:
    r = returns.values
    n = r.size
    nlags = min(p.arch_lags, max(1, n // 5))
    lm, lmp, fval, fp = het_arch(r, nlags=nlags)
    # ACF des |rendements| et LB sur rendements^2
    absr = np.abs(r)
    ac_abs = acf(absr, nlags=min(20, n - 2), fft=False)
    persist = float(np.mean(np.abs(ac_abs[1:min(11, len(ac_abs))])))  # persistance |r|
    lb2 = acorr_ljungbox(r**2, lags=[min(10, n - 2)], return_df=True)
    p_lb2 = float(lb2["lb_pvalue"].iloc[0])
    sig = (lmp < p.alpha)
    persistent = persist > 0.05
    # clustering de volatilité ≈ certain sur actions -> traité par extension régimes
    if sig and persistent:
        verdict = V_EXT
    elif sig:
        verdict = V_NONMAT
    else:
        verdict = V_OK
    return TestResult(
        name="ARCH-LM + ACF|rendements| + LB(rendements²)", axis=AX_VOL,
        h0="H0 : pas d'effet ARCH (variance conditionnelle constante).",
        h1="H1 : hétéroscédasticité conditionnelle (clustering de volatilité).",
        statistic={"LM": round(float(lm), 3), "LB(r²)_p": round(p_lb2, 4)},
        p_value=float(lmp),
        effect_size=round(persist, 4), effect_label="persistance ACF|r| (moy. retards 1-10)",
        n=n,
        threshold=(f"Violation si p<{p.alpha} ET ACF|r| persistante (>0.05). "
                   f"Mais NON invalidant : géré par l'extension à régimes de volatilité "
                   f"(Paulot–Lacroze §7.1)."),
        verdict=verdict,
        interpretation=("Le clustering de volatilité est quasi certain sur actions. Ce n'est "
                        "PAS une raison d'abandonner le modèle : il est absorbé par l'extension "
                        "à régimes. Verdict = « traitée par extension », pas « modèle invalide »."),
        limits="Détecte l'hétéroscédasticité, pas le modèle générateur sous-jacent.",
        details=dict(lm=float(lm), lmp=float(lmp), nlags=nlags,
                     acf_abs=list(map(float, ac_abs)), lb2_p=p_lb2),
    )


# ---- C. Stationnarité / retour à la moyenne --------------------------------
def _half_life(logp: np.ndarray):
    """Demi-vie via Δy_t = a + b*y_{t-1} ; tau = -ln(2)/b si b<0 (en périodes)."""
    y = np.asarray(logp, dtype=float)
    ylag = y[:-1]
    dy = np.diff(y)
    X = np.column_stack([np.ones_like(ylag), ylag])
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    b = beta[1]
    hl = -np.log(2.0) / b if b < 0 else np.inf
    return float(b), float(hl)


def test_stationarity(prices_freq: pd.Series, freq: str, p: Params) -> TestResult:
    logp = np.log(prices_freq.values)
    n = logp.size
    adf = adfuller(logp, autolag="AIC")
    adf_stat, adf_p = float(adf[0]), float(adf[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = kpss(logp, regression="c", nlags="auto")
    kpss_stat, kpss_p = float(kp[0]), float(kp[1])
    b, hl_per = _half_life(logp)
    hl_years = hl_per / periods_per_year(freq) if np.isfinite(hl_per) else np.inf

    adf_rejects = adf_p < p.alpha          # rejette racine unitaire -> stationnaire
    kpss_rejects = kpss_p < p.alpha        # rejette stationnarité
    # iid (Lévy) <=> racine unitaire (marche aléatoire). MR = stationnaire = violation.
    if (not adf_rejects) and kpss_rejects:
        verdict = V_OK                     # racine unitaire confirmée : iid OK sur cet axe
    elif adf_rejects and np.isfinite(hl_years):
        verdict = V_MAT if hl_years < p.maturity_years else V_NONMAT
    elif adf_rejects and (not kpss_rejects):
        verdict = V_IND                    # signaux contradictoires
    else:
        verdict = V_IND
    return TestResult(
        name="ADF + KPSS + demi-vie (sur log-prix)", axis=AX_STA,
        h0="ADF — H0 : racine unitaire (non stationnaire). KPSS — H0 : stationnaire.",
        h1="ADF — H1 : stationnaire. KPSS — H1 : non stationnaire.",
        statistic={"ADF": round(adf_stat, 3), "KPSS": round(kpss_stat, 3),
                   "demi-vie (ans)": (round(hl_years, 2) if np.isfinite(hl_years) else "∞")},
        p_value=adf_p,
        ci=("ADF p", round(adf_p, 4), "KPSS p", round(kpss_p, 4)),
        effect_size=(round(hl_years, 2) if np.isfinite(hl_years) else float("inf")),
        effect_label="demi-vie (années)",
        n=n,
        threshold=(f"Retour à la moyenne MATÉRIEL si ADF rejette (p<{p.alpha}) ET "
                   f"demi-vie < maturité produit ({p.maturity_years:g} ans). Sinon, "
                   f"racine unitaire => iid OK sur cet axe."),
        verdict=verdict,
        interpretation=("L'hypothèse iid = log(prix) marche aléatoire = RACINE UNITAIRE. "
                        "Les indices actions ont en général une racine unitaire (pas de MR) "
                        "=> iid OK ici. Les matières premières montrent souvent du MR. "
                        "ADF et KPSS sont complémentaires (puissance vs spécificité)."),
        limits=("ADF : faible puissance, sensible au choix de retards et de termes "
                "déterministes ; d'où KPSS en complément. La demi-vie suppose un AR(1)."),
        details=dict(adf_stat=adf_stat, adf_p=adf_p, adf_crit=adf[4],
                     kpss_stat=kpss_stat, kpss_p=kpss_p, kpss_crit=kp[3],
                     b=b, half_life_periods=hl_per, half_life_years=hl_years),
    )


# ---- D. Mémoire longue ------------------------------------------------------
def _memory_verdict(H_methods: dict, band: tuple, n: int, p: Params, on_returns: bool):
    finite = [h for h in H_methods.values() if np.isfinite(h)]
    if not finite:
        return V_IND
    devs = [abs(h - 0.5) for h in finite]
    material_flags = [d >= p.hurst_material for d in devs]
    enough_n = n >= p.hurst_min_n
    # bande surrogate iid : 0.5 effectif si la bande contient les H
    lo, hi = band
    inside = (np.isfinite(lo) and np.isfinite(hi) and
              all(lo <= h <= hi for h in finite))
    if on_returns:
        if (not any(material_flags)) or inside:
            return V_OK
        if all(material_flags) and enough_n and not inside:
            return V_MAT
        return V_IND          # méthodes en désaccord / borderline
    else:
        # mémoire en VOLATILITÉ : phénomène de clustering -> extension régimes
        if (not any(material_flags)) or inside:
            return V_OK
        return V_EXT


def test_long_memory(returns: pd.Series, p: Params, on_returns: bool = True) -> TestResult:
    base = returns.values if on_returns else np.abs(returns.values)
    n = base.size
    rs = hurst_rs(base, p.rs_min_window)
    dfa = hurst_dfa(base, p.rs_min_window)
    gph = gph_estimator(base, p.gph_power)
    H = {"R/S": rs["H"], "DFA": dfa["H"], "GPH": gph["H"]}
    # bande iid par surrogate (sur DFA, estimateur stable) — reproductible
    band, _ = _surrogate_hurst_band(base, hurst_dfa, p)
    verdict = _memory_verdict(H, band, n, p, on_returns)
    label = "rendements" if on_returns else "|rendements| (volatilité)"
    spread = (max(h for h in H.values() if np.isfinite(h)) -
              min(h for h in H.values() if np.isfinite(h))) if any(
                  np.isfinite(v) for v in H.values()) else np.nan
    return TestResult(
        name=f"Hurst R/S · DFA · GPH — {label}", axis=AX_MEM,
        h0="H0 : pas de mémoire longue (H=0.5).",
        h1="H1 : persistance (H>0.5) ou anti-persistance (H<0.5).",
        statistic={k: (round(v, 3) if np.isfinite(v) else None) for k, v in H.items()},
        ci=(round(band[0], 3) if np.isfinite(band[0]) else None,
            round(band[1], 3) if np.isfinite(band[1]) else None),
        effect_size={k: (round(abs(v - 0.5), 3) if np.isfinite(v) else None)
                     for k, v in H.items()},
        effect_label="|H-0.5| par méthode",
        n=n,
        threshold=(f"Matériel si |H-0.5|≥{p.hurst_material} (concordant entre méthodes) "
                   f"sur ≥{p.hurst_min_n} points ET hors bande iid surrogate "
                   f"[{band[0]:.3f}, {band[1]:.3f}]."
                   if np.isfinite(band[0]) else
                   f"Matériel si |H-0.5|≥{p.hurst_material} sur ≥{p.hurst_min_n} points."),
        verdict=verdict,
        interpretation=("Sur les RENDEMENTS : H≈0.5 => pas de mémoire => iid OK. "
                        if on_returns else
                        "Sur les |rendements| : H>0.5 documente une mémoire de VOLATILITÉ "
                        "(clustering) — traitée par l'extension à régimes, ce n'est PAS "
                        "une violation de l'indépendance des rendements eux-mêmes. ")
                        + f"Désaccord inter-méthodes (étendue) = {spread:.3f}."
                        if np.isfinite(spread) else "",
        limits=("Estimateurs biaisés en petit échantillon et sensibles à la méthode. "
                "On reporte explicitement le désaccord R/S vs DFA vs GPH et une bande iid "
                "par données de substitution."),
        details=dict(rs=rs, dfa=dfa, gph=gph, band=band, on_returns=on_returns),
    )


# ---- E. Forme de la distribution (CARACTÉRISATION, pas une violation iid) ---
def test_distribution(returns: pd.Series, p: Params) -> TestResult:
    r = returns.values
    n = r.size
    jb, jb_p = sps.jarque_bera(r)
    sk = float(sps.skew(r))
    ku = float(sps.kurtosis(r, fisher=True))   # excès de kurtosis
    # quantiles théoriques (QQ vs normale)
    rs = np.sort(r)
    pp = (np.arange(1, n + 1) - 0.5) / n
    theo = sps.norm.ppf(pp, loc=r.mean(), scale=r.std(ddof=1))
    return TestResult(
        name="Jarque-Bera, asymétrie, excès de kurtosis, QQ", axis=AX_DIS,
        h0="H0 : rendements normaux (asymétrie=0, excès de kurtosis=0).",
        h1="H1 : non-normalité (asymétrie et/ou queues épaisses).",
        statistic={"JB": round(float(jb), 1), "asymétrie": round(sk, 3),
                   "excès kurtosis": round(ku, 3)},
        p_value=float(jb_p),
        effect_size={"asymétrie": round(sk, 3), "excès kurtosis": round(ku, 3)},
        effect_label="moments",
        n=n,
        threshold="Aucun seuil de VIOLATION : axe purement descriptif (voir interprétation).",
        verdict=V_CHAR,
        interpretation=("Ceci décrit la LOI MARGINALE. Des queues épaisses restent "
                        "compatibles avec l'hypothèse iid (famille de Lévy à SAUTS). Ce n'est "
                        "PAS une violation des incréments indépendants, mais l'indication "
                        "qu'une loi à sauts (Kou) est plus adaptée que Black-Scholes."),
        limits="JB sur-rejette sur grand échantillon ; ne dit rien sur la dépendance temporelle.",
        details=dict(jb=float(jb), jb_p=float(jb_p), skew=sk, kurt=ku,
                     sample_q=list(map(float, rs)), theo_q=list(map(float, theo))),
    )


# ---- F. Volatilité locale (HORS SCOPE, documenté) --------------------------
def test_local_vol_placeholder() -> TestResult:
    return TestResult(
        name="Volatilité locale σ(S,t)", axis=AX_LOC,
        h0="—", h1="—",
        statistic=None, p_value=None, effect_size=None, n=0,
        threshold="Non applicable.",
        verdict=V_NT,
        interpretation=("Non testé, faible priorité. Nécessite une surface de volatilité "
                        "implicite (options listées), non disponible pour le MASI. Le gap risk "
                        "d'un CPPI est dominé par les SAUTS, pas par σ(S,t) (Cont–Tankov 2007)."),
        limits="Requiert un marché d'options liquide ; hors périmètre v1.",
        details={},
    )


# ============================================================================
# 5. ORCHESTRATION + SYNTHÈSE HOLISTIQUE (test multiple, conservateur)
# ============================================================================
def run_all_tests(series: pd.Series, kind: str, freq: str, p: Params) -> dict:
    """Lance toute la batterie. Renvoie un dict d'axes -> liste de TestResult,
    plus la table de synthèse et la phrase de verdict."""
    returns = compute_returns(series, kind=kind, freq=freq)
    prices_freq = resample_prices(series, freq)

    res = {
        "ljung_box": test_ljung_box(returns, p),
        "variance_ratio": test_variance_ratio(returns, p),
        "runs": test_runs(returns, p),
        "bds": test_bds(returns, p),
        "arch": test_arch(returns, p),
        "stationarity": test_stationarity(prices_freq, freq, p),
        "memory_returns": test_long_memory(returns, p, on_returns=True),
        "memory_vol": test_long_memory(returns, p, on_returns=False),
        "distribution": test_distribution(returns, p),
        "local_vol": test_local_vol_placeholder(),
    }

    axis_verdicts = _axis_verdicts(res, p)
    res["_axis_verdicts"] = axis_verdicts
    res["_returns"] = returns
    res["_prices_freq"] = prices_freq
    res["_freq"] = freq
    res["_n_returns"] = returns.size
    return res


def _axis_verdicts(res: dict, p: Params) -> dict:
    """Réconciliation holistique par axe (ne pas cueillir une p-value isolée).

    Cœur : distinguer dépendance dans la MOYENNE (dommageable pour Markov)
    de dépendance dans la VARIANCE (clustering -> extension régimes)."""
    lb, vr, runs, bds_r = res["ljung_box"], res["variance_ratio"], res["runs"], res["bds"]
    arch = res["arch"]

    # --- Axe indépendance ---
    linear_material = (lb.verdict == V_MAT) or (vr.verdict == V_MAT)
    linear_clean = (lb.verdict in (V_OK, V_NONMAT)) and (vr.verdict in (V_OK, V_NONMAT))
    bds_rejects = bds_r.verdict in (V_IND, V_MAT)
    arch_present = arch.verdict in (V_EXT, V_NONMAT)

    if linear_material:
        ind_axis = V_MAT                         # prévisibilité linéaire réelle des rendements
    elif bds_rejects and linear_clean and arch_present:
        ind_axis = V_EXT                         # dépendance attribuable à la variance (régimes)
    elif bds_rejects and linear_clean and not arch_present:
        ind_axis = V_IND                         # structure non linéaire inexpliquée -> à confirmer
    elif linear_clean and not bds_rejects:
        ind_axis = V_OK
    else:
        ind_axis = V_IND

    # --- Axe volatilité ---
    vol_axis = arch.verdict

    # --- Axe stationnarité ---
    sta_axis = res["stationarity"].verdict

    # --- Axe mémoire longue (rendements = celui qui compte pour iid) ---
    mem_axis = res["memory_returns"].verdict
    mem_vol_axis = res["memory_vol"].verdict

    return {
        AX_IND: ind_axis,
        AX_VOL: vol_axis,
        AX_STA: sta_axis,
        AX_MEM: mem_axis,
        "Mémoire de volatilité": mem_vol_axis,
        AX_DIS: V_CHAR,
        AX_LOC: V_NT,
    }


def synthesis_table(all_results: dict) -> pd.DataFrame:
    """Table livrable d'un actif : test × stat × p/IC × effet × seuil × verdict."""
    order = ["ljung_box", "variance_ratio", "runs", "bds", "arch",
             "stationarity", "memory_returns", "memory_vol",
             "distribution", "local_vol"]
    rows = [all_results[k].to_row() for k in order if k in all_results]
    return pd.DataFrame(rows)


def multi_asset_table(results_by_asset: dict) -> pd.DataFrame:
    """Table « actif × axe × statistique × p/effet × seuil × VERDICT »."""
    frames = []
    for asset, all_res in results_by_asset.items():
        df = synthesis_table(all_res).copy()
        df.insert(0, "Actif", asset)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plain_language_verdict(all_results: dict, asset_name: str = "le MASI") -> str:
    """Phrase de synthèse conservatrice (cf. point 9 : holistique, pas binaire forcé)."""
    av = all_results["_axis_verdicts"]
    freq = _FREQ_LABEL.get(all_results["_freq"], all_results["_freq"])
    key_axes = {
        "indépendance des incréments": av[AX_IND],
        "stationnarité (pas de retour à la moyenne)": av[AX_STA],
        "absence de mémoire longue": av[AX_MEM],
    }
    materials = [name for name, v in key_axes.items() if v == V_MAT]
    indets = [name for name, v in key_axes.items() if v == V_IND]

    head = f"Sur {asset_name}, à fréquence {freq}, "
    if not materials and not indets:
        verdict_phrase = "l'hypothèse iid TIENT"
        tail = (" : pas de prévisibilité matérielle des rendements, racine unitaire "
                "confirmée, pas de mémoire longue. Le clustering de volatilité, s'il est "
                "présent, est absorbé par l'extension à régimes — il ne remet pas en cause "
                "la réduction de Markov.")
    elif len(materials) == 1 and not indets:
        verdict_phrase = f"l'hypothèse iid TIENT SAUF sur l'axe « {materials[0]} »"
        tail = " : c'est là que la réduction de Markov est challengée (voir l'axe concerné)."
    elif materials:
        verdict_phrase = ("l'hypothèse iid NE TIENT PAS : violations matérielles sur "
                          + ", ".join(f"« {m} »" for m in materials))
        tail = ". La réduction en chaîne de Markov 1-D n'est pas justifiée en l'état."
    else:
        verdict_phrase = ("le verdict est INDÉTERMINÉ / À CONFIRMER sur "
                          + ", ".join(f"« {m} »" for m in indets))
        tail = (" : les tests se contredisent ou sont borderline. Conformément à la "
                "prudence sur le test multiple, on ne force pas un binaire — passer à la "
                "Phase 2 (chiffrage de l'impact prix) pour trancher.")
    return head + verdict_phrase + tail


# ============================================================================
# 6. SENSIBILITÉ À LA FRÉQUENCE (une dépendance quotidienne peut disparaître
#    en mensuel — donc immatérielle pour le pricer ; point 6 du livrable)
# ============================================================================
def frequency_scan(series: pd.Series, kind: str, p: Params,
                   freqs=("D", "W", "M")) -> pd.DataFrame:
    """Compare la statistique d'indépendance directe (rho(1), VR(2)) à chaque
    fréquence, pour montrer si la dépendance survit au rebalancement mensuel."""
    rows = []
    for f in freqs:
        r = compute_returns(series, kind=kind, freq=f)
        n = r.size
        if n < 30:
            rows.append({"Fréquence": _FREQ_LABEL[f], "n": n,
                         "rho(1)": np.nan, "LB(1) p": np.nan,
                         "VR(2)": np.nan, "VR(2) p": np.nan})
            continue
        ac = acf(r.values, nlags=1, fft=False)
        lb = acorr_ljungbox(r.values, lags=[1], return_df=True)
        vr2 = variance_ratio_lomackinlay(r.values, 2)
        rows.append({
            "Fréquence": _FREQ_LABEL[f], "n": n,
            "rho(1)": round(float(ac[1]), 4),
            "LB(1) p": round(float(lb["lb_pvalue"].iloc[0]), 4),
            "VR(2)": round(vr2["vr"], 4),
            "VR(2) p": round(vr2["p_value"], 4),
        })
    return pd.DataFrame(rows)
