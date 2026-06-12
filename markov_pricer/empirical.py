"""
empirical.py
============
Alimente le pricer Markov avec la DISTRIBUTION EMPIRIQUE RÉELLE d'un indice
(ex. MASI) : les rendements observés pilotent directement la loi de transition,
au lieu d'être résumés à une seule volatilité.

Chaîne : fichier Excel -> série de niveaux -> rendements journaliers ->
rendements de PÉRIODE (pas de rebalancement) -> rendement déflaté rho ->
EmpiricalLaw (utilisée telle quelle par markov_cppi).

Loader autonome (le projet reste indépendant du backtest).
"""

from __future__ import annotations

import unicodedata
from typing import Optional

import numpy as np
import pandas as pd

from markov_cppi import EmpiricalLaw, MarkovParams


# ---------------------------------------------------------------------------
# CHARGEMENT ROBUSTE D'UN INDICE
# ---------------------------------------------------------------------------

_DATE_HINTS = ["date", "jour", "day", "seance", "time"]
_VALUE_HINTS = ["valeur", "close", "cloture", "price", "prix", "cours", "level",
                "niveau", "index", "indice", "ref", "last", "px"]
_IGNORE_HINTS = ["code", "ticker", "name", "nom", "symbol", "isin", "devise", "currency"]


def _norm(text) -> str:
    t = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in t if not unicodedata.combining(c)).strip().lower()


def _detect(df: pd.DataFrame):
    labels = {c: _norm(c) for c in df.columns}
    date_col = next((c for c, l in labels.items() if any(h in l for h in _DATE_HINTS)), None)
    if date_col is None:
        for c in df.columns:
            if pd.to_datetime(df[c], errors="coerce", dayfirst=True).notna().mean() > 0.8:
                date_col = c
                break
    if date_col is None:
        return None
    val_col = next((c for c, l in labels.items()
                    if c != date_col and any(h in l for h in _VALUE_HINTS)
                    and not any(h in l for h in _IGNORE_HINTS)), None)
    if val_col is None:
        cands = [c for c in df.columns if c != date_col
                 and not any(h in labels[c] for h in _IGNORE_HINTS)
                 and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
        val_col = cands[-1] if cands else None
    if val_col is None:
        return None
    return date_col, val_col


def load_series(path_or_buffer, name: Optional[str] = None) -> pd.Series:
    """
    Renvoie une Series de niveaux (index = dates triées). Parcourt toutes les
    feuilles et cherche la ligne d'en-tête (ignore pages de garde / titres).
    """
    import warnings
    xl = pd.ExcelFile(path_or_buffer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sheet in xl.sheet_names:
            for hdr in range(0, 12):
                try:
                    raw = xl.parse(sheet_name=sheet, header=hdr)
                except Exception:
                    continue
                raw.columns = [str(c).strip() for c in raw.columns]
                if raw.shape[1] < 2:
                    continue
                det = _detect(raw)
                if not det:
                    continue
                dcol, vcol = det
                d = pd.to_datetime(raw[dcol], errors="coerce", dayfirst=True)
                v = pd.to_numeric(raw[vcol], errors="coerce")
                s = pd.Series(v.values, index=d).dropna().sort_index()
                s = s[~s.index.duplicated(keep="last")]
                if len(s) >= 30:
                    s.name = name or "indice"
                    s.attrs["sheet"] = sheet
                    return s
    raise ValueError(f"Aucune série exploitable. Feuilles : {xl.sheet_names}")


# ---------------------------------------------------------------------------
# CONSTRUCTION DE L'ÉCHANTILLON DE RENDEMENTS DÉFLATÉS rho
# ---------------------------------------------------------------------------

def period_days(p: MarkovParams, trading_days: int = 252) -> int:
    """Nombre de jours de bourse par pas de rebalancement."""
    dt = p.maturity / p.n_rebal
    return max(1, int(round(dt * trading_days)))


def build_rho_sample(series: pd.Series, p: MarkovParams, method: str = "overlap",
                     n_boot: int = 20_000, seed: int = 0,
                     trading_days: int = 252) -> np.ndarray:
    """
    Construit l'échantillon de rendement déflaté rho = (S'/S)·e^{-r·dt} sur un pas.

    method='overlap'   : rendements k-jours glissants observés (toutes les fenêtres).
    method='bootstrap' : k rendements journaliers ré-échantillonnés i.i.d. et
                         composés, n_boot tirages (queues riches, perd le clustering).

    rho N'est PAS recentré ici : le recentrage (mesure risque-neutre) est géré par
    EmpiricalLaw(recenter=True).
    """
    lr = np.log(series.values[1:] / series.values[:-1])
    lr = lr[np.isfinite(lr)]
    k = period_days(p, trading_days)
    dt = p.maturity / p.n_rebal
    deflate = np.exp(-p.rate * dt)

    if method == "overlap":
        if len(lr) <= k:
            raise ValueError(f"Historique trop court pour des fenêtres de {k} jours.")
        csum = np.cumsum(np.insert(lr, 0, 0.0))
        period_lr = csum[k:] - csum[:-k]            # sommes glissantes de k jours
    elif method == "bootstrap":
        rng = np.random.default_rng(seed)
        draws = rng.choice(lr, size=(n_boot, k), replace=True)
        period_lr = draws.sum(axis=1)
    else:
        raise ValueError("method ∈ {'overlap','bootstrap'}")

    return np.exp(period_lr) * deflate


def make_empirical_law(series: pd.Series, p: MarkovParams, recenter: bool,
                       method: str = "overlap", n_boot: int = 20_000) -> EmpiricalLaw:
    rho = build_rho_sample(series, p, method=method, n_boot=n_boot)
    return EmpiricalLaw(rho, dt=p.maturity / p.n_rebal, recenter=recenter)


def empirical_stats(series: pd.Series, p: MarkovParams, method: str = "overlap") -> dict:
    """Statistiques descriptives de l'échantillon de période (diagnostic)."""
    from scipy.stats import skew, kurtosis
    rho = build_rho_sample(series, p, method=method)
    r = np.log(rho)
    dt = p.maturity / p.n_rebal
    return {
        "n_obs_serie": len(series),
        "k_jours_periode": period_days(p),
        "n_echantillon": len(rho),
        "vol_annualisee": float(np.std(r) / np.sqrt(dt)),
        "rendement_moyen_periode": float(rho.mean() - 1),
        "skewness": float(skew(r)),
        "kurtosis_exces": float(kurtosis(r)),         # 0 = normal
        "pire_rendement_periode": float(rho.min() - 1),
    }
