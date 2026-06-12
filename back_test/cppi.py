"""
cppi.py
=======
Coeur "métier" de l'application : chargement des données, moteur de backtest
CPPI et calcul des indicateurs de performance.

Volontairement SANS aucune dépendance à Streamlit : ce sont des fonctions pures,
réutilisables et testables. L'interface (app.py) se contente d'appeler ces
fonctions et d'afficher les résultats.

Conventions :
- une "série de prix" est une pandas.Series indexée par dates (DatetimeIndex
  triée), contenant les niveaux de l'indice ;
- les rendements sont calculés en interne en rendement simple (le mode
  log-return proposé dans l'UI sert surtout aux statistiques affichées) ;
- toutes les valeurs monétaires sont dans la même unité que le capital initial.
"""

from __future__ import annotations

import os
import glob
import unicodedata
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. CHARGEMENT ET NORMALISATION DES DONNÉES
# ---------------------------------------------------------------------------

# Mots-clés acceptés pour détecter chaque colonne, même si la structure varie
# légèrement d'un fichier à l'autre.
DATE_KEYS = ["date", "jour", "time", "périod", "period"]
VALUE_KEYS = ["valeur", "close", "cours", "price", "level", "niveau",
              "index", "ref", "vl", "nav", "cloture", "clôture"]
CODE_KEYS = ["code", "ticker", "symbol", "indice", "isin", "nom"]


def _strip_accents(text: str) -> str:
    """Minuscule + suppression des accents, pour une détection robuste."""
    text = str(text).strip().lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _match_column(columns, keys):
    """Retourne le 1er nom de colonne dont le libellé contient un des mots-clés."""
    normalized = {col: _strip_accents(col) for col in columns}
    for col, norm in normalized.items():
        for k in keys:
            if k in norm:
                return col
    return None


def discover_files(folder: str) -> dict:
    """Trouve tous les .xls/.xlsx d'un dossier. Retourne {label: chemin}."""
    paths = []
    for ext in ("*.xlsx", "*.xls"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths = sorted(set(paths))
    out = {}
    for p in paths:
        label = os.path.splitext(os.path.basename(p))[0]
        out[label] = p
    return out


def load_index_file(path: str) -> pd.DataFrame:
    """
    Charge un fichier Excel d'historique d'indice et renvoie un DataFrame
    standardisé avec les colonnes : ['date', 'value', 'code'].

    La détection des colonnes est souple : on lit la 1ère feuille, on cherche
    une colonne date, une colonne valeur et (optionnellement) un code. On ne
    crée jamais silencieusement une colonne : si la valeur est introuvable, on
    lève une erreur explicite.
    """
    # 1ère feuille quel que soit son nom (ici elle s'appelle " Données ").
    raw = pd.read_excel(path, sheet_name=0, header=0)
    raw.columns = [str(c).strip() for c in raw.columns]

    date_col = _match_column(raw.columns, DATE_KEYS)
    value_col = _match_column(raw.columns, VALUE_KEYS)
    code_col = _match_column(raw.columns, CODE_KEYS)

    # Repli : si on n'a rien trouvé, on suppose date = 1ère colonne,
    # valeur = dernière colonne numérique.
    if date_col is None:
        date_col = raw.columns[0]
    if value_col is None:
        numeric_cols = [c for c in raw.columns
                        if pd.to_numeric(raw[c], errors="coerce").notna().mean() > 0.8]
        if not numeric_cols:
            raise ValueError(
                f"Aucune colonne de valeurs numériques détectée dans {os.path.basename(path)}. "
                f"Colonnes trouvées : {list(raw.columns)}"
            )
        value_col = numeric_cols[-1]

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    df["value"] = pd.to_numeric(raw[value_col], errors="coerce")
    if code_col is not None:
        df["code"] = raw[code_col].astype(str).str.strip()
    else:
        df["code"] = os.path.splitext(os.path.basename(path))[0]

    # Nettoyage : on enlève les lignes inexploitables et on trie par date.
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    df = df.drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Fichier {os.path.basename(path)} vide après nettoyage.")
    return df


def to_price_series(df: pd.DataFrame) -> pd.Series:
    """Transforme le DataFrame standardisé en Series de prix indexée par date."""
    s = df.set_index("date")["value"].astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def align_on_common_dates(series_dict: dict) -> pd.DataFrame:
    """
    Aligne plusieurs séries de prix sur leur ensemble de dates commun.
    Utile pour comparer un indice risqué et un actif sûr (indice obligataire).
    """
    df = pd.DataFrame(series_dict)
    return df.dropna(how="any")


# ---------------------------------------------------------------------------
# 2. CALENDRIER DE REBALANCEMENT
# ---------------------------------------------------------------------------

FREQUENCIES = ["Quotidien", "Hebdomadaire", "Mensuel", "Trimestriel", "Annuel"]
_FREQ_TO_PANDAS = {"Hebdomadaire": "W", "Mensuel": "M",
                   "Trimestriel": "Q", "Annuel": "Y"}


def build_rebalance_mask(dates: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    """
    Renvoie un masque booléen de même longueur que `dates` indiquant les jours
    de rebalancement. On rebalance toujours au tout 1er jour disponible.

    Pour les fréquences > quotidien, on prend le 1er jour de bourse de chaque
    période (semaine/mois/trimestre/année).
    """
    n = len(dates)
    if frequency == "Quotidien":
        mask = np.ones(n, dtype=bool)
        return mask

    mask = np.zeros(n, dtype=bool)
    periods = dates.to_period(_FREQ_TO_PANDAS[frequency])
    seen = set()
    for i, p in enumerate(periods):
        if p not in seen:
            seen.add(p)
            mask[i] = True
    mask[0] = True
    return mask


# ---------------------------------------------------------------------------
# 3. MOTEUR CPPI
# ---------------------------------------------------------------------------

def default_params() -> dict:
    """Jeu de paramètres par défaut "intelligents" pour démarrer."""
    return dict(
        capital=100_000.0,
        # --- floor ---
        floor_mode="% du capital initial (fixe)",  # ou "Montant fixe absolu" / "Croissant au taux sans risque"
        floor_pct=0.80,         # niveau de floor en % du capital (modes pct & croissant)
        floor_abs=80_000.0,     # montant absolu (mode "Montant fixe absolu")
        # --- coeur CPPI ---
        multiplier=4.0,
        rf=0.03,                # taux sans risque annuel
        frequency="Mensuel",
        exp_min=0.0,            # exposition risquée minimale (fraction du portefeuille)
        exp_max=1.0,            # exposition risquée maximale
        return_mode="Simple",   # "Simple" ou "Log" (pour les stats affichées)
        # --- avancé ---
        use_cushion_limit=False,
        cushion_limit=0.50,     # plafond du coussin en fraction du portefeuille
        use_lockin=False,
        lockin_level=0.90,      # nouveau floor = lockin_level * valeur au moment du verrou
        lockin_frequency="Mensuel",
        fee_mode="Aucun",       # "Aucun" / "Fixes" / "Proportionnels"
        fee_fixed=0.0,          # frais fixes par rebalancement (montant)
        fee_prop=0.0010,        # frais proportionnels au turnover (ex : 0.10%)
        # --- actif sûr ---
        use_safe_index=False,   # si True, la poche sûre suit un indice obligataire
    )


def _fee(turnover: float, params: dict) -> float:
    """Frais facturés lors d'un rebalancement, selon le mode choisi."""
    mode = params["fee_mode"]
    if mode == "Fixes":
        return float(params["fee_fixed"])
    if mode == "Proportionnels":
        return float(params["fee_prop"]) * float(turnover)
    return 0.0


def run_cppi(risky_prices: pd.Series,
             params: dict,
             safe_prices: pd.Series | None = None) -> pd.DataFrame:
    """
    Exécute un backtest CPPI quotidien sur la série de prix `risky_prices`.

    Logique (rappel) :
      - Coussin   C = max(V - Floor, 0)
      - Exposition risquée cible E = m * C, bornée par [exp_min, exp_max] * V
      - Poche sûre = V - E
      - Entre deux rebalancements, la poche risquée dérive avec l'indice et la
        poche sûre capitalise au taux sans risque (ou suit un indice obligataire
        si `safe_prices` est fourni).
      - Le floor peut être fixe, en % du capital, ou croissant au taux sans risque.
      - Option "lock-in" (cliquet) : on relève le floor pour verrouiller les gains.

    Retourne un DataFrame indexé par date avec, pour chaque jour :
      value, floor, cushion, risky_value, safe_value, risky_weight,
      safe_weight, index, index_return, rebalanced, fees, breach.
    """
    if safe_prices is not None:
        # On aligne actif risqué et actif sûr sur les dates communes.
        merged = align_on_common_dates({"r": risky_prices, "s": safe_prices})
        risky_prices = merged["r"]
        safe_prices = merged["s"]

    dates = risky_prices.index
    px = risky_prices.to_numpy(dtype=float)
    n = len(px)
    if n < 2:
        raise ValueError("Série trop courte pour un backtest (< 2 points).")

    # --- rendements quotidiens de l'actif risqué ---
    rret = np.zeros(n)
    rret[1:] = px[1:] / px[:-1] - 1.0

    # --- rendements quotidiens de la poche sûre ---
    if safe_prices is not None:
        sp = safe_prices.to_numpy(dtype=float)
        sret = np.zeros(n)
        sret[1:] = sp[1:] / sp[:-1] - 1.0
    else:
        rf_daily = (1.0 + params["rf"]) ** (1.0 / 252.0) - 1.0
        sret = np.full(n, rf_daily)
        sret[0] = 0.0

    # --- calendriers ---
    rebal_mask = build_rebalance_mask(dates, params["frequency"])
    if params["use_lockin"]:
        lockin_mask = build_rebalance_mask(dates, params["lockin_frequency"])
    else:
        lockin_mask = np.zeros(n, dtype=bool)

    # --- floor de base selon le mode ---
    capital = float(params["capital"])
    if params["floor_mode"] == "Montant fixe absolu":
        floor0 = float(params["floor_abs"])
    else:  # "% du capital initial (fixe)" ou "Croissant au taux sans risque"
        floor0 = float(params["floor_pct"]) * capital

    year_frac = (dates - dates[0]).days / 365.25  # pour le floor croissant
    rf = float(params["rf"])

    # --- état initial ---
    m = float(params["multiplier"])
    w_min, w_max = float(params["exp_min"]), float(params["exp_max"])

    V = capital
    risky = 0.0
    safe = 0.0
    ratchet_floor = -np.inf  # floor verrouillé par le lock-in (cliquet)

    rows = []
    for i in range(n):
        # 1) Dérive du portefeuille (sauf le 1er jour)
        if i > 0:
            risky *= (1.0 + rret[i])
            safe *= (1.0 + sret[i])
            V = risky + safe

        # 2) Floor de base à la date i
        if params["floor_mode"] == "Croissant au taux sans risque":
            base_floor = floor0 * (1.0 + rf) ** year_frac[i]
        else:
            base_floor = floor0

        # 3) Lock-in (cliquet) : on relève éventuellement le floor verrouillé
        if lockin_mask[i]:
            candidate = float(params["lockin_level"]) * V
            ratchet_floor = max(ratchet_floor, candidate)

        floor = max(base_floor, ratchet_floor)

        fees = 0.0
        rebalanced = False

        # 4) Rebalancement
        if rebal_mask[i]:
            rebalanced = True
            cushion = max(V - floor, 0.0)
            if params["use_cushion_limit"]:
                cushion = min(cushion, float(params["cushion_limit"]) * V)

            target_risky = m * cushion
            # Bornes d'exposition (en fraction du portefeuille)
            target_risky = min(max(target_risky, w_min * V), w_max * V)

            turnover = abs(target_risky - risky)
            fees = _fee(turnover, params)
            V = max(V - fees, 0.0)

            # Re-clamp après prélèvement des frais, puis re-split des poches
            target_risky = min(max(target_risky, w_min * V), w_max * V)
            risky = min(target_risky, V)
            safe = V - risky
        else:
            cushion = max(V - floor, 0.0)

        rows.append({
            "date": dates[i],
            "index": px[i],
            "index_return": rret[i],
            "value": V,
            "floor": floor,
            "cushion": V - floor,          # coussin réel (peut être négatif si breach)
            "risky_value": risky,
            "safe_value": safe,
            "risky_weight": (risky / V) if V > 0 else 0.0,
            "safe_weight": (safe / V) if V > 0 else 0.0,
            "rebalanced": rebalanced,
            "fees": fees,
            "breach": V < floor,           # le portefeuille passe sous le floor
        })

    hist = pd.DataFrame(rows).set_index("date")
    return hist


# ---------------------------------------------------------------------------
# 4. INDICATEURS DE PERFORMANCE
# ---------------------------------------------------------------------------

def _annualization_factor(dates: pd.DatetimeIndex) -> float:
    """Estime le nombre d'observations par an (≈ 252 pour du quotidien boursier)."""
    if len(dates) < 2:
        return 252.0
    days = (dates[-1] - dates[0]).days
    if days <= 0:
        return 252.0
    return len(dates) / (days / 365.25)


def max_drawdown(series: pd.Series) -> float:
    """Drawdown maximal (valeur négative, ex : -0.35 = -35%)."""
    running_max = series.cummax()
    dd = series / running_max - 1.0
    return float(dd.min())


def compute_metrics(hist: pd.DataFrame,
                    params: dict,
                    benchmark_prices: pd.Series | None = None) -> dict:
    """
    Calcule les indicateurs synthétiques à partir de l'historique du backtest.
    `benchmark_prices` sert à comparer le CPPI à un indice de référence.
    """
    V = hist["value"]
    dates = hist.index
    capital = float(params["capital"])
    ann = _annualization_factor(dates)

    # Rendements du portefeuille (mode simple ou log selon l'UI)
    if params.get("return_mode") == "Log":
        port_ret = np.log(V / V.shift(1)).dropna()
    else:
        port_ret = V.pct_change().dropna()

    final_value = float(V.iloc[-1])
    total_return = final_value / capital - 1.0

    n_years = (dates[-1] - dates[0]).days / 365.25
    cagr = (final_value / capital) ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan

    vol = float(port_ret.std() * np.sqrt(ann))
    sharpe = (cagr - float(params["rf"])) / vol if vol > 0 else np.nan
    mdd = max_drawdown(V)

    n_breaches = int(hist["breach"].sum())
    below = (hist["value"] - hist["floor"])
    max_loss_below_floor = float(below[below < 0].min()) if (below < 0).any() else 0.0
    total_fees = float(hist["fees"].sum())

    metrics = {
        "Capital initial": capital,
        "Valeur finale": final_value,
        "Rendement total": total_return,
        "Rendement annualisé (CAGR)": cagr,
        "Volatilité annualisée": vol,
        "Sharpe (simplifié)": sharpe,
        "Drawdown maximal": mdd,
        "Nb de breaches du floor": n_breaches,
        "Perte max sous le floor": max_loss_below_floor,
        "Frais totaux": total_fees,
    }

    # Comparaison au benchmark
    if benchmark_prices is not None:
        bench = benchmark_prices.reindex(dates).ffill().dropna()
        if len(bench) >= 2:
            bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1.0)
            bench_cagr = ((bench.iloc[-1] / bench.iloc[0]) ** (1.0 / n_years) - 1.0
                          if n_years > 0 else np.nan)
            metrics["Performance benchmark (total)"] = bench_total
            metrics["Performance benchmark (CAGR)"] = bench_cagr
            metrics["Sur/sous-performance (total)"] = total_return - bench_total

    return metrics


def metrics_to_frame(metrics: dict) -> pd.DataFrame:
    """Met en forme le dictionnaire d'indicateurs pour affichage / export."""
    pct_keys = [
        "Rendement total", "Rendement annualisé (CAGR)", "Volatilité annualisée",
        "Drawdown maximal", "Performance benchmark (total)",
        "Performance benchmark (CAGR)", "Sur/sous-performance (total)",
    ]
    rows = []
    for k, v in metrics.items():
        if isinstance(v, float) and k in pct_keys:
            disp = f"{v * 100:,.2f} %"
        elif isinstance(v, float) and k in ("Sharpe (simplifié)",):
            disp = f"{v:,.2f}"
        elif isinstance(v, float):
            disp = f"{v:,.2f}"
        else:
            disp = str(v)
        rows.append({"Indicateur": k, "Valeur": disp})
    return pd.DataFrame(rows)
