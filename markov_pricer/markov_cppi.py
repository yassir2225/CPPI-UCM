"""
markov_cppi.py
==============
Pricer CPPI par OPERATEURS DE MARKOV (v1, lognormal auto-financé).

Projet INDEPENDANT du backtest historique : aucun import croisé.

Idée (Paulot & Lacroze)
-----------------------
La valeur CPPI renormalisée  X_i = C_i / H_i  (portefeuille / seuil actualisé)
est un processus de Markov à une dimension. Au lieu de simuler des trajectoires,
on discrétise X sur une grille et on propage des PROBABILITES via une matrice de
transition, par produit matriciel, jusqu'à la maturité.

Dynamique déflatée (martingale, E[rho]=1) entre deux rebalancements :
    rho      = exp(-0.5*sigma^2*dt + sigma*sqrt(dt)*Z),   Z ~ N(0,1)
    w(X)     = clip( m*(1 - 1/X), w_min, w_max )
    X_{i+1}  = X_i * (1 + w(X_i)*(rho - 1))

Tout noeud X <= 1 est ABSORBANT (cushion <= 0 -> défaisance -> X constant).

Sorties : prix du put de gap risk, proba de gap, prix de la stratégie (≈ C0),
greeks (par différences finies), et distribution finale de X_T.

Un MONTE CARLO utilisant EXACTEMENT la même dynamique est fourni comme contrôle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# LOIS DE RENDEMENT (rho = rendement déflaté sur un pas)
# Interface commune : F(r)=CDF, PE(r)=E[rho 1_{rho<=r}], sample(n,rng), mean,
# sigma_ann (vol annualisée effective pour dimensionner la grille).
# ---------------------------------------------------------------------------

class LognormalLaw:
    """rho lognormal martingale (E[rho]=1) : modèle paramétrique GBM."""

    def __init__(self, sigma: float, dt: float):
        self.sigma = sigma
        self.dt = dt
        self.mu = -0.5 * sigma ** 2 * dt
        self.sd = sigma * np.sqrt(dt)
        self.sigma_ann = sigma
        self.mean = 1.0
        self.label = "lognormale (paramétrique)"

    def F(self, r):
        r = np.asarray(r, float)
        out = np.zeros_like(r)
        m = r > 0
        out[m] = norm.cdf((np.log(r[m]) - self.mu) / self.sd)
        out[np.isposinf(r)] = 1.0
        return out

    def PE(self, r):
        r = np.asarray(r, float)
        out = np.zeros_like(r)
        m = r > 0
        out[m] = norm.cdf((np.log(r[m]) - self.mu - self.sd ** 2) / self.sd)
        out[np.isposinf(r)] = 1.0
        return out

    def sample(self, n, rng):
        return np.exp(self.mu + self.sd * rng.standard_normal(n))


class EmpiricalLaw:
    """
    Loi EMPIRIQUE du rendement déflaté, construite à partir d'un échantillon
    historique {rho_s} (ex. rendements MASS observés sur un pas de rebalancement).
    F et PE sont les versions empiriques évaluées sur l'échantillon trié, donc
    les queues épaisses et l'asymétrie réelles pilotent directement la dynamique.

    recenter=True : recentre à moyenne 1 -> mesure RISQUE-NEUTRE (forme empirique,
                    dérive d'arbitrage) -> donne un PRIX.
    recenter=False: garde la dérive historique -> mesure RÉELLE -> donne une
                    proba de gap / perte attendue HISTORIQUES (pas un prix).
    """

    def __init__(self, rho_sample, dt: float, recenter: bool = False):
        r = np.asarray(rho_sample, float)
        r = r[np.isfinite(r) & (r > 0)]
        if r.size < 30:
            raise ValueError("Échantillon empirique trop petit (<30 points).")
        self.raw_mean = float(r.mean())
        if recenter:
            r = r / self.raw_mean
        self.rho = np.sort(r)
        self.S = self.rho.size
        self.cum = np.concatenate([[0.0], np.cumsum(self.rho)])
        self.mean = float(self.rho.mean())
        self.dt = dt
        self.sigma_ann = float(np.std(np.log(self.rho)) / np.sqrt(dt))
        self.recenter = recenter
        self.label = "empirique " + ("risque-neutre" if recenter else "historique")

    def F(self, r):
        r = np.asarray(r, float)
        idx = np.searchsorted(self.rho, r, side="right")
        out = idx / self.S
        out = np.where(np.isposinf(r), 1.0, out)
        out = np.where(r <= 0, 0.0, out)
        return out

    def PE(self, r):
        """E[rho 1_{rho<=r}] empirique."""
        r = np.asarray(r, float)
        idx = np.clip(np.searchsorted(self.rho, r, side="right"), 0, self.S)
        out = self.cum[idx] / self.S
        out = np.where(np.isposinf(r), self.mean, out)
        out = np.where(r <= 0, 0.0, out)
        return out

    def sample(self, n, rng):
        return self.rho[rng.integers(0, self.S, size=n)]


# ---------------------------------------------------------------------------
# PARAMETRES
# ---------------------------------------------------------------------------

@dataclass
class MarkovParams:
    # --- modèle ---
    spot: float = 100.0           # spot de l'actif risqué (info ; n'entre pas dans X)
    sigma: float = 0.15           # volatilité annualisée
    rate: float = 0.03            # taux sans risque (continu)
    maturity: float = 5.0         # maturité en années
    n_rebal: int = 60             # nombre de périodes de rebalancement

    # --- produit ---
    initial: float = 100.0        # capital investi C0
    guarantee: float = 100.0      # nominal garanti G
    multiplier: float = 4.0       # multiplicateur m
    w_min: float = 0.0            # poids risqué minimal
    w_max: float = 1.5            # poids risqué maximal (>1 = levier)

    # --- numérique ---
    n_grid: int = 400             # nombre de noeuds de grille
    x_min: Optional[float] = None # borne basse de grille (auto si None)
    x_max: Optional[float] = None # borne haute de grille (auto si None)


# ---------------------------------------------------------------------------
# OUTILS
# ---------------------------------------------------------------------------

def weight(x: np.ndarray | float, m: float, w_min: float, w_max: float) -> np.ndarray:
    """Poids risqué CPPI w(X) = clip(m*(1 - 1/X), w_min, w_max)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = m * (1.0 - 1.0 / x)
    w = np.where(x <= 0, w_min, w)          # X<=0 : défaisance totale
    return np.clip(w, w_min, w_max)


def _auto_bounds(p: MarkovParams, law=None) -> tuple[float, float]:
    """
    Bornes de grille (grille log, positives). La zone (x_lo, 1) — le gap — est
    finement résolue. eff_vol vient de la loi (lognormale ou empirique) ; un
    facteur de dérive couvre le cas historique (E[rho] > 1).
    """
    x0 = (p.initial / p.guarantee) * np.exp(p.rate * p.maturity)
    sig = law.sigma_ann if law is not None else p.sigma
    drift = (law.mean ** p.n_rebal) if (law is not None and law.mean > 1.0) else 1.0
    eff_vol = min(p.w_max, p.multiplier) * sig
    x_max = p.x_max if p.x_max is not None else max(x0, 2.0) * np.exp(6.0 * eff_vol * np.sqrt(p.maturity)) * drift
    x_lo = p.x_min if p.x_min is not None else 1e-2
    return float(x_lo), float(x_max)


def build_grid(p: MarkovParams) -> np.ndarray:
    """
    Grille LOG-espacée de x_lo (>0) à x_max. La masse passant sous x_lo
    (gap profond, éventuellement X<0 sous levier) est repliée sur le 1er noeud.
    """
    x_lo, x_max = _auto_bounds(p)
    nodes = np.exp(np.linspace(np.log(x_lo), np.log(x_max), p.n_grid))
    return nodes


# ---------------------------------------------------------------------------
# MATRICE DE TRANSITION
# ---------------------------------------------------------------------------

def build_operators(p: MarkovParams, law=None):
    """
    Opérateurs d'un pas avec comptabilité EXACTE de la défaisance, pour une LOI
    de rendement quelconque (lognormale ou empirique).

    Depuis chaque noeud vivant j, via la loi de rho :
        c_j  = P(X' < 1 | j)              s_j = E[(1-X')1_{X'<1}]    gx_j = E[X' 1_{X'<1}]
    La masse vivante (X' >= 1) est redistribuée par cellules de Voronoï.
    Renvoie : alive_nodes, A, c, s, gx.
    """
    dt = p.maturity / p.n_rebal
    if law is None:
        law = LognormalLaw(p.sigma, dt)

    _, x_max = _auto_bounds(p, law)
    nodes = np.exp(np.linspace(np.log(1.0 + 1e-6), np.log(x_max), p.n_grid))
    N = len(nodes)
    w = weight(nodes, p.multiplier, p.w_min, p.w_max)
    F, PE = law.F, law.PE

    # bords des cellules vivantes : 1 (barrière), milieux géométriques, +inf
    edges = np.empty(N + 1)
    edges[0] = 1.0
    edges[1:-1] = np.sqrt(nodes[:-1] * nodes[1:])
    edges[-1] = np.inf

    A = np.zeros((N, N))
    c = np.zeros(N)
    s = np.zeros(N)
    gx = np.zeros(N)

    for j in range(N):
        xj, wj = nodes[j], w[j]
        base = xj * (1.0 - wj)
        slope = xj * wj
        if slope <= 0:
            A[j, j] = 1.0
            continue
        rho_e = np.where(np.isposinf(edges), np.inf, (edges - base) / slope)
        Fe = F(rho_e)
        r1 = rho_e[0]                            # rho tel que X'=1
        c[j] = F(np.array([r1]))[0]
        PE1 = PE(np.array([r1]))[0]
        s[j] = (1.0 - base) * c[j] - slope * PE1
        gx[j] = base * c[j] + slope * PE1
        A[j, :] = np.diff(Fe)
    return nodes, A, c, s, gx


# ---------------------------------------------------------------------------
# PRICING MARKOV
# ---------------------------------------------------------------------------

def _initial_vector(p: MarkovParams, nodes: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Masse initiale sur la grille vivante (X>1), répartie en log entre les deux
    noeuds voisins de X_0. Si X_0 <= 1, le produit est défaisé d'emblée :
    on renvoie un vecteur nul + (gap_mass0, gap_short0).
    """
    x0 = (p.initial / p.guarantee) * np.exp(p.rate * p.maturity)
    v = np.zeros(len(nodes))
    if x0 <= 1.0:                                   # défaisance immédiate
        return v, 1.0, max(1.0 - x0, 0.0)
    if x0 >= nodes[-1]:
        v[-1] = 1.0
    else:
        k = int(np.searchsorted(nodes, x0)) - 1
        k = max(0, k)
        lo, hi = np.log(nodes[k]), np.log(nodes[k + 1])
        frac = (np.log(x0) - lo) / (hi - lo)
        v[k] = 1.0 - frac
        v[k + 1] = frac
    return v, 0.0, 0.0


def price_markov(p: MarkovParams, law=None) -> dict:
    """
    Price le CPPI et le put de gap risk par opérateurs de Markov, avec
    comptabilité exacte de la défaisance (pas-à-pas explicite).
    law : loi de rendement (None -> lognormale paramétrique p.sigma).
    """
    nodes, A, c, s, gx = build_operators(p, law)
    v, gap_mass, gap_short = _initial_vector(p, nodes)
    # gap_x = E[X 1_{défaisé}] : = x0 si défaisé d'emblée, sinon 0
    x0 = (p.initial / p.guarantee) * np.exp(p.rate * p.maturity)
    gap_x = x0 * gap_mass if gap_mass > 0 else 0.0

    for _ in range(p.n_rebal):                      # périodes identiques
        gap_mass += float(v @ c)
        gap_short += float(v @ s)
        gap_x += float(v @ gx)
        v = v @ A                                   # masse vivante restante

    disc = np.exp(-p.rate * p.maturity)
    G = p.guarantee

    alive_mass = float(v.sum())
    EX = float(v @ nodes) + gap_x                   # E[X_T] (vivant + défaisé)
    strat_price = disc * G * EX                     # ≈ C0 attendu
    put_price = disc * G * gap_short                # disc * G * E[max(1-X_T,0)]

    return {
        "put_price": put_price,
        "p_gap": gap_mass,
        "strategy_price": strat_price,
        "expected_shortfall": gap_short,
        "EX_T": EX,
        "mass": alive_mass + gap_mass,
        "alive_mass": alive_mass,
        "nodes": nodes,
        "dist": v,                                  # distribution vivante (X>1)
    }


# ---------------------------------------------------------------------------
# GREEKS (différences finies, "re-pricing")
# ---------------------------------------------------------------------------

def greeks_markov(p: MarkovParams, law=None, h_rel: float = 1e-3) -> dict:
    """
    Sensibilités du put par re-pricing. Avec une loi empirique, le vega
    lognormal n'a pas de sens (pas de sigma paramétrique) : il est renvoyé NaN.
    """
    def reprice(**over):
        q = MarkovParams(**{**p.__dict__, **over})
        return price_markov(q, law)["put_price"]

    dC = max(p.initial * h_rel, 1e-6)
    base = reprice()
    up = reprice(initial=p.initial + dC)
    dn = reprice(initial=p.initial - dC)
    sens_c0 = (up - dn) / (2 * dC)
    gamma_c0 = (up - 2 * base + dn) / (dC ** 2)

    if law is None:                      # vega seulement en paramétrique
        dS = max(p.sigma * h_rel, 1e-6)
        vega = (reprice(sigma=p.sigma + dS) - reprice(sigma=p.sigma - dS)) / (2 * dS)
    else:
        vega = np.nan

    x0 = (p.initial / p.guarantee) * np.exp(p.rate * p.maturity)
    w0 = float(weight(x0, p.multiplier, p.w_min, p.w_max))
    delta_underlying = sens_c0 * w0 * p.initial / p.spot if p.spot else np.nan

    return {
        "sens_C0": sens_c0,
        "gamma_C0": gamma_c0,
        "vega": vega,
        "delta_underlying": delta_underlying,
        "w0": w0,
    }


# ---------------------------------------------------------------------------
# MONTE CARLO DE CONTROLE (même dynamique exacte)
# ---------------------------------------------------------------------------

def price_monte_carlo(p: MarkovParams, law=None, n_paths: int = 200_000, seed: int = 0) -> dict:
    """
    Monte Carlo utilisant la MEME récurrence et la MEME loi de rendement que le
    moteur Markov. Référence indépendante pour valider le pricing par grille.
    """
    rng = np.random.default_rng(seed)
    dt = p.maturity / p.n_rebal
    if law is None:
        law = LognormalLaw(p.sigma, dt)

    x0 = (p.initial / p.guarantee) * np.exp(p.rate * p.maturity)
    X = np.full(n_paths, x0)
    for _ in range(p.n_rebal):
        rho = law.sample(n_paths, rng)
        w = weight(X, p.multiplier, p.w_min, p.w_max)
        X = X * (1.0 + w * (rho - 1.0))

    disc = np.exp(-p.rate * p.maturity)
    G = p.guarantee
    shortfall = np.clip(1.0 - X, 0.0, None)

    put = disc * G * shortfall
    put_price = float(put.mean())
    put_se = float(put.std(ddof=1) / np.sqrt(n_paths))   # erreur standard
    p_gap = float(np.mean(X < 1.0))
    strat = disc * G * float(X.mean())

    return {
        "put_price": put_price,
        "put_se": put_se,
        "put_ci95": (put_price - 1.96 * put_se, put_price + 1.96 * put_se),
        "p_gap": p_gap,
        "strategy_price": strat,
        "EX_T": float(X.mean()),
        "n_paths": n_paths,
    }
