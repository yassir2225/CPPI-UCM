# Pricer CPPI — Opérateurs de Markov

Outil **autonome** de pricing du *gap risk* d'une stratégie CPPI par opérateurs de
Markov (propagation de probabilités sur grille), avec **comptabilité exacte de la
défaisance** et **Monte Carlo intégré** comme contrôle. Indépendant du backtest.

## Deux dynamiques pour l'actif risqué

- **Lognormale** : modèle paramétrique GBM (volatilité σ saisie).
- **Empirique** : la **distribution réelle** des rendements d'un indice (ex. MASI)
  pilote directement la loi de transition — queues épaisses et asymétrie incluses.
  Construction par fenêtres glissantes observées ou bootstrap journalier composé.
  Deux mesures :
  - **risque-neutre** (rendements recentrés à moyenne 1) → un **prix** d'arbitrage ;
  - **historique réelle** (dérive conservée) → **proba de gap / perte attendue**
    observées (mesure de risque, pas un prix).

## Contenu

| Fichier | Rôle |
|---|---|
| `markov_cppi.py` | Moteur : lois de rendement (`LognormalLaw`, `EmpiricalLaw`), pricing Markov, greeks, Monte Carlo. |
| `empirical.py`   | Chargement d'un indice + construction de l'échantillon de rendements déflatés (MASI). |
| `app_markov.py`  | Interface Streamlit (pricing, dynamique empirique, contrôle, sensibilités). |
| `controle.py`    | Batterie de contrôle reproductible (N2–N5 + validation empirique). |
| `requirements.txt` | Dépendances. |

## Lancement

```bash
pip install -r requirements.txt
streamlit run app_markov.py     # interface : choisir "Empirique" et charger MASI_HISTO.xlsx
python controle.py              # batterie de contrôle en console
```

## Méthode

Variable renormalisée `X_i = C_i/H_i`, `H_i = G·e^{-r(T-tᵢ)}`. Récurrence
`X_{i+1} = X_i·[1 + w(X_i)·(ρ−1)]`, `w(X)=clip(m(1−1/X), w_min, w_max)`. Tout
nœud `X ≤ 1` est absorbant (défaisance). On discrétise la région vivante `X>1`
sur grille log (cellules de Voronoï, faible diffusion) ; la masse franchissant
`X=1` et son *shortfall* sont calculés exactement à chaque pas via la CDF et
l'espérance partielle de la loi `ρ` (lognormale **ou empirique**), ce qui rend la
proba de gap précise indépendamment de la grille.

### Sorties
Prix du put de gap risk `e^{-rT}·G·E[max(1−X_T,0)]`, proba de gap `P(X_T<1)`,
prix de la stratégie (`≈ C₀` en risque-neutre), greeks (le vega n'existe qu'en
lognormal), distribution de `X_T`, statistiques de la distribution empirique.

## Validation (SR 11-7)

- **N2** masse conservée, cas `m=0` exact.
- **N3** convergence en `N` (put et proba de gap).
- **N4** Markov vs Monte Carlo concordants — **en lognormal ET en empirique**.
- **N5** sensibilités économiques au bon signe.

À volatilité **identique**, la loi empirique (queues épaisses, skew négatif) peut
donner un gap **bien plus cher** qu'une gaussienne : une gaussienne sous-estime
le risque de gap. C'est la raison d'être de la dynamique empirique.

## Limites / extensions
Périodes supposées i.i.d. (le bootstrap journalier perd le clustering de
volatilité ; les fenêtres glissantes le conservent mais donnent peu
d'observations indépendantes pour de longues périodes). Le résultat empirique
reflète l'histoire de l'indice (le futur peut différer). Extensions v2 : seuil
linéaire, profit lock-in, frais, coupons, rebalancement conditionnel.
