# Diagnostic de l'hypothèse d'incréments indépendants (iid) — MASI / CPPI Markov

Outil de **validation de modèle (SR 11-7)** qui teste l'hypothèse d'**incréments
indépendants** (« log(prix) est un processus de Lévy ») sur une série de prix,
en priorité le **MASI**. C'est l'hypothèse qui fait tenir le pricer CPPI par
opérateur de Markov (Paulot–Lacroze) : sous incréments iid, X = coussin/seuil est
une chaîne de Markov 1-D et tout le pricing rapide en découle.

L'outil dit, sans surinterprétation : **est-ce que l'hypothèse tient, où casse-t-elle,
et avec quelle sévérité.**

## Architecture

- `increments.py` — moteur : fonctions **pures**, aucune dépendance UI. Chaque test
  renvoie `{ statistique, p_value (ou IC), taille_effet, n, verdict, details, limites }`.
- `app.py` — interface **Streamlit** (7 onglets).
- `requirements.txt` — dépendances.

Estimateurs **implémentés à la main** (formules en commentaire) où la transparence
prime : variance ratio de Lo–MacKinlay (robuste à l'hétéroscédasticité), test des runs,
R/S, DFA, GPH. Tous validés sur données synthétiques (bruit iid → VR≈1, H≈0.5 ;
marche aléatoire → DFA α≈1.5 ; AR(1)+ → persistance).

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Placez les fichiers de données (`MASI_HISTO.xlsx`, etc.) **à côté de `app.py`**
(chemins relatifs ; le MASI est détecté automatiquement). Vous pouvez aussi charger
d'autres séries (CAC40, DAX, Brent) via l'upload pour la comparaison multi-actifs.

## Déploiement Streamlit Community Cloud

Committez `app.py`, `increments.py`, `requirements.txt` **et** les `.xlsx` au même
niveau dans le dépôt, puis pointez l'app sur `app.py`. Aucun chemin absolu.

## Batterie de tests

| Axe | Tests | Verdict-clé |
|-----|-------|-------------|
| Indépendance sérielle | Ljung-Box, Variance Ratio (surrogate), Runs, BDS | cœur du diagnostic |
| Hétéroscédasticité | ARCH-LM, ACF\|r\|, LB(r²) | clustering → *traité par extension régimes* |
| Stationnarité / MR | ADF + KPSS + demi-vie | racine unitaire ⇒ iid OK |
| Mémoire longue | Hurst R/S · DFA · GPH (+ bande surrogate) | H≈0.5 ⇒ pas de mémoire |
| Distribution | Jarque-Bera, asymétrie, kurtosis, QQ | *caractérisation*, PAS une violation iid |
| Volatilité locale σ(S,t) | — | non testé (hors scope, documenté) |

## Principes de droiture (non négociables)

1. Toujours : statistique + p-value/IC + taille d'échantillon ; jamais un « OK » nu.
2. H0/H1 énoncés à l'écran pour chaque test.
3. Seuil de matérialité énoncé **avant** le verdict, avec son origine.
4. **Significativité ≠ matérialité** : on reporte une taille d'effet et on juge sur la matérialité.
5. **Non-normalité ≠ non-indépendance** : queues épaisses = Lévy à sauts (Kou), pas une violation iid.
6. Test à la **fréquence de rebalancement** (mensuelle par défaut) ; une dépendance
   quotidienne qui disparaît en mensuel est immatérielle (montré dans l'onglet Indépendance).
7. Estimateurs fragiles : plusieurs méthodes, désaccord reporté, **bande surrogate / bootstrap**.
8. Note « Limites du test » sur chaque panneau.
9. **Test multiple** : lecture holistique ; verdict conservateur (« indéterminé » si borderline).
10. **Reproductibilité** : graine fixe affichée pour tout bootstrap / Monte Carlo.

## Phase 2 (à prévoir, non bloquante)

Pour toute violation matérielle, **chiffrer l'impact prix** : comparer le put de
garantie sous hypothèse iid (pricer Markov) à un Monte Carlo sous le modèle portant
la caractéristique détectée (AR(1)/OU pour le MR, GARCH pour le clustering, mBf pour
la mémoire longue). Le verdict « matériel » ne devient définitif que si l'écart de
prix dépasse un seuil (ex. 5 %). La colonne « Impact pricing » de la table de synthèse
est réservée à cet effet.
