"""
controle.py — batterie de contrôle du pricer Markov (démarche SR 11-7).
Reproduit les niveaux N2 (implémentation), N3 (convergence), N4 (Markov vs
Monte Carlo) et N5 (sensibilités économiques).

    python controle.py
"""

import numpy as np
import markov_cppi as mk


def P(**over):
    d = dict(sigma=0.40, rate=0.03, maturity=5, n_rebal=60, multiplier=4,
             initial=100, guarantee=100, w_max=1.5, n_grid=600)
    d.update(over)
    return mk.MarkovParams(**d)


def main():
    print("=" * 74)
    print("N2 — IMPLEMENTATION")
    print("=" * 74)
    r = mk.price_markov(P())
    print(f"  conservation de masse : {r['mass']:.8f}   (cible 1.0)")
    r0 = mk.price_markov(P(multiplier=0.0))
    x0 = np.exp(0.03 * 5)
    print(f"  m=0 : put={r0['put_price']:.6f} (0) | gap={r0['p_gap']:.4f} (0) | "
          f"E[X]={r0['EX_T']:.5f} (cible {x0:.5f})")

    print("\n" + "=" * 74)
    print("N3 — CONVERGENCE EN N")
    print("=" * 74)
    print(f"  {'N':>6} | {'put':>10} | {'p_gap':>8} | {'E[X]':>8}")
    for N in [150, 300, 600, 1200, 2400]:
        rr = mk.price_markov(P(n_grid=N))
        print(f"  {N:>6} | {rr['put_price']:>10.5f} | {rr['p_gap']:>8.5f} | {rr['EX_T']:>8.5f}")

    print("\n" + "=" * 74)
    print("N4 — MARKOV vs MONTE CARLO (test décisif)")
    print("=" * 74)
    print(f"  {'sigma':>6} | {'Mk put':>9} {'Mk gap':>8} | {'MC put':>9} {'MC gap':>8} | put∈IC95")
    for sig in [0.15, 0.25, 0.40, 0.60]:
        q = P(sigma=sig, n_grid=1200)
        rm = mk.price_markov(q)
        mc = mk.price_monte_carlo(q, n_paths=1_000_000, seed=1)
        lo, hi = mc["put_ci95"]
        inside = lo - 2e-3 <= rm["put_price"] <= hi + 2e-3
        print(f"  {sig:>6.2f} | {rm['put_price']:>9.4f} {rm['p_gap']:>8.4f} | "
              f"{mc['put_price']:>9.4f} {mc['p_gap']:>8.4f} | {inside}")

    print("\n" + "=" * 74)
    print("N5 — SENSIBILITES ECONOMIQUES (signe attendu)")
    print("=" * 74)
    g = lambda **o: round(mk.price_markov(P(**o))["p_gap"], 4)
    print("  vol   ↑ -> gap ↑ :", [g(sigma=s) for s in (0.15, 0.25, 0.40, 0.60)])
    print("  m     ↑ -> gap ↑ :", [g(multiplier=m) for m in (2, 4, 6, 8)])
    print("\n" + "=" * 74)
    print("N4-bis — LOI EMPIRIQUE : Markov vs Monte Carlo (même loi)")
    print("=" * 74)
    try:
        import pandas as pd
        import empirical as emp
        rng = np.random.default_rng(7)
        n = 6000
        lr = (0.0003 + rng.standard_t(4, n) * 0.008
              - (rng.random(n) < 0.01) * rng.exponential(0.04, n))   # queues + skew
        s = pd.Series(100 * np.exp(np.cumsum(lr)),
                      index=pd.bdate_range("2002-01-01", periods=n), name="SYNTH")
        q = P(n_grid=1200)
        lawRN = emp.make_empirical_law(s, q, recenter=True)
        rE = mk.price_markov(q, law=lawRN)
        mcE = mk.price_monte_carlo(q, law=lawRN, n_paths=600_000, seed=2)
        lo, hi = mcE["put_ci95"]
        inside = lo - 2e-3 <= rE["put_price"] <= hi + 2e-3
        print(f"  vol empirique={lawRN.sigma_ann:.3f}")
        print(f"  Markov put={rE['put_price']:.4f} gap={rE['p_gap']:.4f} E[X]={rE['EX_T']:.4f}")
        print(f"  MC     put={mcE['put_price']:.4f} gap={mcE['p_gap']:.4f}  -> put dans IC95 : {inside}")
        rLN = mk.price_markov(P(sigma=lawRN.sigma_ann, n_grid=1200))
        print(f"  lognormal MÊME vol : put={rLN['put_price']:.4f} gap={rLN['p_gap']:.4f}  "
              f"(les queues empiriques changent le gap)")
    except Exception as e:
        print("  (section empirique ignorée :", e, ")")


if __name__ == "__main__":
    main()
