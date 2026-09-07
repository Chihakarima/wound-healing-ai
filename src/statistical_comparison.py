"""Comparaison statistique appariée U-Net vs baseline sur les 14 images de test.

Le README rapporte déjà les moyennes (Dice 0,882 vs 0,576) mais pas de test de
significativité : avec seulement 14 images, un écart de moyennes ne suffit pas à
lui seul à écarter un effet du hasard d'échantillonnage. Ce script complète les
métriques déjà calculées (test_metrics.json / test_baseline_metrics.json) par :
- un test des rangs signés de Wilcoxon (non paramétrique, adapté à un petit
  échantillon sans hypothèse de normalité) sur les paires (métrique U-Net,
  métrique baseline) de chaque image, pour le Dice et l'IoU ;
- un intervalle de confiance bootstrap (paires rééchantillonnées avec remise)
  sur la différence moyenne, pour quantifier l'incertitude sans hypothèse
  distributionnelle non plus.

Usage:
    python -m src.statistical_comparison
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

PRED_DIR = "outputs/predictions"
N_BOOTSTRAP = 10000
SEED = 42


def load_paired_metric(unet_path, baseline_path, metric="dice"):
    """Charge la métrique demandée pour les deux méthodes, appariée par id
    d'image. Lève une erreur si les deux fichiers ne couvrent pas exactement
    les mêmes images (une comparaison appariée n'a de sens que sur des paires
    complètes)."""
    unet = {r["id"]: r[metric] for r in json.load(open(unet_path))["per_image"]}
    baseline = {r["id"]: r[metric] for r in json.load(open(baseline_path))["per_image"]}
    if set(unet) != set(baseline):
        raise ValueError(
            f"IDs non appariés entre {unet_path} et {baseline_path}: "
            f"{set(unet) ^ set(baseline)}"
        )
    ids = sorted(unet, key=int)
    return ids, np.array([unet[i] for i in ids]), np.array([baseline[i] for i in ids])


def bootstrap_ci_mean_diff(diffs, n_bootstrap=N_BOOTSTRAP, seed=SEED, ci=0.95):
    """IC bootstrap percentile sur la différence moyenne (paires rééchantillonnées
    avec remise) : ne suppose pas que les différences suivent une loi normale,
    contrairement à un IC paramétrique classique -- plus prudent sur 14 points."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    resampled_means = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(resampled_means, [alpha, 1 - alpha])
    return float(lo), float(hi)


def compare_metric(ids, unet_vals, baseline_vals, metric_name):
    diffs = unet_vals - baseline_vals
    statistic, p_value = wilcoxon(unet_vals, baseline_vals)
    ci_lo, ci_hi = bootstrap_ci_mean_diff(diffs)
    return {
        "metric": metric_name,
        "n": len(ids),
        "mean_unet": float(unet_vals.mean()),
        "mean_baseline": float(baseline_vals.mean()),
        "mean_diff": float(diffs.mean()),
        "n_positive_diff": int((diffs > 0).sum()),
        "n_negative_diff": int((diffs < 0).sum()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value": float(p_value),
        "bootstrap_ci95_mean_diff": [ci_lo, ci_hi],
    }


def main():
    unet_path = os.path.join(PRED_DIR, "test_metrics.json")
    baseline_path = os.path.join(PRED_DIR, "test_baseline_metrics.json")

    results = []
    for metric in ["dice", "iou"]:
        ids, unet_vals, baseline_vals = load_paired_metric(unet_path, baseline_path, metric)
        result = compare_metric(ids, unet_vals, baseline_vals, metric)
        results.append(result)

        print(f"\n{metric.upper()} (n={result['n']} images de test appariées)")
        print(f"  Moyenne U-Net={result['mean_unet']:.4f}  Moyenne baseline={result['mean_baseline']:.4f}"
              f"  diff. moyenne={result['mean_diff']:+.4f}")
        print(f"  U-Net meilleur sur {result['n_positive_diff']}/{result['n']} images, "
              f"pire sur {result['n_negative_diff']}/{result['n']}")
        print(f"  Wilcoxon signé : statistique={result['wilcoxon_statistic']:.1f}, "
              f"p={result['wilcoxon_p_value']:.4g}")
        print(f"  IC bootstrap 95% de la diff. moyenne : [{result['bootstrap_ci95_mean_diff'][0]:+.4f}, "
              f"{result['bootstrap_ci95_mean_diff'][1]:+.4f}]")
        verdict = "significatif (p < 0,05)" if result["wilcoxon_p_value"] < 0.05 else "non significatif (p >= 0,05)"
        print(f"  -> {verdict}")

    out_path = os.path.join(PRED_DIR, "test_statistical_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": "test",
            "test": "wilcoxon_signed_rank",
            "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": SEED, "ci": 0.95},
            "results": results,
        }, f, indent=2)
    print(f"\nRésultats -> {out_path}")


if __name__ == "__main__":
    main()
