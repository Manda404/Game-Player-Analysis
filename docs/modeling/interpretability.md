# Interprétabilité de CatBoost

Deux diagnostics complémentaires sont calculés uniquement après le choix du
modèle, sur le holdout final groupé par `gameId`. Ils expliquent donc le modèle
gelé, mais ne participent ni à sa sélection ni à son entraînement.

## Importance par permutation

Chaque variable est mélangée cinq fois sur le holdout, puis la hausse de MAE
est mesurée avec des prédictions bornées à `[0, 1]`. Cette mesure répond à la
question : *combien la performance se dégrade-t-elle si cette information n'est
plus disponible ?*

| Feature | Hausse moyenne de MAE |
|---|---:|
| `killRank` | 0,28197 |
| marche/minute de match | 0,14575 |
| `kills` | 0,09645 |
| `maxRank` | 0,06500 |
| `walkDist` | 0,04195 |

Voir [`permutation_importance.csv`](../../artifacts/metrics/permutation_importance.csv)
et [`12_permutation_importance.png`](../../artifacts/figures/12_permutation_importance.png).

## Valeurs SHAP natives de CatBoost

CatBoost fournit ses valeurs TreeSHAP nativement via
`get_feature_importance(type="ShapValues")` : aucune dépendance externe
`shap` n'est requise. L'analyse explique un échantillon aléatoire déterministe
de 2 000 lignes parmi les 9 872 lignes du holdout final (`random_state=42`).
Pour chaque ligne, le pipeline vérifie l'identité suivante sur la prédiction
brute :

\[
f(x) = \mathbb{E}[f(X)] + \sum_{j=1}^{16}\phi_j(x).
\]

La valeur attendue est `0,47247` sur cet échantillon. Les cinq variables ayant
la plus grande contribution absolue moyenne sont :

| Rang | Feature | Moyenne de `|SHAP|` | Lecture prudente |
|---:|---|---:|---|
| 1 | `killRank` | 0,19127 | signal post-match dominant |
| 2 | marche/minute de match | 0,11120 | contribution locale marquée |
| 3 | `kills` | 0,07024 | effet conditionnel aux autres signaux de combat |
| 4 | `maxRank` | 0,05416 | structure de la grille de classement |
| 5 | `walkDist` | 0,02477 | mobilité résiduelle après normalisation par durée |

Le panneau de droite de la figure montre le sens de chaque contribution : une
valeur SHAP positive pousse la prédiction vers un meilleur classement normalisé,
une valeur négative la pousse vers un classement plus faible. La couleur est
normalisée **à l'intérieur de chaque variable** (bleu : valeur faible ; rouge :
valeur élevée) ; elle ne permet donc pas de comparer les unités entre variables.

Les résultats SHAP confirment le premier rang de `killRank` observé avec la
permutation. Ils ne sont toutefois ni causaux ni indépendants : des variables
corrélées, comme `kills`, `killRank` et la mobilité, se répartissent leurs
contributions conditionnelles. En particulier, ces explications décrivent un
scénario strictement post-partie et ne doivent pas être utilisées comme levier
produit ou causal.

Artefacts publiés :

- [`shap_global_importance.csv`](../../artifacts/metrics/shap_global_importance.csv) :
  importance globale et rangs ;
- [`shap_sample.csv`](../../artifacts/metrics/shap_sample.csv) : cible,
  prédiction, résidu et valeur attendue des 2 000 lignes expliquées ;
- [`shap_values.csv`](../../artifacts/metrics/shap_values.csv) : contribution
  locale de chacune des 16 variables ;
- [`13_catboost_shap_summary.png`](../../artifacts/figures/13_catboost_shap_summary.png) :
  synthèse globale et locale.
