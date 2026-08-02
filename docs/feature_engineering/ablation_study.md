# Étude d'ablation

Le test emploie CatBoost par défaut et les mêmes cinq folds groupés du corpus de
développement. La MAE passe de 0,267714 avec le seul contexte à 0,098414 après
ajout de la mobilité, 0,095515 avec le combat, 0,092661 avec les ressources et
0,061500 avec `killRank`.

La mobilité fournit le gain comportemental principal. Les familles combat et
ressources ajoutent chacune environ 0,0029 de MAE. `killRank` ajoute 0,03116,
mais transforme le cas d'usage en estimation post-match. Le R² ajusté n'est pas
utilisé pour décider ces ajouts : l'ablation hors échantillon est mieux adaptée
à un ensemble d'arbres.

Le protocole et toutes les métriques sont documentés dans
[`docs/modeling/feature_ablation.md`](../modeling/feature_ablation.md).
