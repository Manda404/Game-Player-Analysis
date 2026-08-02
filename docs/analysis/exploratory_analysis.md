# Analyse exploratoire

## Cible

`winRankPercentage` a une moyenne de 0,47233, une médiane de 0,45785 et un
écart-type de 0,30745. La cible suit presque exactement la grille définie par
`maxRank`, sans être présentée comme littéralement parfaite.

## Relations principales

Les corrélations de Spearman les plus fortes avec la cible sont `walkDist`
(0,866), `killRank` (-0,715), `upgrades` (0,681), `weapons` (0,665), `heals`
(0,564), `damages` (0,449), `rideDist` (0,433) et `kills` (0,425).

Les profils par quantiles et intervalles à 95 % sont préférés aux seuls
coefficients : ils montrent la forme non linéaire et les plateaux. Voir
[`05_feature_target_profiles.png`](../../artifacts/figures/05_feature_target_profiles.png).

## Structure et date

Tous les `gameId` valides multi-lignes portent plusieurs dates. Le span médian
train est 45,03 jours. En parallèle, seulement 2,34 % des lignes ont un
coéquipier observé. Ces deux résultats interdisent respectivement une validation
temporelle naïve et des agrégats équipe/lobby crédibles.

Le détail figure dans `date_integrity.csv` et `sampling_coverage.csv`.

## Drift train/test

Le drift avait d'abord été réduit à la différence standardisée des moyennes,
ce qui était insuffisant. L'audit corrigé mesure maintenant, sur les 16 features
du contrat final : PSI, statistique KS, distance de Wasserstein normalisée,
variation de masse à zéro, percentiles, déplacement des catégories et
validation adversariale multivariée.

- PSI numérique maximal : 0,00509 sur `damages` ;
- KS maximal : 0,01312 sur `damages`, p descriptive 0,411 ;
- différence de moyenne standardisée maximale : 0,02347 sur `damages` ;
- plus grand changement de taux de zéro : 0,722 point sur `rideDist`/`heals` ;
- PSI catégoriel maximal : 0,00844 pour `gameType` ;
- plus grand déplacement d'une modalité : -0,622 point pour `legacy` ;
- validation adversariale logistique : ROC AUC 0,49325 ± 0,00542.

Ces tailles d'effet sont faibles et la validation adversariale ne sépare pas
linéairement train et test. On conclut donc à **l'absence de drift matériel
détecté sur les variables mesurées**, et non à une identité parfaite des
distributions. Le drift de performance ou de concept ne peut pas être mesuré
sans cible test ; le drift temporel n'est pas interprétable avec la colonne
`date` incohérente. Les p-values KS restent descriptives et ne remplacent pas
les tailles d'effet.

Preuves :
[`distribution_shift.csv`](../../artifacts/metrics/distribution_shift.csv),
[`categorical_shift.csv`](../../artifacts/metrics/categorical_shift.csv),
[`adversarial_validation.csv`](../../artifacts/metrics/adversarial_validation.csv)
et
[`07b_train_test_drift.png`](../../artifacts/figures/07b_train_test_drift.png).
