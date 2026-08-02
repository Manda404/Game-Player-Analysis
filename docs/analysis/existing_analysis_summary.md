# Synthèse des analyses existantes

## Périmètre

Deux analyses indépendantes ont précédé le refactoring : la première est
conservée intacte dans `docs.old/`, la seconde était répartie entre 17 fichiers
Markdown, trois scripts et 97 tables/figures. L'état complet antérieur reste
récupérable dans `dist/pre_refactor_2026-08-02.zip`.

## Conclusions communes

- 50 000 lignes train, 5 000 test, une ligne par joueur après la partie ;
- aucune cellule manquante au sens CSV, mais des sentinelles métier ;
- `rankPts=-1` signifie absent et certains zéros `killPts`/`winPts` sont absents ;
- `winRankPercentage` est un rang borné et discrétisé par `maxRank` ;
- la mobilité à pied est le signal comportemental dominant ;
- combat, collecte et soins apportent des voies complémentaires ;
- `killRank` est très prédictif mais doit être séparé d'un scénario sans rang final ;
- `gameId` doit grouper la validation ;
- les IDs bruts ne sont pas des prédicteurs ;
- rétention, churn et monétisation sont impossibles à mesurer avec ce snapshot.

## Correction majeure de la seconde analyse

La première analyse interprétait janvier-avril puis mai comme un vrai découpage
temporel. La seconde a montré que les lignes d'un même `gameId` portent des
dates différentes, avec un span médian de 45 jours. `date` est donc une
pseudo-date d'export/anonymisation. Le test reste externe au train, mais il ne
prouve pas une généralisation calendaire.

## Nouveaux constats structurels

| Mesure | Valeur |
|---|---:|
| Lignes observées par match, moyenne / maximum | 1,614 / 8 |
| Équipes `(gameId, teamId)` singletons | 98,824 % |
| Lignes avec un coéquipier observé | 2,340 % |
| Matchs multi-lignes avec plusieurs pseudo-dates | 100 % |

Ces chiffres excluent les agrégats équipe/lobby du modèle final : ils seraient
le plus souvent identiques à la ligne ou calculés sur un roster incomplet.

## État historique de la modélisation

Le benchmark historique plaçait CatBoost à MAE 0,06037, mais le modèle, le CSV
préparé et la sélection ne partageaient plus le même contrat. Le modèle attendait
`killRankPercentile`, alors que les données courantes contenaient
`combatActivity`. Ce score est conservé comme point de comparaison, pas comme
résultat exécutable.

## Décision de consolidation

Les statistiques descriptives et limites métier sont conservées. Les trois
scripts analytiques, les 17 notebooks et les artefacts incompatibles sont
remplacés par un notebook exécuté, une fonction par règle métier et un benchmark
unique.
