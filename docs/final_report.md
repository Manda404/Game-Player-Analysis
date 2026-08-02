# Rapport final

## Résultat

Le projet a été reconstruit autour du livrable demandé par Gameloft : un
notebook clair, exécuté et soutenu par un petit package testable. XGBoost est le
meilleur modèle du nouveau benchmark GroupKFold avec une MAE de **0,06156**,
réduite à **0,06111** après projection sur la grille des rangs possibles.

| Rang | Modèle | MAE | RMSE | R² | Temps fit cumulé | Diagnostic overfit |
|---:|---|---:|---:|---:|---:|---|
| 1 | XGBoost | 0,06156 | 0,08666 | 0,92053 | 5,18 s | écart modéré |
| 2 | LightGBM | 0,06173 | 0,08671 | 0,92043 | 11,78 s | écart modéré |
| 3 | CatBoost | 0,06177 | 0,08667 | 0,92051 | 9,47 s | faible écart |
| 4 | Random Forest | 0,06555 | 0,09263 | 0,90920 | 26,73 s | écart élevé |
| 5 | Médiane | 0,26799 | 0,30781 | -0,00244 | 0 s | non applicable |

Les temps décrivent cette exécution locale et ne sont pas une propriété
intrinsèque des modèles.

## Méthodologie

1. validation stricte des deux CSV officiels ;
2. conversion des sentinelles sans suppression de ligne ;
3. analyse de la cible, des comportements, de la couverture et de la stabilité ;
4. construction d'un contrat numérique déterministe ;
5. cinq folds groupés par `gameId` avec zéro chevauchement ;
6. comparaison de la baseline et des quatre ensembles demandés ;
7. ablation `killRank`, analyse par taille de match et projection de grille ;
8. entraînement unique du gagnant, manifeste et soumission.

## Enseignements

- La mobilité est le signal comportemental principal, complété par collecte,
  soins et combat.
- Les moyennes de cible varient peu entre solo, duo et squad (0,461 à 0,486) ;
  le mode sert de contexte, sans interprétation causale.
- `killRank` réduit la MAE de 0,03178 : il rend le modèle post-match performant,
  mais interdit toute présentation early-game.
- Un split aléatoire expose 56,5 % des lignes de validation à un match déjà vu.
- Les rosters incomplets rendent les agrégats équipe/lobby non défendables.
- La pseudo-date ne permet pas une validation temporelle réelle.

## Simplification mesurée

| Zone | Avant | Après |
|---|---:|---:|
| Python sous `src/` | 45 fichiers, 7 352 lignes | 9 fichiers, 931 lignes |
| Scripts | 7 fichiers, 3 376 lignes | 1 script |
| Notebooks | 17 | 1 exécuté |
| Tests | 31 fichiers, 337 tests | 6 fichiers, 17 tests ciblés, couverture 91 % |
| Contrats modèle actifs | Plusieurs, incompatibles | Un manifeste aligné |

## Limites

- aucune cible pour le fichier test ;
- aucune identité joueur longitudinale fiable ;
- pas de roster complet, map, MMR ou plateforme ;
- pas de timestamp réel ni snapshot early-game ;
- pas d'interprétation causale des comportements ;
- score brut légèrement inférieur au benchmark historique incohérent, différence
  explicitée dans le journal des décisions.

## Livrables

- `notebooks/game_player_analysis.ipynb` ;
- `artifacts/model.joblib` ;
- `artifacts/model_manifest.json` ;
- `artifacts/model_comparison.csv` ;
- `artifacts/killrank_ablation.csv` ;
- `data/output/submission.csv` ;
- documentation d'analyse et de refactoring sous `docs/`.
