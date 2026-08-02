# Matrice de couverture Data Science

> La colonne d'état initial décrit le projet audité avant intervention. La
> réalisation finale de chaque action est résumée dans `03_final_changes.md`.

## Échelle

- Présent : une implémentation ou une preuve existe dans le projet courant.
- Suffisant : la preuve permet une décision défendable.
- Visualisé : un graphique lisible existe dans le notebook final.
- Justifié : la décision est chiffrée, limitée et reliée au cas d'usage.

Cette première matrice décrit l'état audité avant consolidation. La colonne
Action constitue le contrat de sortie et sera vérifiée dans
03_final_changes.md.

| Domaine | Présent | Suffisant | Visualisé | Justifié | Action |
|---|---|---|---|---|---|
| Compréhension métier | Oui | Partiel | Non nécessaire | Partiel | préciser cible équipe sur ligne joueur et scénario post-match |
| Sujet officiel | Externe | Oui | Non nécessaire | Oui | tracer les exigences dans notebook et rapport |
| Inventaire des données | Oui | Oui | Faible | Oui | condenser dans une vue générale |
| Dictionnaire des variables | Historique | Partiel | Non nécessaire | Partiel | conserver définitions officielles et réserves |
| Qualité des données | Oui | Partiel | Non | Partiel | produire table et figure décisionnelles |
| Valeurs manquantes NaN | Oui | Oui | Non | Oui | conserver constat zéro NaN |
| Valeurs codées / sentinelles | Oui | Oui | Non | Oui | visualiser prévalence et connecter au pipeline |
| Doublons | Oui | Oui | Non nécessaire | Oui | conserver |
| Identifiants corrompus | Oui | Non | Non | Non | groupement conservateur + audit brut |
| Anomalies sémantiques | Oui | Partiel | Non | Partiel | chiffrer kills sans dégâts, combat sans distance |
| Valeurs extrêmes | Historique | Partiel | Non | Partiel | profils brut/log sans suppression arbitraire |
| Distributions numériques | Historique | Partiel | Très faible | Partiel | panneau ciblé zéros/queues |
| Analyse de la cible | Oui | Partiel | Oui | Oui | ajouter grille maxRank et extrêmes |
| Analyse temporelle | Oui | Non | Non | Non | documenter contradiction officielle/données |
| Stabilité train-test | Oui | Partiel | Non | Partiel | compléter moyennes par KS/queues/modes |
| Analyse par mode | Oui | Partiel | Non | Partiel | visualiser effectifs et erreurs |
| Analyse par équipe | Oui | Oui pour rejet | Non | Oui | visualiser couverture sparse ; rejeter agrégats |
| Analyse par partie | Oui | Partiel | Non | Oui | distinguer lignes observées de taille réelle |
| Relations features-cible | Oui | Partiel | Corrélations seules | Partiel | profils par quantiles et IC |
| Relations entre features | Historique | Partiel | Non | Partiel | heatmap Spearman ciblée |
| Multicolinéarité | Historique | Partiel | Non | Partiel | documenter maxRank/numTeams et composites |
| Information mutuelle | Historique | Partiel | Non | Partiel | garder diagnostic, prudence mécanique cible |
| Interactions | Historique | Faible | Non | Faible | mobilité × combat ciblée |
| KPI | Historique | Partiel | Non | Partiel | formules, zéros, distribution, redondance, usage |
| Segmentation | Historique | Partiel | Non | Partiel | archive-only ; ne pas alourdir le modèle final |
| Feature engineering | Oui | Partiel | Non | Partiel | relier chaque feature à preuve actuelle |
| Sélection de variables | Oui | Non | Non | Non | ablation progressive par familles |
| Features équipe/contexte | Rejetées | Oui | Non | Oui | rendre le rejet visible |
| Fuite de cible | Oui | Oui | Non nécessaire | Oui | conserver target/IDs hors contrat |
| Fuite par gameId | Oui | Non pour IDs invalides | Non | Non | grouper les collisions invalides conservativement |
| Stratégie de split | GroupKFold | Partiel | Non | Partiel | comparer ligne, groupe, pseudo-temps, pseudo-temps purgé |
| Baseline moyenne | Non | Non | Non | Non | ajouter |
| Baseline médiane | Oui | Oui | Oui avec modèles | Oui | conserver |
| Modèle linéaire | Non | Non | Non | Non | ajouter Ridge standardisé |
| Comparaison de modèles | Oui | Oui | MAE seulement | Oui | MAE/RMSE/R², folds, temps, gap |
| Hyperparameter tuning | Historique seulement | Non | Non | Non | recherche bornée sur contrat courant |
| Overfitting | Étiquette simple | Non | Non | Non | folds, train-validation gap, stabilité |
| Robustesse entre folds | Partiel | Partiel | Non | Partiel | points par fold et écart-type |
| Post-traitement | Oui | Oui | Non | Oui | montrer gain MAE et coût RMSE |
| Incertitude prédictive | Historique | Non courant | Non | Partiel | ne pas déployer ; documenter hors périmètre |
| Interprétabilité | Non courant | Non | Non | Non | permutation importance sur holdout |
| SHAP / PDP | Historique | Non | Non | Non | n'ajouter que si gain interprétatif stable |
| Analyse des erreurs globale | Très faible | Non | Prédiction-réel | Non | résidus, quantiles, biais |
| Erreurs par cible | Non | Non | Non | Non | ajouter |
| Erreurs par mode | Non courant | Non | Non | Non | ajouter avec seuil d'effectif |
| Erreurs par équipe observée | Non | Non | Non | Non | ajouter avec avertissement de couverture |
| Erreurs par grille maxRank | Oui | Partiel | Non | Partiel | renommer ; maxRank n'est pas taille réelle |
| Erreurs par période | Non | Non | Non | Non | stress test, pas causalité temporelle |
| Sauvegarde du modèle | Oui | Partiel | Non nécessaire | Partiel | enrichir manifeste et checksum |
| Chargement du modèle | Oui | Partiel | Non nécessaire | Partiel | vérifier classe, features et schéma |
| Pipeline d'inférence | Non | Non | Non | Non | predict_from_csv brut → soumission |
| Bornes des prédictions | Oui | Oui | Non nécessaire | Oui | conserver et tester |
| Ordre des colonnes | Oui | Partiel | Non nécessaire | Oui | valider au chargement et à l'inférence |
| Logging | Non | Non | Non nécessaire | Non | configuration standard centralisée |
| Reproductibilité data | Oui | Oui | Non nécessaire | Oui | hashes conservés |
| Reproductibilité environnement | Non | Non | Non nécessaire | Non | réparer groupe dev Poetry et lock |
| Traçabilité des artefacts | Partiel | Non | Non nécessaire | Partiel | versions, paramètres, métriques, date, checksum |
| Tests chargement | Oui | Oui | Non nécessaire | Oui | conserver |
| Tests nettoyage | Oui | Partiel | Non nécessaire | Oui | ajouter cas sentinelles critiques |
| Tests features/division zéro | Oui | Oui | Non nécessaire | Oui | conserver |
| Tests split | Oui | Non IDs invalides | Non nécessaire | Non | ajouter overlap brut et pseudo-temps |
| Tests sauvegarde/chargement | Oui | Partiel | Non nécessaire | Partiel | checksum/manifeste |
| Tests inférence | Non | Non | Non nécessaire | Non | ajouter raw CSV, colonnes, types, bornes |
| Documentation | Oui | Non consolidée | Non nécessaire | Partiel | supprimer copies et liens morts |
| Notebook final | Oui | Partiel | Faible | Partiel | 13 sections et 8–12 figures décisionnelles |

## Risques P0

1. Les gameId invalides sont actuellement isolés par ligne, ce qui permet aux
   mêmes chaînes brutes corrompues d'apparaître dans plusieurs folds.
2. Le champ date contredit sa définition officielle et ne peut pas porter seul
   une validation temporelle.
3. Aucun chemin d'inférence autonome ne démontre le rechargement du modèle.
4. Le groupe de dépendances dev n'est pas reconnu par Poetry 2.1.3, alors que
   README.md documente poetry install puis pytest.

## Critère de complétude

Le projet sera considéré complet lorsque chaque ligne marquée Non ou Partiel
critique aura soit :

- une preuve régénérée et reliée au notebook ;
- une implémentation testée ;
- ou une décision explicite de rejet avec données, limites et impact.
