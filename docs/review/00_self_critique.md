# Autocritique finale du projet

> Ce document fige l'état **avant** les corrections de la revue finale. Les
> réponses apportées sont tracées dans `03_final_changes.md` et les chiffres de
> référence courants dans `docs/final_report.md`.

## Référence d'évaluation

L'audit utilise l'énoncé officiel
`description_Gameloft_DS_technical_test.pdf`. Il demande avant tout un notebook
lisible qui rende visibles quatre étapes : **Data Cleaning**, **Data Analysis &
Visualization**, **Feature Engineering** et **Modeling**. Le raisonnement, les
commentaires et la qualité du code comptent davantage qu'un gain marginal de
score.

## Ce qui est déjà solide

| Composant | État | Preuve | Décision |
|---|---|---|---|
| Chargement des CSV | Solide | séparateur, types, schéma, cible et bornes validés dans `data.py` | KEEP |
| Traçabilité des sources | Solide | SHA-256 des deux fichiers testés | KEEP |
| Sentinelles métier | Solide | `rankPts=-1` et zéros conditionnels de `killPts`/`winPts` traités explicitement | KEEP |
| Prévention de la fuite par match | Solide | `GroupKFold(gameId)`, audit de zéro groupe partagé | KEEP |
| Contrat train/test | Solide | une seule fonction de features, même ordre, cible et identifiants exclus | KEEP |
| Benchmark principal | Solide | mêmes folds et métriques pour quatre ensembles d'arbres | KEEP |
| Ablation de `killRank` | Solide | MAE 0,06156 avec et 0,09335 sans dans l'exécution de référence | KEEP |
| Persistance | Bonne base | modèle, ordre des features, benchmark et hashes réunis | IMPROVE |
| Tests | Ciblés et rapides | 17 tests, 91 % de couverture lors de la publication initiale | IMPROVE |
| Simplicité | Solide | package compact, sans ancienne architecture à quatre couches | KEEP |

## Faiblesses reconnues

### Analyse et visualisation trop courtes

Le notebook actuel ne conserve que quatre graphiques : distribution de la
cible, corrélations avec la cible, MAE des modèles et prédictions contre valeurs
réelles. Cette sélection établit le résultat principal, mais elle ne suffit pas
à montrer le cheminement demandé.

Il manque notamment :

- une vue explicite des sentinelles, zéros et longues queues ;
- la structure observée des matchs et équipes ;
- des relations non linéaires feature-cible avec effectifs et incertitude ;
- une heatmap ciblée des redondances ;
- une progression d'ablation par famille de features ;
- la comparaison visuelle des stratégies de split ;
- l'écart train-validation et sa stabilité ;
- une analyse des résidus et des sous-groupes ;
- une importance par permutation calculée hors entraînement ;
- une démonstration d'inférence depuis un CSV brut.

Le problème n'est pas l'absence de dizaines de graphiques. Il est l'absence de
preuves visuelles entre l'observation et la décision.

### Démarche scientifique insuffisamment explicite

La narration comporte des questions et décisions, mais peu d'hypothèses sont
formalisées avec le cycle complet question → hypothèse → méthode → résultat →
statut → impact. Les conclusions sur la mobilité, le combat, les ressources et
le mode sont plausibles et chiffrées dans les anciens rapports, mais trop peu
de ces preuves sont visibles dans le notebook final.

### Contradiction sur la date

L'énoncé officiel définit `date` comme la date de la partie et décrit le train
de janvier à avril 2024, puis le test en mai. Les fichiers respectent ces
plages. Pourtant, **100 % des `gameId` ayant plusieurs lignes portent plusieurs
dates**, avec un span médian de 45 jours. La documentation actuelle conclut
trop fortement que la date est une simple pseudo-date d'export.

La conclusion défendable est plus nuancée :

- la séparation officielle train/test est bien calendaire ;
- la date est incohérente avec `gameId` dans l'échantillon fourni ;
- un split temporel par ligne fuit donc des matchs ;
- un split temporel groupé peut servir de stress test, mais pas de preuve d'une
  chronologie de match parfaitement observée ;
- `GroupKFold(gameId)` reste la validation principale.

### Feature engineering et sélection incomplets

Les 16 variables finales sont lisibles, mais la preuve de leur sélection est
incomplète. L'étude actuelle ne compare que le contrat complet avec/sans
`killRank`. Elle ne mesure pas la contribution progressive de la mobilité, du
combat et des ressources. Les anciennes analyses montrent aussi que certaines
features proposées historiquement n'apportaient pas de gain stable ; cette
décision doit être régénérée sur le contrat courant.

Les agrégats équipe/lobby ont été correctement écartés : 98,824 % des couples
`(gameId, teamId)` observés sont des singletons et seulement 2,34 % des lignes
ont un coéquipier visible. Il faut rendre ce rejet visible comme un résultat
d'ablation conceptuelle, pas comme une omission.

### Modélisation incomplète

- seule la médiane est utilisée comme baseline ; la moyenne et un modèle
  linéaire simple manquent ;
- le tuning historique a gagné environ 0,000066 de MAE sur un ancien contrat,
  mais aucun tuning proportionné n'est traçable sur le contrat courant ;
- l'analyse d'overfitting se résume à une étiquette calculée sur l'écart de
  MAE ; les folds et la variance doivent être visibles ;
- l'analyse par taille de match est trop étroite ;
- il n'existe ni permutation importance finale, ni SHAP, ni PDP. Une
  permutation importance sur holdout suffit ici si elle est stable et reliée à
  l'EDA ;
- les plus grandes erreurs et les biais aux extrêmes de la cible ne sont pas
  documentés.

### Inférence et exploitation

Le bundle peut être chargé, mais le projet ne fournit pas la fonction demandée
`predict_from_csv(input_path, model_path, output_path)`. Le script principal
réentraîne avant de prédire ; il ne démontre donc pas une inférence indépendante
depuis un artefact déjà sauvegardé. Les contrôles de colonnes manquantes,
d'ordre, de types, de bornes et d'export doivent être testés dans ce chemin.

### Logging

Le package n'a pas de configuration centralisée et le script utilise `print()`
pour les événements du pipeline. Ce choix était acceptable pour un prototype,
mais il ne répond pas à la demande de logging cohérent. Un logger standard,
configuré une fois et sans fichier implicite, suffit.

### Documentation redondante

`docs/report/`, `docs/analysis/`, `docs/refactoring/` et `docs/final_report.md`
répètent plusieurs conclusions. Les rapports historiques sont utiles comme
preuves et restent dans l'archive ZIP, mais ne doivent pas redevenir une seconde
source de vérité. La version finale doit conserver :

- les trois documents de revue ;
- trois synthèses analytiques ;
- les décisions de feature engineering ;
- les résultats de validation/modélisation ;
- un rapport final ;
- le notebook comme narration principale.

## Réponse directe aux questions de revue

| Question | Réponse honnête avant consolidation | Action |
|---|---|---|
| Parties trop superficielles ? | EDA, tuning, interprétabilité, erreurs, inférence | REWRITE / ADD |
| Analyses insuffisamment visualisées ? | qualité, structure match/équipe, relations, splits, overfit, sous-groupes | ADD |
| Décisions sans justification chiffrée ? | sélection progressive des features, rejet du tuning courant, choix du split temporel | VERIFY |
| Graphiques essentiels absents ? | oui, au moins six familles de preuves | ADD |
| Conclusions répétées ? | mobilité, `killRank`, date et rosters dans plus de dix documents | MERGE |
| Sections notebook trop longues ? | aucune ; le notebook est plutôt trop condensé | IMPROVE |
| Sections notebook trop courtes ? | EDA, tuning, interprétabilité, erreurs, inférence | REWRITE |
| Fonctions dupliquées ? | peu dans le package courant ; duplication surtout documentaire et entre notebook/script | MERGE |
| Modélisation complète ? | non, baselines/tuning/diagnostics incomplets | IMPROVE |
| Tuning rigoureux ? | non sur le contrat courant | VERIFY |
| Analyse des erreurs présente ? | seulement par taille déclarée du match | REWRITE |
| Inférence réellement testée ? | chargement du bundle testé, CSV brut non testé | IMPROVE |
| Logging cohérent ? | non | ADD |
| Reproductibilité ? | bonne pour le benchmark, incomplète pour les analyses finales | IMPROVE |
| Conformité complète à l'énoncé ? | structure minimale oui, profondeur et visualisation non | PARTIAL |

## Priorités finales

1. Corriger la lecture de la date et comparer quatre stratégies de split.
2. Régénérer un petit ensemble de visualisations décisionnelles.
3. Mesurer l'ablation progressive des familles de features.
4. Ajouter baselines, tuning proportionné, overfitting et diagnostics d'erreur.
5. Ajouter importance par permutation et inférence depuis CSV brut.
6. Centraliser le logging et tester les risques critiques.
7. Réécrire le notebook et consolider les documents redondants.

Le critère de sortie n'est pas le nombre de nouveaux artefacts. Chaque élément
final devra justifier une décision scientifique, analytique ou technique.
