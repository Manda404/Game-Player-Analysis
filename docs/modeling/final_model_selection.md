# Sélection finale du modèle

## Décision

Le modèle publié est **CatBoost avec ses hyperparamètres d'apprentissage par
défaut**. La décision est gelée après comparaison et tuning sur le développement,
puis évaluée une seule fois sur le holdout groupé du cycle d'audit.

| Évaluation finale | Valeur |
|---|---:|
| développement / holdout | 40 128 / 9 872 lignes |
| `gameId` partagés | 0 |
| MAE train | 0,055770 |
| MAE holdout | **0,060797** |
| RMSE holdout | 0,086596 |
| R² holdout | 0,920827 |
| écart train-holdout | 0,005027 |

Ce holdout est indépendant des décisions de ce cycle, mais pas historiquement
vierge puisque l'EDA antérieure avait vu toutes les lignes. Le modèle de
publication est ensuite réentraîné sur les 50 000 lignes et produit uniquement
des prédictions pour le test officiel sans cible.

## Réponses explicites aux 15 questions de l'audit

1. **Paramètres par défaut ?** Non avant l'audit ; oui désormais pour tous les
   hyperparamètres d'apprentissage de la comparaison initiale.
2. **Paramètres non par défaut ?** Seuls seed, parallélisme, silence, objectif ou
   loss et interdiction des fichiers CatBoost sont conservés pour des raisons
   techniques. Les anciens réglages d'arbres, profondeur, learning rate,
   régularisation et subsampling ont été retirés de la comparaison initiale.
3. **Comparaison équitable ?** Non auparavant ; oui après correction, avec
   mêmes lignes, features, folds, cible et métriques.
4. **Choix avant tuning ?** Oui dans le pipeline corrigé : CatBoost est choisi
   sur la comparaison initiale.
5. **Tuning réservé au gagnant ?** Oui, seul CatBoost est optimisé.
6. **Validation surexploitée ?** L'historique avait réutilisé le corpus. Le
   cycle corrigé gèle un holdout avant sélection, avec la limite historique
   explicitement documentée.
7. **Pourquoi XGBoost était-il meilleur ?** À cause de configurations initiales
   personnalisées : elles amélioraient XGBoost et dégradaient CatBoost. Son
   avantage résiduel n'était que de 0,000054 MAE.
8. **Différence stable ?** Avec les défauts équitables, CatBoost bat XGBoost
   dans les cinq folds ; différence moyenne 0,002000, intervalle descriptif
   [0,001829 ; 0,002192]. La pseudo-temporalité ne peut pas trancher car la date
   est incohérente au sein des `gameId`.
9. **Ancienne conclusion CatBoost incorrecte ?** Pas directionnellement, mais
   son ancien score utilise un autre contrat de features et n'est pas une preuve
   directe pour la version actuelle.
10. **Nouvelle conclusion XGBoost robuste ?** Non ; elle reposait sur une
    comparaison initiale inéquitable et disparaît après correction.
11. **Métrique principale ?** MAE de validation ; RMSE, R², stabilité et écart
    train-validation sont secondaires.
12. **R² ajusté utile ?** Seulement descriptif pour des modèles linéaires
    emboîtés ; pas comme règle de sélection pour les ensembles d'arbres.
13. **Validation d'une feature ?** Ablation sur folds groupés, variation de MAE,
    stabilité, permutation et sous-groupes.
14. **Outils sklearn ?** Oui : `GroupKFold`, `GroupShuffleSplit`,
    `cross_validate` et `RandomizedSearchCV` standardisent le protocole. Les
    boucles manuelles restent légitimes pour comparer des stratégies de split
    différentes ou produire des prédictions OOF détaillées.
15. **Stratégie finale ?** Dummy, Ridge, candidats par défaut sur développement
    groupé, ablation, sélection de CatBoost, recherche aléatoire rejetée,
    holdout groupé ouvert une fois, refit complet, puis inférence test.

La décision machine-readable est dans
[`final_selection_decision.json`](../../artifacts/metadata/final_selection_decision.json).
