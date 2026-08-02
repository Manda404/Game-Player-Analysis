# Justification du feature engineering

## Contrats

- `behavior_without_killRank` : 15 variables ;
- `post_match_with_killRank` : les mêmes 15 variables plus `killRank`.

La target, les identifiants, `date`, les scores externes, les segments et les
agrégats de groupe sont interdits. Le train et le test passent par la même
fonction déterministe `build_model_features`.

| Feature | Formule / source | Niveau | Insight | Intérêt | Risque et traitement |
|---|---|---|---|---|---|
| `killRank` | valeur officielle | joueur post-match | Rang d'élimination très lié à la cible | Très fort | Structurel; scénario séparé et ablation |
| `walkDist` | brut | joueur | Mobilité dominante | Très fort | Accumulé pendant la partie |
| `rideDist` | brut | joueur | Rotation distincte de la marche | Moyen | Longue queue conservée |
| `damages` | brut | joueur | Engagement même sans kill | Fort | 53 cas kill sans dégâts conservés |
| `kills` | brut | joueur | Résultat du combat | Fort | Zéro légitime |
| `weapons` | brut | joueur | Collecte/progression | Fort | Accumulé avec le temps de jeu |
| `upgrades` | brut | joueur | Progression d'équipement | Fort | Définition métier à confirmer finement |
| `heals` | brut | joueur | Continuité d'activité/survie | Moyen à fort | Zéro légitime |
| `maxRank` | brut | match | Support et granularité de la cible | Contexte | N'est pas corrigé par `numTeams` |
| `walk_distance_per_match_minute` | `walkDist/(gameTime/60)` | joueur+match | Compare l'activité entre durées de match | Fort historiquement | N'est pas un rythme de survie individuel; zéro si durée nulle |
| `damage_per_kill` | `damages/kills` | joueur | Dégâts versus conversion | Interaction lisible | Zéro si aucun kill; `damages` brut reste présent |
| `combat_activity` | `kills+assists+knocks` | joueur | Volume d'engagement | Compaction | Composantes hétérogènes, pas un KPI causal |
| `resource_activity` | `weapons+upgrades+heals` | joueur | Collecte/progression | Compaction | Pas de normalisation par faux temps individuel |
| `mode_solo` | famille de `gameType` | match | Structure principale du mode | Contexte | Les modes spéciaux sont la référence |
| `mode_duo` | famille de `gameType` | match | Structure principale du mode | Contexte | Binaire, identique train/test |
| `mode_squad` | famille de `gameType` | match | Structure principale du mode | Contexte | Binaire, identique train/test |

## Features écartées

| Feature ou famille | Motif |
|---|---|
| `killRankPercentile` | Transformation redondante de `killRank` et ancien contrat incompatible |
| `highestKill` | Signal secondaire, très asymétrique et sémantiquement trompeur |
| `killStreaks` | Très redondant avec `kills` |
| `estimatedTeamSize` | Proxy théorique faible, pas taille observée réelle |
| `lootActivityPerMinute` | Dénominateur global du match présenté comme rythme joueur |
| `rankPts`, `killPts`, `winPts` | Systèmes externes dépréciés et sentinelles fréquentes |
| agrégats équipe/match | Roster trop incomplet et risque de fuite |
| date/calendrier | Pseudo-date non liée au match réel |

## Résultat de l'ablation

| Scénario | Features | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Post-match avec `killRank` | 16 | 0,06156 | 0,08667 | 0,92053 |
| Comportement sans `killRank` | 15 | 0,09335 | 0,13044 | 0,81996 |
| Post-match projeté sur la grille | 16 | 0,06111 | 0,08743 | 0,91913 |

L'ingénierie sert ici la lisibilité et la compaction; elle n'est pas présentée
comme une preuve causale ni comme un gain universel face à toutes les variables
brutes.
