# Insights consolidés

| Fait établi | Preuve | Décision | Limite |
|---|---|---|---|
| La cible appartient à `[0,1]` et suit une grille par `maxRank` | 1 445 valeurs, formule de rang quasi exacte | Régression + MAE; projection de grille évaluée | Le test n'est pas étiqueté |
| `walkDist` domine le signal comportemental | Spearman historique ≈0,866 | Conserver valeur brute et rythme par minute de match | Corrélation, pas causalité |
| Loot/soins et combat complètent la mobilité | Corrélations, profils top/bottom, ablations | Conserver familles séparées et deux composites | Variables accumulées pendant la partie |
| `killRank` porte un signal final majeur | MAE 0,06156 avec, 0,09335 sans | Scénario post-match explicite + ablation obligatoire | Interdit pour un produit early-game |
| Les matchs sont partiellement observés | 1,614 ligne/match, maximum 8 | Pas d'agrégat lobby | Impossible de reconstruire les adversaires |
| Les équipes visibles sont presque toutes singletons | 98,824 % | Pas d'agrégat coéquipier | Seulement 2,340 % des lignes ont un pair |
| La date n'est pas un timestamp de match | Un même match traverse plusieurs dates | Exclure calendrier et validation temporelle | Le test reste un fichier externe |
| Un split ligne fuit les matchs | 5 100 matchs communs, 56,5 % des lignes validation vues | `GroupKFold(gameId)` à cinq folds | IDs mal formés isolés par ligne |
| Train/test sont proches sur les variables principales | écart moyen standardisé maximal 0,0235 | Pas de correction de dérive | Pas de garantie après mai ou changement produit |
| Les cibles moyennes sont proches entre solo, duo et squad | 0,461 à 0,486 ; `maxRank` moyen très différent | Garder trois indicateurs de contexte | Les taux de victoire bruts sont mécaniques à la grille |
| Les extrêmes sont souvent des comportements légitimes | zero inflation et longues queues | Ne supprimer ni clipper par défaut | Alertes télémétrie conservées |

## KPI utiles

- cible moyenne : 0,47233 ;
- taux de victoire historique : 2,86 % ;
- joueurs avec au moins un kill : 43,034 % ;
- distance à pied moyenne : 1 155,22 ;
- anomalies à surveiller : 53 kills sans dégâts et 102 combats sans distance ;
- contrôle ML bloquant : zéro `gameId` partagé entre apprentissage et validation ;
- référence finale : MAE groupée 0,06156 avant projection, 0,06111 après.

Les KPI longitudinaux ou financiers restent hors périmètre faute d'identité
joueur stable, sessions et transactions.
