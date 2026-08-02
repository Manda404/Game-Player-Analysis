# Contrat des données officielles

Cette synthèse reprend le PDF du test technique fourni séparément. Le PDF n'est
pas redistribué dans le dépôt. Une ligne représente les statistiques d'un joueur
après une partie ; la cible est le classement normalisé de son équipe.

| Champ | Sens utilisé dans le projet |
|---|---|
| `playerId`, `teamId`, `gameId` | identifiants joueur, équipe et partie |
| `assists`, `knocks`, `revives` | assistance, mise à terre et réanimation |
| `kills`, `headshots`, `killStreaks` | éliminations et indicateurs associés |
| `damages`, `highestKill` | dégâts et plus longue élimination |
| `walkDist`, `rideDist`, `swimDist` | distances par type de déplacement |
| `roadKills`, `teamKills` | kills en véhicule et alliés tués |
| `vehicleDestr` | véhicules détruits |
| `heals`, `weapons`, `upgrades` | soins, armes acquises et améliorations |
| `gameTime`, `gameType` | durée globale et type de partie |
| `maxRank`, `numTeams` | nombre maximal de rangs et groupes déclarés |
| `killRank` | rang final du joueur selon les kills |
| `killPts`, `rankPts`, `winPts` | scores issus de systèmes de ranking |
| `date` | officiellement date de la partie ; incohérente dans les CSV |
| `winRankPercentage` | cible équipe normalisée, absente du test |

Sentinelles officielles : `rankPts=-1` est indisponible ; lorsque `rankPts` est
disponible, les zéros de `killPts` et `winPts` signalent l'indisponibilité.
`maxRank` définit la grille légale de cible, pas le nombre de joueurs observés.
