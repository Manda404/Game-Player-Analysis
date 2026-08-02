# Rationale des features

## Contrat comportemental (15)

- brutes : `walkDist`, `rideDist`, `damages`, `kills`, `weapons`, `upgrades`,
  `heals`, `maxRank` ;
- dérivées : marche par minute de match, dégâts/kill, activité combat, activité
  ressources ;
- contexte : indicateurs solo, duo et squad ; les modes spéciaux sont la
  référence.

## Contrat post-match (16)

Le scénario final ajoute uniquement `killRank`. Cette séparation rend visible
son effet et empêche de qualifier le scénario sans `killRank` d'early-game :
les autres statistiques sont elles aussi observées après la partie.

## Variables exclues

- IDs et cible : fuite ou non numériques ;
- `date` : incohérente au sein des matchs ;
- `killPts`, `rankPts`, `winPts` : systèmes hétérogènes et sentinelles ;
- agrégats équipe/lobby : couverture trop partielle ;
- KPI redondants/clairsemés : valeur analytique, pas de gain défendable.

Les formules sont centralisées dans `features.py`; le train, le test et
l'inférence appellent exactement la même fonction.
