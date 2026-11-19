# Mini-Projet : Caractérisation d'une couche métallique

Modélisation de la propagation d'une onde électromagnétique dans une structure Air/Métal/Air et estimation de l'épaisseur métallique par les coefficients de Fresnel.

## 📋 Description

Ce projet implémente un modèle numérique (Python) pour calculer les coefficients de réflexion (R), de transmission (T) et d'absorption (A) d'une couche métallique mince. Il permet d'étudier l'influence de la fréquence, de l'épaisseur et du type de métal (Cuivre, Aluminium, Acier).

Le point clé est l'**Étude 5** : elle simule une mesure bruitée pour estimer l'épaisseur inconnue d'une couche via une méthode d'optimisation.

## 📦 Contenu du dépôt

- `modele.py` : Moteur physique (calcul de Gamma, T, R, A).
- `validation.py` : Tests de validation du modèle (conservation de l'énergie, limites asymptotiques).
- `etudes.py` : Les 6 études paramétriques (génération des figures).
- `schema_situation.py` : Génération du schéma de la structure.
- `rapport.pdf` : Le rapport complet du mini-projet.

## 🛠️ Installation et Utilisation

1. Clone le dépôt :
   ```bash
   git clone https://github.com/TON_NOM_UTILISATEUR/mini-projet-propagation.git
   cd mini-projet-propagation
