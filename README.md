# Analyse de sentiment d’avis de films (NLP)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Deployed-success)

Application de **traitement du langage naturel (NLP)** permettant de prédire le **sentiment (positif / négatif)** d’un avis de film en français, à l’aide d’un modèle **TF-IDF + Régression Logistique**, déployé avec **Streamlit**.

---

## Démo en ligne

 **Accéder à l’application Streamlit** :  
https://nlp-analyse-sentiment-films-w5ccbwfrfdx3nolmuajeoy.streamlit.app/

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nlp-analyse-sentiment-films-w5ccbwfrfdx3nolmuajeoy.streamlit.app/)

---

## Objectif du projet

- Comprendre et explorer un dataset d’avis textuels
- Construire un pipeline NLP complet
- Entraîner un modèle de classification supervisée
- Interpréter les prédictions (mots influents)
- Déployer une application web interactive

---

## Dataset

- Avis de films en français
- Colonnes principales :
  - `text` : texte de l’avis
  - `sentiment` : label (0 = négatif, 1 = positif)
- Environ **160 000 avis**

---

## Exploration des données (EDA)

Réalisée dans le notebook `01_eda.ipynb` :
- Analyse de la distribution des sentiments
- Étude de la longueur des avis
- Vérification de la qualité du dataset
- Nettoyage de base et sauvegarde du dataset prêt pour le ML

---

## Pipeline NLP

1. Nettoyage du texte (lowercase, ponctuation, caractères spéciaux)
2. Vectorisation **TF-IDF** (unigrammes + bigrammes)
3. Séparation train / test avec `stratify=y`
4. Modèle baseline : **Régression Logistique**
5. Évaluation avec précision, recall, F1-score et matrice de confusion

**Performance obtenue** :
- Accuracy ≈ **93 %**
- F1-score ≈ **0.93**

---

## Interprétabilité

L’application affiche :
- La prédiction du sentiment
- La probabilité associée
- Les **mots les plus influents** dans la décision du modèle  
  (pondération TF-IDF × coefficients du modèle)

---

## Application Streamlit

Fonctionnalités :
- Saisie libre d’un avis
- Bouton **Analyser**
- Bouton **Reset**
- Affichage du sentiment avec style visuel
- Affichage des mots importants

---

## Structure du projet

NLP_avis_buches_noel/
├── app.py
├── notebooks/
│ ├── 01_eda.ipynb
│ └── 02_nlp_preprocessing.ipynb
├── data/
│ ├── raw/
│ └── processed/
├── models/
│ ├── logistic_regression_tfidf.pkl
│ └── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md


---

## Lancer le projet en local

### 1️ - Installer les dépendances
```bash
pip install -r requirements.txt

### 2 - Lancer l’application
streamlit run app.py

---

## Déploiement

Déployé avec Streamlit Cloud

Code versionné sur GitHub

Modèle et vectorizer chargés via joblib

---

## Technologies utilisées

Python

pandas / numpy

scikit-learn

Streamlit

joblib

Git & GitHub

---

## Auteur

Projet réalisé par Samuel M
📌 Dans le cadre d’un projet personnel NLP / Data Science.

---

## Améliorations possibles

Ajout d’une classe neutre

Utilisation de modèles deep learning (CamemBERT)

Ajout d’un mode batch (CSV)

Dockerisation de l’application

...