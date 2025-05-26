
![banner](img/banner.png)

# Transfer Learning : Détection de pneumonie à partir de modèles pré-entrainés type CNN

*Projet réalisé dans un contexte pédagogique durant la formation Développeur en Intelligence Artificielle, chez Simplon Hauts-de-France.*

## Table of Contents
1. [Description du projet](#description-du-projet)
2. [Outils utilisés](#outils-utilisés)
3. [Installation](#installation)
4. [MLflow](#mlflow)
5. [Analyse du projet](#analyse-du-projet)


## Description du projet
Projet de mise en place d'un système de classification binaire permettant de détecter des cas de pneumonie à partir de radios thoraciques, via utilisation d'un *modèle de vision par ordinateur préentraîné* type *CNN*. Utilisation du dataset *[Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)*.

Appréhension d'une démarche de suivi des expérimentations par *MLflow*, afin de tracer les paramètres d’entraînement, les métriques de performance et les versions de modèle.

**Contexte du projet** : Une équipe médicale souhaite tester l’apport de l’intelligence artificielle dans le diagnostic automatisé de la pneumonie à partir de radios thoraciques. Elle a besoin d’un prototype fonctionnel (Proof of Concept) permettant de démontrer la faisabilité d’un système de classification binaire d’images médicales.


## Outils utilisés
Liste des outils et modules utilisés pour le projet :
* [Python](https://www.python.org/downloads/release/python-31010/) : Version 3.10.10 
* Numpy
* Matplotlib
* Opencv-python
* Tensorflow : Version 2.19.0
* Scikit-learn
* MLflow

  
## Installation
Pour utiliser ce projet, téléchargez le .zip [ici](https://github.com/Aurelien-L/CNN_Transfer_Learning/archive/refs/heads/main.zip), ou clonez le projet sur votre ordinateur avec la commande suivante : 
```
$ git clone https://github.com/Aurelien-L/CNN_Transfer_Learning.git
```
>[!WARNING]
>Assurez-vous d'avoir Python 3.10.10 d'installé sur votre machine, et installez les dépendances nécessaires avec cette commande :
```
pip install -r requirements.txt
```


## MLflow
Pour pouvoir accéder au serveur local MLflow, utilisez la commande suivante dans le terminal.\

>[!WARNING]
>Attention, veillez à bien exécuter MLflow **avant** de lancer l'exécution du Jupyter NoteBook, sinon la cellule dédiée amènera à une erreur.

```
mlflow ui
```
Le serveur sera disponible à l'adresse suivante : *http://127.0.0.1:5000/*


## Analyse du projet

Suite à comparatif de différents modèles pré-entraîné, les premiers essais ont été réalisé avec *DenseNet121*. Ce dernier est particulièremet adapté au cas de ce projet, étant très utilisé dans le milieu médical pour la détection de maladie pulmonaire. Il demande cependant beaucoup de puissance de calcul, et donc de temps d'exécution, ce qui m'empêchait de travailler efficacement.  

J'ai donc décidé de travailler avec un autre modèle plus léger, bien que moins adapté, par soucis pédagogique, dans le but de me concentrer principalement sur l'exécution globale et la démarche de travail, plus que sur les résultats du modèle.


**description test MobileNetV2 et V3**


Les résultats des différents modèles utilisés et des différentes simulations sont visible grâce à l'historique MLflow.