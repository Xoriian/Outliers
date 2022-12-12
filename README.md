# Classification de texte - Compétition 2 - IFT 6390A (A2022)
# Equipe *Outliers* (Maylis Heussner - Khadidja Yasmine Bourega - Alexis Raffier)
## 1. Contexte
Ce code contient différents algorithmes de *Machine Learning* pour la classification de texte. Le but est de reconnaître l'humeur du texte, entre "négatif", "neutre", et "positif".

## 2. Algorithmes utilisés
4 algorithmes différents ont été utilisés. Chaque algorithme possède son propre code (.py ou .ipynb) contenant tout ce qui est nécessaire à son exécution. Les 4 algorithmes sont :
- Classifieur de Bayes (avec sac de mots)
- SVM (avec noyaux de phrases)
- RNN (en particulier un GRU)
- Transformer

## 3. Fonctionnement du code
### 3.a Exécution des fichiers .py
Pour faire fonctionner le code, il faut tout d'abord s'assurer que le fichier .py contenant les fonctions utilitaires, *utils_data.py* est bien inclus dans le même dossier (ou dans l'environnement d'exécution pour Google Colab). Il suffit ensuite de lancer toutes les cellules du code souhaité.

### 3.b Explicabilité (modèle de RNN seulement)
Le code associé à l'explicabilité (à travers les matrices de confusion et l'algorithme *LIME*) est directement inclus dans le fichier .py du RNN, à la fin du fichier *RNN.py*. Il suffit donc de lancer toutes les cellules pour exécuter le code associé à l'explicabilité. Deux librairies seront téléchargées pour cela (*lime* et *scikit-plot*) au début du fichier.

**Pour les algorithmes 3 et 4** : il est fortement recommandé de lancer le code sur *Google Colab* avec un run GPU, sinon l'exécution sera (trop) lente. L'exécution avec *Google Colab* est déjà gérée dans le code, prenez juste soin de bien mettre le jeu de données dans un dossier de votre *Drive*.

## 4. Hyperparamètres des algorithmes
Les hyperparamètres choisis par défaut sont ceux ayant donnés les meilleurs résultats lors des tests. Pour modifier un hyperparamètre, il faut le changer directement dans le code (dans la cellule correspondante)

## 5. Références
Nous avons utilisé les sources suivantes pour nos implémentations : 
- Code inspiré de https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-rnn-for-text-classification-tasks pour le GRU et l'explicabilité.
- Code inspiré de https://n8henrie.com/2021/08/writing-a-transformer-classifier-in-pytorch/ pour le Transformer.
