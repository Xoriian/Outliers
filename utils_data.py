# %%
import numpy as np
import sys

# Constantes sur le nombre de classes cible
NB_CLASSES = 3 # (de 0 à 2)

# Variable pour l'utilisation du code dans Google Colab - Dans ce cas le jeu de données doit être dans le Drive !
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    dossier = '/content/drive/My Drive/data/'
else:
    dossier = 'data/'

# %% [markdown]
# ### Récupérer les données

# %%
# Récupération des données de train et de validation
def get_data():
    file = dossier + "train_data.csv"
    with open(file, encoding="utf-8") as f:
        lines = f.readlines()
    data = lines[1:]
    for i, string in enumerate(data):
        _, data[i] = string.split(',', 1)
    limit = int(len(data) * 0.8) + 1 # Séparation entre les données de train et de validation (80% - 20%)

    return data[:limit], data[limit:]

# Récupération des données de test
def get_test():
    file = dossier + "train_data.csv"
    with open(file, encoding="utf-8") as f:
        lines = f.readlines()
    data = lines[1:]
    for i, string in enumerate(data):
        _, data[i] = string.split(',', 1)

    return data

# Récupération des labels des données de train et de validation
def get_labels():
    labels = np.loadtxt(dossier + "train_results.csv", delimiter=",", dtype=str)
    labels = labels[1:]
    labels[labels[: , 1] == 'negative', 1] = 0
    labels[labels[: , 1] == 'neutral', 1] = 1
    labels[labels[: , 1] == 'positive', 1] = 2
    limit = int(len(labels) * 0.8) + 1 # Séparation entre les données de train et de validation (80% - 20%)

    return labels[:limit].astype(int), labels[limit:].astype(int)

# Affichage d'une image
def display_data(data, labels, k):
    print(f"{k} - Phrase : {data[k]} | Label : {labels[k]}")
 

# %%
# Récupération des données
train, val = get_data()
test = get_test()
labels_train, labels_val = get_labels()

# Test et affichage d'une donnée
#nombre = int(np.random.uniform(0, len(train), 1))
#display_data(train, labels_train, nombre)

# %% [markdown]
# ### Fonction de transformation des labels en format "One Hot"

# %%
# Permet de transformer un label "y" en label "one-hot" : labels[y] = 1 et labels[z != y] = 0
def to_one_hot(labels, nb_classes):
    one_hot_labels = np.zeros((labels.shape[0], nb_classes))
    for i, label in enumerate(labels[:, 1]):
        one_hot_labels[i][label] = 1
        
    return one_hot_labels

one_hot_train = to_one_hot(labels_train, NB_CLASSES)
one_hot_val = to_one_hot(labels_val, NB_CLASSES)

#print(one_hot_train[:10])


