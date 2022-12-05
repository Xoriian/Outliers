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
def get_data_split():
    file = dossier + "train_data.csv"
    with open(file, encoding="utf-8") as f:
        lines = f.readlines()
    data = lines[1:]
    for i, string in enumerate(data):
        _, data[i] = string.split(',', 1)
    limit = int(len(data) * 0.8) + 1 # Séparation entre les données de train et de validation (80% - 20%)

    return data[:limit], data[limit:]

def get_data():
    file = dossier + "train_data.csv"
    with open(file, encoding="utf-8") as f:
        lines = f.readlines()
    data = lines[1:]
    for i, string in enumerate(data):
        _, data[i] = string.split(',', 1)

    return data

# Récupération des données de test
def get_test():
    file = dossier + "test_data.csv"
    with open(file, encoding="utf-8") as f:
        lines = f.readlines()
    data = lines[1:]
    for i, string in enumerate(data):
        _, data[i] = string.split(',', 1)

    return data

# Récupération des labels des données de train et de validation
def get_labels_split():
    labels = np.loadtxt(dossier + "train_results.csv", delimiter=",", dtype=str)
    labels = labels[1:]
    labels[labels[: , 1] == 'negative', 1] = 0
    labels[labels[: , 1] == 'neutral', 1] = 1
    labels[labels[: , 1] == 'positive', 1] = 2
    limit = int(len(labels) * 0.8) + 1 # Séparation entre les données de train et de validation (80% - 20%)

    return labels[:limit].astype(int), labels[limit:].astype(int)

def get_labels():
    labels = np.loadtxt(dossier + "train_results.csv", delimiter=",", dtype=str)
    labels = labels[1:]
    labels[labels[: , 1] == 'negative', 1] = 0
    labels[labels[: , 1] == 'neutral', 1] = 1
    labels[labels[: , 1] == 'positive', 1] = 2

    return labels.astype(int)

# Fusionner les phrases et les labels
def merge_data_labels(data, labels):
    merged = []
    for i in range(len(data)):
        merged.append((data[i], labels[i][1]))

    return merged

# %% [markdown]
# ### Fonction de transformation des labels en format "One Hot"

# %%
# Permet de transformer un label "y" en label "one-hot" : labels[y] = 1 et labels[z != y] = 0
def to_one_hot(labels, nb_classes):
    one_hot_labels = np.zeros((len(labels), nb_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i][label] = 1
        
    return one_hot_labels

# %% [markdown]
# ### Fonction d'enregistrement des résultats

# %%
# Permet d'enregistrer les résultats sous le format voulu par Kaggle
def save_results(predictions, name):
    results = np.empty((len(predictions), 2), dtype = int)

    # Enregistrement des résultats
    for i, prediction in enumerate(predictions):
        results[i][0] = i
        results[i][1] = prediction

    np.savetxt(dossier + name + '.csv', results, fmt='%d', delimiter=',', header='id,target', comments='')

