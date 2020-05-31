# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data"
FIGURES_DIR = "figures"

# %%
# os.chdir("lab5")
os.listdir(DATA_DIR)
# MDS()

# %%
cars_df = pd.read_csv(os.path.join(DATA_DIR, "cars.csv"), header=None, index_col=0)

# %% pycharm={"name": "#%%\n"}
cars_df.head()

# %%

swiss_roll_df = pd.read_csv(os.path.join(DATA_DIR, "swissroll.dat"), header=None, delim_whitespace=True)

# %%
# Mnist dataset load

mnist_X, mnist_y = fetch_openml('mnist_784', version=1, return_X_y=True)

# %%
mnist_colors = np.array([float(label) for label in mnist_y])
mnist_colors /= np.max(mnist_colors)

indxes = np.random.choice(range(len(mnist_X)), 1000, replace=False)

mnist_X_sampled = mnist_X[indxes]
mnist_colors_sampled = mnist_colors[indxes]

# %% pycharm={"name": "#%%\n"}
def plot_reduced(ax, X_transformed, annot_labels=None, colors=None):
    x, y = X_transformed.T
    ax.scatter(x, y, c=colors)
    if annot_labels is not None:
        for i, label in enumerate(annot_labels):
            ax.annotate(label, (x[i], y[i]))

methods = [
    (MDS(n_components=2), "MDS sklearn, SMACOF"),
    (PCA(n_components=2), "MDS classic, PCA"),
    (Isomap(n_components=2, n_neighbors=3), "Isomap(k=3)"),
    (Isomap(n_components=2, n_neighbors=5), "Isomap(k=5)"),
    (TSNE(n_components=2), "TSNE"),
    (LocallyLinearEmbedding(n_components=2), "LLE"),
]

data = [
    (cars_df, cars_df.index, "Cars", None),
    (swiss_roll_df, None, "SwissRoll", None),
    (mnist_X_sampled, None, "Mnist", mnist_colors_sampled)
]


# %%

for dataframe, annot, dataset_name, color in data:
    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(18.5, 10.5))
    for i, (method, title) in enumerate(methods):
        ax = axs.reshape(-1)[i]
        ax.set_title(title)
        X_transformed = method.fit_transform(dataframe)
        plot_reduced(ax, X_transformed, annot, color)
    plt.savefig(os.path.join(FIGURES_DIR, dataset_name) + ".pdf")
    plt.show()

# %%
for dataframe, annot, dataset_name, color in data[-1:]:
    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(18.5, 10.5))
    for i, (method, title) in enumerate(methods):
        ax = axs.reshape(-1)[i]
        ax.set_title(title)
        X_transformed = method.fit_transform(dataframe)
        plot_reduced(ax, X_transformed, annot, color)
    plt.savefig(os.path.join(FIGURES_DIR, dataset_name) + ".pdf")
    plt.show()
