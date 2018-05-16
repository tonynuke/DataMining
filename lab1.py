import pandas as pd
import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n_clusters = 4

data_set_path = 'character-deaths.csv'
data_set = pd.read_csv(data_set_path, sep=',')

columns_of_interest = ['Allegiances']
two_columns_of_data = data_set[columns_of_interest]

# выбираем то чта нам интересно для анализа
columns_of_interest = ['Death Year', 'Book of Death', 'Death Chapter', 'Gender']
data_model = data_set[columns_of_interest]

data_model = pd.concat([data_model, pd.get_dummies(data_set['Allegiances'])], axis=1)

# оставим только погибших персонажей
data_model = data_model[np.isfinite(data_model['Death Year'])]
data_model = data_model[np.isfinite(data_model['Book of Death'])]
data_model = data_model[np.isfinite(data_model['Death Chapter'])]

# значения атрибутов в пространтсве
X = data_model.as_matrix()

alldata = X.T

# Regenerate fuzzy model with 3 cluster centers - note that center ordering
# is random in this clustering algorithm, so the centers may change places
cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
    alldata, n_clusters, 2, error=0.005, maxiter=1000)

# Show n-cluster model
fig = plt.figure()
ax2 = Axes3D(fig)
ax2.set_title('Trained model')
for j in range(n_clusters):
    ax2.plot(alldata[0, u_orig.argmax(axis=0) == j],
             alldata[1, u_orig.argmax(axis=0) == j],
             alldata[2, u_orig.argmax(axis=0) == j],
             'o',
             label='series ' + str(j))
    ax2.plot(cntr[:, 0], cntr[:, 2], cntr[:, 3], 'o', c='black')
plt.xlabel('Death Year')
plt.ylabel('Book of Death')
plt.clabel('Death Chapter')
ax2.legend()

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    alldata, cntr, 2, error=0.005, maxiter=1000)

# Plot the classified uniform data. Note for visualization the maximum
# membership value has been taken at each point (i.e. these are hardened,
# not fuzzy results visualized) but the full fuzzy result is the output
# from cmeans_predict.
cluster_membership = np.argmax(u, axis=0)  # Hardening for visualization

fig3, ax3 = plt.subplots()
ax3.set_title('2d Trained model')
for j in range(n_clusters):
    ax3.plot(alldata[0, u_orig.argmax(axis=0) == j],
             alldata[2, u_orig.argmax(axis=0) == j],
             'o',
             label='series ' + str(j))
    ax3.plot(cntr[:, 0], cntr[:, 2], 'o', c='black')
plt.xlabel('Death Year')
plt.ylabel('Death Chapter')
ax3.legend()

# fig3, ax3 = plt.subplots()
# ax3.set_title('Random points classifed according to known centers')
# for j in range(n_clusters):
#     ax3.plot(alldata[0, cluster_membership == j],
#              alldata[3, cluster_membership == j], 'o',
#              label='series ' + str(j))
#     ax3.plot(cntr[:, 0], cntr[:, 3], 'o', c='black')
# ax3.legend()

plt.show()