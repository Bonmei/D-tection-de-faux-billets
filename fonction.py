import pandas as pd # (version pandas 1.4.2)
import numpy as np  # (version numpy 1.22.4)
import seaborn as sns # (version seaborn 0.11.2)
import matplotlib.pyplot as plt



    # Détection clé primaire

def testerCle(df, colonnes):
    if df.size == df.drop_duplicates(colonnes).size :
        print("La clé n'est pas présente plusieurs fois dans le dataframe.")
        print("Elle peut être utilisée comme clé primaire.".format(colonnes))
    else :
        print("La clé est présente plusieurs fois dans le dataframe.")
        print("Elle ne peut pas être utilisée comme clé primaire.".format(colonnes))
        
    print("Le dataframe est de la forme : " + str(df.shape) + " (lignes, colonnes)")


    # Détection des outliers avec la méthode interquantile

def detection_outliers_interquantile(data, col):
    
    Q3 = np.quantile(data[col], 0.75)
    Q1 = np.quantile(data[col], 0.25)

    IQR = Q3 - Q1
    
    print("La valeur de Q1 pour le '%s' est de %f." % (col,Q1)) 
    
    print("La valeur de Q3 pour le '%s' est de %f." % (col,Q3))
    
    print("La valeur de l'IQR pour le '%s' est de %s." % (col, IQR))
   
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR
    
    outliers = [x for x in data[col] if (
        (x < borne_inf) | (x > borne_sup))]
    
    notoutliers=[x for x in data[col] if (
        (x > borne_inf) & (x < borne_sup))]
    
    filtre_data = data.loc[data[col].isin(notoutliers)]
    
    print('Les outliers sont :' ,outliers)
    
    print('Dimensions du dataframe sans les outliers :',filtre_data.shape)
   
    print('Le nombre de outliers est de :',len(outliers))


# Fonction moyenne mobile1

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Fonction moyenne mobile2

def moyenne_mobile_impaire(y,k):
    def mobile(y,k,n):
        if n + k//2 >= len(y):
            return []
        else:
            return [sum(y[n - k//2 : n + 1 + k//2]) / k] + mobile(y,k,n + 1)
    return list(repeat(None,k//2)) + mobile(y,k,k//2)

# Fonction Z-score

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Fonction dendrogram

def plot_dendrogram(Z, names, figsize=(15,30)):
    '''Plot a dendrogram to illustrate hierarchical clustering'''

    plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=20)
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
        leaf_font_size=12,
    )


#########################
#########################

#distortions = []
#K = range(1,10)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k)
#    kmeanModel.fit(X)
#    distortions.append(kmeanModel.inertia_)


    #distortions = []
#inertias = []
#mapping1 = {}
#mapping2 = {}
#K = range(1, 10)
  
#for k in K:
    # Building and fitting the model
#    kmeanModel = KMeans(n_clusters=k).fit(X)
#    kmeanModel.fit(X)
  
#    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
#                                        'euclidean'), axis=1)) / X.shape[0])
#    inertias.append(kmeanModel.inertia_)
  
#    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
#                                   'euclidean'), axis=1)) / X.shape[0]
#    mapping2[k] = kmeanModel.inertia_



#for key, val in mapping1.items():
#    print(f'{key} : {val}')

#plt.plot(K, distortions, 'bx-')
#plt.xlabel('Values of K')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method using Distortion')
#plt.show()

#########################
#########################

# Définition de la fonction pour le graphique Cercle de corrélation


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks:  # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(10, 10))

            # détermination des limites du graphique
            if lims is not None:
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else:
                xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(
                    pcs[d1, :]), min(pcs[d2, :]), max(pcs[d2, :])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30:
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                           pcs[d1, :], pcs[d2, :],
                           angles='xy', scale_units='xy', scale=1, color="grey")
            else:
                lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                ax.add_collection(LineCollection(
                    lines, axes=ax, alpha=.1, color='black'))

            # affichage des noms des variables
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(x, y, labels[i], fontsize='14', ha='center',
                                 va='center', rotation=label_rotation, color="blue", alpha=0.5)

            # affichage du cercle
            circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

# Définition de la fonction pour le graphique Projection des individus sur les plans factoriels

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # initialisation de la figure
            fig = plt.figure(figsize=(10, 10))

            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1],
                            X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(
                        X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x, y, labels[i],
                             fontsize='14', ha='center', va='center')

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))

            plt.title(
                "Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)



# Définition de la fonction pour le graphique Éboulis des valeurs propres
def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(), c="red", marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie(%)")
    plt.title("Éboulis des valeurs propres")
    plt.show(block=False)



