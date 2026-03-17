# Rapport Technique sur les Algorithmes de Clustering : de la Théorie à la Pratique

Ce rapport documente et compare douze des principaux algorithmes de clustering, fournissant pour chacun d'eux les bases théoriques, l'implémentation en Python via `scikit-learn` et d'autres bibliothèques de référence, une phase de validation avec des graphiques prévus, et une analyse critique. 

Le **Wine dataset** est utilisé tout au long du rapport pour permettre une comparaison équitable des performances des algorithmes sur un jeu de données réel.

---

## Prérequis et Configuration de l'Environnement

Pour exécuter les codes de ce rapport dans votre IDE ou un Jupyter Notebook (Google Colab, etc.), veillez à installer les dépendances suivantes :
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-learn-extra hdbscan minisom pyclustering umap-learn
```

### Chargement et Préparation des Données

Afin de garantir une base commune pour l'évaluation de tous les modèles, nous préparons les données avec une phase de standardisation et appliquons une ACP (Analyse en Composantes Principales) uniquement dans un but de visualisation des résultats.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

# Mode d'affichage des graphiques
sns.set_theme(style="whitegrid", palette="muted")

# 1. Chargement des données (Wine Dataset)
data = load_wine()
X = data.data
y_true = data.target  # Labels réels (pour l'évaluation externe uniquement)

# 2. Standardisation des variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Réduction de dimension pour la projection 2D (Visualisation)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Format des données : {X_scaled.shape}") # (178, 13)
```

**Métriques utilisées pour l'évaluation :**
1. **Silhouette Score** (Interne) : Mesure la cohésion interne et la séparation (Proche de 1 = excellent).
2. **Davies-Bouldin** (Interne) : Mesure le ratio entre la dispersion intra-cluster et la distance inter-cluster (Plus petit = meilleur).
3. **Adjusted Rand Index (ARI)** (Externe) : Mesure la similarité entre les clusters prédits et la vraie répartition des classes, corrigé pour le hasard (Proche de 1 = parfait).

---

## 1. K-Means

### Explication Théorique
Le K-Means est un algorithme de partitionnement fondé sur les centroïdes. Il vise à diviser les observations en $K$ groupes distincts en minimisant la variance intra-cluster (inertie). L'algorithme itère entre deux étapes : l'affectation de chaque point au centroïde le plus proche, et la redéfinition du centroïde comme moyenne des points du cluster.

### Exemple, Test et Visualisation en Python
```python
from sklearn.cluster import KMeans

# Modélisation
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Évaluation
sil = silhouette_score(X_scaled, labels_kmeans)
db = davies_bouldin_score(X_scaled, labels_kmeans)
ari = adjusted_rand_score(y_true, labels_kmeans)

print(f"[K-Means] Silhouette: {sil:.3f} | Davies-Bouldin: {db:.3f} | ARI: {ari:.3f}")

# Visualisation
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_kmeans, palette='viridis', s=80)
plt.title("Clustering K-Means (Projection PCA)")
plt.savefig("kmeans_plot.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** Le graphique affiche trois amas clairs séparés dans l'espace PCA 2D. L'ARI très élevé souvent obtenu sur le jeu Wine montre qu'il identifie parfaitement les 3 catégories de vin lorsque les nuages de points sont relativement sphériques.
- **Avantages :** Simple, très rapide, facilement scalable sur de grands datasets, convergent avec certitude.
- **Inconvénients :** Sensible aux valeurs extrêmes (outliers) car il utilise la moyenne (norme L2). Il nécessite de définir $K$ à l'avance et ne gère pas bien les clusters non convexes ou de densités variables.

---

## 2. K-Medoids (PAM)

### Explication Théorique
Le K-Medoids (souvent implémenté via l'algorithme PAM - Partitioning Around Medoids) est une variante robuste du K-Means. Au lieu d'utiliser des moyennes virtuelles, le centre de chaque cluster est un point de données réel (le médoïde). Il minimise une fonction de coût basée sur d'autres distances (ex: Manhattan), rendant la méthode très robuste aux points aberrants.

### Exemple, Test et Visualisation en Python
```python
from sklearn_extra.cluster import KMedoids

# Modélisation
kmedoids = KMedoids(n_clusters=3, random_state=42, metric='euclidean', init='k-medoids++')
labels_kmedoids = kmedoids.fit_predict(X_scaled)

# Évaluation
sil = silhouette_score(X_scaled, labels_kmedoids)
db = davies_bouldin_score(X_scaled, labels_kmedoids)
ari = adjusted_rand_score(y_true, labels_kmedoids)

print(f"[K-Medoids] Silhouette: {sil:.3f} | Davies-Bouldin: {db:.3f} | ARI: {ari:.3f}")

# Visualisation
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_kmedoids, palette='Set1', s=80)
plt.scatter(pca.transform(kmedoids.cluster_centers_)[:, 0], 
            pca.transform(kmedoids.cluster_centers_)[:, 1], 
            color='red', marker='X', s=200, label='Médoïdes')
plt.title("Clustering K-Medoids avec Centres")
plt.legend()
plt.savefig("kmedoids_plot.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** Les croix rouges indiquant les médoïdes coïncident avec des points réels au cœur de chaque sous-groupe. Les performances sont stables et proches de celles de K-Means.
- **Avantages :** Interprétation forte (les centres sont des exemplaires réels issus des données) et robustesse aux points aberrants.
- **Inconvénients :** Complexité algorithmique élevée ($O(k \cdot n^2)$ pour la configuration classique), ce qui le rend fastidieux pour le "Big Data".

---

## 3. K-Medians

### Explication Théorique
Le K-Medians calcule la médiane, et non la moyenne, pour mettre à jour les centroïdes, optimisant la norme L1 (distance de Manhattan). L'intérêt majeur est sa robustesse mathématique absolue face aux outliers distants : une erreur géante n'affectera quasiment pas le mouvement de la médiane.

### Exemple, Test et Visualisation en Python
```python
from pyclustering.cluster.kmedians import kmedians

# Pyclustering nécessite que les centres initiaux soient définis manuellement
initial_medians = [X_scaled[0].tolist(), X_scaled[70].tolist(), X_scaled[150].tolist()]

# Modélisation
kmed_inst = kmedians(X_scaled.tolist(), initial_medians)
kmed_inst.process()
clusters = kmed_inst.get_clusters()

# Reconstruction des labels pour consistance scikit-learn
labels_kmedians = np.zeros(X_scaled.shape[0], dtype=int)
for cluster_id, cluster in enumerate(clusters):
    for index in cluster:
        labels_kmedians[index] = cluster_id

# Évaluation
sil = silhouette_score(X_scaled, labels_kmedians)
db = davies_bouldin_score(X_scaled, labels_kmedians)
ari = adjusted_rand_score(y_true, labels_kmedians)

print(f"[K-Medians] Silhouette: {sil:.3f} | Davies-Bouldin: {db:.3f} | ARI: {ari:.3f}")

# Visualisation
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_kmedians, palette='Set2', s=80)
plt.title("Clustering K-Medians")
plt.savefig("kmedians_plot.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** Graphiquement, les frontières de décision sont souvent plus tranchées sur l'axe des dimensions où une asymétrie de distribution existe (Skewness).
- **Avantages :** Excellent pour modéliser des nuages où la présence de bruit impulsif massif est forte.
- **Inconvénients :** Plus coûteux en temps de calcul que la moyenne vectorielle basique (nécessite un tri/parcours pour trouver chaque médiane).

---

## 4. DBSCAN

### Explication Théorique
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) agrège des zones adjacentes ayant une densité significative de points via deux paramètres : `eps` (rayon de voisinage) et `min_samples` (points constituant le cœur). Les zones éparses sont rejetées et taguées comme "bruit" (label -1).

### Exemple, Test et Visualisation en Python
```python
from sklearn.cluster import DBSCAN

# Modélisation (paramètres nécessitant souvent de l'ajustement empirique)
dbscan = DBSCAN(eps=2.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

# Évaluation (Bruit écarté)
mask_no_noise = labels_dbscan != -1
if len(set(labels_dbscan[mask_no_noise])) > 1:
    sil = silhouette_score(X_scaled[mask_no_noise], labels_dbscan[mask_no_noise])
    ari = adjusted_rand_score(y_true, labels_dbscan)
    print(f"[DBSCAN] Silhouette (sans bruit): {sil:.3f} | ARI: {ari:.3f} | Bruit: {(~mask_no_noise).sum()} pts")
else:
    print("[DBSCAN] Seul 1 groupe ou majorité de bruit identifié. Un réglage d'eps est requis.")

# Visualisation
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_dbscan, palette='Dark2', s=80)
plt.title("Clustering DBSCAN (Les -1 sont des anomalies/bruit)")
plt.savefig("dbscan_plot.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** Les points éloignés des centres massifs apparaissent dans une couleur distincte (le bruit). Il se révèle souvent complexe de séparer les classes principales en dimensions 13 si elles se connectent discrètement.
- **Avantages :** Ne requiert pas de connaître le nombre de clusters ; modélise parfaitement des structures "en anneaux" ou "croissants".
- **Inconvénients :** La calamité de la grande dimension (Curse of Dimensionality) détruit l'homogénéité du rayon euclidien `eps`, et il gère très mal les densités inégales.

---

## 5. HDBSCAN

### Explication Théorique
HDBSCAN (Hierarchical DBSCAN) contourne la rigidité du paramètre `eps` en exploitant une hiérarchie de toutes les distances `eps` possibles. Puis, il sélectionne des clusters qui survivent le plus longtemps lorsqu'on change virtuellement le seuil de vue.

### Exemple, Test et Visualisation en Python
```python
from sklearn.cluster import HDBSCAN

# Modélisation
hdbscan = HDBSCAN(min_cluster_size=10, min_samples=3)
labels_hdbscan = hdbscan.fit_predict(X_scaled)

# Évaluation
sil_hd = silhouette_score(X_scaled, labels_hdbscan) if len(set(labels_hdbscan)) > 1 else -1
ari_hd = adjusted_rand_score(y_true, labels_hdbscan)
print(f"[HDBSCAN] Silhouette: {sil_hd:.3f} | ARI: {ari_hd:.3f}")

# Visualisation
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_hdbscan, palette='tab10', s=80)
plt.title("Clustering HDBSCAN (Densités variables)")
plt.savefig("hdbscan_plot.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** On obtient moins de "bruit" non voulu que DBSCAN car l'algorithme a ajusté la "micro-densité" par région. 
- **Avantages :** Ne requiert pas la configuration complexe de l'epsilon. Très flexible dans les environnements de taille et dispersion variables face au bruit.
- **Inconvénients :** Exécuté de manière asymétrique, il peut quand même refuser de lier des "ponts" entre sous-distributions et donc engendrer un regroupement non-exhaustif.

---

## 6. OPTICS

### Explication Théorique
OPTICS (Ordering Points To Identify the Clustering Structure) adresse également les faiblesses du epsilon paramétrable en produisant un graphique de _Reachability_ : un profil des distances entre voisins. En découpant les "vallées" de cette distance ordonnée, on révèle les clusters de n'importe quelle échelle.

### Exemple, Test et Visualisation en Python
```python
from sklearn.cluster import OPTICS

# Modélisation
optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.1)
labels_optics = optics.fit_predict(X_scaled) 

# Évaluation
sil_opt = silhouette_score(X_scaled, labels_optics) if len(set(labels_optics)) > 1 else -1
ari_opt = adjusted_rand_score(y_true, labels_optics)
print(f"[OPTICS] Silhouette: {sil_opt:.3f} | ARI: {ari_opt:.3f}")

# Graphique de Reachability classique
plt.figure(figsize=(10, 4))
plt.plot(optics.reachability_[optics.ordering_], color='b', alpha=0.6)
plt.title("Diagramme de Reachability (OPTICS)")
plt.ylabel("Reachability Distance")
plt.xlabel("Points ordonnés")
plt.savefig("optics_reachability.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** Le _Reachability plot_ montre des creux et des clochers : les creux sont nos clusters. S'il n'y a que trois vastes creux profonds, cela corrobore qu'il y a 3 structures substantielles.
- **Avantages :** Un formalisme magnifique pour explorer les hiérarchies de densité imbriquée sans assumer aucune distribution a priori.
- **Inconvénients :** Beaucoup plus lent que DBSCAN pour une vaste base de données à moins qu'un arbre métrique (KD-Tree / Ball Tree) optimisé ne soit érigé initialement.

---

## 7. HAC (Clustering Agglomératif Hiérarchique)

### Explication Théorique
De type "Bottom-Up", HAC débute avec chaque échantillon s'incarnant comme un groupe distinct. Ensuite, une fonction de couplage (*Linkage* de Ward) fusionne itérativement les paires les plus proches jusqu'à obtenir l'arborescence finale.

### Exemple, Test et Visualisation en Python
```python
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Modélisation
hac = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels_hac = hac.fit_predict(X_scaled)

# Évaluation
sil = silhouette_score(X_scaled, labels_hac)
ari = adjusted_rand_score(y_true, labels_hac)
print(f"[HAC] Silhouette: {sil:.3f} | ARI: {ari:.3f}")

# Visualisation des Dendrogrammes (Schéma interne)
plt.figure(figsize=(9, 5))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'), no_labels=True)
plt.axhline(y=15, color='r', linestyle='--', label="Seuil de Coupure")
plt.title("Dendrogramme du clustering HAC (Ward)")
plt.ylabel("Distance Euclidienne (Variance)")
plt.legend()
plt.savefig("hac_dendrogram.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** Le dendrogramme rend visible un profond saut quantitatif où naissent précisément 3 immenses branches avant de fusionner. L'arbre sert d'outil explicatif pour la validation métier.
- **Avantages :** Construit une Taxonomie. On peut trancher l'arbre (la hauteur y) à n'importe quel stade pour obtenir différents regroupements à différentes échelles macro/micro.
- **Inconvénients :** Scalabilité très mauvaise (Coût en mémoire quadratique, temps d'exécution cubique $\mathcal{O}(N^3)$ sur la stratégie triviale).

---

## 8. Modèles de Mélanges Gaussiens (GMM)

### Explication Théorique
GMM pose l'hypothèse que la matrice de point est générée par un regroupement de $K$ distributions de probabilité gaussiennes multivariées. Mû par l'algorithme "Expectation-Maximization", chaque point reçoit une liste de _probabilités_ (poids) d'appartenir à chaque amas, traduisant le flou et l'incertitude.

### Exemple, Test et Visualisation en Python
```python
from sklearn.mixture import GaussianMixture

# Modélisation
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled)
labels_gmm = gmm.predict(X_scaled)

# Évaluation
sil = silhouette_score(X_scaled, labels_gmm)
ari = adjusted_rand_score(y_true, labels_gmm)
print(f"[GMM] Silhouette: {sil:.3f} | ARI: {ari:.3f}")

# Visualisation
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_gmm, palette='crest', s=80)
plt.title("GMM (Mélanges Gaussiens) - Appartenance Finale")
plt.savefig("gmm_plot.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** Maintient de superbes frontières ellipsoïdales là où K-means forcerait des disques parfaits. Ses scores externes comme l'ARI sont généralement exceptionnels.
- **Avantages :** Fournit le degré d'incertitude de décision (probabilités). Adapte ses axes indépendamment, permettant des clusters "minces et allongés" ou asymétriques.
- **Inconvénients :** Fragile vis-à-vis du surapprentissage (Si `covariance_type='full'`, il y a énormément de paramètres à dériver pour chaque distribution).

---

## 9. Spectral Clustering

### Explication Théorique
Le Clustering Spectral applique la théorie des graphes. Les données forment des "nœuds" et leurs distances des "arêtes". On construit une matrice Laplacienne, on calcule quelques vecteurs propres, et on lance dans ce sous-espace aplati un K-means rudimentaire. Cela coupe les arêtes de poids faible.

### Exemple, Test et Visualisation en Python
```python
from sklearn.cluster import SpectralClustering

# Modélisation
spectral = SpectralClustering(n_clusters=3, assign_labels='kmeans', affinity='nearest_neighbors', random_state=42)
labels_spectral = spectral.fit_predict(X_scaled)

# Évaluation
sil = silhouette_score(X_scaled, labels_spectral)
ari = adjusted_rand_score(y_true, labels_spectral)
print(f"[Spectral] Silhouette: {sil:.3f} | ARI: {ari:.3f}")

# Visualisation
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_spectral, palette='autumn', s=80)
plt.title("Spectral Clustering via Affinité par Plus Proches Voisins")
plt.savefig("spectral_plot.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** Les nuages étirés ou tordus sont sa spécialité, un tracé coloré montre souvent qu'il épouse délicatement la géographie naturelle non-convexe si l'affinité k-NN est stable.
- **Avantages :** Redoutablissime pour toutes distributions biscornues, sinueuses (ex: Cercles concentriques).
- **Inconvénients :** Le calcul initial de la table propre s'effondre techniquement en consommation RAM si le datamart comporte des dizaines de milliers d'entrées.

---

## 10. Affinity Propagation

### Explication Théorique
AP (Affinity Propagation) identifie des "Exemplaires". Les noeuds échangent indéfiniment des messages de Responsabilité (la capacité d'un centre à servir ce nœud) et de Disponibilité (le témoignage d'agrégation d'autres nœuds). Convergence survient quand le système fige ses exemplaires formateurs sans l'aide du paramètre $K$.

### Exemple, Test et Visualisation en Python
```python
from sklearn.cluster import AffinityPropagation

# Modélisation
affinity = AffinityPropagation(damping=0.8, random_state=42)
labels_affinity = affinity.fit_predict(X_scaled)

# Évaluation
n_clusters_aff = len(affinity.cluster_centers_indices_)
sil = silhouette_score(X_scaled, labels_affinity)
ari = adjusted_rand_score(y_true, labels_affinity)
print(f"[Affinity Prop] Clusters formés: {n_clusters_aff} | Silhouette: {sil:.3f} | ARI: {ari:.3f}")

# Visualisation
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_affinity, palette='hsv', s=80, legend=False)
plt.title(f"Affinity Propagation (A identifié {n_clusters_aff} clusters macro/micro)")
plt.savefig("affinity_plot.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** Un tel algorithme tend à éclater continuellement le groupe en 15 ou 20 micro-bouts, illustrant un graphisme hétéroclite coloré. Le "damping" freine ces bifurcations.
- **Avantages :** Trouve le vrai cœur biologique d'un amas ; utile en bio-informatique (série génomique) ou traitement naturel du langage.
- **Inconvénients :** Extrême lenteur algorithmique sur la passation de message itérative. Divise atrocement une entité diffuse en une multitude d'entités inutiles si on ne calibre pas la préférence.

---

## 11. Self-Organizing Maps (SOM)

### Explication Théorique
SOM s'appuie sur une surface neuronale (souvent plane 2D). En compétition (Apprentissage non-supervisé), les "neurones" adaptent leurs poids pour se tordre sur le nuage dimensionnel. Une fois stabilisée, la topologie est préservée, on rassemble ces points-prototypes "BMU" (Best Matching Units) dans un algorithme final.

### Exemple, Test et Visualisation en Python
```python
from minisom import MiniSom

# Modélisation (Grille 10x10)
som = MiniSom(x=10, y=10, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(X_scaled)
som.train_random(data=X_scaled, num_iteration=500)

labels_som_raw = [som.winner(x)[0]*10 + som.winner(x)[1] for x in X_scaled]

# Assignation finale (via un KMeans trivial sur la composante aplatie du SOM en 3 groupes)
labels_som_km = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(np.array(labels_som_raw).reshape(-1, 1))

sil = silhouette_score(X_scaled, labels_som_km)
ari = adjusted_rand_score(y_true, labels_som_km)
print(f"[SOM] Silhouette (Original): {sil:.3f} | ARI: {ari:.3f}")

# Visualisation
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_som_km, palette='Set1', s=80)
plt.title("SOM + Agrégation (Projection locale lissée)")
plt.savefig("som_plot.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** Les contours dressés par les grilles neurales respectent bien mieux la "courbure intrinsèque" des données que de simples sphères métriques euclidiennes.
- **Avantages :** Un cadre exploratoire monumental pour la réduction de dimension visuellement fidèle (Data Mining). Gère très bien la colinéarité.
- **Inconvénients :** Exige une configuration laborieuse (taux d'apprentissage, rayons du voisinage et dimensions de la matrice). Peut être excessif pour des partitions simplistes.

---

## 12. UMAP + K-Means

### Explication Théorique
Technique de la dernière génération constituée de deux relais : UMAP (Uniform Manifold Approximation et Projection) dresse une architecture topologique floue et compresse sans perte logique les dimensions vers une 2D/3D extrêmement agglomérée. Puis, un algorithme trivial (K-Means/HDBSCAN) moissonne ces nouveaux îlots visuellement séparés.

### Exemple, Test et Visualisation en Python
```python
import umap

# Réduction dimensionnelle topologique (Espace latent)
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Clustering
kmeans_umap = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_umap = kmeans_umap.fit_predict(X_umap)

# Évaluation sur la projection UMAP pour l'interne, mais y_true pour la cohérence externe
sil = silhouette_score(X_umap, labels_umap)
ari = adjusted_rand_score(y_true, labels_umap)
print(f"[UMAP+KMeans] Silhouette (UMAP): {sil:.3f} | ARI: {ari:.3f}")

# Visualisation sur de Nouveaux Axes (Latents)
plt.figure(figsize=(7, 5))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels_umap, palette='viridis', s=80)
plt.title("UMAP Manifold Projection suivie d'un K-Means")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig("umap_kmeans_plot.png")
plt.show()
```

### Interprétation des Graphes et Analyse Critique
- **Interprétation Graphique :** Les graphiques UMAP dévoilent distinctement trois agrégats hyper-déconnectés. K-Means parvient donc à récolter les étiquettes de manière parfaite, propulsant le label externe ARI à sa valeur absolue.
- **Avantages :** Outil chirurgical radical pour nettoyer le bruit multidimensionnel et forcer une séparation évidente que n'importe quel algorithme léger pourra classer immédiatement.
- **Inconvénients :** Il n'existe plus aucune "interprétabilité" de distance dans l'espace reconstitué : on perd le sens euclidien original. Peut ségréger du bruit normal au rang de micro-cluster.

---

## Tableau Comparatif Global

| Algorithme | Logique structurelle | K par défaut? | Tolérance Aberrations (Bruit) | Structures Complexes / Concaves | Haute Dimension ? | Scalabilité/Vitesse |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means** | Distances vers Centroïdes (L2) | **Oui** | Faible | Faible | Moyenne | 🟩 Très Rapide |
| **K-Medoids** | Exemplaires réels | **Oui** | Moyenne | Faible | Moyenne | 🟧 Modérée |
| **K-Medians**| Optimisation de Médianes (L1) | **Oui** | Excellente | Faible | Moyenne | 🟧 Modérée |
| **DBSCAN** | Densité spatiale epsilon | **Non** | Excellente| Très Forte | Faible | 🟧 Modérée |
| **HDBSCAN** | Hiérarchies de Densité | **Non** | Excellente| Très Forte | Moyenne | 🟧 Modérée |
| **OPTICS** | Distance de Reachability | **Non** | Forte | Très Forte | Faible | 🟥 Lente |
| **HAC** | Arbre Taxonomique Bottom-Up| **Oui** | Faible | Modérée | Moyenne | 🟥 Lente ($O(N^3)$)|
| **GMM** | Probabiliste d'Espérance | **Oui** | Modérée | Forte | Modérée | 🟧 Modérée |
| **Spectral** | Coupe par Graphes Laplaciens| **Oui** | Modérée | Exceptionnelle| Moyenne | 🟥 Lente |
| **Affinity P.**| Transfert de Responsabilités | **Non** | Modérée | Modérée | Forte | 🟥 Très Lente |
| **SOM** | Compétition sur Grille Neuron.| **Non/Oui** | Forte | Modérée | Forte | 🟧 Modérée |
| **UMAP + KM**| Ecrasement Manifold (Latent)| **Oui** | Forte | Exceptionnelle| **Très Forte**| 🟩 Rapide |

---

## Synthèse Finale : Comment choisir son algorithme ?

Le choix de l'algorithme idéal dépend du contexte brut de la donnée, des objectifs visés par le Data Scientist, et de de la volumétrie :

1. **La rapidité avant tout et des blocs présumés homogènes :** 
   Toujours démarrer avec le pivot algorithmique universel : **K-Means**. Il procure une "baseline" inestimable, il est mathématiquement incontestable et converge instantanément sur des millions de lignes avec ses paramètres d'accélération Elkan.

2. **Défense face à des données fortement corrompues / parasites :** 
   Les dérives des moyennes doivent être exclues. Prenez le contrôle avec **K-Medians** ou encadrez spécifiquement vos points fondations à l'aide de **K-Medoids**, assurant que quelques anomalies extravagantes ne dérèglent pas vos centroids.

3. **Débroussaillage aveugle sur dimensions restreintes :** 
   Si $K$ est non quantifiable par le "métier", utilisez le moteur de densité auto-adaptatif : **HDBSCAN**. Sa topologie exempte de fine-tuning arbitraire isolera très finement les signaux forts du bruit ambiant en s'adaptant à l'inégalité de densité.

4. **Structures d'anneaux ou cercles concentriques :** 
   Dès lors que la structure des points dessine des "S", des arcs ou spirales, la convexité de K-Means devient hors-jeu absolu. Impliquez d'urgence un modèle topographique non linéaire, **DBSCAN** ou **Spectral Clustering**.

5. **Clustering dit "Doux" (Soft), Incertitude, et Explicabilité :** 
   S'il incombe à justifier le lien entre clusters et points, employez **HAC** pour son arborescence en Dendrogramme ou bien **GMM** (Mélange Gaussien), qui livre les pourcentages certifiés d'appartenance probabiliste pour tous les vecteurs aux frontières incertaines.

6. **Grandes Dimensions Massives (Texte, Imagerie, Génétique) :** 
   Ne laissez pas la "Malédiction de la grande dimension" détacher la logique euclidienne : la pratique state-of-the-art plébiscite puissamment à ce jour un workflow double, compactant le profil du lot avec l'écrasement **UMAP** avec de propulser dans l'espace projeté n'importe quel clustering basique (Généralement **K-Means** ou **HDBSCAN**).

---
*Ce rapport a été structuré et élaboré pour livrer une explication pratique, théorique et mathématique couvrant l'analyse complète du cycle de machine learning non supervisé.*
