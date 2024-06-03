                                         ___________________________________
                                         |                                  |
                                         |             INDICE               |
                                         |__________________________________|

        1. K-MEANS                                         "Definir cluster con K-Means"
        2. GAUSSIAN MIXTURE MODELS  (GMM)                  "Define cluster, pero a con probabilidades"
        3. EJEMPLOS K-MEANS                                "Ejemplo de modelos K-Means" 
        4. DISTANCIAS Y DBSCAN                             "Cluster con DBSCAN y tipos de distancias"
        5. MALDICION DE DIMENCIONALIDAD                    "Problema de maldicion de dimensionalidad"
        6. DBSCAN Y TSNE                                   "Reduccion de dimensionalidad con TSNE y cluster de imagenes" 
        7. MEAN SHIFT CLUSTERING                           "Cluster con mean shift clustering"
        8. EJEMPLOS DE CLUSTERING                          "Ejemplo de varios modelos para hacer clustering"
        9. PCA                                             "Reduccion de dimensionalidad con PCA"  
       10. SINGULAR-VALUE DESCOMPOSITION (SVD)             "SVD es bueno para detectar movimientos en los videos"
       11. OPERACIONES MATRICIALES                         "Ejemplo de operaciones matriciales en python"
       12. EJEMPLOS REDUCCION DE DIMENSIONALIDAD           "Ejemplos utilizndo modelos con reduccion de dimensionalidad"
       13. MDS                                             "MDS es una familia de reductores de dimensionalidad"
       14. KERNEL PCA                                      "Otra forma de reduccion de dimensionalidad con mas opciones que PCA"
       15. FACTORIZACION MATRICIAL NO NEGATIVA             "Alguna de las formas de factorizacion matricial, conteo de vectores"
       16. IF-IDF                                          "Se ocupa para analizar las frecuencias de palabras"
       17. FACTORIZACION MATRICIAL NO NEGATIVA             "Un ejemplo añadido a matricez"


#######################################################################################################################

                                                 #K-Means


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

#Un datasets de clientes queremos segmentar a los clientes considerando todos sus datos
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/CustomerData.csv', index_col=0)

print(df.head())

num_male = df[df['Gender'] == 'Male'].shape[0]
num_female = df[df['Gender'] == 'Female'].shape[0]
plt.pie(
    [num_male, num_female],
    labels=['Male', 'Female'],
    startangle=90,
    autopct='%1.f%%',
    colors=['lavender', 'thistle'])
plt.title('Gender of survey respondants')
plt.show()

#Un histograma de los ingresos anuales
plt.hist(df['Annual Income (k$)'], bins=10)
plt.show()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

#Para generar un grafico de ingresos a anuales vs porcentaje de gasto
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'])
plt.show()

km = KMeans(n_clusters=5, random_state=42)
km.fit(X)

#Pudimos segementar a los clientes, importante es notar que solo ocupammos 2 caracterisitcas y no hay etiqueta categorica
#la categoria la da Kmeans
for label in np.unique(km.labels_):
    X_ = X[label == km.labels_]
    plt.scatter(X_['Annual Income (k$)'], X_['Spending Score (1-100)'], label=label)
plt.show()

###Ejemplo 2

#Segmentacion de imagenes

#Kmeans puede hacer que imagenes solo se muestren en cierta cantidad de colores que nosotros le digamos, en cierta forma
#Lo que hace es segmentar las imagenes, simplemente dice que pixeles pertenecen a cierto color
#IMPORANTE cada valor de k es que tanto puede distinguir la imagen, entre mas la distinga, mas podra definir diferentes
#colores, no tiene sentido poner mas k que pixeles existentes, entre menos k tenga menos distingira la imagen y por ende
#pondra menos colores, porque distingue menos conglomerados

#Aun asi no inventara colores, si la imagen viene de base de cierta forma, solo mejorara la calidad con k

from skimage import io

#Imagen de una persona tomando una foto en escala de grises
img = io.imread('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/images/cameraman.png')
print(f'The image is {img.shape[0]}px by {img.shape[1]}px')
plt.axis('off')
plt.imshow(img)
plt.show()

k = 2



#En blanco y negro solo tiene 1 caracteristica, color negro
X = img.reshape(-1, 1)
print(X.shape)
km = KMeans(n_clusters=k, random_state=42)

km.fit(X)

seg = np.zeros(X.shape)
for i in range(k):
    seg[km.labels_ == i] = km.cluster_centers_[i]
seg = seg.reshape(img.shape).astype(np.uint8)
plt.imshow(seg)
plt.show()

#Este no me funciono
seg = np.zeros(X.shape)
for i in range(k):
    seg[km.labels_ == i] = 255 if km.cluster_centers_[i] > 0.5 else 0
seg = seg.reshape(img.shape).astype(np.uint8)
plt.imshow(seg)
plt.show()

### Ejemplo 3

img = io.imread('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/images/kingfisher.jpeg')
plt.imshow(img)
plt.show()

k = 2

#Recuerde que para RGB son 3 caracterisitcas, los 3 colores, pero como son colores mezclados se divididen
#en subdiviciones, cada pixel es una mezcla de 3 columnas
X = img.reshape(-1, 3) # Remember, since image is RGB
km = KMeans(n_clusters=k, random_state=42)
km.fit(X)

seg = np.zeros(X.shape)
for i in range(k):
    seg[km.labels_ == i] = km.cluster_centers_[i]
seg = seg.reshape(img.shape).astype(np.uint8)
plt.imshow(seg)
plt.show()


k = 4

X = img.reshape(-1, 3) # Remember, since image is RGB
km = KMeans(n_clusters=k, random_state=42)
km.fit(X)

seg = np.zeros(X.shape)
for i in range(k):
    seg[km.labels_ == i] = km.cluster_centers_[i]
seg = seg.reshape(img.shape).astype(np.uint8)
plt.imshow(seg)
plt.show()


#######################################################################################################################

                                          #Gaussian Mixture Models (GMM)
                                        #Matrices y imagenes, tambien con PCA

#GMM tambien cumple una funcion de segmentacion, es muy utilizado para recomendacion, porque Kmean lo que hace es
#decir si es que un cliente pertenece a un conglomerado si o no, pero GMM lo que hace es entregar una probabilidad
#de que le guste o pertenesca cierto conglomerado, en ese sentido es mucho mas versatir que Kmeans en las recomendaciones

#Como es en probabilidad sus recomenadines pueden cambiar diariamente, etc

import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from itertools import chain
from matplotlib.patches import Ellipse

sns.set_context('notebook')
sns.set_style('white')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

#definiendo funciones utiles

#Returna el simulado 1d de datasets X y una figura
def plot_univariate_mixture(means, stds, weights, N = 10000, seed=10):

    np.random.seed(seed)
    if not len(means)==len(stds)==len(weights):
        raise Exception("Length of mean, std, and weights don't match.") 
    K = len(means)
    
    mixture_idx = np.random.choice(K, size=N, replace=True, p=weights)
    # generate N possible values of the mixture
    X = np.fromiter((ss.norm.rvs(loc=means[i], scale=stds[i]) for i in mixture_idx), dtype=np.float64)
      
    # generate values on the x axis of the plot
    xs = np.linspace(X.min(), X.max(), 300)
    ps = np.zeros_like(xs)
    
    for mu, s, w in zip(means, stds, weights):
        ps += ss.norm.pdf(xs, loc=mu, scale=s) * w
    
    fig, ax = plt.subplots()
    ax.plot(xs, ps, label='pdf of the Gaussian mixture')
    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("P", fontsize=15)
    ax.set_title("Univariate Gaussian mixture", fontsize=15)
    plt.show()
    
    return X.reshape(-1,1), fig, ax
    
 
#Returna la solucioes 2d del datasets X y un grafico de dispercion
def plot_bivariate_mixture(means, covs, weights, N = 10000, seed=10):
    

    np.random.seed(seed)
    if not len(mean)==len(covs)==len(weights):
        raise Exception("Length of mean, std, and weights don't match.") 
    K = len(means)
    M = len(means[0])
    
    mixture_idx = np.random.choice(K, size=N, replace=True, p=weights)
    
    # generate N possible values of the mixture
    X = np.fromiter(chain.from_iterable(multivariate_normal.rvs(mean=means[i], cov=covs[i]) for i in mixture_idx), 
                dtype=float)
    X.shape = N, M
    
    xs1 = X[:,0] 
    xs2 = X[:,1]
    
    plt.scatter(xs1, xs2, label="data")
    
    L = len(means)
    for l, pair in enumerate(means):
        plt.scatter(pair[0], pair[1], color='red')
        if l == L-1:
            break
    plt.scatter(pair[0], pair[1], color='red', label="mean")
    
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Scatter plot of the bivariate Gaussian mixture")
    plt.legend()
    plt.show()
    
    return X


#Dibuja una elipse con el que obtinee la posicion y covarianza
def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))
        
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


#Lo que hace es definir gaussianas, prinero le entregamos el valor promedio de cada gaussiana, con su respectiva
#desviacion estandart, un cierto peso que al parecer es cierto porcentaje de datos que afecta a la probabilidad (altura)
#aunque la altura es una mezcla de los 3 factores, pero el peso afecta mas
X1, fig1, ax1 = plot_univariate_mixture(means=[2,5,8], stds=[0.2, 0.5, 0.8], weights=[0.3, 0.3, 0.4]) 

X2, fig2, ax2 = plot_univariate_mixture(means=[2,5,8], stds=[0.6, 0.9, 1.2], weights=[0.3, 0.3, 0.4]) 

X3, fig3, ax3 = plot_univariate_mixture(means=[2,5,8], stds=[0.6, 0.9, 1.2], weights=[0.05, 0.35, 0.6])


# sort X1 in ascending order for plotting purpose
X1_sorted = np.sort(X1.reshape(-1)).reshape(-1,1)

#Es una cantidad de datos que pueda cubrir a las gaussianas graficadas de 1 a 10
print(X1_sorted)

#Creamos un modelos con 3 diferentes conglomerados
GMM = GaussianMixture(n_components=3, random_state=10)
GMM.fit(X1_sorted)

prob_X1 = GMM.predict_proba(X1_sorted)

#Grafica las probabilidades de pertenecer a cierto gaussiana como le mandamos X con valores de 1 a 10 muestra
#las 3 gaussianas
#La linea negra como en ese lado no hay probabilidad de los otras dos conglomerados muestra que la probabilidad es
#100%
fig1, ax1 = plt.subplots()
ax1.plot(X1_sorted, prob_X1[:,0], label='Predicted Prob of x belonging to cluster 1')
ax1.plot(X1_sorted, prob_X1[:,1], label='Predicted Prob of x belonging to cluster 2')
ax1.plot(X1_sorted, prob_X1[:,2], label='Predicted Prob of x belonging to cluster 3')
ax1.scatter(2, 0.6, color='black')
ax1.scatter(2, 1.0, color='black')
ax1.plot([2, 2], [0.6, 1.0],'--', color='black')
ax1.legend()
plt.show()

### Ejemplo 1

mean = [(1,5), (2,1), (6,2)]
cov1 = np.array([[0.5, 1.0],[1.0, 0.8]])
cov2 = np.array([[0.8, 0.4],[0.4, 1.2]])
cov3 = np.array([[1.2, 1.3],[1.3, 0.9]])
cov = [cov1, cov2, cov3]

cov = np.array([[0.5, 1.0], [1.0, 0.8],
               [0.8, 0.4], [0.4, 1.2],
               [1.2, 1.3], [1.3, 0,9]])


print(cov)
weights = [0.3, 0.3, 0.4]

X4 = plot_bivariate_mixture(means=mean, covs=cov, weights=weights, N=1000)  

print("The dataset we generated has a shape of", X4.shape)

gm = GaussianMixture(n_components=3, random_state=0).fit(X4)
print("Means of the 3 Gaussians fitted by GMM are\n")
print(gm.means_)

print("Covariances of the 3 Gaussians fitted by GMM are")
gm.covariances_

plot_gmm(GaussianMixture(n_components=3, random_state=0), # the model, 
          X4) # simulated Gaussian mixture data

# try Covariance_type = 'tied'
plot_gmm(GaussianMixture(n_components=3, covariance_type='tied',random_state=0), # the model, 
         X4)

# try Covariance_type = 'diag'
plot_gmm(GaussianMixture(n_components=3, covariance_type='diag',random_state=0), # the model, 
         X4)



###Ejemplo 2 con imagenes


#GMM en imagen es muy similar a Kmeans, pero la diferencias es que GMM va tomando los limites en base a eso muestra
#la imagen, puede definir muy bien un objeto de la imagen el que tenga los limites mas marcados y todo lo demas se ve
#borrosa

#Tambien un beneficio que tiene es que logra mejores definiciones con mucho menos colores

from skimage import io

img = io.imread('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%201/images/gauss-cat.jpeg')

# If img is greyscale, then change to .reshape(-1, 1):
X = img.reshape(-1, 3)
# The number of components; you can change this to a positive integer of your choice!:
n = 2
gmm = GaussianMixture(n_components=n, covariance_type='tied')
gmm.fit(X)
labels = gmm.predict(X) # num of pixels x 1

seg = np.zeros(X.shape) # num of pixels x 3

for label in range(n):
    seg[labels == label] = gmm.means_[label]
seg = seg.reshape(img.shape).astype(np.uint8)

plt.figure(figsize=(6,6))
plt.imshow(seg)
plt.show()

n = 8
gmm = GaussianMixture(n_components=n, covariance_type='tied')
gmm.fit(X)
labels = gmm.predict(X) # num of pixels x 1
seg = np.zeros(X.shape) # num of pixels x 3

for label in range(n):
    seg[labels == label] = gmm.means_[label]
seg = seg.reshape(img.shape).astype(np.uint8)
#cv2.imwrite(f'gauss-cat-{n}.jpeg', seg)

plt.figure(figsize=(6,6))
plt.imshow(seg)
plt.show()


###Ejemplo 3 GMM con PCA IMPORTANTE

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%201/customers.csv")
data.head()

print(data.shape)

#Se escala toda la data
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X = SS.fit(data).transform(data)

#Luego se descompone con PCA, las data de 19 columnas pasa solo a tener 2 columnas importantes
from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)
reduced_2_PCA = pca2.fit(X).transform(X)

#Se ocupan 4 conglomerados para los datos, que son 2 columnas generados con PCA
model = GaussianMixture(n_components=4, random_state=0)
model.fit(reduced_2_PCA)

#OBtenemos predicciones con este modelo GMM
PCA_2_pred = model.predict(reduced_2_PCA)

#Ploteamos un grafico 2D gracias a PCA, color depende de las categorizaciones hechas por GMM
x = reduced_2_PCA[:,0]
y = reduced_2_PCA[:,1]
plt.scatter(x, y, c=PCA_2_pred)
plt.title("2d visualization of the clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

#Creando otro grafico con un PCA con 3 columnas, luego se ocupa GMM y se obtienen las predicciones
pca3 = PCA(n_components=3)
reduced_3_PCA = pca3.fit(X).transform(X)
mod = GaussianMixture(n_components=4, random_state=0)
PCA_3_pred = mod.fit(reduced_3_PCA).predict(reduced_3_PCA)

#Se plotean el grafico de 3 dimenciones y enrealidad notamos que es identico al anterior, pero en 3D
reduced_3_PCA = pd.DataFrame(reduced_3_PCA, columns=(['PCA 1', 'PCA 2', 'PCA 3']))
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(reduced_3_PCA['PCA 1'],reduced_3_PCA['PCA 2'],reduced_3_PCA['PCA 3'], c=PCA_3_pred)
ax.set_title("3D projection of the clusters")
plt.show()



#######################################################################################################################

                                             #Kmeans ejemplos

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle

plt.rcParams['figure.figsize'] = [6,6]
sns.set_style("whitegrid")
sns.set_context("talk")

#Algunas funciones imporantes
def display_cluster(X,km=[],num_clusters=0):
    color = 'brgcmyk'
    alpha = 0.5
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_==i,0],X[km.labels_==i,1],c = color[i],alpha = alpha,s=s)
            plt.scatter(km.cluster_centers_[i][0],km.cluster_centers_[i][1],c = color[i], marker = 'x', s = 100)
            plt.show()
            
angle = np.linspace(0,2*np.pi,20, endpoint = False)
X = np.append([np.cos(angle)],[np.sin(angle)],0).transpose()
display_cluster(X)


#Agrupamiento usando datos aleatorios, simplemente muestra donde se posiciones el centro
#Recordar que el punto de partida es importante para los conglomerados que elegira

num_clusters = 2
km = KMeans(n_clusters=num_clusters,random_state=10,n_init=1) # n_init, number of times the K-mean algorithm will run
km.fit(X)
display_cluster(X,km,num_clusters)

km = KMeans(n_clusters=num_clusters,random_state=20,n_init=1)
km.fit(X)
display_cluster(X,km,num_clusters)


n_samples = 1000
n_bins = 4  
centers = [(-3, -3), (0, 0), (3, 3), (6, 6)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)
display_cluster(X)

num_clusters = 7
km = KMeans(n_clusters=num_clusters)
km.fit(X)
display_cluster(X,km,num_clusters)

num_clusters = 4
km = KMeans(n_clusters=num_clusters)
km.fit(X)
display_cluster(X,km,num_clusters)

#Es una  medida de error cuadrado en la distancia al centro del cumulo
#Depende de la cantidad de conglomerados, pero es buena para elegir el mejor numero de conglomerados
print(km.inertia_)

#Esto muestra que a medida que aumentan los cluster, disminuye el error
inertia = []
list_num_clusters = list(range(1,11))
for num_clusters in list_num_clusters:
    km = KMeans(n_clusters=num_clusters)
    km.fit(X)
    inertia.append(km.inertia_)
    
plt.plot(list_num_clusters,inertia)
plt.scatter(list_num_clusters,inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

#Por lo que se deja al criterio cual seria el mejor modelo
#Pero cuando la curva deja de caer tan rapido seria una buena eleccion


###Ejemplo de imagen

from skimage import io

img = io.imread('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%201/images/peppers.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()

print(img.shape)

#Esto es una imagen creada con un solo color
R = 35
G = 95
B = 131
plt.imshow([[np.array([R,G,B]).astype('uint8')]])
plt.axis('off')
plt.show()

#Toma denuevo la imagen y la toma como matrices
img_flat = img.reshape(-1, 3)
print(img_flat[:5,:])

print(img_flat.shape)

#Ajusta un modelo de Kmean con 8 conglomerados
kmeans = KMeans(n_clusters=8, random_state=0).fit(img_flat)

img_flat2 = img_flat.copy()

#Es como que crea montones de centro para poder recrear la imagen, gracias a esto puede interpretarla
#Si esto no se pone ocupa la imagen normalmente como si nada
for i in np.unique(kmeans.labels_):
    img_flat2[kmeans.labels_==i,:] = kmeans.cluster_centers_[i]

img2 = img_flat2.reshape(img.shape)
plt.imshow(img2)
plt.axis('off')
plt.show()


def image_cluster(img, k):
    img_flat = img.reshape(img.shape[0]*img.shape[1],3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_flat)
    img_flat2 = img_flat.copy()

    # loops for each cluster center
    for i in np.unique(kmeans.labels_):
        img_flat2[kmeans.labels_==i,:] = kmeans.cluster_centers_[i]
        
    img2 = img_flat2.reshape(img.shape)
    return img2, kmeans.inertia_

#Para graficar los valores de inercia de los diferentes valores de k
k_vals = list(range(2,21,2))
img_list = []
inertia = []
for k in k_vals:
#    print(k)
    img2, ine = image_cluster(img,k)
    img_list.append(img2)
    inertia.append(ine)  

plt.plot(k_vals,inertia)
plt.scatter(k_vals,inertia)
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()

#Muestra muchos graficos con diferentes k con la imagen de los pimentones, cada k muestra mejor la imagen que un valor
#menor de k anterior
plt.figure(figsize=[10,20])
for i in range(len(k_vals)):
    plt.subplot(5,2,i+1)
    plt.imshow(img_list[i])
    plt.title('k = '+ str(k_vals[i]))
    plt.axis('off')
    plt.show()



#######################################################################################################################

                                              #Distancias y DBSCAN

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import scipy
from scipy.spatial.distance import euclidean, cityblock, cosine
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt


#Funcion para alojar diferentes promedios de distancias
def avg_distance(X1, X2, distance_func):
    from sklearn.metrics import jaccard_score
    #print(distance_func)
    res = 0
    for x1 in X1:
        for x2 in X2:
            if distance_func == jaccard_score: # the jaccard_score function only returns jaccard_similarity
                res += 1 - distance_func(x1, x2)
            else:
                res += distance_func(x1, x2)
    return res / (len(X1) * len(X2))

#Funcion que muestra la distancia entre pares como x1 con su y1
def avg_pairwise_distance(X1, X2, distance_func):
    return sum(map(distance_func, X1, X2)) / min(len(X1), len(X2))

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/iris.csv')
df.head()

df.drop(['petal_width'], axis=1, inplace=True)
df.head()

species = df['species'].unique()
print(species)

#Una grafico 3D hecho con las medidas de los petalos, los separa en colores segun la especie
attrs = ['sepal_length', 'sepal_width', 'petal_length']
markers = ['o', 'v', '^']
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for specie, marker in zip(species, markers):
    specie_data = df.loc[df['species'] == specie][attrs]
    xs, ys, zs = [specie_data[attr] for attr in attrs]
    ax.scatter(xs, ys, zs, marker=marker)
plt.show()

setosa_data = df.loc[df['species'] == 'setosa'][attrs].to_numpy()
versicolor_data = df.loc[df['species'] == 'versicolor'][attrs].to_numpy()
virginica_data = df.loc[df['species'] == 'virginica'][attrs].to_numpy()

print(setosa_data.shape)

#Para ver la distancia euclidiana
print(euclidean([0, 0], [3, 4]))

#Saca la distancia promedio entre los dos conglomerados, calculada dato a dato, segun la distancia euclidiana
print(avg_distance(setosa_data, versicolor_data, euclidean))

print(avg_distance(setosa_data, virginica_data, euclidean))

#Entre pares se le llama a punto por punto la distancia
from sklearn.metrics.pairwise import paired_euclidean_distances

X = np.array([[0, 0]], dtype=float)
Y = np.array([[3, 4]], dtype=float)
print(paired_euclidean_distances(X, Y).mean())

print(avg_pairwise_distance(X, Y, euclidean))

M, N = setosa_data.shape
print(f'{M} points and each column is {N} dimensions')

#Distancia entre cada fila
row_dist=paired_euclidean_distances(setosa_data, versicolor_data)
print(row_dist)

print(row_dist.mean())

#Se puede calcular con esta funcion tambien, fueron separados anteriormente
print(paired_euclidean_distances(setosa_data, virginica_data).mean())

print(avg_pairwise_distance(setosa_data, virginica_data, euclidean))


###Distancia de Manhatan

#Se calcula con esta funcion
#a=la distancia es 0, b=la distancia entre los valores es 4, b-a=4
print(cityblock([1, 1], [-2, 2]))

print(avg_distance(setosa_data, setosa_data, cityblock))

print(avg_distance(setosa_data, versicolor_data, cityblock))

print(avg_distance(setosa_data, virginica_data, cityblock))

#Para determiar la distancia entre pares con manhattan
from sklearn.metrics.pairwise import manhattan_distances

X = np.array([[1, 1]])

Y = np.array([[-2, 2]])

print(manhattan_distances(X, Y))


###Distancia coseno

#Para sacar la distancia coseno, se saca con trigonometria se calcula con el angulo
print(cosine([1, 1], [-1, -1]))

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/auto-mpg.data', header=None, delim_whitespace=True, names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])
print(df.head())

df['car_name'] = df['car_name'].str.split(n=1).apply(lambda lst: lst[0]).replace('chevrolet', 'chevy')
df.rename(columns={'car_name': 'make'}, inplace=True)
#Se toman solo ciertas columnas
df = df[['mpg', 'weight', 'make']]
print(df.head())

#Se normaliza la data
dfn = df[['mpg', 'weight']]
df[['mpg', 'weight']] = (dfn-dfn.min())/(dfn.max()-dfn.min())
print(df.head())

chevy = df.loc[df['make'] == 'chevy']
honda = df.loc[df['make'] == 'honda']

plt.scatter(chevy['mpg'], chevy['weight'], marker='o', label='chevy')
plt.scatter(honda['mpg'], honda['weight'], marker='^', label='honda')
plt.xlabel('mpg')
plt.ylabel('weight')
plt.legend()
plt.show()

chevy_data = chevy[['mpg', 'weight']].to_numpy()
honda_data = honda[['mpg', 'weight']].to_numpy()

print(avg_distance(chevy_data, chevy_data, cosine))

print(avg_distance(honda_data, honda_data, cosine))

print(avg_distance(honda_data, chevy_data, cosine))

#Tambien para calcularla en pares
from sklearn.metrics.pairwise import cosine_distances

X = np.array([[1, 1]])
Y = np.array([[-1, -1]])
print(cosine_distances(X, Y))

#Tambien se puede calcular asi
#cosine_distance = 1 - cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
print(1-cosine_similarity(X,Y))

print(cosine_distances(chevy_data, chevy_data).mean())

print(cosine_distances(honda_data, chevy_data).mean())


### Ocupar distancias con DBSCAN con algoritmos de agrupamiento

from sklearn.cluster import DBSCAN
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/data/synthetic_clustering.csv')
print(df.head())

#Es un data que tiene pequeños grupos juntos
plt.scatter(df['x'], df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Usando DBSCAN euclidiano, los pone de colores ya que logra decifrar su distancia y que estan cerca uno de otro
#Apesar de eso tiene problemas con los outliers
dbscan = DBSCAN(eps=0.1, metric=euclidean)
dbscan.fit(df)
colors = np.random.random(size=3*(dbscan.labels_.max()+1)).reshape(-1, 3)
plt.scatter(df['x'], df['y'], c=[colors[l] for l in dbscan.labels_])
plt.show()

#Utilizando DBSCAN manhattan, tiene menos precision que la euclidiana en distancia
dbscan = DBSCAN(eps=0.1, metric=cityblock)
dbscan.fit(df)
colors = np.random.random(size=3*(dbscan.labels_.max()+1)).reshape(-1, 3)
plt.scatter(df['x'], df['y'], c=[colors[l] for l in dbscan.labels_])
plt.show()

#Utilizando DBSCAN coseno, es completamente diferente, porque secciona los conglomerados por colores dependiendo si se
#ubican en cierto angulo de todos los datos
dbscan = DBSCAN(eps=0.1, metric=cosine)
dbscan.fit(df)
colors = np.random.random(size=3*(dbscan.labels_.max()+1)).reshape(-1, 3)
plt.scatter(df['x'], df['y'], c=[colors[l] for l in dbscan.labels_])
plt.show()

###Con la distancia jaccard

from sklearn.metrics import jaccard_score

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/breast-cancer.data', header=None, names=['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])
print(df.head())

print(sorted(df['age'].unique()))
print(df.age.value_counts())

#Para calcular la distancia jaccard se deben codificar las columnas categoricas como edad, edad en este caso es
#categorica, ayuda a calcular la distancia
from sklearn.preprocessing import OneHotEncoder
OH = OneHotEncoder()

X = OH.fit_transform(df.loc[:, df.columns != 'age']).toarray()
print(X)
print(f"By using onehot encoding, we obtained a 2d array with shape {X.shape} that only has value 0 and 1 ")

X30to39 = X[df[df.age == '30-39'].index]
X60to69 = X[df[df.age == '60-69'].index]

print(X30to39.shape), print(X60to69.shape)

#Promedio de distancia entre este rango de 30 a 39
print(avg_distance(X30to39, X30to39, jaccard_score))

#Promedio de distancia entre este rango de 60 a 69
print(avg_distance(X60to69, X60to69, jaccard_score))

#Promedio de distancia entre este rango de 30 a 39 con 60 a 69
print(avg_distance(X30to39, X60to69, jaccard_score))

### Ejemplo de jaccard

#Con distancia jaccard

print(" ")
#Solo hay una distancia pequeña solo de una palabra
sentence1 = 'Hello everyone and welcome to distance metrics'
sentence2 = 'Hello world and welcome to distance metrics'

s1set = set(sentence1.split())
s2set = set(sentence2.split())
ans = len(s1set.intersection(s2set)) / len(s1set.union(s2set))

print(ans)

#Con distancia euclidiana y manhatan

p1 = np.array([4, -3, 1])
p2 = np.array([-5, 1, -7])

import scipy.special
euclidean = scipy.spatial.distance.euclidean(p1, p2)
manhattan = scipy.spatial.distance.cityblock(p1, p2)
ans = abs(manhattan - euclidean)

print(ans)

#Con distancia coseno

p1 = np.array([1, 2, 3]).reshape(1, -1)
p2 = np.array([-2, -4, -6]).reshape(1, -1)

ans = cosine_distances(p1, p2)

print(ans)

#Usando distancia entre pares

X1 = np.arange(8).reshape(4, 2)
X2 = np.arange(8)[::-1].reshape(4, 2)
print(f'X1:\n{X1}')
print(f'X2:\n{X2}')

paired_euclidean = sklearn.metrics.pairwise.paired_euclidean_distances(X1, X2)
paired_manhattan = sklearn.metrics.pairwise.paired_manhattan_distances(X1, X2)

print(paired_euclidean)
print(paired_manhattan)


#######################################################################################################################

                                           #Maldicion de la dimencionalidad

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

def make_circle(point=0):
    fig = plt.gcf()
    ax = fig.add_subplot(111, aspect='equal')
    fig.gca().add_artist(plt.Circle((0,0),1,alpha=.5))
    ax.scatter(0,0,s=10,color="black")
    ax.plot(np.linspace(0,1,100),np.zeros(100),color="black")
    ax.text(.4,.1,"r",size=48)
    ax.set_xlim(left=-1,right=1)
    ax.set_ylim(bottom=-1,top=1)
    plt.xlabel("Covariate A")
    plt.ylabel("Covariate B")
    plt.title("Unit Circle")
    
    if point:
        ax.text(.55,.9,"Far away",color="purple")
        ax.scatter(.85,.85,s=10,color="purple")
    else: 
        plt.show()
    
print(make_circle())

print(make_circle(1))

#Como mostraran los graficos a continuacion lo que se quiere saber es cuanto cubre la esfera y cuanto queda fuera de
#el en un cubo como muestra el grafico a continuacion

from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

# Create figure 
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
#ax.set_aspect("equal")

# Draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r,r,r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s,e))

# Draw sphere on same axis 
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x=np.cos(u)*np.sin(v)
y=np.sin(u)*np.sin(v)
z=np.cos(v)
ax.plot_wireframe(x, y, z, color="black")
plt.show()

#Gracias a las formulas podemos saber que un 48% del volumen del cubo no esta cubrido

#Pero la gracia es saber si se puede hacer esto con mas dimenciones

# Draw a sample of data in two dimensions
sample_data = np.random.sample((5,2))
print("Sample data:\n", sample_data, '\n')

def norm(x): 
    ''' Measure the distance of each point from the origin.
    
    Input: Sample points, one point per row
    Output: The distance from the origin to each point
    '''
    return np.sqrt( (x**2).sum(1) ) # np.sum() sums an array over a given axis 

def in_the_ball(x): 
    ''' Determine if the sample is in the circle. 
    
    Input: Sample points, one point per row
    Output: A boolean array indicating whether the point is in the ball
    '''
    return norm(x) < 1 # If the distance measure above is <1, we're inside the ball


for x, y in zip(norm(sample_data),in_the_ball(sample_data)):
    print("Norm = ", x.round(2), "; is in circle? ", y)


def what_percent_of_the_ncube_is_in_the_nball(d_dim,
                                              sample_size=10**4):
    shape = sample_size,d_dim
    data = np.array([in_the_ball(np.random.sample(shape)).mean()
                     for iteration in range(100)])
    return data.mean()

dims = range(2,15)
data = np.array(list(map(what_percent_of_the_ncube_is_in_the_nball,dims)))


for dim, percent in zip(dims,data):
    print("Dimension = ", dim, "; percent in ball = ", percent)


plt.plot(dims, data, color='blue')
plt.xlabel("# dimensions")
plt.ylabel("% of area in sphere")
plt.title("What percentage of the cube is in the sphere?")
plt.show()

#Estos graficos muestran que el porcentaje de volumen que cubre una "esfera" es cada vez menor con las dimenciones

def get_min_distance(dimension, sample_size=10**3):
    ''' Sample some random points and find the closet 
    of those random points to the center of the data '''
    points = np.random.sample((sample_size,dimension))-.5   # centering our data
    return np.min(norm(points))

def estimate_closest(dimension):
    ''' For a given dimension, take a random sample in that dimension and then find 
        that sample's closest point to the center of the data. 
        Repeat 100 times for the given dimension and return the min/max/mean 
        of the distance for the nearest point. '''
    data = np.array([get_min_distance(dimension) for _ in range(100)])
    return data.mean(), data.min(), data.max()

# Calculate for dimensions 2-100
dims = range(2,100)
min_distance_data = np.array(list(map(estimate_closest,dims)))

# Test it for dimension 6
print("For dimension 6: ", estimate_closest(6))


plt.plot(dims,min_distance_data[:,0], color='blue')
plt.fill_between(dims, min_distance_data[:,1], min_distance_data[:,2],alpha=.5)
plt.xlabel("# dimensions")
plt.ylabel("Estimated min distance from origin")
plt.title("How far away from the origin is the closest point?")
plt.show()

### PCA soluciona la maldicion de dimensionalidad

#IMPORTANTE los datos no deben estar tratados

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

#make_classification crea una data automatica, con n_features para X, n_cluster debe tener la misma cantidad de clases
#por defecto son dos clases
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=2)
                           
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

DT =  DecisionTreeClassifier()
DT.fit(X_train, y_train)
score = DT.score(X_test, y_test)

print("Score from two-feature classifier: ", score)

X, y = make_classification(n_features=200, n_redundant=0, n_informative=200,
                           random_state=1, n_clusters_per_class=2)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

DT =  DecisionTreeClassifier()
DT.fit(X_train, y_train)
score = DT.score(X_test, y_test)

print("Score from 200-feature classifier: ", score)

scores = []

increment, max_features = 50, 4000

for num in np.linspace(increment, max_features, increment, dtype='int'):

    X, y = make_classification(n_features=num, n_redundant=0, 
                               random_state=1, n_clusters_per_class=1, n_classes = 3)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

    
    DT =  DecisionTreeClassifier()
    DT.fit(X_train, y_train)
    scores.append( DT.score(X_test, y_test) )

#Muestra que la puntuacion oscila dependiendo de la cantidad de features
plt.plot(np.linspace(increment, max_features, increment, dtype='int'),scores)
plt.title("Accuracy of Classification with Increasing Features")
plt.xlabel("Number of features")
plt.ylabel("Classification accuracy")
plt.show()

#En general suele bajar la puntuacion debido a que muchas columnas suelen ser redundantes




#######################################################################################################################

                                          #DBSCAN Y TSNE (imagenes)

#DBSCAN funciona igual que Kmeans, mientos que Kmeans necesita que se le especifique la cantidad de conglomerados
#DBSCAN necesita parametros de distancia y de epsilon, lo que logra mayor argumento matematico que Kmeans

#TSNE es un reductor de dimencionalidad, reconoce ciertos closter y los divide completamente los datos unos de otros

#PCA es similar a TSNE, pero PCA es bueno para representar datos cuando tienen en general estructuras lineales o que
#sean mas achatados en una direccion ejemplo distribuidos mas datos en y que en x en los ejes. Si es circular no es buena
#interpretacion, por lo que se pide que tengan cierta correlacion

#En cambio TSNE mantiene relacion de distancia entre datos, por lo que no es lineal, util para datos por ejemplo que
#tienen forma de conglomerados circulares o irregulares, etc, cuando los datos tienen una alta dimencion probablemente
#sea mejor ocupar TSNE porque abarca mejor datos irregulares

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import string

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

import seaborn as sns

sns.set_context('notebook')
sns.set_style('white')

import matplotlib.pyplot as plt


df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/data/example1.csv')

#Son solo dos columnas con valores, se puede graficar 2D
print(df.head(n=6))

#Grafica varios puntos, solo para mostrarlos
#rcParams es para hacer configuraciones solamente, en este caso el tamaño de la figura
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
#Un scatter de las dos columnas
plt.scatter(df['0'], df['1'])
#Esto solo le aplica las letras a los puntos
for t, p in zip(string.ascii_uppercase, df.iterrows()):
    plt.annotate(t, (p[1][0] + 0.2, p[1][1]))
plt.show()

#DBSCAN con epsilon=3 y min_samples=4, sin distancia especificada
cluster = DBSCAN(eps=3, min_samples=4)
cluster.fit(df)

#Dice que encuentra un cluster y ruido
#Se puede preguntar la existencia de esta forma
print(f'DBSCAN found {len(set(cluster.labels_) - set([-1]))} clusters and {(cluster.labels_ == -1).sum()} points of noise.')

#Todos los datos excepto uno, pertenecen al cluster, pero uno que es un outliers se le llama ruido
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
plt.scatter(df['0'], df['1'], c=[['blue', 'red'][l] for l in cluster.labels_])
plt.scatter(0, 0, c='blue', alpha=0.2, s=90000)
plt.scatter(6, 0, c='red', alpha=0.2, s=9000)
for t, p in zip(string.ascii_uppercase, df.iterrows()):
    plt.annotate(t, (p[1][0] + 0.2, p[1][1]))
plt.show()


### Ejemplo 2 DBSCAN

#Muestra comparaciones de letras escritas a manos, por mi y mi amigo, queremos saber que tanto se diferencian
#La data al parecer son 3 imagenes aplastadas los datos de matricez en total son 64 datos
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/data/012.csv')
print(df.head())
print(df.shape)

#Se muestran imagenes 3 de los numeros de mia amigo en la data, letra por letra las que hizo mi amigo
friend_digits = df.iloc[:, df.columns != 'y'].to_numpy()
plt.rcParams['figure.figsize'] = (8,6)
#la imagenenes aplastadas las convierte en matricez
it = (x.reshape(8, 8) for x in friend_digits)
c = 3
#subplot dice que mostrara una fila de graficos con 3 columnas
fig, ax = plt.subplots(1, c, sharex='col', sharey='row')
for j in range(c):
    ax[j].axis('off')
    ax[j].set_title(f'Sample of friend\'s number {j}')
    ax[j].imshow(next(it))
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

#Aqui carga una nueva data con 1756 imagenes de numeros + y, son imagenes aplastadas con 64 datos
digits, y = load_digits(return_X_y=True)
print(pd.DataFrame(digits).shape)

#Ahora muestra digitos, del 0 al 9 y otros del 1 al 4
plt.rcParams['figure.figsize'] = (8,6)
#Convierte las imagenes aplastadas en matricez
it = (x.reshape(8, 8) for x in digits)
r, c = 3, 5
#Sub plot dice que mostrara 3 filas con graficos con 5 columnas con graficos en cada espacio
fig, ax = plt.subplots(r, c, sharex='col', sharey='row')
for i in range(r):
    for j in range(c):
        ax[i, j].axis('off')
        ax[i, j].imshow(next(it))
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

#Lo que hace es exportar una data de 1800 imagenes, cada una esta aplastada con 64 datos

#np.r_ es para crear una matriz rapidamente con el dataframe, friend digist son las labels finales el numero de la imagen
data = np.r_[digits, friend_digits]

print(data)
print(data.shape)

#Crea una matriz aparte con el y inicial
y = np.r_[y, df['y']]

print(y.shape)

#Se pone un modelo de TSNE en embedding
embedding = TSNE(n_components=2,
        init="pca",
        n_iter=500,
        n_iter_without_progress=150,
        perplexity=10,
        random_state=0)

#Todas las imagenes de la data son transformadas solo a dos columnas
e_data = embedding.fit_transform(data)

plt.rcParams['figure.figsize'] = (20,15)
n = friend_digits.shape[0]
print(e_data)

#La data cada fila, representa la misma imagen, pero en menos datos

#Primero se grafica la data con TSNE, menos las 3 imagenes del amigo que son las ultimas 3, n=3
plt.scatter(
    e_data[:-n, 0],
    e_data[:-n, 1],
    marker='o',
    alpha=0.75,
    label='mnist data',
    s=100)

#Grafica solo las ultimas 3 filas que son las imagenes de el amigo
plt.scatter(
    e_data[-n:, 0],
    e_data[-n:, 1],
    marker='x',
    color='black',
    label='friend\'s data',
    alpha=1,
    s=100)
plt.legend(bbox_to_anchor=[1, 1])
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

#Ahora con DBSCAN creamos conglomerados, recordar que no se necesita especificar cuantos
cluster = DBSCAN(eps=5, min_samples=20)
cluster.fit(e_data)
#Dice cuantos numeros y ruido encontro
print(f'DBSCAN found {len(set(cluster.labels_) - set([-1]))} clusters and {(cluster.labels_ == -1).sum()} points of noise.')

plt.rcParams['figure.figsize'] = (20,15)
#Obtenemos diferentes labels para los conglomerados con DBSCAN
unique_labels = set(cluster.labels_)
n_labels = len(unique_labels)
cmap = plt.cm.get_cmap('brg', n_labels)
#Esto es para etiquetar cada conglomerado en la imagen con su respectivo nombre
for l in unique_labels:
    plt.scatter(
        e_data[cluster.labels_ == l, 0],
        e_data[cluster.labels_ == l, 1],
        c=[cmap(l) if l >= 0 else 'Black'],
        marker='ov'[l%2],
        alpha=0.75,
        s=100,
        label=f'Cluster {l}' if l >= 0 else 'Noise')
plt.legend(bbox_to_anchor=[1, 1])
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

print("The predicted labels of our friend's handwriting:")
print(cluster.labels_[-3:])

#Por ultimo graficos con una fila y 5 columnas, pero creando un grafico para cada labels anterior
#coincidentemente DBSCAN agrupo las imagenes que tenian caracteristicas similares
r, c = 1, 5
plt.rcParams['figure.figsize'] = (4*r,4*c)
for label in unique_labels:
    cluster_data = data[cluster.labels_ == label]
    nums = cluster_data[np.random.choice(len(cluster_data), r * c, replace=False)]
    it = (x.reshape(8, 8) for x in nums)
    fig, ax = plt.subplots(r, c)
    ax = ax.reshape(r, c)
    plt.subplots_adjust(wspace=0.1, hspace=-0.69)
    fig.suptitle(f'Original data from cluster {label}', fontsize=20, y=0.545)
    for i in range(r):
        for j in range(c):
            ax[i, j].axis('off')
            ax[i, j].imshow(next(it))
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

print('Correct labels:')
plt.rcParams['figure.figsize'] = (20,15)

#Grafica nuevamente el grafico anterior, pero va itinerando marcas para que se entienda mejor a cual pertenecen de los labels
unique_labels = set(y)
n_labels = len(unique_labels)
cmap = plt.cm.get_cmap('brg', n_labels)
for l in unique_labels:
    plt.scatter(
        e_data[y == l, 0],
        e_data[y == l, 1],
        c=[cmap(l)],
        marker=f'${l}$',
        alpha=1,
        label=f'{l}',
        s=100)
plt.legend(bbox_to_anchor=[1, 1])
plt.show()

#Para decir que la mayoria de los datos de los conglomerados, se ordenaron correctamente
for i, (l, t) in enumerate(zip(cluster.labels_[-3:], y[-3:])):
    print('-' * 30)
    print(f'Your friend\'s {i}th sample was categorized as being in cluster #{l}')
    if l == -1:
        print('(IE: Noise)')
    else:
        v, c = np.unique(y[cluster.labels_ == l], return_counts=True)
        mfreq = v[np.argmax(c)]
        ratio = c.max() / c.sum()
        print(f'Cluster {l} is {ratio * 100:.2f}% the number {mfreq}')
        
    print(f'Your friend\'s {i}th sample is supposed to be the number {t}')

#Aunque los datos de nuestro amigo, se clasificaron como ruido, enrealidad las imagenes eran pesimas, asi que
#el modelo es correcto


### Ejemplo 3

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/data/DBSCAN_exercises.csv')
print(df.head())

#Muestra un grafico de puntos solamente
plt.scatter(df['x'], df['y'])
plt.show()

#Aplica DBSCAN para formar conglomerados
cluster = DBSCAN(eps=2, min_samples=16, metric="euclidean")
cluster.fit(df)

#Reconoce la cantidad de conglomerados y muestra los punts del centro
print(len(set(cluster.labels_) - {1}))

#Muestra el porcentaje de ruido
print(f'{100 * (cluster.labels_ == -1).sum() / len(cluster.labels_)}%')

plt.rcParams['figure.figsize'] = (20,15)
unique_labels = set(cluster.labels_)
n_labels = len(unique_labels)
cmap = plt.cm.get_cmap('brg', n_labels)
for l in unique_labels:
    plt.scatter(
        df['x'][cluster.labels_ == l],
        df['y'][cluster.labels_ == l],
        c=[cmap(l)],
        marker='ov'[l%2],
        alpha=0.75,
        s=100,
        label=f'Cluster {l}' if l >= 0 else 'Noise')
plt.legend(bbox_to_anchor=[1, 1])
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

#No siempre quedan los datos tan definidos para predecir a que dato pertenecen



#######################################################################################################################

                                            #Mean Shift Clustering



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from mpl_toolkits import mplot3d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import cv2 as cv
import skillsnetwork 
from skimage import io

img = io.imread("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/peppers.jpeg")

print(img.shape)

#img = cv.imread('peppers.jpeg')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()

#La manera para smotear (suavizado) una imagen
#No etendi porque es 7, pero al parecer es una formula detro de medianBlur
img = cv.medianBlur(img, 7)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()

#Es como un grafico de analisis de cada color, Red, Green, Blue siedo x, y, z
ax = plt.axes(projection ="3d")
ax.scatter3D(img[:,:,0],img[:,:,1],img[:,:,2])
ax.set_title('Pixel Values ')
plt.show()

print(img.shape)

#Convertimos la imagen en una fila, es una multiplicacion 194 * 259 por cada canal en este caso 3
#Se transforma en una matriz de 50246, 3, siendo las 3 columnas teniendo los mismos valores

X = img.reshape((-1,3))
print("shape: ",X.shape)
print("data type   : ",X.dtype)

print(X)

X = np.float32(X)

#Para ocupar mean shift hay que estimar bandwidth, toma solo una muestra, n_samples=la cantidad de filas que tomara
#n_features la cantidad de columnas, en este caso las 3 porque no se especifica, quantile, es una valor entre 0 y 1
#la mediana es 0.5 cuando se toma todas las muestras, pero en este caso como se toman 3000, se debe calcular
#3000/50246 = 0.06 aprox
bandwidth = estimate_bandwidth(X, quantile=.06, n_samples=3000)
print(bandwidth)

#Ocupamos MeanShift
ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
ms.fit(X)

labeled=ms.labels_
#Son los 50246
print(labeled.shape)

#Son 12 conglomerdos de 0 a 11
print(np.unique(labeled))

clusters=ms.predict(X)

#Simplemente es el tamaño de los datos 50246
print(clusters.shape)

#Son las cordenadas de los centros de los conglomerados
print(ms.cluster_centers_)

#Solo la parte entera de los cordenadas anteirores
cluster_int8=np.uint8(ms.cluster_centers_)
print(cluster_int8)

ms.predict(X)

#Grafica los puntos, igual que lo anteior no los define como colores
ax = plt.axes(projection ="3d")
ax.scatter3D(img[:,:,0],img[:,:,1],img[:,:,2])
ax.set_title('Pixel Values ')
plt.show()

#Grafica solo los centros de los conglomerados
ax = plt.axes(projection ="3d")
ax.set_title('Pixel Cluster Values  ')
ax.scatter3D(cluster_int8[:,0],cluster_int8[:,1],cluster_int8[:,2],color='red')
plt.show()

#Crea otra matriz con la misma cantidad de datos que X con solo 0
result=np.zeros(X.shape,dtype=np.uint8)

#Cluster_int8 son los centros de los conglomerados son 12, que los itinera con filas
#result es una matriz de 0, Si son iguales los conglomerados los rellena toda la imagen con solo los centros
for label in np.unique(labeled):
    result[labeled==label,:]=cluster_int8[label,:]   



#Todos los valores itinerados, los junta en una sola matriz
#Tiene solo los colores de los centros de los conglomerados
#Recordando que los conglomerados son los colores mas fuertes, los muestra en una sola imagen
result=result.reshape(img.shape)
print(result)

#Grafica solo los colores centrales
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()

#Muestra en la imagen cada conglomerado, cada conglomerado es una capa de la imagen inicial
#Rehace deuevo la matriz, recordando que los conglomerados son definidos por el modelo, con ciertos colores
#Cada conglomerado es un color diferente en la imagen por eso los junta, esto puede servir para reconocer ciertos objetos
#de cierto color, en el ejemplo separa cada vegetal por los colores
for label in np.unique(labeled):
    result=np.zeros(X.shape,dtype=np.uint8)
    result[labeled==label,:]=cluster_int8[label,:]  
    plt.imshow(cv.cvtColor(result.reshape(img.shape), cv.COLOR_BGR2RGB))
    plt.show()

### Ejemplo 2


#Muestra un problema que en las imagennes con un color predominante, no diferencia casi ningun objeto
import requests 
url='https://www.plastform.ca/wp-content/themes/plastform/images/slider-image-2.jpg'
name="my_file.jpg"

with open(name, 'wb') as file:
    file.write(requests.get(url, stream=True).content)
    
img = cv.imread(name)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()

img = cv.medianBlur(img, 7)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()

X = img.reshape((-1,3))
X = np.float32(X)
bandwidth = estimate_bandwidth(X, quantile=.06, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
ms.fit(X)
labeled=ms.labels_
cluster_int8=np.uint8(ms.cluster_centers_)
result=np.zeros(X.shape,dtype=np.uint8)
labeled=ms.labels_

for label in np.unique(labeled):
    result[labeled==label,:]=cluster_int8[label,:]        
    print(cluster_int8[label])
    
result=result.reshape(img.shape)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()

for label in np.unique(labeled):
    result=np.zeros(X.shape,dtype=np.uint8)
    result[labeled==label,:]=cluster_int8[label,:]  
    plt.imshow(cv.cvtColor(result.reshape(img.shape), cv.COLOR_BGR2RGB))
    plt.show()

#Ejercicio 3

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/titanic.csv")
print(df.head())

#Elimina varias columnas
df=df.drop(columns=['Name','Ticket','Cabin','PassengerId','Embarked'])

#Define el valor para mujeres y hombres los codifica
df.loc[df['Sex']!='male','Sex']=0
df.loc[df['Sex']=='male','Sex']=1

print(df.head())

#Dice la cantidad de numeros vacios
print(df.isna().sum())

#Remplaza las na con promedios
print(df['Age'].fillna(df['Age'].mean(),inplace=True))

#Elimina la columas sobrevivientes
X=df.drop(columns=['Survived'])

#Aplica una funcion a todos los datos
X=df.apply(lambda x: (x-x.mean())/(x.std()+0.0000001), axis=0)

print(X.head())

bandwidth = estimate_bandwidth(X)
ms = MeanShift(bandwidth=bandwidth , bin_seeding=True)
ms.fit(X)

X['cluster']=ms.labels_
df['cluster']=ms.labels_

#Los agrupa por los cluster y promedia cada valor para cada cluster, los ordena por la columna sobreviviente
#ojo que sobrevivientes se habia eliminado para X, pero no para df, eso solo fue para entrenarse
print(df.groupby('cluster').mean().sort_values(by=['Survived'], ascending=False))

#Los cluster los hace por caracterisitcas similares, en esto se puede ver que los que tienen mas dinero tienen
#Mas probabilidad de sobrevivir coincidentemente

### Ejemplo 4 Como funciona MeanShift

#En pocas palabras, el algoritmo busca los centros con gaussianas

def gaussian(d, h):
    return np.exp(-0.5*((d/h))**2) / (h*math.sqrt(2*math.pi))

s=1 # a sample point

x = np.linspace(-2, 4, num=200)
dist=np.sqrt(((x-s)**2))
kernel_1=gaussian(dist, 1)
kernel_2=gaussian(dist, 3)

plt.plot(x,kernel_1,label='h=1')
plt.plot(x,kernel_2,label='h=3')
plt.plot(s,0,'x',label="$x_{1}$=1")
plt.hist(s, 10, facecolor='blue', alpha=0.5,label="Histogram")
plt.xlabel('x')
plt.legend()
plt.show()

def kernel_density(S,x,h=1):

    density=np.zeros((200))
    for s in S:
        #Determine the distance and kernel for each point 
        dist=np.sqrt(((x-s)**2))
        kernel=gaussian(dist, h)
        #Find the sum  
        density+=kernel
    #Normalize the sum  
    density=density/density.sum() 
    
    return density

S=np.zeros((200))
S[0:100] = np.random.normal(-10, 1, 100)
S[100:200]=np.random.normal(10, 1, 100)
plt.plot(S,np.zeros((200)),'x')
plt.xlabel("$x_{i}$")
plt.show()

x = np.linspace(S.min()-3, S.max()+3, num=200)
density=kernel_density(S,x)

plt.plot(x,density,label=" KDE")
plt.plot(S,np.zeros((200,1)),'x',label="$x_{i}$")
plt.xlabel('x')
plt.legend()
plt.show()

mean_shift=((density.reshape(-1,1)*S).sum(0) / density.sum())-x

plt.plot(x,density,label=" KDE")
plt.plot(S,np.zeros((200,1)),'x',label="$x_{i}$")
plt.quiver(x, np.zeros((200,1)),mean_shift, np.zeros((200,1)), units='width',label="$m_{h}(x)$")
plt.xlabel('x')
plt.legend()
plt.show()

Xhat=np.copy(S.reshape(-1,1))
S_=S.reshape(-1,1)


for k in range(3):
    plt.plot(x,density,label=" KDE")
    plt.plot(Xhat,np.zeros((200,1)),'x',label="$\hat{x}^{k}_i$,k="+str(k))
    plt.xlabel('x')
    plt.legend()
    plt.show()
  
    for i,xhat in enumerate(Xhat):
        dist=np.sqrt(((xhat-S_)**2).sum(1))
        weight = gaussian(dist, 2.5)
        Xhat[i] = (weight.reshape(-1,1)*S_).sum(0) / weight.sum()

print(np.unique(Xhat.astype(int)))



#######################################################################################################################

                                        #Ejemplos de clustering

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Es la data de los vinos: su color y contenido
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/Wine_Quality_Data.csv")

print(data.head(4).T)

print(data.shape)

print(data.dtypes)

print(data.color.value_counts())

print(data.quality.value_counts().sort_index())

# seaborn styles
sns.set_context('notebook')
sns.set_style('white')

# custom colors
red = sns.color_palette()[2]
white = sns.color_palette()[4]

# set bins for histogram
bin_range = np.array([3, 4, 5, 6, 7, 8, 9])

#Crea los dos histogramas, pero los plotea mas abajo, sirve en caso de tener varias clases esta forma de graficar
ax = plt.axes()
for color, plot_color in zip(['red', 'white'], [red, white]):
    q_data = data.loc[data.color==color, 'quality']
    q_data.hist(bins=bin_range, alpha=0.5, ax=ax, color=plot_color, label=color)


ax.legend()
ax.set(xlabel='Quality', ylabel='Occurrence')

ax.set_xlim(3,10)
ax.set_xticks(bin_range+0.5)
ax.set_xticklabels(bin_range);
ax.grid('off')
plt.show()


### BEGIN SOLUTION
float_columns = [x for x in data.columns if x not in ['color', 'quality']]

# The correlation matrix
corr_mat = data[float_columns].corr()

# Strip out the diagonal values for the next step
for x in range(len(float_columns)):
    corr_mat.iloc[x,x] = 0.0
    
print(corr_mat)

# Pairwise maximal correlations
print(corr_mat.abs().idxmax())

skew_columns = (data[float_columns]
                .skew()
                .sort_values(ascending=False))

skew_columns = skew_columns.loc[skew_columns > 0.75]
print(skew_columns)

# Perform log transform on skewed columns
for col in skew_columns.index.tolist():
    data[col] = np.log1p(data[col])

#Escalamos los datos de todas las coumnas que tienen valores numericos
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
data[float_columns] = sc.fit_transform(data[float_columns])

print(data.head(4))

#Hay poca relacion entre las columnas, so bastante irregulares
#sns.set_context('notebook')
#sns.pairplot(data[float_columns + ['color']], 
#             hue='color', 
#             hue_order=['white', 'red'],
#             palette={'red':red, 'white':'gray'})
#plt.show()

#Hacemos un modelo de kmeans para 2 cluster
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, random_state=42)
km = km.fit(data[float_columns])

#Esto crea una columna adicional con a que conglomerado pertenecen
data['kmeans'] = km.predict(data[float_columns])

#Mostramos el conglomerado y agrupamos para color y conglomerado
print(data[['color','kmeans']]
 .groupby(['kmeans','color'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))


#Probando diferentes cantidad de cluster para un modelo
km_list = list()

for clust in range(1,21):
    km = KMeans(n_clusters=clust, random_state=42)
    km = km.fit(data[float_columns])
    
    km_list.append(pd.Series({'clusters': clust, 
                              'inertia': km.inertia_,
                              'model': km}))

plot_data = (pd.concat(km_list, axis=1)
             .T
             [['clusters','inertia']]
             .set_index('clusters'))

#Para ver un grafico de las inercias, de cada numero de cluster
ax = plot_data.plot(marker='o',ls='-')
ax.set_xticks(range(0,21,2))
ax.set_xlim(0,21)
ax.set(xlabel='Cluster', ylabel='Inertia')
plt.show()

#No entiendo lo que hace AgglomerativeClustering, pero los conglomerados lo toma al reves que Kmeans
from sklearn.cluster import AgglomerativeClustering

ag = AgglomerativeClustering(n_clusters=2, linkage='ward', compute_full_tree=True)
ag = ag.fit(data[float_columns])
data['agglom'] = ag.fit_predict(data[float_columns])

print((data[['color','agglom','kmeans']]
 .groupby(['color','agglom'])
 .size()
 .to_frame()
 .rename(columns={0:'number'})))

#Comparando con Kmean
print(data[['color','agglom','kmeans']]
 .groupby(['color','kmeans'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))

print(data[['color','agglom','kmeans']]
 .groupby(['color','agglom','kmeans'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))

# First, we import the cluster hierarchy module from SciPy (described above) to obtain the linkage and dendrogram functions.
from scipy.cluster import hierarchy

Z = hierarchy.linkage(ag.children_, method='ward')

fig, ax = plt.subplots(figsize=(15,5))

#Muestra un grafico gerarquico
den = hierarchy.dendrogram(Z, orientation='top', 
                           p=30, truncate_mode='lastp',
                           show_leaf_counts=True, ax=ax)

plt.show()

###Usando RandomForest con Kmeans

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

#Tomamos y solo con buena calidad
y = (data['quality'] > 7).astype(int)
#Eliminamos las columnas que no sirven con Kmean
X_with_kmeans = data.drop(['agglom', 'color', 'quality'], axis=1)
#Solo eliminamo la prediccion de Kmeans
X_without_kmeans = X_with_kmeans.drop('kmeans', axis=1)
sss = StratifiedShuffleSplit(n_splits=10, random_state=6532)

#Una funcion que separa en datos de entrenamiento y prueba
#Ademas fitea y saca las putuaciones
def get_avg_roc_10splits(estimator, X, y):
    roc_auc_list = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        estimator.fit(X_train, y_train)
        y_predicted = estimator.predict(X_test)
        y_scored = estimator.predict_proba(X_test)[:, 1]
        roc_auc_list.append(roc_auc_score(y_test, y_scored))
    return np.mean(roc_auc_list)
# return classification_report(y_test, y_predicted)

#Definimos un estimador RandomForest
estimator = RandomForestClassifier()
#Aplicamos la funcion y devuelve la puntuacion, donde los datos de conglomerados adicionales en la data, hacen que varie
#un poco puede mejorar el modelo
roc_with_kmeans = get_avg_roc_10splits(estimator, X_with_kmeans, y)
roc_without_kmeans = get_avg_roc_10splits(estimator, X_without_kmeans, y)
print("Without kmeans cluster as input to Random Forest, roc-auc is \"{0}\"".format(roc_without_kmeans))
print("Using kmeans cluster as input to Random Forest, roc-auc is \"{0}\"".format(roc_with_kmeans))


from sklearn.linear_model import LogisticRegression

#Volvemos solo con los datos numericos
X_basis = data[float_columns]
sss = StratifiedShuffleSplit(n_splits=10, random_state=6532)

#Funcion que hace un modelo de Kmeans y lo fitea, saca las prediciciones y luego junta todo con predicciones incluidas
def create_kmeans_columns(n):
    km = KMeans(n_clusters=n)
    km.fit(X_basis)
    km_col = pd.Series(km.predict(X_basis))
    km_cols = pd.get_dummies(km_col, prefix='kmeans_cluster')
    return pd.concat([X_basis, km_cols], axis=1)

#Ahora con Regresion logistica
estimator = LogisticRegression()
ns = range(1, 21)
#Saca puntuaciones con este estimador, hace variar la cantidad de conglomerados en n del 1 al 21
roc_auc_list = [get_avg_roc_10splits(estimator, create_kmeans_columns(n), y)
                for n in ns]
print(roc_auc_list)

#Con mayor numero de cluster en Kmean tiende a mejorar el modelo, al aplicar otro estimador
ax = plt.axes()
ax.plot(ns, roc_auc_list)
ax.set(
    xticklabels= ns,
    xlabel='Number of clusters as features',
    ylabel='Average ROC-AUC over 10 iterations',
    title='KMeans + LogisticRegression'
)
ax.grid(True)
plt.show()


#######################################################################################################################

                                                        #PCA

#PCA puede reducir la dimencionalidad, es muy util porque se queda con la mayoria de la informacion en menos
#columnas, pero requiere algun tipo de correlacion entre los datos

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import accumulate

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from scipy.stats import loguniform

warnings.filterwarnings('ignore')

sns.set_context('notebook')
sns.set_style('white')

def plot_explained_variance(pca):
    # This function graphs the accumulated explained variance ratio for a fitted PCA object.
    acc = [*accumulate(pca.explained_variance_ratio_)]
    fig, ax = plt.subplots(1, figsize=(50, 20))
    ax.stackplot(range(pca.n_components_), acc)
    ax.scatter(range(pca.n_components_), acc, color='black')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, pca.n_components_-1)
    ax.tick_params(axis='both', labelsize=36)
    ax.set_xlabel('N Components', fontsize=48)
    ax.set_ylabel('Accumulated explained variance', fontsize=48)
    plt.tight_layout()
    plt.show()


hwdf = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/HeightsWeights.csv', index_col=0)
print(hwdf.head())

#IMPORTANTE antes de ocupar PCA, se deben escalar los datos para que tengan la misma media y la misma desviacion estandar,
#Esto dado que se puede inclinar a ciertos componentes que tengan datos mas altos, esto no sucedera y entendera mejor la
#data

scaler = StandardScaler()
hwdf[:] = scaler.fit_transform(hwdf)
hwdf.columns = [f'{c} (scaled)' for c in hwdf.columns]
print(hwdf.head())

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
xs, ys, zs = [hwdf[attr] for attr in hwdf.columns]
ax1.scatter(xs, ys, zs)

ax2 = fig.add_subplot(122, projection='3d')
xs, ys, zs = [hwdf[attr] for attr in hwdf.columns]
ax2.view_init(elev=10, azim=-10)
ax2.scatter(xs, ys, zs)

plt.tight_layout()
plt.show()

sns.pairplot(hwdf)
plt.show()

print(hwdf.corr().style.background_gradient(cmap='coolwarm'))

#Hay dos columnas peso en gramos y peso en kilogramos, es redundando, dicen la misma informacion, por lo que son columnas
#altamente correlacionadas, ayudara a mostrar lo que hace PCA

#Le aplica PCA a toda la data, al parecer por defecto son 3 columnas PCA
pca = PCA()
pca.fit(hwdf)

#Hacemos efectiva la transformacion
Xhat = pca.transform(hwdf)
print(Xhat.shape)

#hace un data frame con los nombres de las columnas cambiadas, cada columna son componentes de proyeccion
hwdf_PCA = pd.DataFrame(columns=[f'Projection on Component {i+1}' for i in range(len(hwdf.columns))], data=Xhat)
print(hwdf_PCA.head())

colors = ['red', 'red', 'green']

#Hacemos un scater con flechas de las proyecciones del PCA
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(121, projection='3d')
xs, ys, zs = [hwdf[attr] for attr in hwdf.columns]
ax1.view_init(elev=10, azim=75)
ax1.scatter(xs, ys, zs)

#ACA le agrega las flechas
for component, color in zip(pca.components_, colors):
    ax1.quiver(*[0, 0, 0], *(8 * component), color=color)

    
ax2 = fig.add_subplot(122, projection='3d')
xs, ys, zs = [hwdf[attr] for attr in hwdf.columns]
ax2.view_init(elev=0, azim=0)
ax2.scatter(xs, ys, zs)

#Le agrega las flechas
for component, color in zip(pca.components_, colors):
    ax2.quiver(*[0, 0, 0], *(8 * component), color=color)

#Plotea los mismos graficos, pero de diferentes angulos
plt.show()

#Muestra la varianza de cada uno de las proyecciones
#caso lineal, varianza=0 porque son lo mismo, caso circular, lado menor=0.2, caso cirular, varianza 0.7
for color, ev in zip(colors, pca.explained_variance_ratio_):
    print(f'{color} component accounts for {ev * 100:.2f}% of explained variance')

#Muestra un grafico de las correlaciones de PCA y son todas circulares
sns.pairplot(hwdf_PCA)
plt.show()

print(hwdf_PCA.corr().style.background_gradient(cmap='coolwarm'))

hwdf_PCA.drop('Projection on Component 3', axis=1, inplace=True)
print(hwdf_PCA.head())

###Volviendo al ejemplo

#Se puede escalar de esta forma o con PCA(whiten=True)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(hwdf), index=hwdf.index, columns=hwdf.columns)
print(X.head())

pca = PCA()
X_PCA = pd.DataFrame(pca.fit_transform(X), index=X.index, columns=[f'Component {i}' for i in range(pca.n_components_)])
# (Remember it's technically "Projection onto Component {i}")
print(X_PCA.head())

plot_explained_variance(pca)

threshold = 0.99
num = next(i for i, x in enumerate(accumulate(pca.explained_variance_ratio_), 1) if x >= threshold)
print(f'We can keep the first {num} components and discard the other {pca.n_components_-num},')
print(f'keeping >={100 * threshold}% of the explained variance!')

print(X_PCA.shape)

#Al parecer reconoce que una columna es totalmente correlaionada con otra y la elimina
X_PCA.drop([f'Component {i}' for i in range(num, pca.n_components_)], axis=1, inplace=True)
print(X_PCA.head())
print(X_PCA.shape)


#Otro ejemplo IMAGENES

#Es una base de datos de las caras de las personas

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

#Toma las dimeciones de las imagenes de las caras
N, h, w = lfw_people.images.shape
target_names = lfw_people.target_names

#h, altura de la imagen, w, ancho de la imagen, N no lo se, pero es 1288
print(N, h, w)
#Nombres de las personas son 7
print(target_names)

#y son los labels, son numeros del 0 al 6 porque son 7 personas
y = lfw_people.target

#Al parecer son ciertas capas de las imagenes, la data tiene 1288 filas que serian las caracteristicas de la imagen
#1850 es la multiplicacion de las dimencionses 50 * 37
X = lfw_people.data

n_features = X.shape[1] #Es 1850

#Muestra las imagenes para cada persona
for person in np.unique(lfw_people.target):
    idx = np.argmax(lfw_people.target == person)
    #En blanco y negro
    plt.imshow(lfw_people.images[idx], cmap='gray')
    plt.title(lfw_people.target_names[person])
    plt.show()

#Separamos entre datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

#Necesitamos dos parametros
param_grid = {
    "C": loguniform(1e3, 1e5),
    "gamma": loguniform(1e-4, 1e-1)}

#Para vectores ocupamos RandomizedSearchCV con kernel de gaussianas
clf = RandomizedSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10)
    
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#Muestra que tal le fue al model, tuvo bastantes aciertos
hmap = sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    xticklabels=lfw_people.target_names,
    yticklabels=lfw_people.target_names,
    fmt='g')
hmap.set_xlabel('Predicted Value')
hmap.set_ylabel('Truth Value')
plt.show()

#Ahora ocupamos PCA recordando que whiten es para escalar, entrenamos al instante
pca = PCA(svd_solver='full',  whiten=True).fit(X_train)

person_index=1

#Es solo una columna, que contiene toda la imagen aplanada 50 * 37
print(X[person_index].shape)

#Aplicamos pca solo para una persona, reformamos solo con 1 fila, y la ultima columna no determinada

Xhat=pca.transform(X[person_index,:].reshape(1, -1))

#Es un poco no especifico, pero lo que hace es optimizar los datos, ahora es una sola fila con reduccion de datos
#los redujo a la mitad
print(Xhat.shape)

#Imagen antes de ocupar PCA
plt.imshow(pca.inverse_transform(Xhat).reshape(h, w), cmap='gray')
plt.title("Image after PCA and inverse transform"  ) 
plt.show()

#Despues de ocupar PCA y es exactamente igual, solo tiene los componentes mas importantes
plt.imshow(lfw_people.images[person_index],cmap='gray')
plt.title("Image")
plt.show()

#Es un grafico, que muestra que la imagen con la varianza, por lo general es suficiente con 80% de aciertos para
#explicar una imagen, en este caso es 95%
plot_explained_variance(pca)
plt.show()

threshold = 0.60

#Podemos mostrar solo los componentes que tengan un mal rendimiento
components = np.cumsum(pca.explained_variance_ratio_) < threshold
print(components.sum())

#Muestra todas las imegenes de componentes con bajo rendimiento, son solo humo
for component in pca.components_[components,:]:
    plt.imshow(component.reshape(h, w),cmap='gray')
    plt.show()

#Podemos especificar la cantidad de componentes
pca = PCA(n_components=150, svd_solver="randomized", whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

person_index = 1

#Solo de la persona 1 imagen origial
plt.figure(figsize=(8, 4))
plt.subplot(1,2,1)
plt.imshow(lfw_people.images[person_index,:,:],cmap='gray')
plt.title("Original image")

#La imagen con pca inverso, se reduce bastante la calidad al hacer esta transformacion inversa
plt.subplot(1,2,2)
plt.imshow(pca.inverse_transform(pca.transform(X[person_index ,:].reshape(1, -1))).reshape(h, w),cmap='gray')
plt.title("PCA transformed and inverse-transformed image ") 

plt.tight_layout()
plt.show()

#Mostrar los mejores hiperparametros
param_grid = {
    "C": loguniform(1e3, 1e5),
    "gamma": loguniform(1e-4, 1e-1),}
    
#Ocupamos SVC como modelo
clf = RandomizedSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10)

clf = clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)

#Sacamos la matriz de confusion
hmap = sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    xticklabels=lfw_people.target_names,
    yticklabels=lfw_people.target_names,
    fmt='g')
hmap.set_xlabel('Predicted Value')
hmap.set_ylabel('Truth Value')
plt.show()

###Ejemplo 3

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/energydata_complete.csv')
df.drop('date', axis=1, inplace=True)
df = df.dropna().astype(np.float64)
print(df.head())

scaler = StandardScaler()
df[:] = scaler.fit_transform(df)
print(df.head())

pca = PCA()
pca.fit(df)

#Esta parte dice que los mejores componentes son 12 por lo que nos quedamos con 12 componentes para el siguiente modelo
print(np.argwhere(pca.explained_variance_ratio_.cumsum() >= 0.95)[0][0] + 1)

pca = PCA(n_components=12)
reduced_data = pca.fit(df).transform(df)
print(pd.DataFrame(reduced_data, columns=[f'Component {i}' for i in range(reduced_data.shape[1])]).head())



#######################################################################################################################

                                      #Singular-Value Descomposition (SVD)  
                                    #Este es SVD no SVM que era el de vectores      

#SVD es bueno para detectar objetos en movimientos en los videos, tambien se ocupa con PCA

import pandas as pd
import numpy as np 

from os import listdir,getcwd
from os.path import isfile, join
from random import randint
from PIL import Image
from io import BytesIO
import seaborn as sns 
import matplotlib.pylab as plt
import requests
from requests.adapters import HTTPAdapter, Retry
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import tarfile
import urllib.request as urllib2
from sympy import Matrix, init_printing,Symbol
from numpy.linalg import qr,eig,inv,matrix_rank,inv,svd
init_printing()

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from zipfile import ZipFile

#"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/traffic.tar.gz"

#"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/peds.tar.gz")

#"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/boats.tar.gz"


def get_data_Matrix (mypath="peds"):
    cwd = getcwd()

    mypath=join(cwd,mypath)
    files = [ join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.startswith(".")==False]
    # Read image
    img = Image.open(files[0])
    I=np.array(img)
    # Output Images

    Length,Width=I.shape
   
    X=np.zeros((len(files),Length*Width))
    for i,file in enumerate(files):
        img = Image.open(file)
        I=np.array(img)
        X[i,:]=I.reshape(1,-1)
    return X,Length,Width

#En SVD se puede utilizar cualquier matriz, debe tener valores singulares y positivos, la matriz tiene columnas
#ortonormales, se les llama vectores singulares

#En la partica una matriz cualquiera, se descompone en 3 matricez, en ellas vemos que solo algunas columnas son
#relevantes, las que tienen valores mas altos en la matriz de importantes, podemos quedarnos con esas operaciones solamente

X=np.array([[1.0,2],[2,1],[3,3]])
Matrix(X)

U, s, VT =svd(X, full_matrices=False)

print(Matrix(U))

S=np.diag(s)

print(Matrix(S))

print(Matrix(VT))

#El @ se ocupa para multiplicar solo matricez con numpy
X_=U@S@VT
X_=np.round(X_)
print(Matrix(X_))

#Aqui no etiendo lo que hace, pero muestra solo los valores enteros
X_2=s[0]*U[:,0:1]@VT[0:1,:]+s[1]*U[:,1:2]@VT[1:2,:]
print(Matrix(X_2))

###SVD trucado

print("SVD truncado")

X=np.array([[1,2],[2,4],[4,8.0001]])
print(Matrix(X))

U, s, VT =svd(X, full_matrices=False)
S=np.diag(s)
print(Matrix(S))

X_hat=np.round(s[0]*U[:,0:1]@VT[0:1,:])
print(Matrix(X_hat))

L=1
Xhat=U[:,:L]@S[0:L,0:L]@VT[:L,:]

#Esta es la multuplicacion de matricez, contiene los valores truncados
print(Matrix(Xhat))

print(f"With {L} singular value and its corresponding singular vectors, {s[0:L]/s.sum()} variance of X is explained")

plt.figure()
plt.plot(np.cumsum(s)/s.sum())
plt.xlabel('L')
plt.title('Cumulative explained singular value')
plt.tight_layout()
plt.show()


###SVD truncado con Sklearn

#Uno de los beneficios de SVD es que las columnas de matricez puede estar en cualquier orden no necesita centrarlos con
#PCA

svd_ = TruncatedSVD(n_components=1, random_state=42)

Z=svd_.fit_transform(X)
print(Z)

Xhat=svd_.inverse_transform(Z)
print(Matrix(np.round(Xhat)))

#Aqui toma las imagees a matriz, peds es una carpeta
X,Length,Width=get_data_Matrix(mypath="peds")

#Muestra las dimenciones
print(X.shape, Length, Width)

#Muestra 5 imagenes
for i in range(5):
    frame=randint(0, X.shape[0]-1)
    plt.imshow(X[randint(0, X.shape[0]-1),:].reshape(Length,Width),cmap="gray")
    plt.title("frame: "+str(frame))
    plt.show()

#Ahora toma la image y  la descompone en las 3 matricez
U, s, VT =svd(X, full_matrices=False)

S=np.diag(s)

L=1

#Ahora multiplicamos las matricez
Xhat=U[:,:L]@S[0:L,0:L]@VT[:L,:]

#Tiene una forma de (170, 35264)
print(Xhat.shape)

#Mostramos el resultado de la multiplicacion solo los componentes mas importantes, solo 1 en este caso
plt.imshow(Xhat[0,:].reshape(Length,Width),cmap="gray")
plt.title('Truncated SVD L=1')
plt.show()

#Es increible, pero tomo todo el fondo sin personas, es el elemento mas importante de la imagen, la entendio perfecto


#Como son imagenes del modelo creado, son todas iguales y solo el fondo
for i in range(5):
    #Simplemente toma un numero al azar entre 0 y el total de filas, muestra ese frame para graficarlo
    frame=randint(0, X.shape[0]-1)
    #Es una fila asi que la pasa a ser imagen en blanco y negro
    plt.imshow(Xhat[randint(0, X.shape[0]-1),:].reshape(Length,Width),cmap="gray")
    plt.title("frame: "+str(frame))
    plt.show()

#Cuanto contenido muestra segun el L, en este caso fue 1, por lo que mostro lo mas importante solamente, crece bastante segun
#la cantidad de L
plt.plot(np.cumsum(s)/s.sum())
plt.xlabel('L')
plt.title('Cumulative explained  singular value')
plt.show()

#Con L=10 son las 10 caracteristicas mas importantes
#En este caso si muestra personas
L=10
Xhat=U[:,:L]@S[0:L,0:L]@VT[:L,:]
for i in range(5):
    frame=randint(0, X.shape[0]-1)
    plt.imshow(Xhat[randint(0, X.shape[0]-1),:].reshape(Length,Width),cmap="gray")
    plt.title("frame: "+str(frame))
    plt.show()

#Ahora toma otra carpeta con datos
X,Length,Width=get_data_Matrix(mypath="boats")


#En este caso es diferente no se le ha aplicado ningun modelo, simplemente muestra el movimiento de la lancha, al azar
for i in range(5):
    frame=randint(0, X.shape[0]-1)
    plt.imshow(X[randint(0, X.shape[0]-1),:].reshape(Length,Width),cmap="gray")
    plt.title("frame: "+str(frame))
    plt.show()

#Ahora solo mostrado el componente mas importate con L=1
U, s, VT =svd(X, full_matrices=False)
L=1
Xhat=U[:,:L]@S[0:L,0:L]@VT[:L,:]

#Vemos la imagen del componente mas importante, que es el fondo sin la lancha, esto dado que la lancha se mueve
#no es un componente que se mantiene igual durante todas las fotos como el fondo
plt.imshow(Xhat[0,:].reshape(Length,Width),cmap="gray")
plt.title('Truncated SVD L=1')

plt.show()

#Con otras imagenes del trafico
X,Length,Width=get_data_Matrix(mypath="traffic")

#Simplemente muestra imagenes al azar de la carpeta de la imagen
for i in range(5):
    frame=randint(0, X.shape[0]-1)
    plt.imshow(X[randint(0, X.shape[0]-1),:].reshape(Length,Width),cmap="gray")
    plt.title("frame: "+str(frame))
    plt.show()

#Ahora ocupado el modelo truncado sin especifcar L=1, poniendo n_component=1, tomamos solo el elemento mas importante
svd_ = TruncatedSVD(n_components=1, n_iter=7, random_state=42)
score=svd_.fit_transform(X)
#Score se queda solo con el elmento mas importante, es si es la imagen, pero tiene solo una columna
Xhat=svd_.inverse_transform(score)
#Al hacer la trasformacion inversa, vuelve a tener las dimenciones de la imagen que son los frame, pero solo con la
#caracteristica mas importate
#Tomamos solo la primera fila y la transformamos en imagen
plt.imshow(Xhat[0,:].reshape(Length,Width),cmap="gray")
plt.title('Truncated SVD L=1')
plt.show()

###Otro ejemplo SVD de scratch

X=np.array([[1,2],[2,4],[4,8]])
print(Matrix(X))

C=X.T@X
eigen_vectors1 , V=eig(C)
print(Matrix(V))

G=X@X.T
eigen_vectors2 , U=eig(G)
print(Matrix(U))

S=np.round((U.T@X@V))
print(Matrix(S))

X_=np.round(U@S@V.T)
print(Matrix(X_))

### Relacion de SVD y PCA

#Creamos una matriz cualquiera
N=200
u=np.array([[1.0,1.0],[0.10,-0.10]])/(2)**(0.5)
#Simplemente hacemos las operaciones
X_=np.dot(4*np.random.randn(N,2),u)+10
X=X_-X_.mean(axis=0)

#Luego aplicamos el modelo SVD
U, s, VT =svd(X, full_matrices=False)

#Luego aplicamos PCA en X, solo tomamos el componente mas importante
pca = PCA(n_components=1)
projection=pca.fit_transform(X)
#Aplicamos trasfromacion inversa para la proyeccion
X_sklearn=pca.inverse_transform(projection)

#Hace de forma manual la operacion, la resta son valores muy pequeños solo porque fue trucando, pero de forma manual
#y con el modelo dan los mismos valores
projection_=X@VT[0,:]
print ("error SVD vs scikit-learn's PCA",(projection_-projection).sum())

projection_=U@np.diag(s)[:,0]
print ("error SVD vs scikit-learn's PCA",(projection_-projection).sum()) 

L=1
Xhat=U[:,:L]@np.diag(s[0:L])@VT[:L,:] # resconstruct X
print ("error SVD vs scikit-learn's PCA",((Xhat-X_sklearn)**2).sum()) 

#De la forma manual como con el modelo da exacamente los mismos resultados
print(s[0]**2/(200-1))

print(pca.explained_variance_)



#######################################################################################################################

                                              #Operaciones matriciales

#Las matrices en este apartado es bastante utilizado por lo que esta es una seccion solo para operaciones

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np 
import matplotlib.pylab as plt
import pandas as pd
from sklearn.decomposition import PCA
from sympy import Matrix, init_printing,Symbol
from numpy.linalg import qr,eig,inv,matrix_rank,inv, norm
from scipy.linalg import null_space
init_printing()

#Fucio para plotear 2d
def plot_2d(dict_):
    for key, value in dict_.items():
        if value.shape[0]>2:
            plt.scatter(value[:, 0], value[:, 1],label=key)
        else:
            print(value)
            plt.quiver([0],[0],value[:,0],value[:,1],label=key)

    plt.legend()
    plt.show()

#De un array crea una matriz, primer elemento fila de arriba, segundo fila de abajo
A=np.array([[2,-3],[4,7]])
print(Matrix(A))

#Muestra la primera columna
a1=A[:,0]
print(a1)

#Muestra la segunda columna
a2=A[:,1]
print(a2)

#Hace la traspuesta del elemento, las filas se cambian por columnas y viceversa
AT=A.T
print(Matrix(AT))

#rank es rango de una matriz es el numero de filas de la matriz, rango completo se le llama si es que es matriz cuadrada
print(matrix_rank(A))

#Esta graficando los columna de la matriz como vectores, al ser vectores perpendiculares
fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver([0, 0],[0, 0],A[0,0], A[1,0],scale=30,label="$a_{1}$")
ax.quiver([0, 0],[0, 0],A[0,1], A[1,1],scale=30,label="$a_{2}$")
plt.title("columns of $A$ ")
plt.legend()
plt.show()

#Si los vectores no van en la misma direccion, cualquier punto del grafico puede ser representado por estos vectores
#Si un vector es multiplo de otro no es de rango completo

#Este es un vector multiplo, el rango es solo 1, no dos como deberia
F=np.array([[2,4],[4,8]])
print(matrix_rank(F))

fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver([0, 0],[0, 0],F[0,1], F[1,1],scale=30,label="$f_{2}$",color='red')
ax.quiver([0, 0],[0, 0],F[0,0], F[1,0],scale=30,label="$f_{1}$")
plt.title("columns of $F$ ")
plt.legend()
plt.show()

F=np.array([[1,2],[1,-2],[-1,1]])
print(Matrix(F))

#A pesar de ser vectores en 3 dimensiones solo puede describir a puntos en un plano de dos dimenciones, a los cual los
#vectores esten juntos
ax = plt.figure().add_subplot(projection='3d')
p=null_space(F.T)
xx, yy = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1))
z=(p[0]*xx+p[1]*yy)/p[2]
ax.plot_surface(xx, yy, z, alpha=0.1)
ax.quiver([0,0], [0,0], [0,0], F[0,:], F[1,:], F[2,:])
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
plt.show()

print(matrix_rank(F))

#Para calcular la norma de una matriz, norma euclidiana
#La norma es la np.sqrt(norma de la primera columna**2 + la norma de la segunda columna**2), cada columna es un vector 
print(Matrix(A), norm(A))

m, n = A.shape[0], A.shape[1] # get number of rows and columns 
ss = 0

for i in range(m):
    for j in range(n):
        ss += A[i,j] ** 2
print(np.sqrt(ss))


###Suma de matricez

B=np.array([[1,1],[1,-1]])
print(Matrix(B))

#Simplemente es la suma de componente a componenete
C=A+B
print(Matrix(C))

B_T=B.T
print(Matrix(B_T))

C=np.random.randn(2,2)
S=C+C.T

print(Matrix(S))

print(Matrix(S.T))

#Si le decimos que la diagonal es tal, crea unna matriz cuadrada con esa diagonal, rellena con zeros
print(Matrix(np.diag(np.array([1,2,3]))))

#Hace una matriz cuadrada con una diagonal de solo unos, lo demas rellena con 0
print(Matrix(np.eye(3)))

#Matriz y vector multiplicacion

a=np.array([1,1])
b=np.array([1,2])

#La cantidad de columnas
print(a.ndim)

#Es el producto punto entre los dos vectores
print(a@b)

a=np.array([[1],[1]])
b=np.array([[1],[2]])

#En este caso tambien es producto punto, pero los elementos son diferentes
print(a.T@b)

#UNa matriz de dos por dos con puros 1
one=np.ones(2)

print(a.T@one)

u= np.array([[1],[2],[3],[4]])
v= np.array([[0],[1],[2],[3],[6]])

print(Matrix(u@v.T))

u=np.array([[1],[2]])

print(Matrix(u@np.array([[0,1,0,1]])))

x=np.array([1,1])
print(x.shape)
A=np.array([[-1,1],[1,2]])

b=A@x
print(Matrix(b))

fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver([0, 0],[0, 0],A[0,0], A[1,0],scale=10,label="$a_{1}$")
ax.quiver([0, 0],[0, 0],A[0,1], A[1,1],scale=10,label="$a_{2}$")
ax.quiver([0,0],[0,0],b[0], b[1],scale=10,label="b",color='r')
ax.quiver([0,0],[0,0],x[0], x[1],scale=10,label="x",color='b')
ax.set_xlim([-10,10])
ax.set_ylim([-5,10])
fig.legend()
plt.show()

### Multiplicando matricez

#Una multiplicacion de matricez
C=A@B
print(Matrix(C))

#Para encotrar la matriz inversa
A_inv=inv(A)
print(Matrix(A_inv))

#Cuando se multiplica la matriz inversa de A por A, da la matriz identidad
I=A_inv@A
print(Matrix(I))

#La multiplicacion con la matriz identidad devuelve la misma matriz
print(A@I)


x_=A_inv@b
#Es una minima diferencia
#x como es se muestra con el 1.
print("x_ :",x_)
#x solo se muestra sin el punto solo 1
print("x:",x)

#Se toma primero el elevado 1/raiz(2), esto es 0.7 y se multplica por todos los numeros dentro de la matriz
Q=np.array([[1,1],[1,-1]])*2**(-1/2)
print(Q)

I=Q@Q.T
#Es coincidente en este caso con el signo negativo
print(Matrix(I))

#Es un plot de vectores
fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver([0, 0],[0, 0],B[0,0], B[1,0],scale=10,label="$q_{1}$")
ax.quiver([0, 0],[0, 0],B[0,1], B[1,1],scale=10,label="$q_{2}$")
plt.title("columns of $B$ ")
plt.legend()
plt.show()

samples=200

u=np.array([[1.0,1.0],[0.10,-0.10]])/(2)**(0.5)

#Para operar matricez producto punto, primera matriz, por segunda matriz "u"
X_=np.dot(4*np.random.randn(samples,2),u)+10
print(X_[0:5])

#Plotear relacion de los datos
dict_={"design matrix samples":X_}
plot_2d(dict_)

#Con tiene las dimenciones de X_
N,D=X_.shape
print("number of smaples {}, dimensions is {}".format(N,D))

#Es el promedio, dividio por la cantidad de filas
mean=(np.ones((1,N))/N)@X_
print(mean)

print(X_.mean(axis=0))

#Matriz identidad de 200 * 200
I=np.identity(N)
print(pd.DataFrame(I))

#Una fila solo con 1, 200 datos
col1=np.ones((1,N))
print(pd.DataFrame(col1))

#Una columna con solo 0.005 debido a que son 1 dividios en 200
row1=np.ones((N,1))/N  
print(pd.DataFrame(row1))  	  	

#La multiplicacion de la columan da una matriz es solo una matriz solo con una diagonal con los 0.005, restada a la identidad
no_mean=(I-row1@col1)

print(pd.DataFrame(X_))

#Se multiplica X contiene dos columnas con datos aleatorios, (200*200)*(200*2)= (200,2)
X=no_mean@X_

#Muestra el promedio de las dos columnas, valores muy pequeños
print("mean of X",X.mean(axis=0))

dict_={"original data":X_,"zero mean data":X,"mean":mean}
plot_2d(dict_)   

#Nuevamente una matriz identidad, (2, 200) * (200, 2) = (2,2) por lo tanto su ranking es 2
C=X.T@X/N
print(Matrix(C))

print(matrix_rank(C))


###Descomposicion propia

#Solo se puede hacer esta descomposicion si la matriz tiene rango completo, aun asi los vectores propios podrian ser
#complejos

#Un ejemplo es la matriz simetria util es que podemos saber que sus valores y vectores propios son reales
print(pd.DataFrame(A))
#eig(A) toma los valores propios de una matriz automaticamente, tira 2 array, el primero con los valores propios
#el segundo contiene los vectores propios
print(eig(A))
eigen_values , eigen_vectors = eig(A)

#Los muestra como vector ojo, un vector con dos filas
print(pd.DataFrame(eigen_values))

#Los muestra como una matriz, cada columna es un vector propio
print(pd.DataFrame(eigen_vectors))

#Crea una matriz donde los valores propios los pone en una matriz
print(Matrix(np.diag(eigen_values)))
print(inv(eigen_vectors))
#La matriz de los vectores propios multplicada por una matriz de los vectores propios por la matriz intertida de los vectores
#propios
A=eigen_vectors@np.diag(eigen_values)@inv(eigen_vectors)
print(Matrix(A))
print(pd.DataFrame(inv(eigen_vectors)))

eigen_values , eigen_vectors = eig(C)

v=eigen_vectors[:, np.argmax(eigen_values)].reshape(-1,1)
print(v)

Z=X@v

#X es una matriz que tiene valores aleatorios de 200 * 2
print(pd.DataFrame(X))

#Por ultimo occupamos PCA
pca = PCA(n_components=1)
#Lo ocupamos en X
X_transformed=pca.fit_transform(X)
X_=pca.inverse_transform(X_transformed)

#Se restablece completamente la matriz original
print(pd.DataFrame(X_))

Xhat=Z@v.T

#Principal componente es solo una fila que representa la matriz
print(pd.DataFrame(v.T))
print(pd.DataFrame(Xhat))

dict_ = {"Sklearn inverse_transform": X_, "Matrix inverse transform": Xhat, "First Principal Component": v.T}
plot_2d(dict_)


#######################################################################################################################

                                       #Reduccion de dimensionalidad Ejemplos

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/Wholesale_Customers_Data.csv', sep=',')

print(data.shape)

print(data.head())

#Eliminamos algunas columnas clasificatorias
data = data.drop(['Channel', 'Region'], axis=1)

#Todas las columnas son int
print(data.dtypes)

#Los pasamos a float para ser escalados
for col in data.columns:
    data[col] = data[col].astype(float)

data_orig = data.copy()

corr_mat = data.corr()

#Solo para borrar la diagonal porque tomara como 1
for x in range(corr_mat.shape[0]):
    corr_mat.iloc[x,x] = 0.0
    
#Vemos las correlaciones, en general son bastante bajas
print(corr_mat)

#Muestra las correlaciones mas altas para cada columna
print(corr_mat.abs().idxmax())

#Para tomar solo algunas columnas
log_columns = data.skew().sort_values(ascending=False)
log_columns = log_columns.loc[log_columns > 0.75]

#Muestra el logaritmos de los datos de las columnas, cada numero es muy alto con respecto a los otros
print(log_columns)

#Le aplicamos logaritmos naturales a algunas columnas para tener los datos relativamente normalizados
for col in log_columns.index:
    data[col] = np.log1p(data[col])
    
#Escalamos los datos
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

#Escala datos que se les habia aplicado logaritmos
#squeeze le quita un dimension a los array [[[1,2]]] con squezee pasa a ser solo [[1,2]]
for col in data.columns:
    print(data[col])
    data[col] = mms.fit_transform(data[[col]]).squeeze()

#Son bastante irregulares los datos
sns.set_context('notebook')
sns.set_style('white')
sns.pairplot(data)
plt.show()

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

#Para tener un funcion para aplicar
log_transformer = FunctionTransformer(np.log1p)

#Aplicamos el pipe, primero normalizar, luego escalar
estimators = [('log1p', log_transformer), ('minmaxscale', MinMaxScaler())]
pipeline = Pipeline(estimators)

#Le aplicamos los estimadores otra vez a la data original
data_pipe = pipeline.fit_transform(data_orig)

print(np.allclose(data_pipe, data))

#Lo anterior es para aplicar PCA
from sklearn.decomposition import PCA

pca_list = list()
feature_weight_list = list()

#Aplicamos PCA a la data
for n in range(1, 6):
    
    # Create and fit the model
    PCAmod = PCA(n_components=n)
    PCAmod.fit(data)
    
    #Agregamos a la lista la catniadad de estimadores, modelo, modelo explicado varianza
    pca_list.append(pd.Series({'n':n, 'model':PCAmod,
                               'var': PCAmod.explained_variance_ratio_.sum()}))
    
    #Agregamos las caracterisitcas mas importantes
    abs_feature_values = np.abs(PCAmod.components_).sum(axis=0)
    feature_weight_list.append(pd.DataFrame({'n':n, 
                                             'features': data.columns,
                                             'values':abs_feature_values/abs_feature_values.sum()}))
    
pca_df = pd.concat(pca_list, axis=1).T.set_index('n')
print(pca_df)

features_df = (pd.concat(feature_weight_list)
               .pivot(index='n', columns='features', values='values'))

print(features_df)

#Gracias al modelo explained
#Muestra que con 5 columnas, se explica el modelo casi completamente,  con 3 es aceptable
sns.set_context('talk')
ax = pca_df['var'].plot(kind='bar')

ax.set(xlabel='Number of dimensions',
       ylabel='Percent explained variance',
       title='Explained Variance vs Dimensions')
plt.show()

#Es un grafico super completo que muestra si dejamos mas columnas componentes, que columnas iniciales son mas importantes
#dependiendo de la cantidad de columnas
ax = features_df.plot(kind='bar', figsize=(13,8))
ax.legend(loc='upper right')
ax.set(xlabel='Number of dimensions',
       ylabel='Relative importance',
       title='Feature importance vs Dimensions')
plt.show()



#Ahora con KernelPCA simplemente es lo mismo, pero con 'rbf'
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

#Para saber la puntuacion del modelo
# Custom scorer--use negative rmse of inverse transform
def scorer(pcamodel, X, y=None):

    try:
        X_val = X.values
    except:
        X_val = X
        
    #Aplicamos el modelo que le especifiquemos en X y transformacion automatica
    data_inv = pcamodel.fit(X_val).transform(X_val)
    #Luego aplicamos la inversa
    data_inv = pcamodel.inverse_transform(data_inv)
    
    #Vemos si existe diferencia entre la data invertida y el X inicial
    mse = mean_squared_error(data_inv.ravel(), X_val.ravel())
    
    # Larger values are better for scorers, so take negative value
    return -1.0 * mse

param_grid = {'gamma':[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
              'n_components': [2, 3, 4]}

#Aplicamos GridSearch para encontrar los mejores parametros, hay que tener ojo aca, el scorer lo inventamos
#debido a que PCA no tiene puntuacion como tal
kernelPCA = GridSearchCV(KernelPCA(kernel='rbf', fit_inverse_transform=True),
                         param_grid=param_grid,
                         scoring=scorer,
                         n_jobs=-1)

#Aplicamos PCA a la data
kernelPCA = kernelPCA.fit(data)

print(kernelPCA.best_estimator_)



###Ejemplo 2

data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/Human_Activity_Recognition_Using_Smartphones_Data.csv', sep=',')

print(data.columns)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = data.drop('Activity', axis=1)
y = data.Activity
sss = StratifiedShuffleSplit(n_splits=5, random_state=42)

#Una funcion que obtiene la puntuacion promedio, pero antes le aplica a los datos escalamiento, PCA y luego
#regression logistica, lo que hace es aplicar distintos PCA y sacar la puntuacion, para ver como mejor el modelo
#en este caso la puntuacion depende netamente del PCA, no siempre es recomendable ocupar PCA, en este caso
#solo si necesita acotar el almacenamiento, pero igual necesita 100 componentes para tener una puntuacion
#aceptable
def get_avg_score(n):
    pipe = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n)),
        ('estimator', LogisticRegression(solver='liblinear'))
    ]
    pipe = Pipeline(pipe)
    scores = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        pipe.fit(X_train, y_train)
        scores.append(accuracy_score(y_test, pipe.predict(X_test)))
    return np.mean(scores)


ns = [10, 20, 50, 100, 150, 200, 300, 400]

#La puntuacion la obtenemos desde la funcion
score_list = [get_avg_score(n) for n in ns]

sns.set_context('talk')

ax = plt.axes()
ax.plot(ns, score_list)
ax.set(xlabel='Number of Dimensions',
       ylabel='Average Accuracy',
       title='LogisticRegression Accuracy vs Number of dimensions on the Human Activity Dataset')
ax.grid(True)
plt.show()


#######################################################################################################################

                                                   #MDS
                                          #Escalado Multidimencional

#El escalado multidimencional es una familia de algoritmos como PCA, pero MDS tambien permite hace algunas cosas
#mas visuales
#Hay diferentes categorias de escalado multidimencional

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
import sklearn.metrics.pairwise
import seaborn as sns
import folium
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from matplotlib import offsetbox


#Definimos algunas funciones, graficar puntos con scatter
def plot_points(df,color="red",title=""):

    X=df['lon']
    Y=df['lat']

    annotations=df.index

    plt.figure(figsize=(8,6))
    plt.scatter(X,Y,s=100,color=color)
    plt.title(title)
    plt.xlabel("lat")
    plt.ylabel("log")
    for i, label in enumerate(annotations):
        plt.annotate(label, (X[i], Y[i]))
    plt.axis('equal')
    plt.show()

#Definimos embedding para escalar, hacer cosas con las imagenes
def plot_embedding(X, title, ax):
    X = MinMaxScaler().fit_transform(X)
    for digit in digits.target_names:
        ax.scatter(
            *X[y == digit].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.425,
            zorder=2,
        )
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")

#Una data de distancias entre ciudades, contiene los angulos
distance=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/distance.csv').set_index('name')
print(distance.head(8))

from sklearn.manifold import MDS

#Hacemos el modelo para dos componentes
embedding =  MDS(dissimilarity='precomputed',n_components=2,random_state=0,max_iter=300,eps=1e-3)

#La data de las distancias que era enrealidad una especie de matriz de distancias con los angulos,
#Le aplicamos la transformacion, al separarlo en dos automaticamente mantiene las distancias entre todos
#por lo que llegaremos a la latitud y a la longuitud
X_transformed = embedding.fit_transform(distance)

#Le ponemos nombre, ahora hay solo dos columnas
df_t=pd.DataFrame(X_transformed , columns=["lon","lat"], index=distance.columns)
print(df_t.head(8))

#Es la data sin nombres de filas y columnas
print(embedding.embedding_)

#Un parametro llamado estres
print(embedding.stress_)

#Matriz de disimilaridad, es el data frame inicial, pero solo con valores enteros
print(embedding.dissimilarity_matrix_)

#No entiendo porque lo hace manual, quizas verifico la latitud y longuitu en internet
df = pd.DataFrame({
   'lon':[-58, 2, 145, 30.32, -4.03, -73.57, 36.82, -38.5],
   'lat':[-34, 49, -38, 59.93, 5.33, 45.52, -1.29, -12.97],
   'name':['Buenos Aires', 'Paris', 'Melbourne', 'St Petersbourg', 'Abidjan', 'Montreal', 'Nairobi', 'Salvador']})
df=df.set_index('name')
print(df.head(10))

#Grafica los puntos originales
plot_points(df,title='original dataset')

#Grafica los puntos con el modelo y estan cambiados solo en una dimencion
#Una dimencion parece estar distoricionada, esto dado que el modelo intenta conservar el stress
#Es como si los puntos iniciales tienen cercania, esto sigue presente y los clasifica como por tipos de stress
plot_points(df_t,color='blue',title='Embedded Coordinates using Euclidean distance ')


###Ejemplo 2

from scipy.spatial.distance import squareform, pdist

distance=pd.DataFrame(squareform(pdist(df.iloc[:, 1:])), columns=df.index, index=df.index)

print(distance)

#Lo que hacemos aca es poner todas los tipos de distancias posibles para el modelo
dist=['cosine','cityblock','seuclidean','sqeuclidean','cosine','hamming','jaccard','chebyshev','canberra','braycurtis']
#Ultima muestra de como era el grafico original
plot_points(df,title='original dataset')

#Lo que hace es probar diferentes tipo de distancias y hace efectos totalmente difentes en los graficos
for d in dist:

    distance=pd.DataFrame(squareform(pdist(df.iloc[:, 1:],metric=d)), columns=df.index, index=df.index)

    embedding =  MDS(dissimilarity='precomputed', random_state=0,n_components=2)
    X_transformed = embedding.fit_transform(distance)
    df_t=pd.DataFrame(X_transformed , columns=df.columns, index=df.index)

    plot_points(df_t,title='Embedded Coordinates using '+d ,color='blue')


###MDS no metrico

#Si nosotros aplicamos una distancia, esta distancia persevera en el modelo antes y despues
#La distacias pueden no aplicarseles ninguna metrica

metric=False
embedding =  MDS(dissimilarity='precomputed',n_components=2,metric=metric,random_state=0)

X_transformed = embedding.fit_transform(distance)
df_t=pd.DataFrame(X_transformed , columns=df.columns, index=df.index)
print(df_t.head(8))

#Graficamos nuevamente el grafico original
plot_points(df,title='original dataset')

#Hace diferentes efectos, pero no tan severos como los anteriores
plot_points(df_t,color='blue',title='Embedded Coordinates using Euclidean distance ')

dist=['cosine','cityblock','seuclidean','sqeuclidean','cosine','hamming','jaccard','chebyshev','canberra','braycurtis']
plot_points(df,title='original dataset')
metric=False

#Tiene resultados diferentes a los anteiores, alguas distancias tienen los mismos resultados
for d in dist:

    distance=pd.DataFrame(squareform(pdist(df.iloc[:, 1:],metric=d)), columns=df.index, index=df.index)

    embedding =  MDS(dissimilarity='precomputed', random_state=0,n_components=2,metric=False)
    X_transformed = embedding.fit_transform(distance)
    df_t=pd.DataFrame(X_transformed , columns=df.columns, index=df.index)

    plot_points(df_t,title='Embedded Coordinates using '+d ,color='blue')


###Reduccion de dimensionalidad con MDS

#importamos un datasets
from sklearn.datasets import load_digits

digits = load_digits(n_class=6)
#target es el numero real, digit.data es la imagen
X, y = digits.data, digits.target
#Es una matriz de 1083 * 64
n_samples, n_features = X.shape

print("samples:", n_samples, "features", n_features)

#Lo que se hace es graficar 10 * 10 osea 100 imagenes juntas, como eran filas se pasan a imagen
fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))
for idx, ax in enumerate(axs.ravel()):
    ax.imshow(X[idx].reshape((8, 8)), cmap=plt.cm.binary)
    ax.axis("off")
    
#Se las graficamos todas juntas
_ = fig.suptitle("A selection from the 64-dimensional digits dataset", fontsize=16)
plt.show()

#Ahora ocupamos MDS para 2 componentes
embedding=MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2)

#Transoformamos
X_transformed=embedding.fit_transform(X)

#Graficamos y automaticamete los agrupa en ciertos cluster con la fucion plot_embedding
fig, ax = plt.subplots()
plot_embedding(X_transformed, "Metric MDS ", ax)
plt.show()


### Otros ejercicios

dist=['cosine','cityblock','hamming','jaccard','chebyshev','canberra','braycurtis']
scaler = MinMaxScaler()
X_norm=scaler.fit_transform(X)

#En el caso de los cluster y los numeros, tiene resultados bastante similares, solo cambia un poco la distancia entre los
#puntos, pero poco
for d in dist:

    distance=squareform(pdist(X_norm,metric=d))
    print( d)

    embedding =  MDS(dissimilarity='precomputed', random_state=0,n_components=2)
    X_transformed = embedding.fit_transform(distance)
    fig, ax = plt.subplots()
    plot_embedding(X_transformed, "Metric MDS ", ax)
    plt.show()


###MDS con TSNE

from sklearn.manifold import TSNE

#Se diferencia bastante de MDS, MDS no separa tanto los numeros en los cluster, acepta distancias mas cercanas
#Se mantiene la relacion de distancia, en el caso TSNE fuerza demaciado a que formen los cluster, poco suceptible
#a la cercania de los datos
X_embedded = TSNE(n_components=2, init='random').fit_transform(X)

fig, ax = plt.subplots()
plot_embedding(X_embedded , "test", ax)
plt.show()


#######################################################################################################################

                                              #Kernel PCA

#Kernel PCA es una variante de PCA, lo que hace es aumentar las dimenciones, para luego reducirlas con 
#Tanto PCA como Kernel PCA, pueden tener resultados similares, pero Kernel PCA puede ayudar en la visualizacion
#de los datos ya que reduce el ruido


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
from itertools import accumulate

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

sns.set_context('notebook')
sns.set_style('white')

#Definimos una funcion de proyeccion
def plot_proj(A,v,y,name=None):

    plt.scatter(A[:,0],A[:,1],label='data',c=y,cmap='viridis')
    
    #plt.plot(np.linspace(A[:,0].min(),A[:,0].max()),np.linspace(A[:,1].min(),A[:,1].max())*(v[1]/v[0]),color='black',linestyle='--',linewidth=1.5,label=name)   
    plt.plot(np.linspace(-1,1),np.linspace(-1,1)*(v[1]/v[0]),color='black',linestyle='--',linewidth=1.5,label=name)  
    # Run through all the data

    for i in range(len(A[:,0])-1):
        #data point 
        w=A[i,:]

        # projection
        cv = (np.dot( A[i,:],v))/np.dot(v,np.transpose(v))*v

        # line between data point and projection
        plt.plot([A[i,0],cv[0]],[A[i,1],cv[1]],'r--',linewidth=1.5)
    plt.plot([A[-1,0],cv[0]],[A[-1,1],cv[1]],'r--',linewidth=1.5,label='projections' )
    plt.legend()
    plt.show()

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Create the toy dataset
X, y = make_circles(n_samples=1000, factor=0.01, noise=0.05, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

_, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))


#Grafico de los datos de los datos de entrenamiento como de los datos de prueba, no tienen ningun modelo aplicado
train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,cmap='viridis')
train_ax.set_xlabel("$x_{0}$")
train_ax.set_ylabel("$x_{1}$")
train_ax.set_title("Training data")

test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,cmap='viridis')
test_ax.set_xlabel("$x_{0}$")
test_ax.set_ylabel("$x_{1}$")
test_ax.set_title("Test data")
plt.show()

#Ahora aplicamos PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

score_pca = pca.fit(X_train).transform(X_test)
print(pca)

#Este grafico tambien muestra los vectores propios quiver es para plotear flechas
plt.scatter(score_pca[:, 0], score_pca[:, 1], c=y_test,label="data points", cmap='viridis')
plt.quiver([0,0],[0,0], pca.components_[0,:], pca.components_[1,:], label="eigenvectors")
plt.xlabel("$x_{0}$")
plt.ylabel("$x_{1}$")
plt.legend(loc='center right')
plt.show()

#En componentes principales busca, que los datos se distribuyan de cierta forma, que pasen por ellos 2 vectores
#Lo que hace entonces, primero en el siguiente grafico, una linea horizontal que es el vector principal
#y las proyecciones son todas las diferencias del vector principal y cada dato
plot_proj(X_train,pca.components_[0,:],y_train,"first principal component")

#Grafico de las proyecciones solo los puntos en la linea horizontal
plt.scatter(score_pca [:,0],np.zeros(score_pca[:,0].shape[0]),c=y_test,cmap='viridis')
plt.title("Projection of testing data\n using PCA")
plt.show()

plt.scatter(score_pca[:, 0], score_pca[:, 1], c=y_test,cmap='viridis')
plt.title("Projection of testing data\n using PCA")
plt.show()

#Ahora aplicando la regression logistica tuvo una puntuacion bastante mala de 0.5
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression().fit(X_train, y_train)
print(str.format("Test set  mean accuracy score for for PCA: {}", lr.score(X_test, y_test)))


###Ahora con KernelPCA

#Aca agrega una nueva columna, X_train tiene dos, ahora con esta tiene 3, suma los dos valores de las columnas originales
#agrega una nueva columna con eso
PHI_train=np.concatenate((X_train, (X_train**2).sum(axis=1).reshape(-1,1)),axis=1)
PHI_test=np.concatenate((X_test, (X_test**2).sum(axis=1).reshape(-1,1)),axis=1)

#El resultado es un grafico de 3 dimenciones donde vemos que hay dos clases que tienen datos muy diferentes
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(PHI_train[:,0], PHI_train[:,1],  PHI_train[:,2], c=y_train, cmap='viridis', linewidth=0.5);
ax.set_xlabel('$x_{1}$')
ax.set_ylabel('$x_{2}$')
ax.set_zlabel('$x_{1}^2+x_{2}^2$')
plt.show()

#Ocupamos un PCA a 3 componentes
pca = PCA(n_components=3)
score_polly = pca.fit(PHI_train).transform(PHI_test)

#Primero un grafico de las proyecciones en el eje principal por eso el y lo hace 0
plt.scatter(score_polly[:,0],np.zeros(score_polly[:,1].shape[0]),c=y_test,cmap='viridis')
plt.title("Projection onto the \nfirst principal component")
plt.show()

#Luego un grafico total de prediccines donde se muestra que los componentes principales, estan mal graficados
#no se ve una clara diferencia
plt.scatter(score_polly[:,0], score_polly[:,1],c=y_test, cmap='viridis', linewidth=0.5);
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
plt.title("Projection onto PCs")
plt.show()

#Pero como habiamos agrandado las dimenciones, notamos que enrealidad estamos ocupando KernelPCA
#Al ocupar regression logistica tuvo, prediccion perfecta
lr= LogisticRegression().fit(PHI_train, y_train)
print(str.format("Test set  mean accuracy score for for Kernal PCA: {}", lr.score(PHI_test, y_test)))


###Aplicando Kernel PCA

kernel_pca = KernelPCA( kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1)

kernel_pca.fit(X_train)

score_kernel_pca = kernel_pca.transform(X_test)

#Grafica el valor de valores propios despues de 25 componentes es muy cercana a 0
plt.plot(kernel_pca.eigenvalues_)
plt.title("Principal component and their eigenvalues")
plt.xlabel("nth principal component")
plt.ylabel("eigenvalue magnitude")
plt.show()

#Grafica el kernelPCA siendo buena representacion de los datos, pero no totalmente correcta
plt.scatter(score_kernel_pca[:,0],score_kernel_pca[:,1] ,c=y_test,cmap='viridis')
plt.title("Projection onto PCs (kernel)")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
plt.show()

#Hace la trasnformacion inversa
X_hat_kpca = kernel_pca.inverse_transform(kernel_pca.transform(X_test))

#Aplica PCA
pca = PCA(n_components=2)
pca.fit(X_train)
#Hace la trasnformacion inversa
X_hat_pca = pca.inverse_transform(pca.transform(X_test))

#Mostramos los puntos originales de la data
plt.scatter(X_test[:,0],X_test[:,1] ,c=y_test,cmap='viridis')
plt.title("Original data")
plt.show()

#La primera trasnformacion inversa de kernelPCA
plt.scatter(X_hat_kpca[:,0],X_hat_kpca[:,1] ,c=y_test,cmap='viridis')
plt.title("Inversely Transformed Data (Kernel PCA)")
plt.show()

#La segunda trasnformacion iversa de PCA
plt.scatter(X_hat_pca[:,0],X_hat_pca[:,1] ,c=y_test,cmap='viridis')
plt.title("Inversely Transformed Data (PCA)")
plt.show()

print("Mean squared error for Kernel PCA is:",((X_test-X_hat_kpca)**2).mean())

print("Mean squared error PCA is:" ,((X_test-X_hat_pca)**2).mean())

#Lo que se muestra es que en este caso PCA tiene mucho menor error al recuperar una data, por lo que es mejor en esto
#En cambio KernelPCA no tiene buen rendimiento, tiene un error mucho mayor al recuperar una data


###Usando PCA para predecir tendencias, esto solo se hace visualmente
#Data de los millonarios

df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/billionaires.csv',index_col="Unnamed: 0")
print(df.head())

#2600 * 7
print(df.shape)

#Dice cuantos valores unicos tiene cada columna
for col in df:
    print(str.format("{} has {} unique values.", col, len(df[col].unique())))

print(df[-100:-1])

#Grafica las todas las categorias de estas dos columnas son muchas
for column in ['country','industry']:
    
    df[column].hist(bins=len(df[column].unique()))
    plt.xticks(rotation='vertical')
    plt.show()

sns.pairplot(df[['age','rank']])
plt.show()
#Las dos columnas tienen una correlacion muy baja
print(df[['age','rank']].corr())

B_names,networths,sources,industrys=df['name'],df['networth'],df['source'],df['industry']
print(B_names,networths,sources,industrys)

#El ranking sera la etiqueta
y=df['rank']
print(y.head())

#Botamos algunas columnas no utiles
df.drop(columns=['name','networth','source'],inplace=True)
print(df.head())

#Codificamos con OneHot dos columnas clasificatorias
one_hot = ColumnTransformer(transformers=[("one_hot", OneHotEncoder(), ['country','industry']) ],remainder="passthrough")
data=one_hot.fit_transform(df)


names=one_hot.get_feature_names_out()
#Creo que le pone nombre a las columnas codificadas
column_names=[name[name.find("_")+1:] for name in  [name[name.find("__")+2:] for name in names]]
new_data=pd.DataFrame(data.toarray(),columns=column_names)
print(new_data.head())

#Usando PCA, para mejorar la visualizacion

kernel_pca = KernelPCA(kernel="rbf" ,fit_inverse_transform=True, alpha=0.1)
kernel_score=kernel_pca.fit_transform(new_data)

ranking=13

fig, ax = plt.subplots()

#Grafica dos proyecciones, da una figura un poco extraña
sc=ax.scatter(kernel_score[:,0],kernel_score[:,1] ,c=y,cmap='viridis')
fig.colorbar(sc, orientation='vertical')
ax.annotate(industrys[ranking], (kernel_score[ranking,0],kernel_score[ranking,1]))
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
plt.title("Projection on the top 2 \nprincipal components (colored by ranking)")
plt.show()

#Las mismas proyecciones, pero ahora en 3 dimenciones
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc=ax.scatter(kernel_score[:,0], kernel_score[:,1],  kernel_score[:,2], c=y, cmap='viridis', linewidth=0.5);
fig.colorbar(sc, orientation='horizontal')
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
ax.set_zlabel('3rd PC')
plt.show()


#Ahora aplicando PCA
pca = PCA()
score_pca = pca.fit_transform(new_data)

#Primero un grafico de la proyeccon unidimencional, donde la categorizacion
#Es que los mas millonarios se ubican justo en cierto lado de los datos mostrado con los colores
fig, ax = plt.subplots()
sc=ax.scatter(score_pca[:,0],np.zeros(score_pca[:,1].shape ),c=y,cmap='viridis')
ax.set_title('1-dimensional projection space\n (1st PC)')
fig.colorbar(sc, orientation='vertical')
plt.show()

#Ahora en dos dimenciones, si bien se distribuyen equitativamente segun sus caracterisitcas, las primeras
#2 componentes, el color dice que tienden a ganarse en cierto lado segun la etiqueta
fig, ax = plt.subplots()
sc=ax.scatter(score_pca[:,0],score_pca[:,1] ,c=y,cmap='viridis')
fig.colorbar(sc, orientation='vertical')
ax.set_title('2-dimensional projection space\n (Top 2 PCs)')
plt.xlabel("1st PC")
plt.ylabel("2nd PC")
plt.show()

#En 3 dimenciones dice exactemente lo mismo sus componentes se distribuyen casi aleatoriamente, pero cuando le ponemos
#la etiqueta con color estos mas ricos estan en en cierta ubicacion del grafico
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc=ax.scatter(score_pca[:,0], score_pca[:,1],  score_pca[:,2], c=y, cmap='viridis', linewidth=0.5);
ax.set_title('3-dimensional projection space\n (Top 3 PCs)')
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
ax.set_zlabel('3rd PC')
plt.show()

#Podemos notar que en cierto componente en este caso el primero es el que logra categorizar las etiqueta
#de cual es mas ricos que otros, es posible que sea una combinacion de edad, industria y pais
#Aunque debido a la naturaleza del PCA no podemos saber o interpretar claramente el grafico


#KernelPCA tiene mayor puntuacion en sus modelos con metrica R2

from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(kernel_score, y, test_size=0.4, random_state=0)
lr = Ridge(alpha=0).fit(X_train, y_train)
print(str.format("Test set R^2 score for Kernel PCA: {}", lr.score(X_test, y_test)))

X_train, X_test, y_train, y_test = train_test_split(score_pca, y, test_size=0.40, random_state=0)
lr= Ridge(alpha=0).fit(X_train, y_train)
print(str.format("Test set R^2 score for PCA: {}", lr.score(X_test, y_test)))

#En este caso estuvieron bastante parejas las puntuaciones

X_train_noisy = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/X_train_noisy.csv').to_numpy()
X_test_noisy = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/X_test_noisy.csv').to_numpy()

def plot_digits(X, title):
    "Small helper function to plot 100 digits."
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap="Greys")
        ax.axis("off")
    fig.suptitle(title, fontsize=24)
    plt.show()
    
#Muestra 100 imagenes con ruido de numeros, no se ven claramente
plot_digits(X_test_noisy, "Noisy test images")

#Aplicamos PCA y KernelPCA
pca = PCA(n_components=35)
pca.fit(X_train_noisy)

kernel_pca = KernelPCA(n_components=400, kernel="rbf", gamma=0.01, fit_inverse_transform=True, alpha=0.1)
kernel_pca.fit(X_train_noisy)

#Aplicamos las trasnformaciones inversas
X_hat_pca = pca.inverse_transform(pca.transform(X_test_noisy))

X_hat_kpca = kernel_pca.inverse_transform(kernel_pca.transform(X_test_noisy))

#Mostramos denuevo las 100 imagenes juntas, donde se ve que el ruido se redujo, pero tambien
#se ve un poco borrosas
plot_digits(X_hat_pca, "Reconstructed Test Set (PCA)")

#Tambien se redujo el ruido, pero en este caso le bajo calidad a las imageenes, pero ruido
#lo redujo casi en su totalidad, pero la imagen de base venia muy mala asi que tampoco
#pudo hacer mas
plot_digits(X_hat_kpca, "Reconstructed Test Set (Kernel PCA)")


#######################################################################################################################

                                     #Factorizacion matricial no negativa
                                            #Bases para copyright
                                      
#Se ocupa para el reconocimiento de que las imagenes no incumplan los tratados de propiedad intelectual, logrando
#reconocer ciertas marcas, etc. Gracias a bases de datos potentes

import warnings
warnings.simplefilter('ignore')

import logging
import numpy as np
import pandas as pd
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from os import listdir,getcwd
from os.path import isfile, join
from PIL import Image, ImageOps
import os 


def get_data_matrix(test=False,Length=100,Width=100,mypath="images/"):

    files = [join(mypath,f) for f in listdir(mypath) if isfile(join(mypath, f)) and f[0] != '.']
    if mypath + '/.DS_Store' in files:
        files.remove(mypath + '/.DS_Store')
  
    if test:
        print("test data")
        files=files[9000:10000]
        
    else:
        print("training data")
        files=files[0:9000]
        
    print(len(files))
    X=np.zeros((len(files),Length*Width))
    for i,file in enumerate(files):
        img = Image.open(file).resize((Width, Length))
        img =  ImageOps.grayscale(img)

        I=np.array(img)
 
        X[i,:]=I.reshape(1,-1)
    return X

def reshape_row(x) :
    plt.imshow(x.reshape(Length,Width),cmap="gray")

def threshold(similar_distance,max_=0.1,min_=0):
    dataset_index=np.where(np.logical_and(similar_distance>min_ ,similar_distance<max_))[0]
    query_index=similar_index[np.logical_and(similar_distance>min_ ,similar_distance<max_)]
    return dataset_index,query_index

def plot_data_query(dataset_index,query_index,N):
    for data_sample,query_sample in zip(dataset_index[0:N],query_index[0:N]):

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        reshape_row(X[data_sample])
        plt.title("dataset sample {}".format(data_sample))
        plt.subplot(1,2,2)
        reshape_row(X_q[query_sample])
        plt.title("query sample match {}".format(query_sample))
        plt.show()

        print("-----------------------------------------------------")

#Aca este modelo cada fila, se llama base

#Tenemos imagenes en este caso 100 * 100 pixeles
Length,Width=100,100
#Pero en la carpeta la dimencion son filas 9000 cada imagen por 10000 columas que son en total la cantidad de pixeles
X=get_data_matrix(test=False,Length=100,Width=100,mypath="images")
print(X.shape)

n_components=10

nmf_estimator = NMF(n_components=n_components, tol=5e-3,max_iter=2000)
nmf_estimator.fit(X)  # original non- negative dataset

H = nmf_estimator.components_

#Toma solo 10 componentes del modelo que podrian ser 9000 imagenes, de 10000 pixeles
print(H)
print(H.shape)
plt.figure(figsize=(25, 8))

#Lo que hacemos es mostrar 2 filas por 5 columnas de imagenes, es decir 10 imagenes, esas imagenes solo tomaron
#10 componentes al azar, que son 10 imagenes
#con enumerate toma, i = todas las filas, h=empieza de 0, no se especifico
#como son 10 componentes, todas las filas, son las 10 imagenes, las recorre todas es el itinerable
#en cambio h, empieza de 0 y termina en 9
#Las filas las convierte en columnas y las grafica, muchas columnas son 0, por lo que es solo una base
for i,h in enumerate(H):   
    plt.subplot(2, 5, i+1)
    reshape_row(h)
    plt.title("basis images {}".format(str(i+1)), fontsize=20) 
    
plt.tight_layout()
plt.show()

#Transoformamos la data completa
W = nmf_estimator.transform(X)

print(W.shape)

i=0
#Tomamos la primera fila solamente
w=W[i,:]

#Podemos ver que los componentes de la fila, muchos tienen valores enteros diferentes, algunos 0
plt.bar([n+1 for n in range(len(w))],w)
plt.title("encodings for image {} ".format (i+1))
plt.show()

plt.figure(figsize=(15,4))

#ahora tomamos 131 filas, simplemente de X veamos la diferencia
#Es la imagen original en blanco y negro, la imagen 0
plt.subplot(131)
reshape_row(X[i,:])
plt.title("Original image")

#Es una imagen base diferente casi negra por completo, es otra imagen, es una base, las bases tienen mas manchas negras
#donde le faltan pixeles
plt.subplot(132)
reshape_row(H[1,:])
plt.title("Similar basis 2")

#Por ultimo otra base diferente de otra imagen, tiene casi nada de puntos negros, es simplemente una imagen mal hecha
plt.subplot(133)
reshape_row(H[8,:])
plt.title("Dissimilar basis 9")
plt.show()

#Hacemos la transformacion inversa
Xhat=nmf_estimator.inverse_transform(W)


plt.figure(figsize=(20,8))


#Muestra 2 filas de imagenes por 4 columnas, es decir 8 imagenes, pero solo muestra 2, porque no hay mas ploteadas
#al mismo tiempo, lo hace 4 veces, recordando que el ultimo numero no lo toma
#Primero toma la imagen original, luego solo una aproximacion de ella porque es una imagen recostruida
for i in range(1,5):
    plt.subplot(2,4,i)
    reshape_row(X[i])
    plt.title(f"Original image {i}")
    
    plt.subplot(2,4,i+4)
    reshape_row(Xhat[i])
    plt.title(f"Approximated image {i}")
    plt.show()

#Es primero la tranformacion multiplicada por los componentes
Xhat_M=W@H

print(Xhat[0,:10], Xhat_M[0,:10])

plt.figure(figsize=(20,8))

#Grafico de la transformacion inversa y luego de la multiplicacion de la trasnformacion por los componentes
#muy parecida las dos imagenes, practicamente iguales
for i in range(1,5):
    plt.subplot(2,4,i)
    reshape_row(Xhat[i])
    plt.title(f"Approximated image {i} with sklearn")
    
    plt.subplot(2,4,i+4)
    reshape_row(Xhat_M[i])
    plt.title(f"Approximated image {i} with matrix operation")
    plt.show()
   
# initialize an image array with 10000 zeros which will be reshaped as 100x100  
image=np.zeros((1,10000))

#w es solo una fila de W[i,:], pero con i fijo i=0, por lo que es la misma image trasformada * h que empieza de 0,
#i que itinera todos los filas
#La agregacion de valores cada vez mas altos, hace que la imagen tenga mas claridad en los tonos blanco y negro
plt.figure(figsize=(25,8))
for i, (w_, h) in enumerate(zip(w, H)):
    
    # w is the encoding vector of the first image in X
    # reconstruction of the image is a linear combination of H 
    plt.subplot(2,5,i+1)
    #La imagen suma cada vez mas valores, de 1 a 10, todas las imagenes varian
    image += w_*h
    reshape_row(image)
    plt.title(f"{i+1} components added", fontsize=20)
plt.tight_layout()
plt.show()



###Sistema de recuperacion de imagenes

#Encontrar imagenes similares por derechos de autor

X_q=get_data_matrix(test=True,Length=100,Width=100,mypath="images")
print(X_q.shape)


#El trasnformador hace cosas muy importante
#1. obtenemos la gran parte de a informacion importante
#2. las imagenes que son diferentes, pero similares, las reconoce como tal (similares)
#3. reduce los factores, que no dejan reconocerlos como similares, escala, rotaciones, ruido
#4. al ser una base de menos calidad requiere menos calculo que la imagen original

#Solo con 10 componentes
W_q=nmf_estimator.transform(X_q)
print(W_q.shape)


#Lo que se hace es ocupar la distancias en 2D, ocupamos la distancia coseno
#Lo que se hace entonces es poner las imagenes mas diferentes mas separadas unas de otras en un grafico 2D
#Al coseno regirse por los angulos, las imagenes que tengas caracteristicas similares, estaran todas juntas
#por lo tanto las agrupara como iguales o similares, esto proboca que cree solo una base


#Importamos parwise que mide la distancia entre imagenes
from sklearn.metrics import pairwise_distances

#Las imagenes ya transformadas, W y W_q son transformaciones, lo que diferencia es que X y X_q, el primero son
#los datos de entrenamiento y el segundo los datos de prueba
D=pairwise_distances(W,W_q,metric='cosine')

#son 9000 y 1000 elementos respectivamente
print(D.shape)

#Los puntos que minimizan la funcion
similar_index=np.argmin(D, axis=1)

#Los menores valores
similar_distance=np.min(D, axis=1)

#La mayoria de las distancias son muy cercanas a 0 en angulo
plt.hist(similar_distance,bins=100)
plt.title("Distance values")
plt.show()

#Son los datos que cumplen con ese umbral
dataset_index,query_index=threshold(similar_distance,max_=0.00001,min_=0)

#son (43, todo) (43, todo)
print(dataset_index.shape, query_index.shape)

#Son solo una muestra de 10 de los datos
print(dataset_index[:10])

#Son solo una muestra de 10 de los datos
print(query_index[:10])

#Tomamos otro umbral y sacamos diferentes datos
dataset_index,query_index=threshold(similar_distance,max_=0.005,min_=0.00001)

#Solo son una muestra de las imagenes (pares) que tienen distancias pequeñas
#por ende las toma como similares, no quiere decir que todas las imagenes sean iguales,
#solo las 2 que tomo tienen distancias similares, pero entre 4 por ejemplo 2 similares y las otras 2 son
#diferentes a las otras 2 por completo
#Plotea 5 pares
plot_data_query(dataset_index,query_index,5)

###Ejemplo 2

from sklearn.datasets import fetch_olivetti_faces

#Importamos una nueva data de rostros
rng = RandomState(0)
data = fetch_olivetti_faces(shuffle=True, random_state=rng)
X = data.images
X1 = X.reshape(400, -1)
print(X1.shape)
#Son 400 imagenes con 4096 componentes

image_shape = (64, 64)

#Defiimos una funcion para mostrarlas
def plot_faces(title, images, n_col, n_row, cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows = n_row,
        ncols = n_col,
        figsize = (2.0*n_col, 2.3*n_row),
        facecolor='white',
        constrained_layout=True)
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax)
        ax.axis('off')
    
    fig.colorbar(im, ax=axs, orientation='horizontal', shrink=0.99, 
                aspect=40, pad=0.01)
    plt.show()

#Solo muestra 6 caras del datasets, cantidad de imagenes, 3 columnas, 2 filas para plotear
plot_faces("6 Faces from the Original Dataset", X[:6], 3, 2)

#Dividimos la data X_r=son 300 imagenes, y X_q las ultimas 100 imagenes para prueba
X_r = X[:300].reshape((300, 64*64))
X_q = X[300:].reshape((100, 64*64))

#NMF para los difrentes datos por separado, tenemos de entrenamiento y prueba con 10 componentes
nmf = NMF(n_components=10, tol=5e-3,max_iter=2000)
X_r_W = nmf.fit_transform(X_r)
X_q_W = nmf.fit_transform(X_q)

print(X.shape)

#Nuevamente lo hacemos para la data entera con 6 componentes
nmf = NMF(n_components=6, tol=5e-3,max_iter=2000)
nmf.fit(X1)

from sklearn.metrics import pairwise_distances

#Vemos la distancia entre los pares, Las imagenes transformadas de entrenamiento y de prueba
D = pairwise_distances(X_r_W, X_q_W, metric='cosine')

#Sacamos los componentes que estan en los modelos
H = nmf.components_

#Grafico solo de los componentes, recordado que los componentes es una especie de base, tienen menos calidad
#por lo tanto son bases, muestra 3 columnas con 2 filas, en total 6 imagenes
plot_faces("Basis from dataset", H, 3, 2)

#Tomamos los minimos de las distancias
similar_index = np.argmin(D, axis=1)
similar_distance = np.min(D, axis=1)

#Ahora transformamos toda la data a 6 componentes
W = nmf.transform(X1)
w6 = W[:6, :] #tomamos solo 6 imagenes

#Tomamos la transformacion inversa, la que disminuye la calidad el ruido y las ploteamos
X_hat = nmf.inverse_transform(w6)
plot_faces("6 Reconstructed faces using inverse_transform", X_hat, 3, 2)

#tomamos de umbral minimo de las distancias
o_index, q_index = threshold(similar_distance)
Length=64
Width=64
#Ploteamos 5 de las que cumplen lo minimo de distancia, son diferentes a pesar que las toma como iguales
plot_data_query(o_index, q_index, 5)

#Ṕor ultimo tomamos esas 6 trasnformadas y las multiplicamos por los componentes
X_hat_M = w6@H

#Graficamos las transformadas * componentes y luego las mismas, pero con inversa, parece que fueran iguales
plot_faces("6 Reconstructed faces using matrix operations", X_hat_M, 3,2)
print("-------------------------------------------------------")
plot_faces("6 Reconstructed faces using inverse_transform", X_hat, 3, 2)


#######################################################################################################################

                                #Frecuencia de termino-Documento frecencia inversa
                                                     #if-idf
                                
#if-idf se ocupa para analizar las frecuencia de las palabras en un texto

import re
import skillsnetwork
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# You can also use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

#Tenemos dos frases
D = ["We like dogs and cats", "We like cars and planes"]

#Con countVectorizer creamos los termino es una matriz
#Lo que hace es si tenemos seleccionada la opcion los vuelve a minuscula todas las palabras
#Luego reconoce todas las palaras iguales y las cuenta, en case a eso hace la vectorizacion
#la primera frase que le pongamos sera la primera fila de la vectorizacion y la segundo, la segunda fila
#si la palabra aparece en la primera fila, marcara con un 1, si aparece 2 veces con un 2, si no aparece 0
#si aparece en la segunda fila solamente la palabra, marcara un 1 en la segunda fila
#      "and" "or"
#frase1 0      1
#frase2 1      2
cv = CountVectorizer()
tf_mat = cv.fit_transform(D)
tf = pd.DataFrame(tf_mat.toarray(), columns = cv.get_feature_names_out())
print(tf)

#En la matriz anterior, crea una normalizacion para los datos, de modo que cada fila es una vector, estos
#vectores estan normalizados cuando tienen norma 1, sqrt(componentes**2 +...)=1
tfidf_trans = TfidfTransformer(smooth_idf=False)
tfidf_mat = tfidf_trans.fit_transform(tf)
tfidf = pd.DataFrame(tfidf_mat.toarray(), columns = tfidf_trans.get_feature_names_out())

pd.DataFrame(tfidf_trans.idf_ * tf.to_numpy(), columns = tfidf_trans.get_feature_names_out())

print(tfidf)

print(tfidf.iloc[0,:])
# d * d
print(np.multiply(tfidf.iloc[0,:], tfidf.iloc[0,:]).sum().round())

###Ejemplo 2

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/tfidf.csv').iloc[:,1]
# df = pd.read_json('tfidf.json')
#df = pd.read_csv('tfidf.csv').iloc[:,1]

#Cada fila es una frase muy larga
print(df.head(5))

#Definimos una funcion para texto, lo que hace es pasarlo a minuscula
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text

#Pasamos a una matriz vector
cv = CountVectorizer(max_features = 500, preprocessor = preprocess_text)
tf = cv.fit_transform(df)
pd.DataFrame(tf.toarray(), columns = cv.get_feature_names_out())

#la normalizamos
tfidf_trans = TfidfTransformer()
tfidf_mat = tfidf_trans.fit_transform(tf.toarray())
tfidf = pd.DataFrame(tfidf_mat.toarray(), columns = cv.get_feature_names_out())
print(tfidf)

#Es lo mismo, pero con formato denso de la lista, es otra version solamente
tfidf = TfidfTransformer()
tfidf_mat = tfidf.fit_transform(tf.toarray())
pd.DataFrame(tfidf_mat.toarray(), columns = cv.get_feature_names_out())
print(tfidf)

#No me fuciono stack
#dense_tfidf = tfidf.stack()
#print(dense_tfidf[dense_tfidf != 0])


#######################################################################################################################

                                        #Factorizacion matricial no negativa
                                        
                                        

import urllib3
import urllib
import requests
from scipy.io import mmread
from io import BytesIO
with urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/bbc.mtx') as r: content = r.readlines()[2:]


sparsemat = [tuple(map(int,map(float, c.split()))) for c in content]
# Let's examine the first few elements
#ES una lista con tuplas
print(sparsemat[:8])


import numpy as np
from scipy.sparse import coo_matrix
rows = [x[0] for x in sparsemat]
cols = [x[1] for x in sparsemat]
values = [x[2] for x in sparsemat]
coo = coo_matrix((values, (rows, cols)))

#otra forma de obtener la misma matriz

#response = requests.get('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/bbc.mtx')
#coo = mmread(BytesIO(response.content))


# Surpress warnings from using older version of sklearn:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#Ocupando NMF el mismo que reconoce copyright
from sklearn.decomposition import NMF
model = NMF(n_components=5, init='random', random_state=818)
doc_topic = model.fit_transform(coo)

#Solo toma 5 componentes, son 5 columnas
print(doc_topic.shape)
# we should have 9636 observations (articles) and five latent features

# find feature with highest value per doc
#Tomamos los que tienen los valores maximos
np.argmax(doc_topic, axis=1)

#Aqui toma los componentes como filas
print(model.components_.shape)


#Con otra data
with urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/bbc.terms') as r:
    content = r.readlines()
words = [c.split()[0] for c in content]


topic_words = []
for r in model.components_:
    a = sorted([(v,i) for i,v in enumerate(r)],reverse=True)[0:12]
    topic_words.append([words[e[1]] for e in a])

print(topic_words[:5])

with urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/bbc.docs') as r:
    doc_content = r.readlines()
    
print(doc_content[:8])










                           
