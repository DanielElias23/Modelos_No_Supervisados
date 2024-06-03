#En este ejemplo en realidad hay 3 modelos debido a las funcionalidades que tienen
#1. Componentes principales (version graficos)
#2. Clustering
#3. Imagenes con componentes principales


#######################################################################################################################
"""
                                        #Modelo de componentes principales
                                        
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

#Una data de clientes de un mayorista, son valores anuales
data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/Wholesale_Customers_Data.csv', sep=',')

print(data.shape)

print(data.head())

#Eliminamos algunas columnas clasificatorias, en el caso de PCA no nos sirve, esta informacion no se reduce
#con la combinacion lineal
data = data.drop(['Channel', 'Region'], axis=1)

#Esto muestra que los datos tienen una relacion demaciado irrgular con diferentes formas, no se puede ocuapr PCA
#sns.set_context('notebook')
#sns.set_style('white')
#sns.pairplot(data)
#plt.show()

#Debemos ver el tipo de datos, vemos que son todos int
print(data.dtypes)

#Debemos pasarlos a float normalizarlos y  luego escalarlos, porque PCA no puede procesar datos en bruto, porque se 
#sesgan al ser algunos valores mucho mayores que otros
for col in data.columns:
    data[col] = data[col].astype(float)

data_orig = data.copy()

log_columns = data.skew().sort_values(ascending=False)

#Primero debemos normalizar lo mas posible los datos para que el PCA sea efectivamente represetativo
for col in log_columns.index:
    data[col] = np.log1p(data[col])

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

#Escalamos las columnas
#squeeze le quita un dimension a los array [[[1,2]]] con squezee pasa a ser solo [[1,2]], es necesario para operar
#el for incluirle una dimencion mas y luego quitarsela
for col in data.columns:
    data[col] = mms.fit_transform(data[[col]]).squeeze()
    
print(data)

#Los datos contiene cierta relacion de datos de comglomerados ahora
#sns.set_context('notebook')
#sns.set_style('white')
#sns.pairplot(data)
#plt.show()

#Ahora bien muchas veces vamos a querer disolver todo lo que les hemos hecho a la data, por lo que se recomienda ocupar
#un PIPE, debido a que la data queda asi por todo el programa

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

#Para tener un funcion para aplicar
log_transformer = FunctionTransformer(np.log1p)

estimators = [('log1p', log_transformer), ('minmaxscale', MinMaxScaler())]
pipeline = Pipeline(estimators)

#Le aplicamos los estimadores otra vez a la data original
data_pipe = pipeline.fit_transform(data_orig)

#Nos dice si las dos data son iguales dentro de cierta tolerancia de decimales, esto dado que los decimales
#pueden ser diferentes
print(np.allclose(data_pipe, data))

from sklearn.decomposition import PCA

#Aplicamos PCA a la data
PCAmod = PCA(n_components=2)
PCAmod.fit(data)
data2 = PCAmod.transform(data)

#Ahora paso a ser una data de 2 columnas y la misma cantidad de filas
#Recordar que cada fila, es una compresion de las columas anteriores
print(data2.shape)

#Cambia los nombres de las columnas de las nueva data, porque quedan sin nombre
hwdf_PCA = pd.DataFrame(columns=[f'Projection on Component {i+1}' for i in range(len(pd.DataFrame(data2).columns))], data=data2)
print(hwdf_PCA.head())

plt.scatter(x=hwdf_PCA.iloc[:,0], y=hwdf_PCA.iloc[:,1])
plt.show()

pca_list = list()
feature_weight_list = list()


#Aplicamos denuevo PCA a la data para ver con cuantos n_componentes se puede explicar
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
#Pero no es necesario, solo es mas informacion explicativa de la data anterior, lo importante es comprimir la informacion
sns.set_context('talk')
ax = pca_df['var'].plot(kind="bar")

ax.set(xlabel='Number of dimensions',
       ylabel='Percent explained variance',
       title='Explained Variance vs Dimensions')
plt.show()


###KernelPCA hace exactamente lo mismo, pero generalmente es peor que PCA, excepto en imagenes, lo que hace es
#aumentar la dimencion y luego reducirla, puede elegir diferenes kernel en este caso

#Apesar de que ocupo gridsearch no sirve tanto, porque siempre elige lo mas explicativo, tendera a casi tener
#la data original

from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

#Para saber la puntuacion del modelo
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

param_grid = {'gamma':[0.001, 0.01, 0.05, 0.1, 0.5, 1.0,1.5],
              'n_components': [2, 3, 4,5,6]}

#Aplicamos GridSearch para encontrar los mejores parametros, hay que tener ojo aca, el scorer lo inventamos
#debido a que PCA no tiene puntuacion como tal
kernelPCA = GridSearchCV(KernelPCA(kernel='rbf', fit_inverse_transform=True),
                         param_grid=param_grid,
                         scoring=scorer,
                         n_jobs=-1)

#Aplicamos PCA a la data
kernelPCA = kernelPCA.fit(data)

print(kernelPCA.best_estimator_)

KPCAmod = KernelPCA(n_components=2, kernel="sigmoid", fit_inverse_transform=False, gamma=1)
KPCAmod.fit(data)
data3 = KPCAmod.transform(data)

print(data3.shape)

#Cambia los nombres de las columnas de las nueva data, porque quedan sin nombre
hwdf_PCA2 = pd.DataFrame(columns=[f'Projection on Component {i+1}' for i in range(len(pd.DataFrame(data3).columns))], data=data3)
print(hwdf_PCA2.head())

#Vemos que KernelPCA depende el kernel que se le aplique
#kernel=linear es igual a usar PCA
#kernel=rbf es un poco diferente a PCA, pero se parecen bastante
#kernel=sigmoid es mas comprimido que PCA
#kernel=poly se expande mucho mas que PCA
plt.scatter(x=hwdf_PCA2.iloc[:,0], y=hwdf_PCA2.iloc[:,1], c="blue")
plt.scatter(x=hwdf_PCA.iloc[:,0], y=hwdf_PCA.iloc[:,1], c="red")
plt.show()


#TSNE es para datos que no tienen mucha correlacion, para eso esta PCA
#la metrica tiene que ver mas con las imagenes y su interpretacion
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,
#init="pca" es mas estable
        init="random",
        n_iter=500,
        n_iter_without_progress=150,
        perplexity=10,
        random_state=0,
        #puede ser "cosine", "jaccard", "cityblock" y otras mas genera diferentes resultados
        metric="euclidean")

e_data = tsne.fit_transform(data)

print(pd.DataFrame(e_data))



###MDS es solo para datos cuadradas, con la misma cantidad de filas y columnas
#lo que hace es diferente eso si, entiende que es una especie de data cuadrada donde, columnas y filas estan
#relacionadas por 1 dato.
#Ese dato lo divide en la cantidad de n_component que le especifiquemos, pero hace un computo de toda la data
#para hacerlo, cosa que las relaciones a pesar de tener 2 componentes se mantenga

#Esto es importante, puesto que encuentra mas relaciones que son veridicas, ejemplo en este caso solo tenia 1 dato
#al dividirlo entendio que cuando los separaba en 2, encontraba la latitud y longuitud

distance=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/distance.csv').set_index('name')
print(distance.head(8))

from sklearn.manifold import MDS

MDSmod =  MDS(dissimilarity='precomputed',n_components=2,random_state=0,max_iter=300,eps=1e-3)

data4 = MDSmod.fit_transform(distance)

df_t=pd.DataFrame(data4 , columns=["lon","lat"], index=distance.columns)
print(df_t.head(8))

#tambien podemos cambiarle la metrica de distancia que tomara, con squareform

from scipy.spatial.distance import squareform, pdist

#Lo hace manual
df = pd.DataFrame({
   'lon':[-58, 2, 145, 30.32, -4.03, -73.57, 36.82, -38.5],
   'lat':[-34, 49, -38, 59.93, 5.33, 45.52, -1.29, -12.97],
   'name':['Buenos Aires', 'Paris', 'Melbourne', 'St Petersbourg', 'Abidjan', 'Montreal', 'Nairobi', 'Salvador']})

#Puede ser cualquiera de estas distancias y logra diferentes resultados de computo
#d=['cosine','cityblock','seuclidean','sqeuclidean','cosine','hamming','jaccard','chebyshev','canberra','braycurtis']

df=df.set_index('name')

distance=pd.DataFrame(squareform(pdist(df.iloc[:, 1:],metric="cosine")), columns=df.index, index=df.index)

embedding =  MDS(dissimilarity='precomputed', random_state=0,n_components=2)
data5 = embedding.fit_transform(distance)
df_t=pd.DataFrame(data5 , columns=df.columns, index=df.index)

print(df_t.head(8))


#######################################################################################################################

                                                  #Clustering
                                                  
#Es una forma de clasificar a los datos, pero entendiendo que existen filas que son similares en sus datos, comparten
#atributos

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt                                                  

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#Queremos segmentar a los clientes
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/CustomerData.csv', index_col=0)

#Hay un problema con el indice debe empezar de 0
df=df.reset_index(drop=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data = le.fit_transform(df["Gender"])
data = pd.DataFrame(data, columns=["Gender"])
df=df.drop(columns=["Gender"])
df = pd.concat([df, data], axis=1, sort=True)
print(df.shape)
print(df.head())

df['Gender'] = df['Gender'].astype('category')

km = KMeans(n_clusters=5, random_state=42)
km.fit(df)

print(np.unique(km.labels_))

lab = pd.DataFrame(km.labels_, columns=["cluster KM"])
df1 = pd.concat([df, lab], axis=1, sort=True)

print(df.head())

print(pd.DataFrame(df1.groupby(by=["cluster KM"]).agg("mean")))
print(pd.DataFrame(df1.groupby(by=["cluster KM"]).size()))

for label in np.unique(km.labels_):
    X_ = df1[label == km.labels_]
    plt.scatter(X_['Annual Income (k$)'], X_['Spending Score (1-100)'], label=label)
plt.show()


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
for label in np.unique(km.labels_):
    X_ = df1[label == km.labels_]
    ax.scatter(X_['Annual Income (k$)'], X_['Spending Score (1-100)'], X_["Age"], label=label)
plt.show()


from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from itertools import chain
from matplotlib.patches import Ellipse

#Creamos un modelos con 5 diferentes conglomerados
#La diferencia es que GMM hace una prediccion mas probable y esa es la que muestra en colores
GMM = GaussianMixture(n_components=5, random_state=10)
GMM.fit(df)

pred = GMM.predict(df)
prob_X1 = GMM.predict_proba(df)

print(pred.shape)
print(pd.DataFrame(prob_X1))

#Es mas util cuando existe mas variedad, porque los define con un clustering con una probabilidad muy alta
print(pd.DataFrame(prob_X1).agg("max", axis=1))

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=pred)
plt.show()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], df["Age"], c=pred)
plt.show()


###DBSCAN no necesita especificar los cluster, sino que los define segun la metrica que ocupemos
from sklearn.cluster import DBSCAN

#metricas euclidean, cityblock (manhattan) son por distancias
#metrica cosine, es por distancias, pero definidas por angulos
#metrica jaccard, es especial para palabras y definir distancias entre ellas
#Hay que ir probando eps, necesita valores altos, pero no tantos cercanos a 10 a 20
dbscan = DBSCAN(eps=10, metric="euclidean")
dbscan.fit(df)
#-1 lo considera ruido no cluster
#print(dbscan.labels_)
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=dbscan.labels_)
plt.show()

#Mostrandolo en 3D se entiende porque no los pone como cluster muchos punto y los reconoce como ruido
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], df["Age"], c=dbscan.labels_)
plt.show()


from sklearn.cluster import MeanShift, estimate_bandwidth
###Meanshift es mas utilizado en imagenes, pero se puede ocupar en clustering, descubre cosas interesantes
#logra categorizar todas las instancias segun sus caracterisiticas similares

#Hay que ajustar el quantile y las muestras en ese caso para obtener buenos resultados, define los cluster automaticamente
bandwidth = estimate_bandwidth(df, quantile=0.1, n_samples=20)
ms = MeanShift(bandwidth=bandwidth , bin_seeding=True)
ms.fit(df)
#el cluster para cada instancia
#print(pd.DataFrame(ms.labels_))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=ms.labels_)
plt.show()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], df["Age"], c=ms.labels_)
plt.show()


#funciona muy parecido a KMeans, pero el gran beneficio es que podemos elegir el vinculo como se relacionan
#los puntos
from sklearn.cluster import AgglomerativeClustering

ag = AgglomerativeClustering(n_clusters=5, linkage='ward', compute_full_tree=True)
ag = ag.fit(df)
pred_agg = ag.fit_predict(df)

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=pred_agg)
plt.show()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], df["Age"], c=pred_agg)
plt.show()

lab2 = pd.DataFrame(pred_agg, columns=["AggClu"])
df2 = pd.concat([df, lab2], axis=1, sort=True)

print(df2.head())

print(pd.DataFrame(df2.groupby(by=["AggClu"]).agg("mean")))
print(pd.DataFrame(df2.groupby(by=["AggClu"]).size()))
"""

#######################################################################################################################

                                               #Imagenes

#Los usos de los modelos para imagenes son muy diferentes, por lo que se ocuparan datas diferentes

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from skimage import io

img = io.imread('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/images/kingfisher.jpeg')
plt.imshow(img)
plt.show()

#Recuerde que para RGB son 3 caracterisitcas, los 3 colores, pero como son colores mezclados se divididen
#en subdiviciones, cada pixel es una mezcla de 3 columnas
X = img.reshape(-1, 3)

#Lo que hace KMeans es a una images es decolorar la imagen segun los contornos
#Cada k de KMeans es como interpreta los bordes de la imagen, menos k menos lo interpreta y por lo tanto no ocupa
#tantos colores
#entre mas k, mas contornos interpreta y mas colores ocupa de la imagen original, colorea mejor la imagen
#En la practica es como ocupar ciertos filtros
#Los fondos en general se deinen mucho peor que las cosas mas cercanas que logra mejor definicion de colores
km = KMeans(n_clusters=3, random_state=42)
km.fit(X)

seg = np.zeros(X.shape)
for i in range(3):
    seg[km.labels_ == i] = km.cluster_centers_[i]
seg = seg.reshape(img.shape).astype(np.uint8)
plt.imshow(seg)
plt.show()

from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
# If img is greyscale, then change to .reshape(-1, 1):
X = img.reshape(-1, 3)

#Es muy parecido a KMeans, pero es un poco mas variable, muestra diferentes definiciones con cada intento
#Logra definiciones un poco peores que KMeans
gmm = GaussianMixture(n_components=3, covariance_type='tied')
gmm.fit(X)

labels = gmm.predict(X)

seg = np.zeros(X.shape)

for label in range(3):
    seg[labels == label] = gmm.means_[label]
seg = seg.reshape(img.shape).astype(np.uint8)

plt.figure(figsize=(6,6))
plt.imshow(seg)
plt.show()


from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
X = img.reshape(-1, 3)
cluster = DBSCAN(eps=10, min_samples=4)
cluster.fit(X)

seg = np.zeros(X.shape)

for label in range(3):
    seg[labels == label] = cluster.labels_[label]
seg = seg.reshape(img.shape).astype(np.uint8)

plt.figure(figsize=(6,6))
plt.imshow(seg)
plt.show()

#Ademas TSNE y DBSCAN, puede entender digitos en imagenes, pero un ejemplo un poco largo
#TSNE le reduce las dimenciones a las imagenes, luego se puede mostrar como en cluster define que numero es
#con DBSCAN, se pueden ocupar los otros modelos de cluster, aunque DBSCAN es el que logra mejores resultados

#Como meanshift define ciertos centros, estos centros son centros de cluster, en las imagenes estos se representan
#como si fueran capas, cada cluster es cierto color de la imagen

#Esto es importante porque si existen objetos de diferente color, los reconocera
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2 as cv
X = img.reshape(-1, 3)
print(X.shape)

bandwidth = estimate_bandwidth(X, quantile=.03, n_samples=1500)
print(bandwidth)

ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
ms.fit(X)

ms.predict(X)

#Grafica los puntos, igual que lo anteior no los define como colores
#ax.scatter3D(img[:,:,0],img[:,:,1],img[:,:,2])
#ax.set_title('Pixel Values ')
#plt.show()

cluster_int8=np.uint8(ms.cluster_centers_)
labeled=ms.labels_
for label in np.unique(labeled):
    result=np.zeros(X.shape,dtype=np.uint8)
    result[labeled==label,:]=cluster_int8[label,:]  
    plt.imshow(cv.cvtColor(result.reshape(img.shape), cv.COLOR_BGR2RGB))
    plt.show()


#Truncate necesita una carpeta con muchas imagenes del mismo lugar, reconoce cuando un objeto se mueve
#porque define que cosas son importantes en las imagenes y dice que es el fondo, logra mostrar solo el fondo
#eliminando lo que se mueve
from sklearn.decomposition import TruncatedSVD
svd_ = TruncatedSVD(n_components=1, random_state=42)
Z=svd_.fit_transform(X)


#Se ocupa para detectar objetos similares en las imagenes, puede ocurse para detectar infraccion a copyright
from sklearn.decomposition import NMF

#PCA y KernelPCA se pueden ocupar para quitar el ruido en una imagen, la limpia


#if- se ocupa para el recuento de palabras de los textos






























