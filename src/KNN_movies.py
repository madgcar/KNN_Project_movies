# Importo las librerias y cargo los Dataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import shutil

movies = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv')
shutil.unpack_archive('/workspace/KNN_Project_movies/data/raw/tmdb_5000_credits.zip')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Unire ambos Dataset por la variable en comun que es titulo

movies = movies.merge(credits, on='title')

# Seleccionamos las columnas para nuestro nuevo Dataset
movies = movies[['movie_id','title','overview','popularity','vote_average','genres','keywords','cast','crew']]

# Eliminamos los valores nulos del Dataset

movies.dropna(inplace = True)

# Transformaremos las columnas que tienen formato JSON, y extraeremos los valores que nos interesan con una funcion
import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies.dropna(inplace = True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()

# Haremos algo similar para las columnas cast

def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
        count +=1  
    return L

movies['cast'] = movies['cast'].apply(convert3)

# Ahora para Crew

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Convertiremos a la columna Overview en una lista

movies['overview'] = movies['overview'].apply(lambda x : x.split())

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

# Aplicamos la funcion collapse para que elimine los espacios

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# Creamos una nueva columna llamada etiquetas

movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

# Nuevamente reduzco el Dataset solo a las columnas que usara el sistema de recomendacion

new_df = movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))

# Usamos el algoritmo KNN para crear el sistema de recomendacion
#Iniciando con la vectorizacion

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000 ,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

# Trabajamos con la similitud del coseno

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors).shape

similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])),reverse =True , key = lambda x:x[1])[1:6]

#Finalmente, crearemos una función de recomendación basada en cosine_similarity. Esta función debe recomendar las 5 películas más similares.

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] #obteniendo el índice de la película
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate( distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# Probamos el sistema de recomendacion

recommend('John Carter') ### Esta es muy buena :)