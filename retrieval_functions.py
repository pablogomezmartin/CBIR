
# imports
import cv2
from typing import List
import pandas as pd
import numpy as np
from scipy.spatial import distance as sci_distance
from numpy import linalg

from feature_extraction_functions import sift_descriptors_and_kp, fast_kp, orb_descriptors
from feature_extraction_functions import preprocessing_for_CNN, feature_map, get_histogram



#---------------------------- SIFT n nearest ----------------------------------
def return_n_nearest_SIFT(query_image, df, n=10) -> List:
    
    '''
    Dada una query image, el df de features y n, devuelve una lista ordenada 
    (según cercanía con la query image para descriptores SIFT) con n tuplas 
    (i, distance) siendo i el índice del dataframe y distance su distancia a 
    la query image
    '''

    distances_l = []
    bf = cv2.BFMatcher(crossCheck=True)
    _, descriptors1 = sift_descriptors_and_kp(query_image)

    for i in df.index:
        descriptors2 = df.iloc[i, 2]
        if descriptors2 is None:
            pass
        else:
            matches = bf.match(descriptors1, descriptors2)    
            distance = np.mean([match.distance for match in matches])
            
            if not np.array_equal(query_image, df.iloc[i, 0]):
                distances_l.append((i, distance))

    distances_l = sorted(distances_l, key = lambda x:x[1])

    return distances_l[0:n]



#---------------------------- ORB n nearest -----------------------------------
def return_n_nearest_ORB(query_image, df, n=10) -> List:
        
    '''
    Dada una query image, el df de features y n, devuelve una lista ordenada 
    (según cercanía con la query image para descriptores ORB) con n tuplas 
    (i, distance) siendo i el índice del dataframe y distance su distancia a 
    la query image
    '''
        
    distances_l = []

    # Nótese que al ser binarios se usa la norma de Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp_fast = fast_kp(query_image)
    descriptors1 = orb_descriptors(query_image, kp_fast)

    for i in df.index:
        descriptors2 = df.iloc[i, 3]
        if descriptors2 is None:
            pass
        else:
            matches = bf.match(descriptors1, descriptors2)    
            distance = np.mean([match.distance for match in matches])
            
            if not np.array_equal(query_image, df.iloc[i, 0]):
                distances_l.append((i, distance))

    distances_l = sorted(distances_l, key = lambda x:x[1])

    return distances_l[0:n]



#---------------------------- CNN n nearest -----------------------------------
def return_n_nearest_CNN(query_image, df, model, norm='euclidean', n=10) -> List:

    '''
    Dada la query image, el df de features, la VGG-16 preentrenada, la norma
    y el número de imágenes que quieres que devuelva n, devuelve una lista ordenada 
    (según cercanía con la query image para mapas de features) con n tuplas 
    (i, distance) siendo i el índice del dataframe y distance su distancia a 
    la query image
    '''

    # Preprocessing de la imagen query
    query_image2 = cv2.resize(query_image, (300, 300))
    query_image2 = preprocessing_for_CNN(query_image2)

    # Obtener el mapa de características de la imagen query
    query_map = feature_map(model, query_image2, layer='conv_11')

    # Calcular distancias con todos los mapas del dataset
    distances = []
    for i in df.index:
        map = df.iloc[i, 4]
        if norm == 'euclidean':
            distance = np.linalg.norm(query_map - map)
        elif norm == 'cosine':
            distance = sci_distance.cosine(query_map, map)
        
        if not np.array_equal(query_image, df.iloc[i, 0]):
            distances.append((i, distance))
    
    # Ordenarlos según la distancia
    distances = sorted(distances, key = lambda x: x[1])
    
    return distances[:n]



#---------------------------- color histogram n nearest -----------------------------------
def return_n_nearest_hist(query_image, df, norm='euclidean' ,n=10) -> List:

    '''
    Dada la query image, el df de features, la norma y el número de imágenes 
    que quieres que devuelva n, devuelve una lista ordenada (según cercanía 
    con la query image para el hist de color) con n tuplas (i, distance) 
    siendo i el índice del dataframe y distance su distancia a la query image
    '''

    # Calculamos el query image histogram
    query_hist = get_histogram(query_image)

    # Calcular distancias con todos los histogramas
    distances = []
    for i in df.index:
        hist = df.iloc[i, 5]
        if norm == 'euclidean':
            distance = np.linalg.norm(query_hist-hist)
        elif norm == 'cosine':
            distance = sci_distance.cosine(query_hist, hist)
        
        if not np.array_equal(query_image, df.iloc[i, 0]):
            distances.append((i, distance))
    
    # Ordenarlos según la distancia
    distances = sorted(distances, key = lambda x: x[1])
    
    return distances[:n]



#---------------------------- Bag of Words n nearest -----------------------------------
def return_n_nearest_bow(query_image, df, norm='euclidean', n=10) -> List:

    '''
    Dada la query image, el df de features, la norma y el número de imágenes 
    que quieres que devuelva n, devuelve una lista ordenada (según cercanía 
    con la query image para tf-idf) con n tuplas (i, distance) siendo i el 
    índice del dataframe y distance su distancia a la query image
    '''

    index = df[df.image.apply(lambda x: np.array_equal(x, query_image))].index
    # Recuperar los resultados originales
    tfidf = df['tfidf'].apply(lambda x: pd.Series(x))
    query_tfidf = list(df.iloc[index, 6])[0]
    
    if norm == 'euclidean':
        euclidean_distances = np.sqrt(np.sum((query_tfidf - tfidf)**2, axis=1))
        idx = np.argsort(euclidean_distances)[1:n+1]
        distances = euclidean_distances[idx]

    elif norm == 'cosine':
        cosine_similarity = 1-np.dot(query_tfidf, tfidf.T)/(linalg.norm(query_tfidf) * linalg.norm(tfidf, axis=1))
        idx = np.argsort(cosine_similarity)[1:n+1]
        distances = cosine_similarity[idx]
    
    distances = list(zip(idx, distances))
    return distances


#---------------------------percentage of correct images in retrieval----------------------------------------
def correct_percentage(query_label, nearest_list, df) -> int:

    '''
    Devuelve el porcentaje de aciertos (número de imagenes de la misma clase que la query image) del CBIR
    '''
    
    total = len(nearest_list)
    counter = 0
    for i, _ in nearest_list:
        label = df.iloc[i, 1]
        if label == query_label:
            counter += 1
    print(f'\033[1mPorcentaje de aciertos: {(counter*100)/total}%\033[0m')

    return int((counter*100)/total)