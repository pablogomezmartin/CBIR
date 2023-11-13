
# imports
from typing import Tuple
import cv2
import torch
import numpy as np
from scipy.cluster.vq import kmeans, vq

#------------------------------------------SIFT extraction functions-------------------------------
def sift_descriptors_and_kp(image) -> Tuple:

    '''
    Dada una imagen devuelve una tupla (kp, des) siendo
    kp los keypoints SIFT y des los descriptors SIFT
    '''
 
    sift = cv2.SIFT_create()

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    return (keypoints, descriptors)



#------------------------------------------ORB extraction functions-------------------------------
def fast_kp(image) -> Tuple[cv2.KeyPoint]:
    '''
    Dada una imagen devuelve los puntos clave FAST
    '''
    fast = cv2.FastFeatureDetector_create(35)
    fast.setNonmaxSuppression(False)
    keypoints = fast.detect(image, None)
    
    return keypoints


def orb_descriptors(image, kp_fast) -> Tuple:

    '''
    Dada una imagen y sus keypoints calculados con FAST 
    devuelve una tupla los descriptores binarios ORB
    '''

    # edgeThreshold: This is size of the border where the features are not detected.
    orb = cv2.ORB_create(edgeThreshold=10)

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #kp = orb.detect(img_gray,None)
    _, descriptors = orb.compute(img_gray, kp_fast)

    return descriptors



#------------------------------------------CNN features extraction functions-------------------------------
def preprocessing_for_CNN(image) -> np.array:

    '''
    Dada una imagen la preprocesa para que la CNN la utilice
    '''

    img = image.astype(np.uint8)
    x = np.expand_dims(img, axis=0) # batch x width x height x channels
    x = np.rollaxis(x, 3, 1) # batch x channels x width x height
    x = x.astype(np.float32)
    x = x - x.mean()
    x = x / x.std()
    image = torch.from_numpy(x).float()

    return image


def feature_map(model, image, layer='conv_1') -> np.array:
     
    '''
    Dada una imagen, la VGG-16 y la capa de la que se quieren
    las features, ddevuelve un vector con las mismas
    '''

    layers = {'conv_1': 1, 'conv_2': 3, 'conv_3': 6,
          'conv_4': 8, 'conv_5': 11, 'conv_5': 13,
          'conv_6': 15, 'conv_7': 18, 'conv_8': 20,
          'conv_9': 22, 'conv_10': 25, 'conv_11': 27,
          'conv_12': 29}
    assert layer in layers, 'layer not found'
     
    feat_maps = model.features[:layers[layer]](image).detach().squeeze()
    return feat_maps.numpy().flatten()



#------------------------------------------Color histogram extraction functions-------------------------------
def get_histogram(image, bins=64) -> np.array:

    '''
    Dada una imagen y el número bins para el histograma
    devuelve los tres histogramas de color concatenados
    y normalizados
    '''

    red = cv2.calcHist([image], [0], None, [bins], [0, 256])
    green = cv2.calcHist([image], [1], None, [bins], [0, 256])
    blue = cv2.calcHist([image], [2], None, [bins], [0, 256])
    
    vector = np.concatenate([red, green, blue], axis=0)
    vector = np.divide(vector.reshape(-1), image.shape[0]*image.shape[1])
    return vector



#------------------------------------------Bag of Words extraction functions-------------------------------
def get_tfidf(df) -> np.array:
    
    '''
    Dada una imagen devuelve su array TF-IDF
    '''
    
    N = len(df['desc_sift'])  # Número de imágenes en el dataset
    desc_sift = df['desc_sift']
    all_descriptors = []
    # Unir todos los descriptores en un solo array
    for img_descriptors in desc_sift: 
        for descriptor in img_descriptors:
            all_descriptors.append(descriptor)
    
    # Aplanar el array
    all_descriptors = np.stack(all_descriptors)
    
    # Creamos el codebook / diccionario haciendo clustering con k-means
    k = 200
    iters = 1
    codebook, _ = kmeans(all_descriptors, k, iters)
    
    # A cada imagen le asignamos sus palabras visuales y las almacenamos en un array
    visual_words = []
    for img_descriptors in desc_sift:
        img_visual_words, _ = vq(img_descriptors, codebook)
        visual_words.append(img_visual_words)

    # Obtenemos los vectores de frecuencias de palabras visuales para cada imagen
    frequency_vectors = []
    for img_visual_words in visual_words:
        img_frequency_vector = np.zeros(k)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)

    frequency_vectors = np.stack(frequency_vectors)
    
    # Obtenemos el número de imágenes en las que aparece cada palabra visual
    freq = np.sum(frequency_vectors > 0, axis=0)
    idf = np.log(N/freq)
    
    tfidf = frequency_vectors * idf
    return tfidf
