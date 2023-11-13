
#imports
import streamlit as st
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt

from retrieval_functions import return_n_nearest_CNN, return_n_nearest_SIFT, return_n_nearest_ORB, return_n_nearest_hist, return_n_nearest_bow
from visualization_functions import interface_view_results

# Para que no salgan los warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Texto de la pestaña y titulo de la página
st.set_page_config(layout="wide", page_title="CBIR Interface")
st.title('WELCOME TO OUR CBIR INTERFACE!📸')

# Desplegable 1
with st.expander('What is this app about?'):
    st.write('''En el mundo digital actual, la explosión de datos visuales y el crecimiento
exponencial de imágenes han planteado desafíos significativos en términos de
organización y recuperación de información visual. En este contexto, el concepto
de Recuperación de Imágenes Basado en el Contenido (CBIR, por sus siglas en
inglés) ha emergido como una poderosa técnica para abordar estos desafíos de
manera eficiente y efectiva.
CBIR es un enfoque revolucionario que busca transformar la forma en que
interactuamos con imágenes digitales. A diferencia de los sistemas tradicionales
de recuperación de imágenes que dependen de etiquetas o metadatos asociados
a las imágenes, CBIR se basa en el contenido visual de las mismas. Este enfoque
permite a los usuarios buscar imágenes similares a una consulta en función de
sus características visuales, como colores, texturas, formas y patrones.
En esencia, CBIR permite que las imágenes hablen por sí mismas, permitiendo a los usuarios explorar vastas colecciones de imágenes sin depender de
etiquetas o descripciones previas. Imagina poder buscar una imagen de un paisaje específico simplemente proporcionando una imagen similar como consulta,
o encontrar una obra de arte única en una extensa galería en línea con solo
cargar una imagen de referencia.
En este proyecto, exploraremos el emocionante mundo de CBIR y nos embarcaremos en el desarrollo de un sistema funcional de Recuperación de Imágenes
Basado en el Contenido. Nuestro objetivo es aplicar los conceptos teóricos y las
técnicas prácticas de procesamiento de imágenes y aprendizaje automático para
construir un sistema que pueda aprender a identificar y recuperar imágenes similares en función de sus características visuales, allanando así el camino hacia
una experiencia de búsqueda de imágenes más intuitiva y enriquecedora.
A lo largo de las próximas dos semanas, nuestros estudiantes se sumergirán en la investigación, el diseño y la implementación de un sistema CBIR,
enfrentando desafíos técnicos apasionantes y adquiriendo habilidades valiosas
en el proceso. Al finalizar este proyecto, esperamos que los participantes no solo hayan ganado experiencia en el campo del procesamiento de imágenes y el
aprendizaje automático, sino que también hayan contribuido al avance de la
tecnología de recuperación de imágenes basada en el contenido''')

# Desplegable 2
with st.expander('Our Dataset'):
    st.write(''' Para el desarrollo de esta práctica se ha utilizado un DataSet público que hemos obtenido de kaggle llamado [Caltech 256 Image Dataset](https://www.kaggle.com/datasets/jessicali9530/caltech256) creado por Griffin, G. Holub, AD. Perona, P.
            \nEn este conjunto de datos hay 30.607 imágenes, que abarcan 257 categorías de objetos. Las categorías de objetos son muy diversas, desde saltamontes hasta diapasones. Nosotros hemos realizado una selección del conjunto de datos original, reduciéndolo a 15 categorías específicas y limitando la cantidad de imágenes a 50 por cada una de ellas. De este modo, hemos obtenido un subconjunto que consta de 750 imágenes con las siguientes categorías
             ''')
    
    st.image('clases.png', 'Las 15 catergorías seleccionadas con ejemplos')


# Elegir el método a sacar
st.sidebar.write("## Upload and download :gear:")

selected_option = st.sidebar.selectbox("What method would you like to use to find the most similar images??", ["CNN", "SIFT", "ORB","Color histogram","Bag of Words"])

if selected_option == "CNN":
    st.sidebar.subheader("CNN Options")
    cnn_option = st.sidebar.radio("Choose distance", ["euclidean", "cosine"])

elif selected_option == "Color histogram":
    st.sidebar.text('Remind that for color histogram \nyou must use an image of the \ndataset')
    st.sidebar.subheader("Color Histogram Options")
    hist_option = st.sidebar.radio("Choose distance", ["euclidean", "cosine"])

elif selected_option == "Bag of Words":
    st.sidebar.subheader("BoW Options")
    bow_option = st.sidebar.radio("Choose distance", ["euclidean", "cosine"])



# Elegir cuantas fotos recuperar 
n_fotos = st.sidebar.slider('How much images do you want to retrieve?', 4, 10, step=2)

# Tamaño máximo de la foto que se va a insertar (5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024  

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    
    else:
        image = Image.open(my_upload)
        st.header("Query Image", divider='grey')
        st.image(image, use_column_width='auto')

        my_upload = np.array(image)

        df_images = pd.read_pickle('Images_df.pkl', compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 1})
        df_features = pd.read_pickle('Features_df.pkl', compression={'method': 'gzip', 'compresslevel': 3, 'mtime': 1})
        df = pd.concat([df_images, df_features], axis=1)

        # Descargamos el modelo preentrenado
        model = models.vgg16(weights='DEFAULT')
    
        if selected_option == "CNN":
            nearest = return_n_nearest_CNN(my_upload, df, model, norm=cnn_option , n=n_fotos)
            st.header(f'Retrieval Images with CNN and {cnn_option} distance', divider='grey')
            st.pyplot(interface_view_results(nearest, df, n=n_fotos))
        
        elif selected_option == "SIFT":
            nearest = return_n_nearest_SIFT(my_upload, df, n=n_fotos)
            st.header("Retrieval Images with SIFT", divider='grey')
            st.pyplot(interface_view_results(nearest, df, n=n_fotos))

        elif selected_option == "ORB":
            nearest = return_n_nearest_ORB(my_upload, df, n=n_fotos)
            st.header("Retrieval Images with ORB", divider='grey')
            st.pyplot(interface_view_results(nearest, df, n=n_fotos))
        
        elif selected_option == "Color histogram":
            nearest = return_n_nearest_hist(my_upload, df, norm=hist_option)
            st.header(f'Retrieval Images with color histogram and {hist_option} distance', divider='grey')
            st.pyplot(interface_view_results(nearest, df, n=n_fotos))

        elif selected_option == 'Bag of Words':
            nearest = return_n_nearest_bow(my_upload, df, norm=bow_option, n=n_fotos)
            st.header(f'Retrieval Images with BoW and {bow_option} distance', divider='grey')
            st.pyplot(interface_view_results(nearest, df, n=n_fotos))
