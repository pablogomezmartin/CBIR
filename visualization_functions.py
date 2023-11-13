
# imports
import cv2
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt



def show_kp(image, label, kp) -> None:
    '''
    Dada una imagen, su clase y sus keypoints los dibuja
    '''
    kp_img = cv2.drawKeypoints(image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.title(f'{label}')
    plt.axis('off')
    plt.imshow(kp_img)



def view_results(query_image, query_label, nearest_list, df) -> None:
    ''' 
    Función que dada una query image y la lista con las imagenes más cercana las plotea
    '''
    fig = plt.figure(figsize=(18, 4))

    # Dividimos el área de la figura en una cuadrícula de 2x6 subplots
    grid = plt.GridSpec(2, 6, wspace=0.7, hspace=0.3, width_ratios=[3, 1, 1, 1, 1, 1])

    # Colocamos la imagen de consulta a la izquierda
    query_ax = fig.add_subplot(grid[:, 0])
    query_ax.imshow(query_image)
    query_ax.set_title("Query image")
    query_ax.set_xticks([])
    query_ax.set_xlabel(f'{query_label}')
    query_ax.set_yticks([])

    for i in range(len(nearest_list)):
        row = i // 5
        col = i % 5 + 1
        ax = fig.add_subplot(grid[row, col])

        ax.set_title(f'Metric value: {nearest_list[i][1]:.4f}', size=10)
        ax.set_xlabel(f'{df.iloc[nearest_list[i][0], 1]}')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(df.iloc[nearest_list[i][0], 0])

    ax.autoscale()
    plt.show()    




def interface_view_results(nearest_list, df, n) -> None:
    ''' 
    Función que dada una query image y la lista con las imágenes más cercanas las plotea
    '''

    # Calcula la cantidad de filas y columnas en la cuadrícula
    if n == 4:
        num_cols = 4
    else:
        num_cols = int(n/2)

    num_rows = (n + num_cols - 1) // num_cols

    fig = plt.figure(figsize=(12, 6))

    # Dividimos el área de la figura en una cuadrícula dinámica
    grid = plt.GridSpec(num_rows, num_cols, wspace=0.7, hspace=0.3, width_ratios= [1] * num_cols)

    for i in range(n):
        row = i // num_cols
        col = i % num_cols
        ax = fig.add_subplot(grid[row, col])

        ax.set_title(f'Metric value: {nearest_list[i][1]:.4f}', size=10)
        ax.set_xlabel(f'{df.iloc[nearest_list[i][0], 1]}')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(df.iloc[nearest_list[i][0], 0])

    ax.autoscale()
    plt.show()