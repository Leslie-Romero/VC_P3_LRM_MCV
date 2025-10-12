# Práctica 3 de VC (Visión por Computador)

## Autores

Leslie Liu Romero Martín
<br>
María Cabrera Vérgez

## Tareas realizadas

Para la tercera práctica de la asignatura se debe de poder adquirir información de objetos a partir de imágenes. El fin de la misma es obtener las características que identifican a cada figura de manera automática. Se divide en dos tareas:

1. Los ejemplos ilustrativos anteriores permiten saber el número de monedas presentes en la imagen. ¿Cómo saber la cantidad de dinero presente en ella? Sugerimos identificar de forma interactiva (por ejemplo haciendo clic en la imagen) una moneda de un valor determinado en la imagen (por ejemplo de 1€). Tras obtener esa información y las dimensiones en milímetros de las distintas monedas, realiza una propuesta para estimar la cantidad de dinero en la imagen. Muestra la cuenta de monedas y dinero sobre la imagen. No hay restricciones sobre utilizar medidas geométricas o de color.

Una vez resuelto el reto con la imagen ideal proporcionada, captura una o varias imágenes con monedas. Aplica el mismo esquema, tras identificar la moneda del valor determinado, calcula el dinero presente en la imagen. ¿Funciona correctamente? ¿Se observan problemas?

2. La tarea consiste en extraer características (geométricas y/o visuales) de las tres imágenes completas de partida, y *aprender* patrones que permitan identificar las partículas en nuevas imágenes. Para ello se proporciona como imagen de test *MPs_test.jpg* y sus correpondientes anotaciones *MPs_test_bbs.csv* con la que deben obtener las métricas para su propuesta de clasificación de microplásticos, además de la matriz de confusión. La matriz de confusión permitirá mostrar para cada clase el número de muestras que se clasifican correctamente de dicha clase, y el número de muestras que se clasifican incorrectamente como perteneciente a una de las otras dos clases.

## Instalación
```py
import cv2  
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
```

Además, para poder elaborar la matriz de confusión, es requisito instalar los paquetes scikit-learn y seaborn.

```bash
pip install scikit-learn seaborn
```

## Tareas

### Tarea 1

Para esta tarea, comenzamos con el preprocesamiento de la imagen. Convertimos la imagen a escala de grises y aplicamos el umbralizado de OTSU junto con un umbralizado invertido, este umbralizado siempre viene precedido por un Guassian Blur para producir mejores resultados.

```py
# Leer la imagen de las monedas
img = cv2.imread('./content/Monedas.jpg') 

# Convierte la imagen a todos de gris, mostrando el resultado
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Umbralizado binario invertido
blur = cv2.GaussianBlur(img_gris,(5,5),0)
ret, img_th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

Después de preparar la imagen, extraemos los contornos con la función de OpenCV findCountours(). Nos aseguramos de extraer solo los contornos externos y no todos los que se encuentran en la imagen, para poder encontrar aquellos contornos que rodean a las monedas más fácilmente.

```py
# Buscamos los contornos
contours, hierarchy = cv2.findContours(img_th1, 
    cv2.RETR_EXTERNAL , 
    cv2.CHAIN_APPROX_SIMPLE)
```

Ya en cuanto a la implementación de la tarea, que incluye identificar las monedas correctamente en la imagen y extraer sus datos, usamos la función de OpenCV minEnclosingCircle() que, dado un contorno, aproxima el círculo mínimo. De esta manera, podemos obtener el diámetro de la moneda en píxeles fácilmente.

```py
(x,y), radius = cv2.minEnclosingCircle(approx)
```

Una vez hemos guardado todos los círculos que los contornos nos han permitido detectar de la imagen, procedemos a la parte interactiva del ejercicio. La función click() se encarga de leer los eventos del mouse y registrar cuando se ha pulsado el botón izquierdo.

```py
def click(event, x, y, flags, param):
    global selected
    moneda_2 = None
    escala = 0
    total_monedas = 0
    # Si presionoo el botón izquierdo del ratón
    if event == cv2.EVENT_LBUTTONDOWN:
```

Para nuestra tarea, hemos decidido utilizar la moneda de 2€ como referencia. De esta manera, el usuario pulsará sobre la moneda y nosotros extraemos el tamaño en píxeles del radio del círculo seleccionado y, a partir de ahí, establecemos la escala de conversión de píxeles (px) a milímetros (mm) y podemos obtener el valor del resto de monedas presentes en la imagen. Una vez el usuario ha seleccionado la moneda de 2€, mostramos los contornos que hemos considerado como monedas en color azul.

Para poder comprobar de qué moneda se trata, usamos un array (valores) donde tenemos los diámetros de cada moneda en mm y en px y dejamos un margen de 0.5 mm para no ser demasiado restrictivos.

```py 
valores = [
    [25.75, 2.00],
    [23.25, 1.00],
    [24.25, 0.50,],
    [22.25, 0.20,],
    [19.75, 0.10,],
    [21.25, 0.05,],
    [18.75, 0.02,],
    [16.25, 0.01]
]
```

La variable total_monedas suma el valor en euros que vamos acumulando.

```py
r2 = circles[moneda_2][1]
escala = 25.75/(r2*2)
for (center, r) in circles:
    tamaño = (r*2)*escala
    for [mm, moneda] in valores:
        if tamaño <= (mm + 0.5) and tamaño >= (mm - 0.5):
            cv2.circle(img, center, r, (255, 0, 0), 3)
            moneda = int(moneda*100)
            total_monedas += moneda
```

Finalmente, mostramos el valor total en euros que hemos calculado en forma de texto.

```py
# Show text
text = f'Total: {total_monedas/100}'
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = img.shape[0]/400
grosor = int(font_scale)+3
text_size = cv2.getTextSize(text, font_face, font_scale, grosor)[0][1]
print(text_size)
cv2.putText(img, text, (40, (text_size+50)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), grosor)
cv2.imshow("Testing", img)
```


### Tarea 2

Se ha creado en primer lugar una función que será usada luego para poder mostrar

## Referencias

- https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
- https://ieeexplore.ieee.org/document/8976153
- https://docs.kanaries.net/es/topics/Python/export-dataframe-to-csv
