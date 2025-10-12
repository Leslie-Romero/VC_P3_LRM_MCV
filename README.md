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
```
import cv2  
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
```

Además, para poder elaborar la matriz de confusión, es requisito instalar los paquetes scikit-learn y seaborn.

```
pip install scikit-learn seaborn
```

## Tareas

### Tarea 2

Se ha creado en primer lugar una función que será usada luego para poder mostrar

## Referencias

- https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
- https://ieeexplore.ieee.org/document/8976153
- https://docs.kanaries.net/es/topics/Python/export-dataframe-to-csv
