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

Se ha creado en primer lugar una función que será usada luego para poder mostrar tres imágenes, una al lado de otra. Además, se recibe un booleano para saber si están en escala de grises.

``` py
ax1.set_axis_off()
    ax1.set_title('Fragmentos')
    if is_gray:
        ax1.imshow(img1, cmap='gray')
    else:
        ax1.imshow(img1)
```

Se muestra una versión de cada imagen limpia, tanto a color como en escala de grises. De esta forma, obtenemos las imágenes con las que vamos a estar trabajando.

La siguiente función, process_image(), lee la imagen en escala de grises, aplica un suavizado gaussiano para eliminar frecuencias altas. Se aplicarán dos umbrales: un primero, adaptativo, para separar los objetos del fondo, y un segundo, limpiando la imagen.

De esta forma, las figuras con las que se va a trabajar estarán de color blanco y el fondo de color negro, marcando una diferencia.

Lo siguiente que se necesita es poder detectar los contornos. Esto se hace creado una lista que tendrá los contornos en array de puntos.

``` py
cnt, hierarchy = cv2.findContours(img_th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

Para poder visualizarlos, se pintan los contornos sobre un fondo negro.

Lo siguiente es calcular las características de cada una de las formas para obtener su medias para más tarde. Para ello, calculamos: el área, perímetro, compacidad, relación área/rectángulo, relación ancho/alto y la relación de los ejes.

#### Área

Se cogen solo aquellos áreas que estén por encima de un cierto umbral (410), para evitar casos donde se tomen medidas que no son ciertas.

``` py
areas = [cv2.contourArea(c) for c in cnt if cv2.contourArea(c) > min_area]
    areas_aux = [[cv2.contourArea(c), c] for c in cnt if cv2.contourArea(c) > min_area]
```

#### Perímetro

Al igual que en el caso del área, se coge solo aquellos perímetros que superen un umbral establecido (250, 195, 150, según el elemento con el que se trabaje).

``` py
    perims = [cv2.arcLength(c, True) for c in cnt if cv2.arcLength(c, True) > min_perim]
```

#### Compacidad

La compacidad mide que tan circular es una forma. Su fórmula se basa en el perímetro al cuadrado dividido el área. 

``` py
    compact = [(p**2)/a for p, a in zip(perims, areas)]
```

#### Relación área/rectángulo contenedor

Se debe calcular el rectángulo mínimo con el contorno. Primero se calcula el perímetro del contorno y luego se simplifica, disminuyendo la cantidad de vérticas que tiene. Con las dimensiones del contenedor, se calcula el área. Finalmente, se calcula su relación con una división.

Para poder hacerlo, se hizo uso de un array que además de contener el área, tenía los contornos.

``` py
a2c_ratio = []
    selected_cnt = []
    for (area, c) in areas_aux:
        epsilon = 0.1*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        container_area = w*h
        a2c_ratio.append(area/container_area)
        # Guardamos los contornos seleccionados en un array aparte
        selected_cnt.append(c)
```

#### Relación ancho/alto

Se repite lo que se hizo para el caso anterior, pero en esta ocasión se gaurdará la división entre el ancho y alto.

``` py
w2h_ratio = []
    for (area, c) in areas_aux:
        epsilon = 0.1*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        if h > 0:
            w2h_ratio.append(w/h)

```

#### Relación ejes de elipse

Se obtienen los dos ejes: el mayor y menor. Esto es para poder calcular la relación entre estos dos, como pide el apartado. Se ajustó una ellipse para poder obtener los datos que se necesitaban.

``` py
# Axis ratio
    axis_ratio = []
    for (area, c) in areas_aux:
        ellipse = cv2.fitEllipse(c)
        (center, axes, angle) = ellipse
        axis_ratio.append(min(axes)/max(axes))
```

Para comprobar si todo ha sido calculado de manera correcta, se pinta los contornos que hemos recibido en cada imagen y se imprimen las diferentes características que hemos recibido de cada figura.

El siguiente paso es clasifica lo que tenemos en una imagen con una mezcla de elementos. Para ello, sea crea la función classifier().

El inicio de la función es casi el mismo que el de la función anterior: se lee la imagen en escala de grises, se suavisa y usan dos umbrales.

Se ha calculado la media de cada array para saber que datos muestran mayor diferencia entre las figuras y, por lo tanto, se pueden usar.

```
Area fragmentos 1473.15
Area pellets 3301.5
Area tar 2831.3125
--------------------------------------
Perimetro fragmentos 435.15456955432893
Perimetro pellets 319.8555981571024
Perimetro tar 217.5460925102234
--------------------------------------
Compactness fragmentos 230.3932154595233
Compactness pellets 72.6674463079967
Compactness tar 19.04169939876367
--------------------------------------
Area to container ratio fragmentos 0.6070461349579281
Area to container ratio pellets 0.7619148276035955
Area to container ratio tar 0.7921912698722103
--------------------------------------
Width to height ratio fragmentos 5.591190173520857
Width to height ratio pellets 0.9013230050366623
Width to height ratio tar 1.0102915355596893
--------------------------------------
Axis ratio fragmentos 0.6126878747522126
Axis ratio pellets 0.8655030311474872
Axis ratio tar 0.6711621225963896

```

Dentro de estos datos, se observa que en la compacidad, los fragmentos presentar valores muy por encima de los pellets y alquitrán, por lo que se busca aquellos elementos que cumplan el rango de compacidad. Además, los fragmentos también tienen un valor mayor en la relación entre el ancho y altom por lo que debe de ajustarse también a esta medida.

``` py
if comp >= min_comp and comp <= max_comp and w2h_ratio >= min_w2hr and w2h_ratio <= max_w2hr:
            # Fragments = RED
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (255, 0, 0), 3)
            classified = True
```

Se dibuja un rectángulo rojo para identificarlos.

Si, por el contrario, nos fijamos en la relación entre los ejes, estaremos buscando los elementos que coinciden con los pellets, pues el axis ratio de los fragmentos y alquitrán es bastante parecida, destacando solo los pellets. Si se está dentro del rango, se pinta un cuadrado de color verde.

``` py
 ellipse = cv2.fitEllipse(c)
        (center, axes, angle) = ellipse
        axis_ratio = min(axes)/max(axes)
        if classified == False and axis_ratio >= min_axr and axis_ratio <= max_axr:
            # Pellets = GREEN
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 3)
            classified = True
```

Si no entra dentro de ninguno de los rangos, se deduce que tiene que ser alquitrán. Esto se debe a que las medias de esta figura no destacan antes las otras dos.

``` py
if classified == False:
            # Tar = BLUE
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 0, 255), 3)

    return img_color
```

El área se tuvo que escalar para conseguir el área mínima, cambiando según la resolución de la imagen. Se llama al clasificador, indicándole los rangos que se van a estar usando.

``` py
classified_img = classifier('./content/MPs_test.jpg', min(compact1), max(compact1), min(w2h_ratio1), max(w2h_ratio1), min(axis_ratio2), max(axis_ratio2), min_area)
```

Se compara entre una imagen sin clasificar y la imagen después del clasificado, para que se pueda ver que era cada elemento que fue detectado o no.

Para poder comparar los datos, se elaboró una función llamada build_dataframe() que servía para guardar los elementos en un csv, donde los datos estarían separados por columnas.

``` py
def build_dataframe(label, areas, perims, compacts, a2c_ratios, w2h_ratios, axis_ratios):
    data = {
        'Label': label,
        'Area': areas,
        'Perimeter': perims,
        'Compactness': compacts,
        'Area_to_Container_Ratio': a2c_ratios,
        'Width_to_Height_Ratio': w2h_ratios,
        'Axis_Ratio': axis_ratios
    }
    return pd.DataFrame(data)

df_fragments = build_dataframe('Fragmentos', area1, perim1, compact1, a2c_ratio1, w2h_ratio1, axis_ratio1)
df_pellets = build_dataframe('Pellet', area2, perim2, compact2, a2c_ratio2, w2h_ratio2, axis_ratio2)
df_tar = build_dataframe('Alquitrán', area3, perim3, compact3, a2c_ratio3, w2h_ratio3, axis_ratio3)

df = pd.concat([df_fragments, df_pellets, df_tar], ignore_index=True)

df_numeric = df.select_dtypes(include=['float', 'int']).applymap(lambda x: str(x).replace('.', ','))

for col in df_numeric.columns:
    df[col] = df_numeric[col]

df.to_csv('particle_features.csv', index=False, sep=';')
```

Esto genera el csv "particle_features".

El último paso de la tarea es crear la matriz de confusión. Dentro del array real, y, se ecuentran los datos de referencia (contenido en MPS_test_bbs.csv). El segundo array contiene las predicciones del clasificador, cuyos datos han sido alineados.

Una vez los dos array están hechos, se crea la matriz pasándole los datos. Se presentarán los números enteros, con colores azules.

Al imprimirla, se puede comprobar que coinciden 36 fragmentos, 26 pellets y 6 de alquitrán.









## Referencias

- https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
- https://ieeexplore.ieee.org/document/8976153
- https://docs.kanaries.net/es/topics/Python/export-dataframe-to-csv
