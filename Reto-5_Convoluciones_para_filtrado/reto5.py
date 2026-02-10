"""
Reto 5 — “Convoluciones para filtrado”

Objetivo:
1) Partir de una imagen limpia (grises)
2) Añadir ruido Salt & Pepper
3) Probar filtros (media, gaussiano, mediana, moda) IMPLEMENTADOS EN PYTHON/NUMPY (sin OpenCV para el filtrado)
4) Comparar con OpenCV (recomendado: cv2.medianBlur)

Nota:
- Aquí SOLO usamos OpenCV al final para la comparación (y opcionalmente para mostrar).
- El filtrado principal se hace con funciones propias (bucles + NumPy).
"""

import numpy as np

# -----------------------------
# 1) Cargar imagen SIN OpenCV
# -----------------------------
from PIL import Image

# Cambia el nombre de tu archivo si hace falta
RUTA_IMAGEN = "imagen.png"

img_rgb = Image.open(RUTA_IMAGEN).convert("RGB")
img_rgb = np.array(img_rgb, dtype=np.uint8) #Se extraen alto, ancho y canales

# Pasar a escala de grises "a mano" (luma aproximada)
# (Si ya tienes una imagen en gris, puedes saltarte esto)
gris_limpia = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]).astype(np.uint8)
#Los decimales son intensidad con que el ojo humano percibe un color
#Tomar las columnas y filas pero solo de este canal ´(..., 0)´ para convertir a escala de grises

# -------------------------------------------------------
# 2) Función: añadir ruido Salt & Pepper (sal y pimienta)
# -------------------------------------------------------
def ruido_salt_pepper(imagen_gray: np.ndarray, cantidad: float = 0.05, sal_vs_pimienta: float = 0.5) -> np.ndarray:
    """
    imagen_gray: uint8 (H,W)
    cantidad: proporción de píxeles a corromper (0.0 a 1.0)
    sal_vs_pimienta: fracción de 'sal' (255) vs 'pimienta' (0)
    """
    #cantidad: densidad (porcentaje de la imagen) con ruido
    #sal_vs_pimienta: dentro de ese porcentaje de ruido, cuánto es sal (blanco) vs pimienta (negro)

    salida = imagen_gray.copy()
    h, w = salida.shape

    # Total de pixeles a afectar
    n = int(cantidad * h * w)

    # Cuántos serán "sal" y cuántos "pimienta"
    n_sal = int(n * sal_vs_pimienta)
    n_pimienta = n - n_sal

    # Coordenadas aleatorias (sal)
    ys = np.random.randint(0, h, size=n_sal)
    xs = np.random.randint(0, w, size=n_sal)
    salida[ys, xs] = 255

    # Coordenadas aleatorias (pimienta)
    ys = np.random.randint(0, h, size=n_pimienta)
    xs = np.random.randint(0, w, size=n_pimienta)
    salida[ys, xs] = 0

    return salida

gris_ruidosa = ruido_salt_pepper(gris_limpia, cantidad=0.08, sal_vs_pimienta=0.5)

# ------------------------------------------
# 3) Utilidades: padding + recorte de valores
# ------------------------------------------
def pad_replicate(imagen: np.ndarray, pad: int) -> np.ndarray:
    """
    Padding tipo "replicate" (bordes se repiten) parecido a cv2.BORDER_REPLICATE
    """
    return np.pad(imagen, ((pad, pad), (pad, pad)), mode="edge")
#ndarray: n-dimensional array
#((pad, pad), (pad, pad)): padding vertical y horizontal
#mode="edge": los bordes se repiten

def clip_uint8(x: np.ndarray) -> np.ndarray:
    """
    Asegura rango [0..255] y tipo uint8
    """
    return np.clip(x, 0, 255).astype(np.uint8)
#np.clip: limita los valores a un rango específico -> entre 0 y 255

# -----------------------------------------
# 4) Convolución 2D MANUAL (sin OpenCV)
# -----------------------------------------
def convolucion2d(imagen_gray: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convolución 2D manual:
    - imagen_gray: uint8 (H,W)
    - K: kernel float (kh,kw) con kh y kw impares.
    Retorna: uint8 (H,W)
    """
    img = imagen_gray.astype(np.float32) #para que no de la vuelva de nuevo si se pasa de 255
    kh, kw = K.shape

    # Validación simple
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("El kernel debe tener dimensiones impares (ej: 3x3, 5x5, 7x7).")

    pad = kh // 2 #de cuanto debe ser el pad
    img_pad = pad_replicate(img, pad) #la imagen original + el pad

    h, w = img.shape
    salida = np.zeros((h, w), dtype=np.float32)

    # Bucle recorrer cada píxel y multiplicar por kernel
    for y in range(h):
        for x in range(w):
            roi = img_pad[y:y+kh, x:x+kw]#movmiento de ventana (region of interest)
            salida[y, x] = np.sum(roi * K) #multiplicar elemento a elemento y sumar

    return clip_uint8(salida)

# -----------------------------------------
# 5) Kernels típicos: media y gaussiano
# -----------------------------------------
def kernel_media(k: int) -> np.ndarray:
    """
    Kernel promedio kxk
    """
    #k debe ser impar para tener un centro definido
    #kernel blur
    K = np.ones((k, k), dtype=np.float32) / (k * k)#cada elemento del kernel es 1/(k*k) para que la suma total sea 1 (conserva el brillo)
    return K

def kernel_gaussiano(k: int, sigma: float) -> np.ndarray:
    """
    Kernel gaussiano kxk con sigma dado.
    """
    #se intenta forma una campana gaussiana en el kernel, con el centro más brillante y los bordes más oscuros, dependiendo de la distancia al centro y el valor de sigma
    if k % 2 == 0:
        raise ValueError("k debe ser impar.")
    ax = np.arange(-(k//2), k//2 + 1, dtype=np.float32)#k//2: radio del kernel, va de -radio a +radio(no se incluye este por eso +1). y en el centro siempre va cero
    xx, yy = np.meshgrid(ax, ax)#crea dos matrices 2D con las coordenadas x e y de cada punto en el kernel, con los valores anteriores organizandolos en eje x e y
    K = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))#aplica la fórmula de la función gaussiana a cada punto del kernel, donde xx**2 + yy**2 es la distancia al centro al cuadrado, y sigma controla el ancho de la campana
    #sigam indica el tamaño de la campana. las cosas se desenfocan si estan lejos de ella
    #el menos indica que a mayor distancia menor importancia
    #esto es una formula matematica, funcion gaussiana en 2d
    K = K / np.sum(K)  # normalizar para conservar brillo
    return K.astype(np.float32)

# -----------------------------------------
# 6) Filtro mediana MANUAL (sin OpenCV)
# -----------------------------------------
def filtro_mediana(imagen_gray: np.ndarray, k: int) -> np.ndarray:
    """
    Filtro de mediana manual kxk (k impar).
    """
    if k % 2 == 0:
        raise ValueError("k debe ser impar.")
    pad = k // 2
    img_pad = pad_replicate(imagen_gray, pad)
    h, w = imagen_gray.shape
    salida = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            roi = img_pad[y:y+k, x:x+k]#ventana de interés
            salida[y, x] = np.median(roi).astype(np.uint8)

    return salida

# -----------------------------------------
# 7) Filtro moda MANUAL (sin OpenCV)
# -----------------------------------------
def filtro_moda(imagen_gray: np.ndarray, k: int) -> np.ndarray:
    """
    Filtro de moda (valor más repetido) en ventana kxk.
    Ojo: puede ser más lento que mediana/media.
    """
    if k % 2 == 0:
        raise ValueError("k debe ser impar.")
    pad = k // 2
    img_pad = pad_replicate(imagen_gray, pad)
    h, w = imagen_gray.shape
    salida = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            roi = img_pad[y:y+k, x:x+k].ravel()#ventana de interés
            # bincount funciona bien para uint8 (0..255)
            #ravel convierte la matriz en un arreglo 1d
            hist = np.bincount(roi, minlength=256)
            #bincount cuenta la cantidad de ocurrencias de cada valor en el arreglo, y minlength=256 asegura que el histograma tenga 256 bins (0 a 255)
            salida[y, x] = np.argmax(hist).astype(np.uint8)
            #np.argmax dice en qué posición (índice) se encuentra ese número más grande

    return salida

# -----------------------------------------
# 8) Aplicar filtros (SIN OpenCV)
# -----------------------------------------
# (a) Media 7x7
K_media_7 = kernel_media(7)
fil_media = convolucion2d(gris_ruidosa, K_media_7)

# (b) Gaussiano 7x7 (sigma típico: 1.0 a 2.0)
K_gauss_7 = kernel_gaussiano(7, sigma=1.5)
fil_gauss = convolucion2d(gris_ruidosa, K_gauss_7)

# (c) Mediana 5x5 (muy buena para salt&pepper)
fil_mediana = filtro_mediana(gris_ruidosa, 5)

# (d) Moda 5x5 (a veces ayuda con salt&pepper, depende mucho del caso)
fil_moda = filtro_moda(gris_ruidosa, 5)

# -----------------------------------------
# 9) Métricas simples: MSE / PSNR
# -----------------------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))#formula matematica
#np.mean: calcula el promedio de los elementos del arreglo resultante de (a - b) ** 2, que es la diferencia al cuadrado entre las dos imágenes. Esto da una medida de cuánto difieren las imágenes en promedio por píxel.
#mse: Error Cuadrático Medio, cuanto más bajo mejor (0 es perfecto)

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """
    PSNR en dB (a vs b). MAX=255
    """
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / m)#formula matematica para calcular PSNR
#PSNR: Relación Señal-Ruido en decibelios, cuanto más alto mejor (inf es perfecto)

print("=== Comparación contra la imagen LIMPIA (gris_limpia) ===")
print("Ruido S&P      -> MSE:", mse(gris_limpia, gris_ruidosa), "PSNR:", psnr(gris_limpia, gris_ruidosa))
print("Media 7x7      -> MSE:", mse(gris_limpia, fil_media),    "PSNR:", psnr(gris_limpia, fil_media))
print("Gauss 7x7      -> MSE:", mse(gris_limpia, fil_gauss),    "PSNR:", psnr(gris_limpia, fil_gauss))
print("Mediana 5x5    -> MSE:", mse(gris_limpia, fil_mediana),  "PSNR:", psnr(gris_limpia, fil_mediana))
print("Moda 5x5       -> MSE:", mse(gris_limpia, fil_moda),     "PSNR:", psnr(gris_limpia, fil_moda))


# ---------------------------------------------------------
# 10) Mostrar resultados con matplotlib - Navegación interactiva
# ---------------------------------------------------------
import matplotlib.pyplot as plt

# Lista de imágenes y sus títulos para navegación
imagenes = [
    ("Gris limpia", gris_limpia),
    ("Ruido Salt & Pepper", gris_ruidosa),
    ("Filtro Media 7x7", fil_media),
    ("Filtro Gaussiano 7x7", fil_gauss),
    ("Filtro Mediana 5x5", fil_mediana),
    ("Filtro Moda 5x5", fil_moda)
]

# -------------------------------------------------------------------
# 11) Comparación con OpenCV (recomendada: medianBlur) + filter2D
# -------------------------------------------------------------------
# Esta parte es SOLO para comparar (no es el "sin OpenCV").
# Si no tienes OpenCV instalado, comenta este bloque.
try:
    import cv2

    # OpenCV espera uint8 y, para medianBlur, k impar.
    cv_mediana = cv2.medianBlur(gris_ruidosa, 5)

    # Para comparar: filter2D con el mismo kernel de media/gauss
    cv_media = cv2.filter2D(gris_ruidosa, ddepth=-1, kernel=K_media_7)
    cv_gauss = cv2.filter2D(gris_ruidosa, ddepth=-1, kernel=K_gauss_7)

    print("\n=== Comparación OpenCV contra la imagen LIMPIA ===")
    print("cv2.filter2D Media 7x7   -> MSE:", mse(gris_limpia, cv_media),   "PSNR:", psnr(gris_limpia, cv_media))
    print("cv2.filter2D Gauss 7x7   -> MSE:", mse(gris_limpia, cv_gauss),   "PSNR:", psnr(gris_limpia, cv_gauss))
    print("cv2.medianBlur 5x5       -> MSE:", mse(gris_limpia, cv_mediana), "PSNR:", psnr(gris_limpia, cv_mediana))

    # Agregar imágenes de OpenCV a la lista
    imagenes.extend([
        ("OpenCV - filter2D Media 7x7", cv_media),
        ("OpenCV - filter2D Gauss 7x7", cv_gauss),
        ("OpenCV - medianBlur 5x5", cv_mediana)
    ])

except Exception as e:
    print("\n[INFO] Bloque OpenCV no se ejecutó (quizá no tienes cv2 instalado). Error:", e)

# ---------------------------------------------------------
# Navegación interactiva entre imágenes
# ---------------------------------------------------------
class VisorImagenes:
    def __init__(self, imagenes_lista):
        self.imagenes = imagenes_lista
        self.indice = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.mostrar_imagen()
        
    def mostrar_imagen(self):
        self.ax.clear()
        titulo, imagen = self.imagenes[self.indice]
        self.ax.imshow(imagen, cmap="gray")
        self.ax.set_title(f"{titulo}\n[Imagen {self.indice + 1}/{len(self.imagenes)}] - Usa ← → para navegar, 'q' para salir", 
                         fontsize=12, pad=10)
        self.ax.axis("off")
        self.fig.canvas.draw()
        
    def on_key(self, event):
        if event.key == 'right' or event.key == 'n':
            self.indice = (self.indice + 1) % len(self.imagenes)
            self.mostrar_imagen()
        elif event.key == 'left' or event.key == 'p':
            self.indice = (self.indice - 1) % len(self.imagenes)
            self.mostrar_imagen()
        elif event.key == 'q' or event.key == 'escape':
            plt.close(self.fig)

print("\n=== NAVEGACIÓN DE IMÁGENES ===")
print("Usa las flechas ← → (o 'p'/'n') para navegar entre imágenes")
print("Presiona 'q' o 'Esc' para salir")

visor = VisorImagenes(imagenes)
plt.show()