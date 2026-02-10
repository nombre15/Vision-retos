import random
import time

# ============== DEFINICIONES DE FUNCIONES ==============

# Convertir matriz a formato PPM (Portable Pixmap)
def matrix_to_ppm(matrix, filename):
    height = len(matrix)
    width = len(matrix[0]) if height > 0 else 0 # Verificar que la matriz no esté vacía
    
    # Escribir el archivo PPM
    with open(filename, 'w') as f:
        
        f.write('P3\n') # P3 indica formato ASCII
        f.write(f'{width} {height}\n') # Ancho y alto de la imagen
        f.write('255\n') # Valor máximo de color (255 para RGB)
        
        # Crear contenido de la imagen
        for row in matrix:
            for pixel in row:
                if isinstance(pixel, int): # Si es un valor entero, es una imagen en escala de grises
                    f.write(f'{pixel} {pixel} {pixel} ')
                else: # En cambio, si es una tupla (r, g, b), es una imagen a color
                    r, g, b = pixel
                    f.write(f'{r} {g} {b} ')
            f.write('\n')

# Función para crear una matriz de colores aleatorios (1000x1000)
def create_random_matrix(width, height):
    matrix = []
    for i in range(height):
        row = []
        for j in range(width):
            # Elegir valores aleatorios para R, G y B (0-255)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            row.append((r, g, b))
        matrix.append(row)
    return matrix

def calculate_statistics(matrix):
    """
    Calcular el valor mínimo, máximo, promedio y desviación estándar de los
    valores de color en la matriz sin hacer uso de librerias
    """
    # Extraer todos los valores de color en una sola lista para facilitar cálculos
    all_values = []
    
    for row in matrix:
        for pixel in row:
            if isinstance(pixel, int):
                # Escala de grises: un solo valor
                all_values.append(pixel)
            else:
                # RGB: tres valores
                r, g, b = pixel
                all_values.append(r)
                all_values.append(g)
                all_values.append(b)
    
    # Calcular valor mínimo
    minimum = all_values[0]
    for value in all_values:
        if value < minimum:
            minimum = value
    
    # Calculatr valor máximo
    maximum = all_values[0]
    for value in all_values:
        if value > maximum:
            maximum = value
    
    # Calcular promedio
    total = 0
    count = 0
    for value in all_values:
        total += value
        count += 1
    average = total / count
    
    # Calcular desviación estándar
    sum_squared_diff = 0
    for value in all_values:
        diff = value - average
        sum_squared_diff += diff * diff
    
    variance = sum_squared_diff / count
    std_dev = variance ** 0.5
    
    # Obtener resultados
    return {
        'minimum': minimum,
        'maximum': maximum,
        'average': average,
        'std_dev': std_dev
    }

# Convertir matriz 2D a vector 1D
def matrix_to_vector(matrix):

    vector = []
    
    for row in matrix:
        for pixel in row:
            if isinstance(pixel, int):
                # Escala de grises: agregar un solo valor
                vector.append(pixel)
            else:
                # RGB: agregar los tres valores
                r, g, b = pixel
                vector.append(r)
                vector.append(g)
                vector.append(b)
    
    return vector

def save_vector_to_txt(vector, filename):
    """
    Guardar vector en un archivo de texto
    Cada valor en una línea separada para facilitar lectura y procesamiento posterior
    """
    with open(filename, 'w') as f:
        for value in vector:
            f.write(f'{value}\n')

# ===== PROGRAMA PRINCIPAL =====

print("=" * 60)
print("Generador de imagen aleatoria y cálculo de estadísticas")
print("=" * 60)

# Iniciar cronometro
total_start = time.time()

# Paso 1: Generar matriz de colores aleatorios
print("\nGenerando matriz aleatoria de 1000x1000...")
gen_start = time.time() # Inicio de generacion
random_image = create_random_matrix(1000, 1000)
gen_time = time.time() - gen_start
print(f"    Matriz generada en {gen_time:.3f} segundos")

# Paso 2: Calcular estadísticas
print("\nCalculando estadísticas...")
stats_start = time.time() # Inicio de calculo de estadisticas
stats = calculate_statistics(random_image)
stats_time = time.time() - stats_start
print(f"    Estadísticas calculadas en {stats_time:.3f} segundos")

total_time = time.time() - total_start # Tiempo total de la generación de la matriz y calculo de estadisticas

# Paso 3: Guardar imagen en formato PPM
matrix_to_ppm(random_image, 'random_colors.ppm')

# Paso 4: Convertir matriz a vector y guardar en formato TXT
vector = matrix_to_vector(random_image)
print(f"    Longitud de vector: {len(vector):,} valores")
save_vector_to_txt(vector, 'vector_data.txt')   

# Mostrar estadísticas y tiempos de ejecución de manera clara y organizada
print("\n" + "=" * 60)
print("Resultados de estadísticas")
print("=" * 60)
print(f"Valor minimo:       {stats['minimum']}")
print(f"Valor maximo:       {stats['maximum']}")
print(f"Promedio:       {stats['average']:.2f}")
print(f"Desviación estándar:  {stats['std_dev']:.2f}")

print("\n" + "=" * 60)
print("Tiempos de ejecución")
print("=" * 60)
print(f"Generación de matriz:   {gen_time:.3f} segundos ({gen_time/total_time*100:.1f}%)")
print(f"Calculo de estadisticas:     {stats_time:.3f} segundos ({stats_time/total_time*100:.1f}%)")
print(f"Tiempo total:          {total_time:.3f} segundos")
print("=" * 60)