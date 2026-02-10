import cv2
import numpy as np

image_path = 'minion.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Crear una copia de la imagen para dibujar los rectángulos
image_with_rois = image.copy()

# Listar regiones de interés (ROI) con sus colores y coordenadas (x, y)
regions = [
    ("Blanco", 197, 151),
    ("Rojo", 47, 299),
    ("Amarillo", 127, 183),
    ("Azul", 15, 255),
    ("Negro", 146, 295),
    ("Varios colores", 215, 340)
]

square_size = 10

print("\n" + "="*70)
print("Estadisticas por cada region de interes")
print("="*70)

# Guardar resultados para mostrar en formato tabla
results = []

# Procesar cada región de interés
for i, (name, x, y) in enumerate(regions, 1):
    # Calcular coordenadas del cuadrado de 10x10 centrado en (x, y)
    x1 = x - square_size // 2
    y1 = y - square_size // 2
    x2 = x1 + square_size
    y2 = y1 + square_size
    
    # Extraer Region de interes
    roi = image[y1:y2, x1:x2]
    
    # Dibujar rectangulos y etiquetas en la imagen principal
    cv2.rectangle(image_with_rois, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_with_rois, str(i), (x1-15, y1), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Calcular estadisticas para cada canal de color
    # Canal 0 = Azul, Canal 1 = Verde, Canal 2 = Rojo
    blue_channel = roi[:, :, 0]
    green_channel = roi[:, :, 1]
    red_channel = roi[:, :, 2]
    
    # Promedio por canal
    avg_blue = np.mean(blue_channel)
    avg_green = np.mean(green_channel)
    avg_red = np.mean(red_channel)
    
    # Desviacion estandar por canal
    std_blue = np.std(blue_channel)
    std_green = np.std(green_channel)
    std_red = np.std(red_channel)
    
    # Mostrar resultados para cada region
    print(f"\n{i}. {name} (Posicion: x={x}, y={y})")
    print(f"   Coordenadas: [{y1}:{y2}, {x1}:{x2}]")
    print(f"   Forma: {roi.shape}")
    print(f"   ---")
    print(f"   B:")
    print(f"      Promedio: {avg_blue:.2f}")
    print(f"      Desviacion estandar: {std_blue:.2f}")
    print(f"   G:")
    print(f"      Promedio: {avg_green:.2f}")
    print(f"      Desviacion estandar: {std_green:.2f}")
    print(f"   R:")
    print(f"      Promedio: {avg_red:.2f}")
    print(f"      Desviacion estandar: {std_red:.2f}")
    
    # Guardar resultados 
    results.append({
        'name': name,
        'avg_b': avg_blue, 'avg_g': avg_green, 'avg_r': avg_red,
        'std_b': std_blue, 'std_g': std_green, 'std_r': std_red
    })
    
    # Mostrar ROI ampliada para mejor visualizacion
    roi_enlarged = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(f"{i}. {name}", roi_enlarged)

# Mostrar tabla resumen de resultados
print("\n" + "="*70)
print("TABLA DE RESULTADOS")
print("="*70)
print(f"{'Color':<20} {'Promedio B':<10} {'Promedio G':<10} {'Promedio R':<10} {'Desv. B':<10} {'Desv. G':<10} {'Desv. R':<10}")
print("-"*70)

for result in results:
    print(f"{result['name']:<20} "
          f"{result['avg_b']:<10.2f} {result['avg_g']:<10.2f} {result['avg_r']:<10.2f} "
          f"{result['std_b']:<10.2f} {result['std_g']:<10.2f} {result['std_r']:<10.2f}")

# Mostrar imagen con todas las regiones de interes marcadas
cv2.imshow('Imagen principal', image_with_rois)

print("\n" + "="*70)
print("Presiona cualquier boton para cerrar todas las ventanas")
print("="*70)

cv2.waitKey(0)
cv2.destroyAllWindows()