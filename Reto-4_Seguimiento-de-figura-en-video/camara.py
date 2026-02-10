import cv2
import numpy as np

# Definir rangos de colores en HSV (Hue, Saturation, Value)
color_ranges = [
    ("Rojo", np.array([0, 120, 70]), np.array([10, 255, 255]), (0, 0, 255)),
    ("Rojo", np.array([170, 120, 70]), np.array([180, 255, 255]), (0, 0, 255)), 
    ("Azul", np.array([100, 120, 70]), np.array([130, 255, 255]), (255, 0, 0)),
    ("Verde", np.array([40, 120, 70]), np.array([80, 255, 255]), (0, 255, 0)),
    ("Amarillo", np.array([20, 120, 70]), np.array([35, 255, 255]), (0, 255, 255)),
    ("Naranja", np.array([10, 120, 70]), np.array([20, 255, 255]), (0, 165, 255)),
    ("Morado", np.array([130, 120, 70]), np.array([160, 255, 255]), (255, 0, 255)),
]

# Abrir camara
cap = cv2.VideoCapture(0)

if not cap.isOpened(): 
    print("Error: no se pudo abrir la camara")
    exit()

# Mientras la camara esté en uso
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error inesperado de la camara") 
        break
    
    # Girar horizontalmente para dar efecto espejo
    frame = cv2.flip(frame, 1)
    
    # Mejorar deteccion de color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    detected_objects = []
    
    for color_name, lower, upper, display_color in color_ranges:  # Intentar detectar cada color
        # Crear mascara para detectar los colores
        mask = cv2.inRange(hsv, lower, upper)
        
        # Remover ruido de la mascara para detectar los colores de mejor manera
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar encontornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Procesar cada contorno
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrar areas muy pequeñas
            if area > 500: # Solo mostrar areas mayores a 500 pixeles de diametro
                
                x, y, w, h = cv2.boundingRect(contour) # Obtener informacion de bounding box
                
                # Guardar deteccion
                detected_objects.append({
                    'name': color_name,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'area': area,
                    'color': display_color
                })
    
    # Dibujar cajas para los objetos detectados (bounding boxes)
    for obj in detected_objects:
        x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
        color_name = obj['name']
        display_color = obj['color']
        
        # Dibujar rectangulo
        cv2.rectangle(frame, (x, y), (x + w, y + h), display_color, 3)
        
        # Preparar texto de acuerdo al color
        label = f"{color_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Obtener informacion para dibujar el texto correctamente
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Dibujar fondo blanco para el texto
        cv2.rectangle(frame, 
                      (x, y - text_height - 10), 
                      (x + text_width + 10, y),
                      (255, 255, 255), -1)
        
        # Dibujar texto
        cv2.putText(frame, label, (x + 5, y - 5), 
                    font, font_scale, display_color, thickness)
        
        # Dibujar informacion del area
        area_text = f"Area: {obj['area']:.0f}"
        cv2.putText(frame, area_text, (x + 5, y + h + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 1)
    
    # Mostrar instrucciones para salir
    cv2.putText(frame, "Presiona q para salir", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Mostrar la camara en una ventana
    cv2.imshow('Deteccion de colores', frame)
    
    # Salir presionando '1'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cierre
cap.release()
cv2.destroyAllWindows()