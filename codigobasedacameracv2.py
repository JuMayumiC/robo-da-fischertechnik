import cv2

# Função para processar o frame e detectar a linha
def detect_line(frame):
    # Converter imagem para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicar threshold para binarizar a imagem
    _, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    # Detectar contornos na imagem binarizada
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Encontrar o maior contorno (presumindo que seja a linha)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Encontrar o centro da linha
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return None, None

# Inicializar a câmera
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame da câmera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Processar o frame e detectar a linha
    line_center_x, line_center_y = detect_line(frame)
    
    # Desenhar a linha detectada
    if line_center_x is not None and line_center_y is not None:
        cv2.circle(frame, (line_center_x, line_center_y), 5, (0, 255, 0), -1)
    
    # Mostrar o frame
    cv2.imshow("Line Following", frame)
    
    # Verificar se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
