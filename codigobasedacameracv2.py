import cv2
import numpy as np

# Função para pré-processamento da imagem
def preprocess_image(image):
    # Conversão para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicação de filtro gaussiano para suavização
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detecção de bordas usando o algoritmo Canny
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Função para detecção da linha
def detect_line(image):
    # Realizar a detecção de linhas usando a transformada de Hough
    lines = cv2.HoughLinesP(image, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        # Desenhar as linhas detectadas na imagem original (apenas para visualização)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Calcular a posição média da linha
        mean_x = np.mean(lines[:,:,0:2])
        mean_y = np.mean(lines[:,:,1:3])
        return (mean_x, mean_y)
    else:
        return None

# Função principal
def main():
    # Inicializar a captura de vídeo da câmera (0 representa a câmera padrão)
    cap = cv2.VideoCapture(0)

    while True:
        # Capturar frame da câmera
        ret, frame = cap.read()
        if not ret:
            break

        # Pré-processar a imagem
        processed_image = preprocess_image(frame)

        # Detectar a linha
        line_position = detect_line(processed_image)

        if line_position is not None:
            # Controlar o movimento do robô com base na posição da linha
            # Aqui você pode adicionar lógica para controlar o movimento do robô com base na posição da linha
            # Por exemplo, você pode girar para a esquerda ou direita com base na posição da linha em relação ao centro da imagem

        # Exibir a imagem processada
        cv2.imshow('Processed Image', processed_image)

        # Verificar se o usuário pressionou a tecla 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
