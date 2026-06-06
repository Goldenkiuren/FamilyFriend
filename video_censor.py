import cv2
import numpy as np

class VideoCensor:
    def __init__(self, blur_kernel_size=81):
        """
        blur_kernel_size: Define a força do desfoque (blur). O OpenCV exige que seja um número ímpar.
                          Quanto maior, mais forte o borrado. 81 é excelente para esconder texto.
        """
        # Garante matematicamente que o kernel do OpenCV seja sempre ímpar
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
            
        self.blur_kernel = (blur_kernel_size, blur_kernel_size)

    def process_frame(self, frame, bboxes):
        """
        Recebe um frame (numpy array do OpenCV/mss) e uma lista de bounding boxes.
        Aplica um desfoque forte nas regiões exatas.

        bboxes: Lista de coordenadas no formato devolvido pelo EasyOCR:
                [ [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], ... ]
        """
        # Trabalhamos em uma cópia para não alterar o frame de origem acidentalmente
        censored_frame = frame.copy()
        height, width = censored_frame.shape[:2]

        for bbox in bboxes:
            # Extrai todas as coordenadas X e Y do polígono retornado pelo EasyOCR
            xs = [int(pt[0]) for pt in bbox]
            ys = [int(pt[1]) for pt in bbox]

            # Encontra os limites do retângulo
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Segurança contra quebra: Garante que o blur não tente desenhar fora da tela
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            # Se a área for inválida (ex: coordenada negativa bizarra), pula silenciosamente
            if x_min >= x_max or y_min >= y_max:
                continue

            # 1. Corta a Região de Interesse (ROI - a área do palavrão)
            roi = censored_frame[y_min:y_max, x_min:x_max]

            # 2. Aplica um Gaussian Blur extremamente forte apenas na ROI
            blurred_roi = cv2.GaussianBlur(roi, self.blur_kernel, 0)

            # 3. Cola o trecho borrado de volta no frame principal
            censored_frame[y_min:y_max, x_min:x_max] = blurred_roi

        return censored_frame

# ==========================================
# Teste isolado do módulo
# ==========================================
if __name__ == "__main__":
    print("Módulo VideoCensor: Teste de Manipulação de Imagem")

    # Cria um frame falso (tela branca 800x600)
    mock_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # Simula uma ofensa desenhada no meio da tela
    cv2.putText(mock_frame, "OFFENSIVE TEXT", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Simula a "bounding box" que o EasyOCR retornaria para aquela região
    mock_bbox = [[[290, 270], [550, 270], [550, 310], [290, 310]]]

    censor = VideoCensor(blur_kernel_size=91)

    # Aplica a censura
    final_frame = censor.process_frame(mock_frame, mock_bbox)

    print("✅ Frame processado com sucesso com cálculos de limite protegidos.")
    
    # Se quiser ver a mágica funcionando, descomente as linhas abaixo e rode o script:
    # cv2.imshow("Teste Censor Visual", final_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()