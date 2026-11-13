
import cv2
import numpy as np
import onnxruntime as ort


class AntiSpoofing:

    def __init__(self, ruta_modelo):
        self.sesion = ort.InferenceSession(
            str(ruta_modelo),
            providers=["CPUExecutionProvider"]
        )

        self.entrada = self.sesion.get_inputs()[0].name
        self.salida = self.sesion.get_outputs()[0].name

    def preprocesar(self, frame, rostro):
        alto_img, ancho_img = frame.shape[:2]

        x, y, w, h = rostro[:4].astype(int)

        if w <= 0 or h <= 0:
            raise ValueError("Region facial invalida")

        escala = min(
            2.7,
            (alto_img - 1) / h,
            (ancho_img - 1) / w
        )

        centro_x = x + w / 2
        centro_y = y + h / 2

        nuevo_w = w * escala
        nuevo_h = h * escala

        x1 = max(0, int(centro_x - nuevo_w / 2))
        y1 = max(0, int(centro_y - nuevo_h / 2))
        x2 = min(ancho_img, int(centro_x + nuevo_w / 2))
        y2 = min(alto_img, int(centro_y + nuevo_h / 2))

        recorte = frame[y1:y2, x1:x2]

        if recorte.size == 0:
            raise ValueError("Recorte facial vacio")

        recorte = cv2.resize(recorte, (80, 80))

        # BGR, float32, NCHW
        tensor = recorte.astype(np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor

    def evaluar(self, frame, rostro):
        tensor = self.preprocesar(frame, rostro)

        salida = self.sesion.run(
            [self.salida],
            {self.entrada: tensor}
        )[0]

        # Convertir logits a puntuaciones de clase
        logits = salida - np.max(
            salida, axis=1, keepdims=True
        )

        exp = np.exp(logits)
        puntuaciones = exp / np.sum(
            exp, axis=1, keepdims=True
        )

        # En este modelo, la clase 1 corresponde
        # a una presentación genuina.
        score_real = float(puntuaciones[0, 1])

        return score_real
