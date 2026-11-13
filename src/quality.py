
import cv2
import numpy as np


def evaluar_calidad(frame, rostro):
    alto_img, ancho_img = frame.shape[:2]

    x, y, w, h = rostro[:4].astype(int)

    if w <= 0 or h <= 0:
        return False, "Region facial invalida"

    if (
        x < 0 or y < 0
        or x + w > ancho_img
        or y + h > alto_img
    ):
        return False, "Centra tu rostro en la camara"

    recorte = frame[y:y + h, x:x + w]

    if recorte.size == 0:
        return False, "Region facial invalida"

    gris = cv2.cvtColor(
        recorte,
        cv2.COLOR_BGR2GRAY
    )

    proporcion = w / ancho_img

    if proporcion < 0.20:
        return False, "Acercate a la camara"

    if proporcion > 0.65:
        return False, "Alejate un poco de la camara"

    nitidez = cv2.Laplacian(
        gris,
        cv2.CV_64F
    ).var()

    if nitidez < 80:
        return False, "La imagen esta borrosa"

    brillo = float(np.mean(gris))

    if brillo < 50:
        return False, "Mejora la iluminacion"

    if brillo > 210:
        return False, "Reduce la iluminacion"

    return True, "Calidad aceptable"
