import cv2


def generar_embedding(imagen, rostro, reconocedor):
    rostro_alineado = reconocedor.alignCrop(
        imagen,
        rostro
    )

    embedding = reconocedor.feature(
        rostro_alineado
    )

    return embedding

def registrar_rostro(frame, rostro, reconocedor):
    return generar_embedding(
        frame,
        rostro,
        reconocedor
    )

def verificar_rostro(
    frame,
    rostro,
    embedding_registrado,
    reconocedor
):
    embedding_actual = generar_embedding(
        frame,
        rostro,
        reconocedor
    )

    similitud = reconocedor.match(
        embedding_registrado,
        embedding_actual,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    return float(similitud)