
import cv2
import time

from src.landing_file import imagen, detector, reconocedor
from src.utils import generar_embedding
from src.quality import evaluar_calidad


UMBRAL_DEMO = 0.363



alto_img, ancho_img = imagen.shape[:2]
detector.setInputSize((ancho_img, alto_img))

_, rostros_img = detector.detect(imagen)

if rostros_img is None or len(rostros_img) != 1:
    raise RuntimeError(
        "La fotografia de referencia debe contener un solo rostro"
    )

rostro_referencia = rostros_img[0]

valido, mensaje = evaluar_calidad(
    imagen,
    rostro_referencia
)

if not valido:
    raise RuntimeError(
        f"Imagen de referencia invalida: {mensaje}"
    )

embedding_img = generar_embedding(
    imagen,
    rostro_referencia,
    reconocedor
)

print("Fotografia de referencia procesada correctamente")



webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise RuntimeError("No se pudo abrir la webcam")

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camara abierta correctamente")

tiempo_anterior = time.perf_counter()
fps_suavizados = 0.0


try:
    while True:

        ret, frame = webcam.read()

        if not ret:
            print("No se pudo capturar el frame")
            break

        alto, ancho = frame.shape[:2]

        detector.setInputSize((ancho, alto))

        _, rostros = detector.detect(frame)

        cantidad = 0 if rostros is None else len(rostros)

        mensaje = ""
        color = (255, 255, 255)

        # ----------------------------------
        # Validar cantidad de rostros
        # ----------------------------------

        if cantidad == 0:
            mensaje = "No se detecta ningun rostro"
            color = (0, 0, 255)

        elif cantidad > 1:
            mensaje = "Solo debe aparecer una persona"
            color = (0, 0, 255)

            for rostro in rostros:
                x, y, w, h = rostro[:4].astype(int)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

        else:
            rostro = rostros[0]

            x, y, w, h = rostro[:4].astype(int)
            confianza = float(rostro[-1])


            valido, mensaje = evaluar_calidad(
                frame,
                rostro
            )

            if not valido:
                color = (0, 165, 255)

            else:


                embedding_web = generar_embedding(
                    frame,
                    rostro,
                    reconocedor
                )


                similitud = reconocedor.match(
                    embedding_img,
                    embedding_web,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                coincide = similitud >= UMBRAL_DEMO

                color = (
                    (0, 255, 0) if coincide
                    else (0, 0, 255)
                )

                mensaje = (
                    f"Coincidencia DEMO: {similitud:.3f}"
                    if coincide
                    else f"No coincide DEMO: {similitud:.3f}"
                )



            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                f"Deteccion: {confianza:.2f}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            puntos = rostro[4:14].reshape(5, 2)

            for px, py in puntos.astype(int):
                cv2.circle(
                    frame,
                    (px, py),
                    3,
                    (0, 0, 255),
                    -1
                )


        tiempo_actual = time.perf_counter()
        diferencia = tiempo_actual - tiempo_anterior

        fps = 1 / diferencia if diferencia > 0 else 0

        tiempo_anterior = tiempo_actual

        fps_suavizados = (
            0.9 * fps_suavizados
            + 0.1 * fps
        )

        cv2.putText(
            frame,
            f"Rostros: {cantidad}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps_suavizados:.1f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            mensaje,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        cv2.imshow("Biometria facial - YuNet + SFace", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

finally:
    webcam.release()
    cv2.destroyAllWindows()
