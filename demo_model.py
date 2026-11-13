
import cv2
import time
import mediapipe as mp

from pathlib import Path



raiz = Path(__file__).resolve().parent
ruta_modelo = raiz / "models" / "face_landmarker.task"

if not ruta_modelo.is_file():
    raise FileNotFoundError(f"No existe: {ruta_modelo}")

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

opciones = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(ruta_modelo)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True
)

UMBRAL_CERRADO = 0.60
UMBRAL_ABIERTO = 0.30
FRAMES_MINIMOS_CERRADO = 2

estado = "ESPERANDO_ABIERTO"
frames_cerrados = 0
parpadeos = 0



webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise RuntimeError("No se pudo abrir la webcam")

inicio = time.perf_counter()
ultimo_timestamp = -1

try:
    with FaceLandmarker.create_from_options(opciones) as landmarker:

        while True:
            ret, frame = webcam.read()

            if not ret:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            imagen_mp = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (time.perf_counter() - inicio) * 1000
            )
            timestamp_ms = max(
                timestamp_ms,
                ultimo_timestamp + 1
            )
            ultimo_timestamp = timestamp_ms

            resultado = landmarker.detect_for_video(
                imagen_mp,
                timestamp_ms
            )

            mensaje = "Colocate frente a la camara"

            if resultado.face_blendshapes:

                expresiones = {
                    item.category_name: item.score
                    for item in resultado.face_blendshapes[0]
                }

                ojo_izquierdo = expresiones.get(
                    "eyeBlinkLeft", 0.0
                )

                ojo_derecho = expresiones.get(
                    "eyeBlinkRight", 0.0
                )

                ambos_cerrados = (
                    ojo_izquierdo >= UMBRAL_CERRADO
                    and ojo_derecho >= UMBRAL_CERRADO
                )

                ambos_abiertos = (
                    ojo_izquierdo <= UMBRAL_ABIERTO
                    and ojo_derecho <= UMBRAL_ABIERTO
                )

                if estado == "ESPERANDO_ABIERTO":
                    if ambos_abiertos:
                        estado = "ESPERANDO_CIERRE"

                elif estado == "ESPERANDO_CIERRE":
                    if ambos_cerrados:
                        frames_cerrados = 1
                        estado = "CERRANDO"

                elif estado == "CERRANDO":
                    if ambos_cerrados:
                        frames_cerrados += 1

                    elif ambos_abiertos:
                        if frames_cerrados >= FRAMES_MINIMOS_CERRADO:
                            parpadeos += 1

                        frames_cerrados = 0
                        estado = "ESPERANDO_CIERRE"

                mensaje = f"Estado: {estado}"

                cv2.putText(
                    frame,
                    f"Ojo Izq: {ojo_izquierdo:.2f}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Ojo Der: {ojo_derecho:.2f}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            else:
                estado = "ESPERANDO_ABIERTO"
                frames_cerrados = 0

            cv2.putText(
                frame,
                f"Parpadeos: {parpadeos}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                mensaje,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            cv2.imshow("Clase 15 - Parpadeos", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    webcam.release()
    cv2.destroyAllWindows()
