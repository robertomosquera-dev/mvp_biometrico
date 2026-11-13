
import cv2
import time

from src.landing_file import detector, antispoof

# =====================================
# 1. INICIAR WEBCAM
# =====================================

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise RuntimeError("No se pudo abrir la webcam")

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam iniciada correctamente")

tiempo_anterior = time.perf_counter()

# =====================================
# 2. PROCESAR FRAMES
# =====================================

try:
    while True:

        ret, frame = webcam.read()

        if not ret:
            print("Error al capturar el frame")
            break

        alto, ancho = frame.shape[:2]

        detector.setInputSize((ancho, alto))

        # Detectar rostros con YuNet
        _, rostros = detector.detect(frame)

        cantidad = 0 if rostros is None else len(rostros)

        if cantidad > 1:
            mensaje = "Solo debe aparecer un rostro"
        elif cantidad == 1:
            rostro = rostros[0]
            score_real = antispoof.evaluar(frame, rostro)

        mensaje = "Coloca tu rostro frente a la camara"
        color = (255, 255, 255)

        if cantidad > 1:
            mensaje = "Solo debe aparecer un rostro"
            color = (0, 0, 255)

        elif cantidad == 1:

            rostro = rostros[0]

            x, y, w, h = rostro[:4].astype(int)

            # Ejecutar MiniFASNetV2
            score_real = antispoof.evaluar(
                frame,
                rostro
            )

            # Solo mostramos la puntuacion
            # No usamos un umbral de autenticacion
            mensaje = f"PAD score: {score_real:.3f}"
            color = (0, 255, 255)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            print(
                f"\rPAD score: {score_real:.4f}",
                end="",
                flush=True
            )

        # =====================================
        # 3. CALCULAR FPS
        # =====================================

        tiempo_actual = time.perf_counter()

        diferencia = tiempo_actual - tiempo_anterior

        fps = 1 / diferencia if diferencia > 0 else 0

        tiempo_anterior = tiempo_actual

        # =====================================
        # 4. MOSTRAR INFORMACION
        # =====================================

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
            f"FPS: {fps:.1f}",
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
            0.7,
            color,
            2
        )

        cv2.imshow(
            "Prueba Anti-Spoofing - MiniFASNetV2",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    webcam.release()
    cv2.destroyAllWindows()
