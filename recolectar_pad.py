
import csv
import cv2
from pathlib import Path
from datetime import datetime, timezone

from src.landing_file import detector, antispoof

RAIZ = Path(__file__).resolve().parent
ARCHIVO = RAIZ / "data" / "pad_scores.csv"
ARCHIVO.parent.mkdir(exist_ok=True)

ESCENARIOS = {
    ord("1"): ("genuine", "real"),
    ord("2"): ("attack", "impresa"),
    ord("3"): ("attack", "pantalla"),
    ord("4"): ("attack", "video"),
}


sesion = input("ID de sesion (ej. real_01): ").strip()

if not sesion:
    raise ValueError("La sesion no puede estar vacia")

escenario = None
score = None
cantidad = 0

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise RuntimeError("No se pudo abrir la webcam")

try:
    while True:
        ret, frame = webcam.read()

        if not ret:
            break

        alto, ancho = frame.shape[:2]
        detector.setInputSize((ancho, alto))

        _, rostros = detector.detect(frame)
        cantidad = 0 if rostros is None else len(rostros)

        score = None

        if cantidad == 1:
            rostro = rostros[0]
            score = antispoof.evaluar(frame, rostro)

        texto = (
            f"Score PAD: {score:.4f}"
            if score is not None
            else f"Rostros: {cantidad} (se requiere 1)"
        )

        cv2.putText(
            frame, texto, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        nombre = escenario[1] if escenario else "sin elegir"

        cv2.putText(
            frame,
            f"Escenario: {nombre}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

        cv2.putText(
            frame,
            "1 Real | 2 Impresa | 3 Pantalla | 4 Video",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

        cv2.putText(
            frame,
            "ESPACIO: guardar | Q: salir",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

        cv2.imshow("Evaluacion PAD", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

        if tecla in ESCENARIOS:
            escenario = ESCENARIOS[tecla]
            print("Escenario seleccionado:", escenario[1])

        if tecla == 32:  # ESPACIO
            if escenario is None or score is None:
                print("Selecciona escenario y muestra un solo rostro")
                continue

            archivo_nuevo = not ARCHIVO.exists()

            with ARCHIVO.open("a", newline="") as archivo:
                escritor = csv.writer(archivo)

                if archivo_nuevo:
                    escritor.writerow([
                        "fecha", "sesion", "label",
                        "tipo", "score"
                    ])

                escritor.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    sesion,
                    escenario[0],
                    escenario[1],
                    f"{score:.8f}"
                ])

            print(
                f"Guardado: {escenario[1]} "
                f"score={score:.4f}"
            )

finally:
    webcam.release()
    cv2.destroyAllWindows()
