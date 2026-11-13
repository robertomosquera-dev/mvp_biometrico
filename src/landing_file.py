
import cv2
from pathlib import Path
from src.antispoof import AntiSpoofing

raiz = Path(__file__).resolve().parent.parent

ruta_modelo_reconocedor = (
    raiz / "models" / "face_recognition_sface_2021dec.onnx"
)

ruta_modelo_detector = (
    raiz / "models" / "face_detection_yunet_2023mar.onnx"
)

ruta_modelo_antispoof = (
    raiz / "models" / "MiniFASNetV2.onnx"
)

ruta_img = (
    raiz / "img" / "rostro_4.jpeg"
)

for ruta in [
    ruta_modelo_reconocedor,
    ruta_modelo_detector,
    ruta_modelo_antispoof,
    ruta_img
]:
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe: {ruta}")

    print("Se encontro el archivo:", ruta)

imagen = cv2.imread(str(ruta_img))

if imagen is None:
    raise RuntimeError("No se pudo cargar la imagen")

reconocedor = cv2.FaceRecognizerSF.create(
    model=str(ruta_modelo_reconocedor),
    config=""
)

detector = cv2.FaceDetectorYN.create(
    model=str(ruta_modelo_detector),
    config="",
    input_size=(640, 480),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

antispoof = AntiSpoofing(
    ruta_modelo_antispoof
)

print("Los tres modelos se cargaron correctamente")
