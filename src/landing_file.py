import cv2
from pathlib import Path

raiz = Path(__file__).resolve().parent.parent

ruta_modelo_reconocedor = (
    raiz / "models" /
    "face_recognition_sface_2021dec.onnx"
)

ruta_modelo_detector = (
    raiz / "models"/ "face_detection_yunet_2023mar.onnx"
)

ruta_img = (
    raiz / "img" / "rostro_4.jpeg"
)

for ruta in [ruta_modelo_reconocedor,ruta_modelo_detector,ruta_img]:
    if not ruta.is_file():
        raise FileNotFoundError(f"No exite: ", ruta)
    else: 
        print("Se encontro el file la ruta: ", ruta)

imagen = cv2.imread(str(ruta_img))

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