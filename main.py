import cv2
import time
import numpy as np

#CORES DAS CLASSES E AJUSTES
COLORS = [(0,255,255), (255,255,0), (255,0,0)]
CONFIDENCE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.2

#CARREGA CLASSES
class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#CAPTURA VIDEO
cap = cv2.VideoCapture(0)

#CARREGA PESOS DA REDE NEURAL
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

#PARAMETROS DA REDE NEURAL
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)

#LENDO FRAMES
while True:
    #CAPTURA DO FRAME
    _, frame = cap.read()

    #INICIO CONTA TEMPO
    start = time.time()

    #DETECÇÃO
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    #FIM CONTA TEMPO
    end = time.time()

    #PERCORRENDO TODAS AS DETECÇÕES
    for (classid, score, box) in zip(classes, scores, boxes):

        #GERANDO COR P CLASSE
        color = COLORS[int(classid) % len (COLORS)]

        #NOME DA CLASSE, SCORE E ACURACIA
        label = f"{class_names[classid[0]]} : {score}"

        #DESENHANDO O BOX
        cv2.rectangle(frame, box, color, 2)

        #cOLOCANDO INFORMAÇÕES EM CIMA DO BOX
        cv2.putText(frame, label, (box[0], box[1] -10 ), cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    #CALCULANDO TEMPO QUE LEVOU A DETECÇÃO
    fps_label = f"fps: {round((1.0/(end - start)),2)}"

    #ESCREVENDO FPS NA IMAGEM
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0),5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    #MOSTRANDO IMAGEM
    cv2.imshow("detections", frame)

    #ESPERA RESPOSTA
    if cv2.waitKey(1) == 27:
        break

    #LIBERANDO A CAMERA E FECHA JANELAS
    #cap.release()
    #cv2.destroyAllWindows()