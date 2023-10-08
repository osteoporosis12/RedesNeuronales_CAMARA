import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as rna

transform = transforms.Compose([transforms.ToTensor()])
model_ft = torch.load("modelos/Pablo.pt")

candidato = cv2.CascadeClassifier("modelos/haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

nombres={
    0:"camba",
    1:"valencia",
    2:"pablo"
}

while True:
    ret, imagen = video.read()
    # encontrar los puntos
    rostros = candidato.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100),
                                         flags=cv2.CASCADE_SCALE_IMAGE)  # con esto tenemos los grupos de los rostros
    # iteramos sobre cada uno de los grupos para dibujar una ventana
    for rostro in rostros:
        x, y, ancho, largo = rostro
        p_error = int(abs(x - ancho) / 2)
        cv2.rectangle(imagen, (x - p_error, y - p_error), (x + ancho + p_error, y + largo + p_error), (0, 255, 0), 2)
        recorte_cara = imagen[y - p_error:largo + y + p_error, x - p_error:x + ancho + p_error]
        print(recorte_cara.shape)

        if (recorte_cara.shape[0] == 0 or recorte_cara.shape[1] == 0):
            break

        cara = cv2.resize(recorte_cara, (64, 64), interpolation=cv2.INTER_AREA)  # redimencion y relleno de espacios
        # CONVERTIMOS A GRISES
        cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)

        cara = cv2.dilate(cara, (3, 3,))
        cara = torch.Tensor(cara.reshape(1, -1))
        cara.unsqueeze_(dim=0)
        cara = Variable(cara)

        cara = cara.view(cara.shape[0], -1)

        predict = rna.softmax(model_ft(cara), dim=1)
        cv2.putText(imagen, str(nombres[ predict.argmax().item()]), (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

    cv2.imshow("Ejemplo", imagen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

# TAREA
# CAPTURAS DE PROPIO ROSTROS 64x64
# HACER UN VIDEO
# DENTRO DE ESTA PORCION DE CODIGO QUE TE ETIQUETEN QUIEN ES QUIEN
# RESUEM
# QUE RECONOZCA TU CARA
# RECORTA LA MATRIZ PARA IR APLANANDO
# LA CAMARA TIENE QUE SACAR Y RECONOCER
