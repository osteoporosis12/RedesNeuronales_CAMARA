import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as rna

transform = transforms.Compose([transforms.ToTensor()])
model_ft = torch.load("modelos/digitos_model_torch_2.pt")

imagen = cv2.imread("test/prueba03.jpg")
imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagenGris = cv2.GaussianBlur(imagenGris, (5, 5), 0)
ret, imagenBN = cv2.threshold(imagenGris, 90, 255, cv2.THRESH_BINARY_INV)
cv2.waitKey()
grupos, _ = cv2.findContours(imagenBN.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ventanas = [cv2.boundingRect(g) for g in grupos]

for g in ventanas:
    cv2.rectangle(imagen, (g[0], g[1]), (g[0] + g[2], g[1] + g[3]), (255, 0, 0), 2)
    espacio = int(g[3] * 1.6)
    p1 = int(g[1] + g[3] // 2) - espacio // 2
    p2 = int(g[0] + g[2] // 2) - espacio // 2

    digito = imagenBN[p1: p1 + espacio, p2: p2 + espacio]

    digito = cv2.resize(digito, (20, 20), interpolation=cv2.INTER_AREA)
    digito = cv2.dilate(digito, (3, 3,))
    digito = digito.T
    digito = torch.Tensor(digito.reshape(1, -1))
    digito.unsqueeze_(dim=0)
    digito = Variable(digito)

    digito = digito.view(digito.shape[0], -1)
    predict = rna.softmax(model_ft(digito), dim=1)

    cv2.putText(imagen, str(predict.argmax().item()), (g[0], g[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

cv2.imshow("Digitos", imagen)
cv2.waitKey()