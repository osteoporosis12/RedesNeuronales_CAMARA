import cv2
import os
from time import time
import numpy
import h5py

candidato = cv2.CascadeClassifier("modelos/haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

intervalo = time()
cont_cap=0
nombre="pablo"
id=2
if not os.path.exists("Fotos/"+nombre):
    os.mkdir("Fotos/"+nombre)
else:
    while os.path.isfile("Fotos/"+nombre+"/"+str(cont_cap)+".png"):
        cont_cap+=1
        print(cont_cap)



X = numpy.empty((0, 4096), dtype=numpy.uint8)
Y = numpy.empty((0, 1))

print(X.shape)
while True:
    ret, imagen= video.read()
    rostros = candidato.detectMultiScale(imagen,scaleFactor=1.1,minNeighbors=6,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE)
    for rostro in rostros:

        x,y,ancho,largo=rostro

        p_error = int(abs(x-ancho)/2)
        cv2.rectangle(imagen,(x-p_error,y-p_error),(x+ancho+p_error,y+largo+p_error), (0,255,0), 2)

        recorte_cara = imagen[ y-p_error:largo + y +p_error,x-p_error:x + ancho+p_error ]
        print(recorte_cara.shape)

        if(recorte_cara.shape[0]==0 or recorte_cara.shape[1]==0):
            break

        cara = cv2.resize(recorte_cara, (64, 64), interpolation=cv2.INTER_AREA)  # redimencion y relleno de espacios

        # CONVERTIMOS A GRISES
        cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
        if(intervalo+0.25<=time()):
            #print(cara.reshape(1,400).shape)
            xdd=cara.reshape(1, -1)
            X = numpy.concatenate((X, xdd), axis=0)
            Y = numpy.concatenate((Y, numpy.array([[id]])), axis=0)

            #x = numpy.vstack((x, cara.reshape(1,400)))
            #print(Y)

            #cv2.imshow("Ejemplo", cara)
            if ret:
                cv2.imwrite('Fotos/'+nombre+'/'+str(cont_cap)+'.png', cara)  # Guarda el cuadro como una imagen PNG
                print("Captura exitosa.")
                cont_cap+=1
            intervalo=time()

    cv2.imshow("Ejemplo", imagen)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' o tecla Esc
        break

if os.path.exists("data/persona.h5"):
    data = h5py.File("data/persona.h5", "r")
    X = numpy.concatenate((X, data['X'][:]), axis=0)
    Y = numpy.concatenate((Y, data['y'][:]), axis=0)
    data.close()

arch = h5py.File("data/persona.h5", "w")
arch.create_dataset("X", data=X)
arch.create_dataset("y", data=Y)
arch.close()

video.release()
cv2.destroyAllWindows()

#sacamos fotos, despues lo redimencionamos, lo aplanamos y etiquetamos

#una vez etiquetado entrenamos con dos capas y cuando coloquemos el vieo salga en name de la person
