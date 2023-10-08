import h5py


import torch
from time import time
from torch import nn, optim
from H5Data import H5Data

if __name__ == "__main__":
    # cargamos data
    carga_entrenamiento = torch.utils.data.DataLoader(H5Data("data/persona.h5"), batch_size=20, shuffle=True)

    # configurar red
    capa_entrada = 4096
    capas_ocultas = [25,25]
    capa_salida = 2
    modelo = nn.Sequential(nn.Linear(capa_entrada, capas_ocultas[0]), nn.ReLU(),
                           nn.Linear(capas_ocultas[0], capas_ocultas[1]), nn.ReLU(),
                           nn.Linear(capas_ocultas[1], capa_salida), nn.LogSoftmax(dim=1))


    j = nn.CrossEntropyLoss()

    optimizador = optim.Adam(modelo.parameters(), lr = 0.003)
    tiempo = time()
    epochs = 10

    for e in range(epochs):
        costo = 0
        for imagen, etiqueta in carga_entrenamiento:
            imagen = imagen.view(imagen.shape[0], -1)
            optimizador.zero_grad()
            h = modelo(imagen.float())
            etiqueta = etiqueta.flatten()
            #print(etiqueta)
            error = j(h, etiqueta.long())
            error.backward()
            optimizador.step()
            costo += error.item()
        else:
            print("Epoch {} - Funcion costo: {}".format(e, costo / len(carga_entrenamiento)))
    torch.save(modelo, 'modelos/Pablo.pt')
