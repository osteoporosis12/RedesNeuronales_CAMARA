import torch
from time import time
from torch import nn, optim
from H5Data import H5Data

if __name__ == "__main__":
    # cargamos data
    carga_entrenamiento = torch.utils.data.DataLoader(H5Data("data/digitos_modificados.h5"), batch_size=64, shuffle=True)

    # configurar red
    capa_entrada = 400
    capas_ocultas = [25]
    capa_salida = 10
    modelo = nn.Sequential(nn.Linear(capa_entrada, capas_ocultas[0]), nn.ReLU(),
                           nn.Linear(capas_ocultas[0], capa_salida), nn.LogSoftmax(dim=1))

    #[0.1 0.2 ... ] =SUMATORIA 1
    #[0.9 0.6 ] SIGMOIDE

    j = nn.CrossEntropyLoss()

    optimizador = optim.Adam(modelo.parameters(), lr = 0.003)
    tiempo = time()
    epochs = 5

    for e in range(epochs):
        costo = 0
        for imagen, etiqueta in carga_entrenamiento:
            imagen = imagen.view(imagen.shape[0], -1)
            optimizador.zero_grad()
            h = modelo(imagen.float())
            etiqueta = etiqueta.flatten()
            print(etiqueta)
            error = j(h, etiqueta.long())
            error.backward()
            optimizador.step()
            costo += error.item()
        else:
            print("Epoch {} - Funcion costo: {}".format(e, costo / len(carga_entrenamiento)))
    torch.save(modelo, 'modelos/digitos_model_torch_2.pt')
