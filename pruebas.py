import h5py
import numpy

def vectorizar_y(y):
    y = y.reshape(-1, 1)
    m, n = y.shape
    y_aux = numpy.zeros((m, 2))
    for i in range(m):
        j = int(y[i, 0])
        y_aux[i, j] = 1

    return y_aux

data=h5py.File("data/persona.h5","r")
print(data['X'][:])
print(data['y'][:])
X=data['X'][:]
#nuevo_y=vectorizar_y(data['y'][:])
#print(nuevo_y)
data.close()

"""
#escribimos
data=h5py.File("data/persona.h5","w")
data.create_dataset("X", data=X)
data.create_dataset("y", data=nuevo_y)
print(data['y'][:].shape)
data.close()"""