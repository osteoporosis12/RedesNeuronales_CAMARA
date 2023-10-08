import h5py
from torch.utils.data import Dataset
import numpy

class H5Data(Dataset):

    def __init__(self, archivo, transform=None):
        self.archivo = h5py.File(archivo, "r")

        self.etiquetas = self.archivo["y"][:]
        self.data = self.archivo["X"][:]
        print(self.etiquetas.shape,self.data.shape)
        self.transform = transform

    def __getitem__(self, index):
        datum = self.data[index]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, self.etiquetas[index]

    def __len__(self):
        return len(self.etiquetas)

    def close(self):
        self.archivo.close()









