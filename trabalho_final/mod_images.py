#Aprendizado de Máquina
#Alunos: Juan e Evaristo
#Professor:Daniel Guerreiro 

import os
import glob
import numpy as np
from scipy.misc import imread, imresize, imsave
from sklearn.model_selection import train_test_split

# test "data_sign/GTSRB/Final_Test"
# train "data_sign/GTSRB/Final_Training/Images"
imagen_list = []
label_list = np.array([])
for (path, ficheros, archivos) in os.walk("data_sign/GTSRB/Final_Training/Images"):
    print(path)
    for infile in glob.glob(os.path.join(path, '*.ppm')):        
        img = imread(infile)
        #print(img.shape)
        img2 = imresize(img, [32, 32])
        #print(img2.shape)
        imagen_list.append(img2)
        imsave(infile, img2)
        clase = infile.split('/')
        label_list = np.append(label_list,clase[-2])
        # print(clase[-2])
print(np.array(imagen_list).shape)
print(len(label_list))