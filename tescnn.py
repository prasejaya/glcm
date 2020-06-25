import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import f1_score
import shutil
import warnings
warnings.filterwarnings('ignore')

path = r'E:\Dataset\testingca'

subfolder = os.listdir(path)
print(subfolder);
img_width, img_height = 32, 32

totalgambar=0

benar=0
salah=0
aksara = ''

y_true = []
y_pred = []
getbenar = []
getsalah = []
getf = []
getjumlah = []
kelas = []
prediksi = []

model = load_model(r'E:\Dataset\lenet\lenet200.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

for i in subfolder:
	file = os.listdir(path+"\\"+i)
	print ("Mengeksekusi subfolder : "+i)
	for j in file:
		totalgambar = totalgambar + 1
		imgpath = path+"\\"+i+"\\"+j
		print ("Mengeksekusi file : "+imgpath)
		
		img = image.load_img(imgpath, target_size=(img_width, img_height))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)

		images = np.vstack([x])
		classes = model.predict_classes(images, batch_size=32)
		y_true.append(str(i))
		y_pred.append(str(classes[0]))
		kelas.append(str(i))
		prediksi.append(classes[0])
		print("Hasil Prediksi : "+str(classes[0]))
		if(classes[0]==1):
			aksara = 'ca'
		else:
			aksara = 'bukan ca'
		if(i==aksara):
			benar = benar + 1
			shutil.move(imgpath, r'E:\Dataset\hasiltesting\benar\ca')
		else:
			salah = salah + 1
			shutil.move(imgpath, r'E:\Dataset\hasiltesting\salah\ca')
	print('Rekapitulasi Hasil Prediksi Aksara ',i)
	print('Jumlah Data',len(file))
	print('Kelas : ', kelas)
	print('Prediksi : ', prediksi)
	print('Benar = ',benar)
	print('Salah = ',salah)
	getbenar.append(benar)
	getsalah.append(salah)
	getjumlah.append(len(file))
	benar=0
	salah=0
	kelas.clear()
	prediksi.clear()
    

print('Rekapitulasi Hasil Prediksi Semua Aksara')
print('Jumlah Data')
print(getjumlah)
print(np.sum(getjumlah))
print('Jumlah Benar')
print(getbenar)
print(np.sum(getbenar))
print('Jumlah Salah')
print(getsalah)
print(np.sum(getsalah))
print('F1 Score = ',f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_true)))
	

