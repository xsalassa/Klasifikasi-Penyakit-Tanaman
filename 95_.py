

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
       # print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import cv2
import os
from contextlib import redirect_stdout
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import DenseNet121
from sklearn.metrics import confusion_matrix as cm

disease_types = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']
data_dir = '/content/plantdisease/PlantVillage/'
train_dir = os.path.join(data_dir)
#test_dir = os.path.join(data_dir, 'test')

"""Pada disease_types terdapat macam  - macam jenis penyakit pada tanaman yang terdiri dari 

1.   Pepper Bell Bacterial Spot
2.   Pepper Bell Healthy
3.   Potato Early Blight
4.   Potato Late Blight
5.   Potato Healthy
6.   Tomato Bacterial Spot
7.   Tomato Early Blight
8.   Tomato Late Blight
9.   Tomato Leaf Mold
10.  Tomato Septoria Leaf Spot
11.  Tomato Spider Mites Two Spotted Spider Mite
12.  Tomato Target Spot
13.  Tomato Yellow Leaf Curl Virus
14.  Tomato Mosaic
15.  Tomato Healthy
"""

train_data = []
for defects_id, sp in enumerate(disease_types):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['{}/{}'.format(sp, file), defects_id, sp])
        
train = pd.DataFrame(train_data, columns=['File', 'DiseaseID','Disease Type'])
train.tail()

"""Pada cell ini, ditampilkan colum yang dipilih untuk ditampilkan, seperti yang terlihat pada cell diatas, yang ditampikan adalah
1. Nama File
2. Disease Id 
3. Jenis Penyakit
"""

# Randomize the order of training set
SEED = 42
train = train.sample(frac=1, random_state=SEED) 
train.index = np.arange(len(train)) # Reset indices
train.head()

# Plot a histogram
plt.hist(train['DiseaseID'])
plt.title('Frequency Histogram of Species')
plt.figure(figsize=(12, 12))
plt.show()

"""Menampilkan persebaran berapa banyak jenis tanaman"""

# Display images for different species
def plot_defects(defect_types, rows, cols):
    fig, ax = plt.subplots(rows, cols, figsize=(12, 12))
    defect_files = train['File'][train['Disease Type'] == defect_types].values
    n = 0
    for i in range(rows):
        for j in range(cols):
            image_path = os.path.join(data_dir, defect_files[n])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].imshow(cv2.imread(image_path))
            n += 1
# Displays first n images of class from training set
plot_defects('Tomato_Bacterial_spot', 5, 5)

IMAGE_SIZE = 64

def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) # Loading a color image is the default flag
# Resize image to target size
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        X_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
# Normalize the data
X_Train = X_train / 255.
print('Train Shape: {}'.format(X_Train.shape))

Y_train = train['DiseaseID'].values
Y_train = to_categorical(Y_train, num_classes=15)

BATCH_SIZE = 64

# Split the train and validation sets 
X_train, X_val, Y_train, Y_val = train_test_split(X_Train, Y_train, test_size=0.2, random_state=SEED)

fig, ax = plt.subplots(1, 3, figsize=(15, 15))
for i in range(3):
    ax[i].set_axis_off()
    ax[i].imshow(X_train[i])
    ax[i].set_title(disease_types[np.argmax(Y_train[i])])

EPOCHS = 200
SIZE=64
N_ch=3

def build_densenet():
    densenet = DenseNet121(weights='imagenet', include_top=False)

    input = Input(shape=(SIZE, SIZE, N_ch))
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    output = Dense(15,activation = 'softmax', name='root')(x)
 

    # model
    model = Model(input,output)
    
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    return model

model = build_densenet()
annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
# Generates batches of image data with data augmentation
datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations
                        width_shift_range=0.2, # Range for random horizontal shifts
                        height_shift_range=0.2, # Range for random vertical shifts
                        zoom_range=0.2, # Range for random zoom
                        horizontal_flip=True, # Randomly flip inputs horizontally
                        vertical_flip=True) # Randomly flip inputs vertically

datagen.fit(X_train)
# Fits the model on batches with real-time data augmentation

hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
               steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
               epochs=EPOCHS,
               verbose=2,
               callbacks=[annealer, checkpoint],
               validation_data=(X_val, Y_val))

acc_history = hist.history['accuracy']
acc_history = np.array(acc_history)
np.savetxt('acc history.csv', acc_history, delimiter=',', fmt='%f')

val_acc_history = hist.history['val_accuracy']
val_acc_history = np.array(val_acc_history)
np.savetxt('val_acc_history.csv', val_acc_history, delimiter=',', fmt='%f')

loss_history = hist.history['loss']
loss_history = np.array(loss_history)
np.savetxt('loss_history.csv', loss_history, delimiter=',', fmt='%f')

val_loss_history = hist.history['val_loss']
val_loss_history = np.array(val_loss_history)
np.savetxt('val_loss_history.csv', val_loss_history, delimiter=',', fmt='%f')

acc_testing  = model.evaluate(X_val, Y_val) 
print('akurasinya adalah {}'.format(acc_testing[1]))

save_acc_testing = np.array(acc_testing)
np.savetxt('akurasi testing.csv', save_acc_testing, delimiter=',', fmt='%f')

with open('modelsummary.txt','w') as f:
    with redirect_stdout(f):
        model.summary()
        
fig,(ax0) = plt.subplots(nrows=1, figsize=(12,5))
ax0.plot(hist.history['accuracy'],'red', label='Akurasi Training')
ax0.plot(hist.history['val_accuracy'], 'blue', label='Akurasi Validasi')
ax0.plot(label='Accuracy', loc='upper left')
ax0.set_title('Model Accuracy')
ax0.set_xlabel("Epoch")
ax0.set_ylabel("Accuracy")
ax0.legend()
plt.savefig('Grafik Akurasi.png')

fig,(ax1) = plt.subplots(nrows=1, figsize=(12,5))
ax1.plot(hist.history['loss'],'red', label='Loss Training')
ax1.plot(hist.history['val_loss'], 'blue', label='Loss Validasi')
ax1.plot(label='Loss', loc='upper left')
ax1.set_title('Model Loss')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
plt.savefig('Grafik Loss.png')

list_sen = []
list_spe = []
list_pre = []
list_f1 = []
list_err = []
list_acc = []

y_pred = model.predict(X_val)  
y_pred = np.argmax(y_pred, axis=1)
Y_true = np.argmax(Y_val, axis=1)

save_y_pred= np.array(y_pred)
np.savetxt('prediksi testing.csv', save_y_pred, delimiter=',', fmt='%f')

#testing_label=np.argmax(testing_label, axis=1)
cm_value = cm(Y_true, y_pred)
cm_value = np.array(cm_value)
np.savetxt('confusion matrix.csv', cm_value, delimiter=',', fmt='%i')

Sen_Class = []
Spe_Class = []
Pre_Class = []
F1_Class = []
Err_Class = []
Acc_Class = []

for idx in range(len(cm_value)):
    TP = cm_value[idx, idx]
    FP = np.sum(cm_value[idx, :]) - TP
    FN = np.sum(cm_value[:, idx]) - TP
    TN = np.sum(cm_value) - (TP + FN + FP)
    
    Sen = TP / (TP + FN)
    Spe = TN / (TN + FP)
    Pre = TP / (TP + FP)
    F1 = (2 * (Pre * Sen)) / (Sen + Pre)
    Err = (FP + FN) / (FP + FN + TN + TP)
    Acc = (TP + TN) / (FP + FN + TN + TP)
    
    Sen_Class.append([Sen, idx])
    Spe_Class.append([Spe, idx])
    Pre_Class.append([Pre, idx])
    F1_Class.append([F1, idx])
    Err_Class.append([Err, idx])
    Acc_Class.append([Acc, idx])
    
list_sen.extend(Sen_Class)
list_spe.extend(Spe_Class)
list_pre.extend(Pre_Class)
list_f1.extend(F1_Class)
list_err.extend(Err_Class)
list_acc.extend(Acc_Class)

save_list_sen = np.array(list_sen)
save_list_spe = np.array(list_spe)
save_list_pre = np.array(list_pre)
save_list_f1 = np.array(list_f1)
save_list_err = np.array(list_err)
save_list_acc = np.array(list_acc)

save_average_sen = np.mean(save_list_sen[:,0]).reshape(-1,1)
save_average_spe = np.mean(save_list_spe[:,0]).reshape(-1,1)
save_average_pre = np.mean(save_list_pre[:,0]).reshape(-1,1)
save_average_f1 = np.mean(save_list_f1[:,0]).reshape(-1,1)
save_average_err = np.mean(save_list_err[:,0]).reshape(-1,1)
save_average_acc = np.mean(save_list_acc[:,0]).reshape(-1,1)
                         
np.savetxt('average sen.csv', save_average_sen, delimiter=',', fmt='%f')
np.savetxt('average spe.csv', save_average_spe, delimiter=',', fmt='%f')
np.savetxt('average pre.csv', save_average_pre, delimiter=',', fmt='%f')
np.savetxt('average f1.csv', save_average_f1, delimiter=',', fmt='%f')
np.savetxt('average err.csv', save_average_err, delimiter=',', fmt='%f')
np.savetxt('average acc.csv', save_average_acc, delimiter=',', fmt='%f')


np.savetxt('list sen.csv', save_list_sen, delimiter=',', fmt='%f')
np.savetxt('list spe.csv', save_list_spe, delimiter=',', fmt='%f')
np.savetxt('list_pre.csv', save_list_pre, delimiter=',', fmt='%f')
np.savetxt('list_f1.csv', save_list_f1, delimiter=',', fmt='%f')
np.savetxt('list_err.csv', save_list_err, delimiter=',', fmt='%f')
np.savetxt('list acc.csv', save_list_acc, delimiter=',', fmt='%f')


'''========================================================='''

column_model = np.array(['acc training', 'acc testing', 'loss training', 'loss testing']).reshape(-1,1)

row_acc = acc_history[-1] * 100
row_val_acc = val_acc_history[-1] * 100
row_loss = loss_history[-1]
row_val_loss = val_loss_history[-1]

row_acc = np.array(row_acc).reshape(-1,1)
row_val_acc = np.array(row_val_acc).reshape(-1,1)
row_loss = np.array(row_loss).reshape(-1,1)
row_val_loss = np.array(row_val_loss).reshape(-1,1)

row = np.concatenate((row_acc, row_val_acc, row_loss, row_val_loss), axis=0)

hasil_model = np.concatenate((column_model, row), axis=1)
np.savetxt('hasil dari model.csv', hasil_model, delimiter=',', fmt='%s')

column_average = np.array(['sen', 'spe', 'pre', 'f1', 'err', 'acc']).reshape(-1,1)

"""# Testing Disease Prediction"""

from skimage import io
from keras.preprocessing import image
#path='imbalanced/Scratch/Scratch_400.jpg'
img = image.load_img('/content/plantdisease/PlantVillage/Potato___Early_blight/042135e2-e126-4900-9212-d42d900b8125___RS_Early.B 8791.JPG', grayscale=False, target_size=(64, 64))
show_img=image.load_img('/content/plantdisease/PlantVillage/Potato___Early_blight/042135e2-e126-4900-9212-d42d900b8125___RS_Early.B 8791.JPG', grayscale=False, target_size=(200, 200))
disease_class = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
#x = np.array(x, 'float32')
x /= 255

custom = model.predict(x)
print(custom[0])



#x = x.reshape([64, 64]);

#plt.gray()
plt.imshow(show_img)
plt.show()

a=custom[0]
ind=np.argmax(a)
        
print('Prediction:',disease_class[ind])

"""# Production Version (Flask Web App) : [Plant Disease Diagnosis](https://github.com/shawon100/Plant-Disease-Diagnosis-Flask)"""