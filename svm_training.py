import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import Car

from imutils import paths
from sklearn import svm
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC
from time import perf_counter
t1_start = perf_counter()  

################ CONFUSION MATRIX ################

class_name = ['Mobil', 'Motor']

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = class_name
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

################ START SVM ################

# Get each class name and save
folder = "trainingSet(dengan_keluar)"
train_path = folder
training_names = os.listdir(folder)
print ("Start program. . .\n0  Pembacaan image dan deteksi fitur. . .")

# image_paths is a location and correct class label
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = list(paths.list_images(dir))
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

# Feature extraction and object detector keypoint
sift = cv2.xfeatures2d.SIFT_create()

# List where every descriptor saved
des_list = []
for image_path in image_paths:
    im = cv2.imread(image_path)
    im = cv2.resize(im, (72, 72))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kpts,des = sift.detectAndCompute (im, None)
    des_list.append((image_path, des))   
print (1, " Pengumpulan fitur. . .")

# Save every descriptor vertically in numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  
print (2, " Iniasi K-means. . .") 

# Iteration(k) can be changed to find best training result
array = []
loop = 1
sl = 0

k = 30

akurasi = []
array_svm = []
print('Mulai untuk k = ' + str(k))
#while k <501:
while sl < loop:   
    print ('---Mulai loop ke-' + str(sl+1))
    # Do k-means clustering
    voc, variance = kmeans(descriptors, k, 1)    
    array.append(voc)
    
    # Calculate histogram fitur
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1
    
    # Scaling fitur
    testi = im_features
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)
    
    X = im_features
    y = np.array(image_classes)
    
    # Parameter regularisasi SVM
    svc = svm.SVC(kernel='rbf', C=1000, gamma=0.001)
    svc.fit(X, y)
    array_svm.append(svc)
    
    # cross validation
    cvscore = cross_val_score(svc, X, y, cv=10)
    
    akurasi.append(cvscore.mean()*100)
    t1_stop = perf_counter() 
    sl += 1

index_max = np.argmax(akurasi)
print ('Pilih akurasi MAX adalah = '+str(akurasi[index_max]))
rerata = sum(akurasi)/len(akurasi)
print ('Akurasi rerata = '+ str(rerata))
# Plot non-normalized confusion matrix with Cross Val
y_pred = cross_val_predict(array_svm[index_max], X, y, cv=10)
#plot_confusion_matrix(y, y_pred, classes=class_name, title='Confusion matrix, without normalization')
print (classification_report(y, y_pred))

t1_stop = perf_counter() 
print("Elapsed time during the whole program in seconds = {:.2f} s".format(t1_stop-t1_start)) 