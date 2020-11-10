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

################ START VIDEO ################

cap = cv2.VideoCapture("Data Uji\Pagi_Uji.mp4")
#cap = cv2.VideoCapture("CCTV Banyumas\Siang_Edit.mp4")
cek, first_frame = cap.read()

#first_frame = cv2.imread("Siang_Uji(2).png")
first_frame = cv2.resize(first_frame,(1280,720))
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

kernel = np.ones((7,7),np.uint8)
#kernel2 = np.ones((9,9),np.uint8)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

font = cv2.FONT_HERSHEY_SIMPLEX
cars = []

cnt_up = 0
cnt_down = 0
motor = 0
mobil = 0
mobil_keluar = 0
motor_keluar = 0

max_p_age = 5
pid = 1
screenshot = 0

flag = False
pause = False
panjang = 0.1
panjang2 = 0.1
limit = 525

# Region of Interest
pa = (328,720)
pb = (613,110)
pc = (730,110)
pd = (1075,720)
vertices = np.array([pa,pb,pc,pd], np.int32)

while (cap.isOpened()):  
    cek, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    difference = cv2.absdiff(first_gray, gray_frame)
    cek, difference = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
#    cek, difference = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)

    # ROI untuk segmentasi
    mask_blank = np.zeros_like(difference)
    cv2.fillPoly(mask_blank, [vertices], (255,255,255))
#    cv2.imshow("Frame", frame)    
    difference = cv2.bitwise_and(difference, mask_blank)     
    
#    cv2.imshow("Before", difference) 
    difference = cv2.medianBlur(difference,5)
    difference = cv2.morphologyEx(difference, cv2.MORPH_CLOSE, kernel)
    difference = cv2.morphologyEx(difference, cv2.MORPH_DILATE, kernel)
    
    # Start Contour
    (image,countours0,hierarchy)=cv2.findContours(difference,cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE)
    index_c = 0
    if len(countours0) == 0:
        panjang = 0
        panjang2 = 0
    for cnt in countours0:    
        area=cv2.contourArea(cnt)
        if area>limit:     
            m=cv2.moments(cnt)
            x,y,w,h=cv2.boundingRect(cnt)
            cx=int(x+(w)/2)
            cy=int(y+h-20)
            cv2.circle(frame, (cx, cy), 0, (0, 0, 255), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            
            new = True
            if cx <= 700:
                panjang = 38670*(y**-1.413)
            if y >= 380 and cx<=715:
                panjang2 = 38670*(y**-1.413)
            elif y >= 380 and cx<=715:
                panjang2 = 0
            index_c += 1
            cv2.putText(frame, str(index_c), (x, y-10), font, 0.7, (255,0,0), 1, cv2.LINE_AA)
            if cy in range(120,180) and flag == True:
                for i in cars:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        new = False
                        i.updateCoords(cx,cy)
                        if i.going_UP(145,155) == True:
                            cnt_up += 1;
                        elif i.going_DOWN(145,155) == True:
                            cnt_down += 1;
                            img_crop = img[y+1:y+h, x+1:x+w]
                            img_crop = cv2.resize(img_crop, (72, 72))
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                            kp_t, des_t = sift.detectAndCompute(img_crop,None)
                                
                            im_features_t = np.zeros((1, k), "float32")
                            for i in range(1):
                                words_t, distance_t = vq(des_t,array[index_max])                                    
                                for w_t in words_t:
                                    im_features_t[0][w_t] += 1
                            
                            impala = im_features_t
                            im_features_t = im_features_t.transpose()
                            stdSlr = StandardScaler().fit(im_features_t)
                            im_features_t = stdSlr.transform(im_features_t)
                            im_features_t = im_features_t.transpose()    
                            
                            X_t = im_features_t
                            y_pred = array_svm[index_max].predict(X_t)
                            if y_pred == 0:
                                mobil+=1
                                cv2.putText(frame, 'Mobil', (x, y-10), font, 
                                            0.7, (255,255,0), 1, cv2.LINE_AA)
                            elif y_pred == 1:
                                motor+=1
                                cv2.putText(frame, 'Motor', (x, y-10), font, 
                                            0.7, (255,255,0), 1, cv2.LINE_AA)
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > 180:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < 120:
                            i.setDone()
                    if i.timedOut():
                        index = cars.index(i)
                        cars.pop(index)
                        del i     
                if new == True:
                    p = Car.MyCar(pid,cx,cy, max_p_age)
                    cars.append(p)
                    pid += 1
            elif cy in range(450,575) and cx in range(715,1280) and area > 5500 and flag == True:
                for i in cars:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        new = False
                        i.updateCoords(cx,cy)
                        if i.going_UP(500,550) == True:
                            cnt_up += 1;
                        elif i.going_DOWN(500,550) == True:
                            cnt_down += 1;
                            img_crop = img[y+1:y+h, x+1:x+w]
                            img_crop = cv2.resize(img_crop, (72, 72))
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                            kp_tx, des_tx = sift.detectAndCompute(img_crop,None)
                                
                            im_features_tx = np.zeros((1, k), "float32")
                            for i in range(1):
                                words_tx, distance_tx = vq(des_tx,array[index_max])                                    
                                for w_tx in words_tx:
                                    im_features_tx[0][w_tx] += 1
                            
                            im_features_tx = im_features_tx.transpose()
                            stdSlr = StandardScaler().fit(im_features_tx)
                            im_features_tx = stdSlr.transform(im_features_tx)
                            im_features_tx = im_features_tx.transpose()    
                            
                            X_t = im_features_tx
                            y_pred = array_svm[index_max].predict(X_t)
                            if y_pred == 0:
                                mobil_keluar+=1
                                cv2.putText(frame, 'Mobil', (x, y-10), font, 
                                            0.7, (255,255,0), 1, cv2.LINE_AA)
                            elif y_pred == 1:
                                motor_keluar+=1
                                cv2.putText(frame, 'Motor', (x, y-10), font, 
                                            0.7, (255,255,0), 1, cv2.LINE_AA)
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > 180:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < 120:
                            i.setDone()
                    if i.timedOut():
                        index = cars.index(i)
                        cars.pop(index)
                        del i     
                if new == True:
                    p = Car.MyCar(pid,cx,cy, max_p_age)
                    cars.append(p)
                    pid += 1
################ VIRTUAL LINE ################
    
    ##Yellow line
    cv2.line(frame,(pa),(pb),(0,255,255),2,2)
    cv2.line(frame,(pb),(pc),(0,255,255),2,2)
    cv2.line(frame,(pc),(pd),(0,255,255),2,2)

    ##Blue line
    cv2.line(frame,(593,150),(752,150), (255, 0,0),3,8)
    cv2.line(frame,(715,525),(964,525), (255, 255, 0),3,8)

    ##Distance line
    cv2.line(frame,(730,110),(800,110),(0,0,255),2,2)
    cv2.line(frame,(752,150),(752+70,150),(0,0,255),2,2)
    cv2.line(frame,(800,235),(800+70,235),(0,0,255),2,2)
    cv2.line(frame,(882,380),(882+70,380),(0,0,255),2,2)
    cv2.line(frame,(964,525),(964+70,525),(0,0,255),2,2)
    cv2.putText(frame, '48m' ,(750,100),font,0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(frame, '32m' ,(772,140),font,0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(frame, '18.8m' ,(820,225),font,0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(frame, '9.8m' ,(902,370),font,0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(frame, '4.9m' ,(984,515),font,0.5,(0,0,255),1,cv2.LINE_AA)
    
    str_mobil = 'Mobil Masuk: '+ str(mobil)
    str_motor = 'Motor Masuk: '+ str(motor)
    cv2.putText(frame, str_mobil ,(10,120),font,0.6,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_mobil ,(10,120),font,0.6,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(frame, str_motor ,(10,140),font,0.6,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_motor ,(10,140),font,0.6,(0,0,0),1,cv2.LINE_AA)
    
    str_mobilk = 'Mobil Keluar: '+ str(mobil_keluar)
    str_motork = 'Motor Keluar: '+ str(motor_keluar)
    cv2.putText(frame, str_mobilk ,(10,420),font,0.6,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_mobilk ,(10,420),font,0.6,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(frame, str_motork ,(10,440),font,0.6,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_motork ,(10,440),font,0.6,(0,0,0),1,cv2.LINE_AA)
    
    str_flag = 'Kondisi Lampu:'
    cv2.putText(frame, str_flag ,(10,200),font,0.7,(0,0,0),2,cv2.LINE_AA)
    if flag == False:
        str_hijau = 'Hijau'
        cv2.putText(frame, str_hijau ,(200,200),font,0.7,(0,255,0),2,cv2.LINE_AA)
        
    elif flag == True:
        str_merah = 'Merah'
        cv2.putText(frame, str_merah ,(200,200),font,0.7,(0,0,255),2,cv2.LINE_AA)   
    
    panjang = round(panjang,1)
    str_panjang = 'Panjang antrean total: '+ str(panjang) + " meter"
    cv2.putText(frame, str_panjang ,(10,240),font,0.7,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, str_panjang ,(10,240),font,0.7,(0,0,0),1,cv2.LINE_AA)
    
    panjang2 = round(panjang2,1)
    str_panjang2 = 'Panjang antrean motor: '+ str(panjang2) + " meter"
    cv2.putText(frame, str_panjang2 ,(10,270),font,0.7,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, str_panjang2 ,(10,270),font,0.7,(0,0,0),1,cv2.LINE_AA)
    
    if pause == False:
#        cv2.imshow("First frame", first_frame)
        cv2.imshow("Frame", frame)
#        cv2.imshow("ROI", roi)
        cv2.imshow("Final_Morphology", difference)
#        cv2.imshow("S)
#        cv2.imshow("GMM", gmm)
    #    cv2.imshow("gabungan", gabungan)
        
    key = cv2.waitKey(1)
    if key == 27:
        break

    if key == ord('q'):
        cv2.waitKey(-1) #wait until any key is pressed   
    
    if cv2.waitKey(1)&0xff==ord('s'):
        cv2.waitKey(-1)
        cv2.imwrite('save'+str(screenshot)+'.jpg',frame)
        screenshot+=1
        print("Batas. . .")
        flag = False
        motor = 0
        mobil = 0
        mobil_keluar = 0
        motor_keluar = 0
    
    if cv2.waitKey(1)&0xff==ord('r'):
        cv2.waitKey(-1)
        print("Mulai Antrian. . .")
        flag = True
        motor = 0
        mobil = 0

cap.release()
cv2.destroyAllWindows()