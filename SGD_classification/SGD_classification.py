import numpy as np
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn import svm

print "********************Initialization********************"
num_class=20 #20  #10 #401
print "\n"

print "********************train_samples********************"
import scipy.io
temp=scipy.io.loadmat("F:/school/Segmentation/ExtractFeatures/VOC2007/SupervisedClassification/transfer/train_samples.mat")
train_samples=temp['train_samples']   

train_samples=train_samples.astype(np.float)

print "train_samples:\n", train_samples

################################################Read in the train_class file##########################################################
temp=scipy.io.loadmat("F:/school/Segmentation/ExtractFeatures/VOC2007/train/AP/class.mat")
train_class=temp['class']

train_class=train_class.astype(np.float)
num_train_samples=train_class.shape[0]

print "train_class:\n", train_class
print "num_train_samples: ", num_train_samples

print "********************test_samples********************"
temp=scipy.io.loadmat("F:/school/Segmentation/ExtractFeatures/VOC2007/SupervisedClassification/transfer/test_samples.mat")
test_samples=temp['test_samples']   #temp['test_samples']  temp['test']

test_samples=test_samples.astype(np.float)

print "test_samples:\n", test_samples

################################################Read in the test_class file################################################
temp=scipy.io.loadmat("F:/school/Segmentation/ExtractFeatures/VOC2007/test/class.mat")
test_class=temp['class']

test_class=test_class.astype(np.float)
num_test_samples=test_class.shape[0]

print "test_class:\n", test_class
print "num_test_samples: ", num_test_samples

############################################################Learning the model##################################################
print "********************Learning the model********************"
predicted_label=np.zeros((num_test_samples,num_class))
confidence=np.zeros((num_test_samples,num_class))


for i in range(0,num_class):

    temp_class=(train_class==i)

    clf=linear_model.SGDClassifier(n_iter=100, shuffle=True, alpha=0.001)   #100,0.001  penalty='elasticnet' shuffle=True
    clf.fit(train_samples,train_class[:,i],class_weight='auto')
  
    ################using svm###############
##    clf = svm.SVC()    #svm.NuSVC()
##    clf.fit(train_samples, train_class[:,i])

    ############Predicting final labels############
    predicted_label[:,i]=clf.predict(test_samples)
    confidence[:,i]=clf.decision_function(test_samples)

    temp=np.sum(np.multiply(predicted_label[:,i]==1,test_class[:,i]==1))+np.sum(np.multiply(predicted_label[:,i]==0,test_class[:,i]==0))
    accuracy=temp/4952.   

    print "tp=", np.sum(np.multiply(predicted_label[:,i]==1,test_class[:,i]==1)),"tn=", np.sum(np.multiply(predicted_label[:,i]==0,test_class[:,i]==0)),"accuracy=", accuracy


    ##confidence=clf.predict_proba(train_samples)

print "********************computing AP********************"
AP=np.zeros((num_class,1))
REC=np.zeros((num_class,num_test_samples))
PREC=np.zeros((num_class,num_test_samples))

for class_number in range (0,num_class):

    gt=test_class[:,class_number]
    gt[gt==0]=-1

    ####map results to ground truth images
    out=np.ones(gt.shape[0])*-np.inf

    for j in range(0,test_class.shape[0]):
        out[j]=confidence[j,class_number]

    ####compute precision/recall
    si=np.argsort(-out)
    tp=gt[si]>0
    fp=gt[si]<0

    fp=np.cumsum(fp)
    tp=np.cumsum(tp)
    rec=tp.astype(np.float)/np.sum(gt>0)
    prec=np.divide(tp.astype(np.float),(fp+tp))

    ####compute average precision
    ap=0
    T=np.linspace(0.0,1.0,num=11)
    for t in T:
        p=np.max(prec[rec>=t])
        if p.size==0:
            p=0;

        ap=ap+p/11.

 
    AP[class_number]=ap
    REC[class_number,:]=rec
    PREC[class_number,:]=prec   
    print "ap= ", ap
MAP=np.sum(AP)/num_class   

print "********************MAP********************"
print "MAP=", MAP,"\n"     


##############################Saving a variable in a mat file format############################
import scipy.io
scipy.io.savemat("E:/school/Segmentation/ExtractFeatures/VOC2007/SupervisedClassification/Data_BOW200/predicted_label.mat", mdict={'predicted_label': (predicted_label)})
scipy.io.savemat("E:/school/Segmentation/ExtractFeatures/VOC2007/SupervisedClassification/Data_BOW200/AP.mat", mdict={'AP': (AP)})
scipy.io.savemat("E:/school/Segmentation/ExtractFeatures/VOC2007/SupervisedClassification/Data_BOW200/confidence.mat", mdict={'confidence': (confidence)})

