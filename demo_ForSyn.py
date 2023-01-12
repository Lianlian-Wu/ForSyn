from gcForest import gcForest
from logger import get_logger
from sklearn.model_selection import train_test_split,StratifiedKFold,RepeatedStratifiedKFold
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from evaluation import accuracy,f1_binary,f1_macro,f1_micro
from load_data import load_data
from feature_selection import select_feature
from sklearn.metrics import average_precision_score,matthews_corrcoef,f1_score,recall_score,confusion_matrix,classification_report,roc_auc_score,auc,precision_recall_curve,accuracy_score,classification_report
from imblearn.ensemble import BalancedBaggingClassifier,RUSBoostClassifier,BalancedRandomForestClassifier
from imblearn.metrics import geometric_mean_score

x,y,dataid=load_data()
for index in range(len(y)): 
    if y[index]==3.0:
        y[index]=0.0    
    if y[index]==1.0:
        y[index]=0.0
    if y[index]==2.0:
        y[index]=1.0
print("x_shape:",x.shape)
print("y_shape:",y.shape)
print("y_distribution:",Counter(y)) 

def get_config():
    config={}
    config["random_state"]=None
    config["max_layers"]=100
    config["early_stop_rounds"]=1
    config["if_stacking"]=False
    config["if_save_model"]=False
    config["train_evaluation"]=f1_macro ##f1_binary,f1_macro,f1_micro,accuracy
    config["estimator_configs"]=[]
    # for i in range(10):
    #     config["estimator_configs"].append({"n_fold":5,"type":"IMRF","n_estimators":40,"splitter":"best"})
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":100,"n_jobs":-1})
    config["output_layer_config"]=[]
    return config

if __name__=="__main__":   
    config=get_config()  
    skf=RepeatedStratifiedKFold(n_splits=5,random_state=0,n_repeats=1)

    f1s=[]
    auprs=[]
    mccs=[]
    recalls=[]
    gmeans=[]
    i=1
    for train_id,test_id in skf.split(x,y):
        print("============{}-th cross validation============".format(i))
        x_train,x_test,y_train,y_test=x[train_id],x[test_id],y[train_id],y[test_id]
        index=select_feature(x_train,y_train,1806)
        x_train=x_train[:,index]
        x_test=x_test[:,index]
        config=get_config()
        gc=gcForest(config)
        gc.fit(x_train,y_train)
        y_pred=gc.predict(x_test)
        
        #calculate y_score        
        y_pred_prob=gc.predict_proba(x_test)
        y_score=[]
        for item in y_pred_prob:
            y_score.append(item[1])
        y_score=np.array(y_score)
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        aupr = auc(recall,precision)
                  
        f1 = f1_score(y_test, y_pred,average='binary')       
        recall = recall_score(y_test, y_pred,average="binary")
        mcc = matthews_corrcoef(y_test, y_pred)
        gmean= geometric_mean_score(y_test, y_pred,average='binary')
          
        f1s.append(f1)
        auprs.append(aupr)
        recalls.append(recall)
        mccs.append(mcc)
        gmeans.append(gmean)    
        i+=1
        
    print("============training finished============")
    f1s=np.array(f1s)
    auprs=np.array(auprs)
    recalls=np.array(recalls)
    mccs=np.array(mccs)
    gmeans=np.array(gmeans)
    
    print("Data:",dataid)
    print("Model: ForSyn")
    print("f1 average:", np.mean(f1s))
    print("aupr average:", np.mean(auprs))
    print("recall average:", np.mean(recalls))
    print("mcc average:", np.mean(mccs))
    print("gmean average:", np.mean(gmeans))
