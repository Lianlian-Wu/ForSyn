from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier,BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from logger import get_logger
# from xgboost import XGBClassifier

LOGGER_2=get_logger("KFoldWrapper")


def get_acc(y_pre,y_true):
    return (y_pre==y_true).sum()/len(y_pre)

class KFoldWapper(object):
    def __init__(self,layer_id,index,config,random_state):
        self.config=config
        self.name="layer_{}, estimstor_{}, {}".format(layer_id,index,self.config["type"])
        if random_state is not None:
            self.random_state=(random_state+hash(self.name))%1000000007
        else:
            self.random_state=None
        # print(self.random_state)
        self.n_fold=self.config["n_fold"]
        self.estimators=[None for i in range(self.config["n_fold"])]
        self.config.pop("n_fold")
        self.estimator_class=globals()[self.config["type"]]
        self.config.pop("type")

        self.metrics={"accuracy":0.0,"f1_score":0.0}
        self.classification_result=None
    
    def _init_estimator(self):
        
        estimator_args=self.config
        est_args=estimator_args.copy()
        # est_args["random_state"]=self.random_state
        return self.estimator_class(**est_args)
    
    def fit(self,x,y):
        
        skf=StratifiedKFold(n_splits=self.n_fold,random_state=None)
        cv=[(t,v) for (t,v) in skf.split(x,y)]
        
        n_label=len(np.unique(y))
        x_probas=np.zeros((x.shape[0],n_label))

        for k in range(self.n_fold):
            est=self._init_estimator()
            train_id, val_id=cv[k]
            # print(x[train_id])
            est.fit(x[train_id],y[train_id])
            x_proba=est.predict_proba(x[val_id])
            y_pre=est.predict(x[val_id])
            acc=accuracy_score(y[val_id],y_pre)
            f1=f1_score(y[val_id],y_pre,average="macro")
            LOGGER_2.info("{}, n_fold_{},Accuracy={:.4f}, f1_macro={:.4f}".format(self.name,k,acc,f1))
            x_probas[val_id]+=x_proba
            self.estimators[k]=est
        # LOGGER_2.info("{}, n_fold_{},Accuracy={:.4f}".format(self.name,"average",np.mean(acc_k)))

        category=np.unique(y)
        y_pred=category[np.argmax(x_probas,axis=1)]
        # 记录每个森林的分类结果是否正确
        result=(y==y_pred).astype(int)
        self.classification_result=result

        self.metrics["accuracy"]=accuracy_score(y,y_pred)
        self.metrics["f1_score"]=f1_score(y,y_pred,average="macro")
        LOGGER_2.info("{}, {},Accuracy={:.4f},f1_macro={:.4f}".format(self.name,"wrapper",self.metrics["accuracy"],self.metrics["f1_score"]))
        # LOGGER_2.info("--------")
        return x_probas

    def predict_proba(self,x_test):
        proba=None
        for est in self.estimators:
            if proba is None:
                proba=est.predict_proba(x_test)
            else:
                proba+=est.predict_proba(x_test)
        proba/=self.n_fold
        # print(proba)
        return proba

    def predict(self,x_test):
        proba=self.predict_proba(x_test)
        return np.argmax(proba,axis=1)

    def get_classification_result(self):
        return self.classification_result