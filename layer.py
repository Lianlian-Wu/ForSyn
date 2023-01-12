import numpy as np 

class layer(object):
    def __init__(self,layer_id):
        self.layer_id=layer_id
        self.estimators=[]
        # self.f1_score=[]
    
    def add_est(self,estimator):
        if estimator!=None:
            self.estimators.append(estimator)
    
    def get_diversity(self):
        n_estimator=len(self.estimators)
        similarities=np.zeros((n_estimator,n_estimator))
        for i in range(n_estimator):
            for j in range(i+1,n_estimator):
                n10=(self.estimators[i].get_classification_result()!=self.estimators[j].get_classification_result()).astype(int).sum()
                similarity=(n10+0.0)/len(self.estimators[i].get_classification_result())
                similarities[i,j],similarities[j,i]=similarity,similarity
        diversity=np.sum(similarities,axis=0)
        diversity/=(n_estimator-1)
        return diversity
        
    def get_layer_id(self):
        return self.layer_id

    def predict_proba(self,x):
        prob=None
        for each in self.estimators:
            if prob is None:
                prob=each.predict_proba(x)
            else:
                prob=np.hstack((prob,each.predict_proba(x)))
        return prob
    
    def _predict_proba(self,x_test):
        proba=None
        for est in self.estimators:
            if proba is None:
                proba=est.predict_proba(x_test)
            else:
                proba+=est.predict_proba(x_test)
        proba/=len(self.estimators)
        # print(proba)
        return proba
    
    def unit_predict(self,x_test):
        labels=np.zeros((x_test.shape[0],len(self.estimators)))
        for i in range(len(self.estimators)):
            label=self.estimators[i].predict(x_test)
            labels[:,i]=label
        return labels

    def predict(self,x_test):
        labels=np.zeros((x_test.shape[0],len(self.estimators)),dtype=np.int64) 
        for i in range(len(self.estimators)):
            label=self.estimators[i].predict(x_test)
            labels[:,i]=label
        
        label_tmp=[]
        for i in range(labels.shape[0]):
            label_tmp.append(np.argmax(np.bincount(labels[i])))
        return label_tmp