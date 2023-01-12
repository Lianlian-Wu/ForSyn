import numpy as np
from sklearn import ensemble
from layer import layer
from logger import get_logger
from k_fold_wrapper import KFoldWapper
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score

LOGGER=get_logger("gcForest")

def get_acc(y_pre,y_true):
    return (y_pre==y_true).sum()/len(y_pre)

def check(last,current):
    a=[i for i in last if i in current]
    LOGGER.info("Uncorrected,num:{}".format(len(a)))
    b=[i for i in last if i not in current]
    LOGGER.info("corrected,num:{}".format(len(b)))
    c=[i for i in current if i not in last]
    LOGGER.info("new error,num:{}".format(len(c)))

class gcForest(object):

    def __init__(self,config):
        self.random_state=config["random_state"]
        self.max_layers=config["max_layers"]
        self.early_stop_rounds=config["early_stop_rounds"]
        self.if_stacking=config["if_stacking"]
        self.if_save_model=config["if_save_model"]
        self.train_evaluation=config["train_evaluation"]
        self.estimator_configs=config["estimator_configs"]
        self.output_layer_config=config["output_layer_config"]
        self.output_layer=[]
        self.category=None
        self.X_enhanced=None
        self.layers=[]
        self.output_layer=[]

    def fit(self,x_train,y_train):
      
        x_train,n_feature,n_label=self.preprocess(x_train,y_train)
        # print(x_train[0])

        early_stop_rounds=self.early_stop_rounds
        if_stacking=self.if_stacking

        evaluate=self.train_evaluation
        best_layer_id=0
        deepth=0
        best_layer_evaluation=0.0

        last_error_index=None
        while deepth<self.max_layers:
            
            x_train_proba=np.zeros((x_train.shape[0],n_label*len(self.estimator_configs)))

            current_layer=layer(deepth)
            LOGGER.info("-----------------------------------------layer-{}--------------------------------------------".format(current_layer.get_layer_id()))
            LOGGER.info("The shape of x_train is {}".format(x_train.shape))
            x_proba_tmp=np.zeros((x_train.shape[0],n_label))
            for index in range(len(self.estimator_configs)):
                config=self.estimator_configs[index].copy()
                k_fold_est=KFoldWapper(current_layer.get_layer_id(),index,config,random_state=self.random_state)

                x_proba=k_fold_est.fit(x_train,y_train)
                current_layer.add_est(k_fold_est)
                x_train_proba[:,index*n_label:index*n_label+n_label]+=x_proba
                x_proba_tmp+=x_proba

            x_proba_tmp/=len(self.estimator_configs)
            label_tmp=self.category[np.argmax(x_proba_tmp,axis=1)]
            current_evaluation=evaluate(label_tmp,y_train)

            self.layers.append(current_layer)

            

            x_train=np.hstack((x_train[:,0:n_feature],x_train_proba))

            if current_evaluation>best_layer_evaluation:
                best_layer_id=current_layer.get_layer_id()
                best_layer_evaluation=current_evaluation
            LOGGER.info("The evaluation[{}] of layer_{} is {:.4f}".format(evaluate.__name__,deepth,current_evaluation))
            LOGGER.info("The diversity of estimators in layer_{} is {}".format(deepth,current_layer.get_diversity()))

            
            if current_layer.get_layer_id()-best_layer_id>=self.early_stop_rounds:
                self.layers=self.layers[0:best_layer_id+1]
                LOGGER.info("**************************************************The num of layer is {}**********************************************".format(len(self.layers)))
                break

            deepth+=1


  

    def predict(self,x):
        prob=self.predict_proba(x)
        #print(prob.shape)
        label=self.category[np.argmax(prob,axis=1)]
        return label

    # def predict(self,x):
    #     x_test=x.copy()
    #     x_test=x_test.reshape((x.shape[0],-1))
    #     n_feature=x_test.shape[1]
    #     x_test_proba=None
    #     label=None
    #     for index in range(len(self.layers)):
    #         if index==len(self.layers)-1:
    #             # print(index)
    #             label=self.layers[index].predict(x_test)
    #         else:
    #             x_test_proba=self.layers[index].predict_proba(x_test)
    #             if (not self.if_stacking):
    #                 x_test=x_test[:,0:n_feature]
    #             x_test=np.hstack((x_test,x_test_proba))
    #     return label
        
        

    def predict_proba(self,x):
        x_test=x.copy()
        x_test=x_test.reshape((x.shape[0],-1))
        n_feature=x_test.shape[1]
        # print(x_test.shape)
        x_test_proba=None
        for index in range(len(self.layers)):
            if index==len(self.layers)-1:
                # print(index)
                x_test_proba=self.layers[index]._predict_proba(x_test)
            else:
                x_test_proba=self.layers[index].predict_proba(x_test)
                if (not self.if_stacking):
                    x_test=x_test[:,0:n_feature]
                x_test=np.hstack((x_test,x_test_proba))
        # print(x_test.shape)
        # prob=np.zeros(len(self.category))
        # for index in range(len(self.output_layer)):
        #     prob=prob+self.output_layer[index].predict_proba(x_test)
        # proba=np.zeros((x.shape[0],len(self.category)))
        # for i in range(len(self.category)):
        #     for j in range(len(self.estimator_configs)):
        #         proba[:,i]+=x_test_proba[:,j*len(self.category)+i]
        return x_test_proba
        
    def unit_predict(self,x):
        x_test=x.copy()
        x_test=x_test.reshape((x.shape[0],-1))
        n_feature=x_test.shape[1]
        # print(x_test.shape)
        x_test_proba=None
        labels=None
        for index in range(len(self.layers)):
            if index==len(self.layers)-1:
                # print(index)
                labels=self.layers[index].unit_predict(x_test)
            else:
                x_test_proba=self.layers[index].predict_proba(x_test)
                if (not self.if_stacking):
                    x_test=x_test[:,0:n_feature]
                x_test=np.hstack((x_test,x_test_proba))
        return labels

    def preprocess(self,x_train,y_train): 
        x_train=x_train.reshape((x_train.shape[0],-1))
        category=np.unique(y_train)
        print(category)
        self.category=category
        #print(len(self.category))
        n_feature=x_train.shape[1]
        n_label=len(np.unique(y_train))
        LOGGER.info("Begin to train model use gcForest")
        LOGGER.info("The number of samples is {}, the shape is {}".format(len(y_train),x_train[0].shape))
        LOGGER.info("use {} as training evaluation".format(self.train_evaluation.__name__))
        LOGGER.info("stacking: {}, save model: {}".format(self.if_stacking,self.if_save_model))
        return x_train,n_feature,n_label
    
    # def set_output_layer(self,x_train,y_train,type="RandomForestClassifier",n_estimator=20,num=4):
    #     if len(self.output_layer_config)==0:
    #         for index in range(num):
    #             estimator_type=getattr(ensemble,type)
    #             estimator=estimator_type(n_estimators=n_estimator)
    #             estimator.fit(x_train,y_train)
    #             self.output_layer.append(estimator)
    #         LOGGER.info("Use default config to initialize output layer..............\n {} {}, {} estimators in each one".format(num,type,n_estimator))
    #     else:
    #         for index in range(len(self.output_layer_config)):
    #             estimator_type=getattr(ensemble,self.output_layer_config[index]["type"])
    #             estimator=estimator_type(n_estimators=self.output_layer_config[index]["n_estimators"])
    #             estimator.fit(self.X_enhanced,self.Y)
    #             self.output_layer.append(estimator)
    #         LOGGER,info("Use custom config to initialize output layer...............\n ")
    
