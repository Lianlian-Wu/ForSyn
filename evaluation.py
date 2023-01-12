from sklearn.metrics import accuracy_score,f1_score

def accuracy(y_true,y_pred):
    return accuracy_score(y_true,y_pred)

def f1_binary(y_true,y_pred):
    f1=f1_score(y_true,y_pred,average="binary")
    return f1

def f1_micro(y_true,y_pred):
    f1=f1_score(y_true,y_pred,average="micro")
    return f1

def f1_macro(y_true,y_pred):
    f1=f1_score(y_true,y_pred,average="macro")
    return f1