import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def report_cls(y_true, y_pred, labels=[0,1,2]):
    rep = classification_report(y_true, y_pred, labels=labels, target_names=["conserv.","moder.","agress."], digits=3)
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    return rep, cm
