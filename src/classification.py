from enum import Enum

from sklearn.linear_model import LogisticRegression
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

class ClassifierType(Enum):

    LogisticRegression = "LogisticRegression"
    ExtraTreeClassifier = "ExtraTreeClassifier"
    RandomForestClassifier = "RandomForestClassifier"
    GradientBoostingClassifier = "GradientBoostingClassifier"
    KNeighborsClassifier = "KNeighborsClassifier"
    SVC = "SVC"



def get_classifier_by_type(classifier_type : ClassifierType):

    if classifier_type == ClassifierType.LogisticRegression:
        return LogisticRegression
    if classifier_type == ClassifierType.ExtraTreeClassifier:
        return ExtraTreeClassifier
    if classifier_type == ClassifierType.RandomForestClassifier:
        return RandomForestClassifier
    if classifier_type == ClassifierType.GradientBoostingClassifier:
        return GradientBoostingClassifier
    if classifier_type == ClassifierType.KNeighborsClassifier:
        return KNeighborsClassifier
    if classifier_type == ClassifierType.SVC:
        return SVC

    raise ValueError("Classifier not supported")


def get_stratified_folds(X, y):
    skf = StratifiedKFold(n_splits=5)
    return list(skf.split(X, y))