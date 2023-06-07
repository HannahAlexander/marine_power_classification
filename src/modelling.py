# models applied to data
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import src.evaluation as eval
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

def log_reg(X_train, Y_train, X_test, Y_test, hyperparams):
    """
        Fits and scores a logistic regression model

        Input arguments:
        X_train - features for training the model
        Y_train - target for training the model
        X_test - features for testing the model
        Y_test - target for testing the model

        Returns:
        A dictionary containing the tuned model, and model performance metrics

    """

    model = LogisticRegression()
        
    classifier = GridSearchCV(model, hyperparams, scoring="f1_micro", verbose=0, cv=5, n_jobs=1)
    best_model = classifier.fit(X_train, Y_train)
    best_params = classifier.best_estimator_.get_params()
    
    best_model = LogisticRegression(**best_params)
    best_model.fit(X_train, Y_train)
    metrics = eval.scoring_func(Y_test, best_model.predict(X_test))

    precision = metrics["Precision"]
    recall =  metrics["Recall"]
    accuracy =  metrics["Accuracy"]
    f1_score =  metrics["F1 Score"]

    return {"best_model": best_model, "accuracy": accuracy, "F1_score": f1_score,
            "precision": precision, 'recall': recall}


def decision_tree(X_train, Y_train, X_test, Y_test, hyperparams):
    """
        Fits and scores a decision tree

        Input arguments:
        X_train - features for training the model
        Y_train - target for training the model
        X_test - features for testing the model
        Y_test - target for testing the model

        Returns:
        A dictionary containing the tuned model, and model performance metrics

    """
    model = DecisionTreeClassifier()
        
    classifier = GridSearchCV(model, hyperparams, scoring="f1_micro", verbose=0, cv=5, n_jobs=1)
    best_model = classifier.fit(X_train, Y_train)
    best_params = classifier.best_estimator_.get_params()
    
    best_model = DecisionTreeClassifier(**best_params)
    best_model.fit(X_train, Y_train)
    metrics = eval.scoring_func(Y_test, best_model.predict(X_test))

    precision = metrics["Precision"]
    recall =  metrics["Recall"]
    accuracy =  metrics["Accuracy"]
    f1_score =  metrics["F1 Score"]

    return {"best_model": best_model, "accuracy": accuracy, "F1_score": f1_score,
            "precision": precision, 'recall': recall}


def lightgbm(X_train, Y_train, X_test, Y_test, hyperparams):
    """
        Fits and scores a lightgbm model

        Input arguments:
        X_train - features for training the model
        Y_train - target for training the model
        X_test - features for testing the model
        Y_test - target for testing the model

        Returns:
        A dictionary containing the tuned model, and model performance metrics

    """
    model = LGBMClassifier()
        
    classifier = GridSearchCV(model, hyperparams, scoring="f1_micro", verbose=0, cv=5, n_jobs=1)
    best_model = classifier.fit(X_train, Y_train)
    best_params = classifier.best_estimator_.get_params()
    
    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train, Y_train)
    metrics = eval.scoring_func(Y_test, best_model.predict(X_test))

    precision = metrics["Precision"]
    recall =  metrics["Recall"]
    accuracy =  metrics["Accuracy"]
    f1_score =  metrics["F1 Score"]

    return {"best_model": best_model, "accuracy": accuracy, "F1_score": f1_score,
            "precision": precision, 'recall': recall}

