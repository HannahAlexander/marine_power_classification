import sklearn.metrics as metrics
import pandas as pd
from sklearn.utils import resample

def scoring_func(y_true, y_pred):
    """
    Print evaluation metrics of trained model, including: precision, recall and accuracy.
    Plots confusion matrix.

    Input arguments:
    y_true: The true values of the varaible being predicted
    y_pred: The predicted value

    Returns:
    None

    """
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()

    print('Precision:', metrics.precision_score(y_true, y_pred))
    print('Recall:', metrics.recall_score(y_true, y_pred))
    print('Accuracy:', metrics.accuracy_score(y_true, y_pred))
    print('F1 Score:', metrics.f1_score(y_true, y_pred))

    return {'Precision': metrics.precision_score(y_true, y_pred), 
            'Recall': metrics.recall_score(y_true, y_pred),
            'Accuracy': metrics.accuracy_score(y_true, y_pred),
            'F1 Score': metrics.f1_score(y_true, y_pred)}


def resample_data(train_df, multiplier, X_test, y_test, classifier, target):
    """
    Applies upsampling to dataset

    Input arguments:
    train_df- training dataset
    multiplier- how much to multiply number of leads by
    X_test- test dataset features only
    y_test- test target variable
    classifier- classifier object

    Returns:
    Model fitted on new upsampled data

    """
    upsample_pos = resample(train_df[train_df[target]==1], replace=True, n_samples=round(sum(train_df[target]==1)*multiplier), random_state=123)
    neg_df = train_df[train_df[target]==0]
    train_df = pd.concat([neg_df, upsample_pos], axis=0)

    X_train = train_df.drop([target], axis = 1)
    y_train = train_df[target]

    y_train = y_train.values
    y_test = y_test

    print("X_train shape is: ", X_train.shape, "X_train shape is: ", y_train.shape)

    clf = classifier
    clf.fit(X_train, y_train)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    predicted_data = X_test.copy()
    predicted_data["y_pred"] = y_pred
    predicted_data["y_actual"] = y_test

    scoring_func(y_test, y_pred)

    return clf, predicted_data


