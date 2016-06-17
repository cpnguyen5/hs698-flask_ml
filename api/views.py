from api import app
import pandas as pd
import json
from flask import jsonify, session
from flask import render_template
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from flask import url_for
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics, grid_search, svm
from sklearn.linear_model import LogisticRegression


def get_abs_path():
    """
    This function takes no parameters and returns the api root directory pathway.
    :return: api directory pathway
    """
    return os.path.abspath(os.path.dirname(__file__))


def get_data():
    """
    This functions takes no parameters, reads in the CSV file and returns it as a pandas Data Frame. The function
    drops any rows with missing values and converts the class into a binary classification (0, 1).
    :return: pandas Data Frame
    """
    f_name = os.path.join(get_abs_path(), 'data', 'breast-cancer-wisconsin.csv') #path to CSV file
    #Create pandas Data Frame from CSV
    columns = ['code', 'clump_thickness', 'size_uniformity', 'shape_uniformity',
               'adhesion', 'cell_size', 'bare_nuclei', 'bland_chromatin',
               'normal_nuclei', 'mitosis', 'class']

    conv= lambda x: 1 if int(x)==4 else 0 #converts class into binary ( 0:benign(2), 1: malignant(4) )
    df = pd.read_csv(f_name, sep=',', header=None, names=columns, na_values='?', converters={10:conv})
    return df.dropna()


def get_numpy(DataFrame):
    """
    This function takes one parameter, a pandas DataFrame, and returns the object as a Numpy array.
    :param DataFrame: pandas DataFrame of data
    :return: Numpy array of data
    """
    data = DataFrame.as_matrix()
    return data


def partition(data):
    """
    This function takes one parameter, a numpy array of data, and partitions the array into Train/Test sets. The function
    will return a tuple of the partitioned data set.
    :param data: numpy array of data
    :return: tuple of partitioned data (X_train, X_test, y_train, y_test)
    """
    # data = data.astype(np.float64)
    data_train, data_test = train_test_split(data, random_state=2, test_size=0.30)

    n_col = data.shape[1] - 1 #last index position
    #Isolate features from outcomes/labels
    X_train = data_train[:, 0:n_col] #training features
    y_train = data_train[:, n_col] #training labels
    X_test = data_test[:, 0:n_col] #testing features
    y_test = data_test[:, n_col] #testing labels
    return (X_train, X_test, y_train, y_test)


def scale(X_train, X_test):
    """
    This function takes two parameters, the Training & Testing samples/features and returns the respective normalized/scaled
    versions.
    :param X_train: Training set samples
    :param X_test: Testing set samples
    :return: tuple of normalized/scaled Training & Testing sets samples
    """
    scaler = MinMaxScaler().fit(X_train) #scaler object fitted on training set of samples
    scaled_X_train = scaler.transform(X_train) #transformed normalized data - Training set samples
    scaled_X_test = scaler.transform(X_test) #transformed normalized data - Testing set samples
    return (scaled_X_train, scaled_X_test)


def feature_select(X_train, X_test, y_train, n_feat):
    """
    This function takes four parameters, numpy arrays of Training set samples, Testing set samples, training set labels,
    and number of features to select/reduce to. The function will perform univariate feature selection using sklearn.feature_selection.SelectKBest
    and a score function. The function will return a tuple of reduced Training set and Testing set samples as well as scores
    and p-values.
    :param X_train: Training set samples
    :param X_test: Testing set samples
    :param y_train: Training set labels
    :param n_feat: number of features to select
    :return: tuple of selected Training set samples, selected Testing set samples, scores, and p-values
    """
    # univariate feature selection
    score_func = SelectKBest(chi2, k=n_feat).fit(X_train, y_train)  # k = # features
    select_X_train = score_func.transform(X_train)  # transform feature selection/reduction on training set samples
    select_X_test = score_func.transform(X_test)  # transform feature selection/reduction on testing set samples

    score = score_func.scores_
    pval = score_func.pvalues_
    return (select_X_train, select_X_test, score, pval)


def gridsearch(X_train, X_test, y_train):
    """
    This function takes three parameters, Training set samples, Testing set samples, and Training set labels. The function
    will determine the optimal parameters of the best classifier model/estimator by performing a grid search. The best model
    will be fitted with the Training set and subsequently applied on the Testing set for predictions.

    :param X_train: Training set samples
    :param X_test:  Testing set samples
    :param y_train: Trainig set labels
    :return: tuple of Best Classifier instance, Best Classifier predictions (y_pred), dictionary of optimal parameters, and grid score
    """
    # Setup Parameter Grid -- dictionary of parameters -- map parameter names to values to be searched
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 100, 1000], 'fit_intercept': [True, False], 'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']},
        {'C': [0.01, 0.1, 1, 10, 100, 1000], 'fit_intercept': [True, False], 'penalty': ['l1'], 'solver': ['liblinear']}
    ]

    # Create "blank" clf instance
    blank_clf = LogisticRegression(random_state=2)

    # Grid search to find "best" classifier -- Hyperparameters Optimization
    clf = grid_search.GridSearchCV(blank_clf, param_grid, n_jobs=-1)  # classifier + optimal parameters
    clf = clf.fit(X_train, y_train)  # fitted classifier -- Training Set
    best_est = clf.best_estimator_
    clf_pred = best_est.predict(X_test)  # apply classifier on test set for label predictions

    best_params = clf.best_params_  # best parameters identified by grid search
    score = clf.best_score_  # best grid score
    return (best_est, clf_pred, best_params, score)


def clf(X_train, X_test, y_train):
    """
    This function takes three parameters, Training set samples, Testing set samples, and Training set labels. The function
    serves as a convenience function and instantiates the model with optimal parameters (previously identified by a grid search).
    The classifier model is applied on the Testing set for classification preductions.

    :param X_train: Training set samples
    :param X_test:  Testing set samples
    :param y_train: Training set labels
    :return: tuple of fitted classifier instance and classifier predictions (y_pred)
    """
    model = LogisticRegression(penalty= 'l2', C= 1, solver= 'liblinear', fit_intercept= True) #best param determined by gridsearch
    model = model.fit(X_train, y_train) #Fit classifier to Training set
    y_pred = model.predict(X_test) # Test classifier on Testing set
    return (model, y_pred)


def sensitivity(model_pred, target):
    """
    This function takes two parameters, numpy array of the model's classification prediction and true target/labels.
    Given these inputs, the function will calculate and return the sensitivity value of the classification as a float.

    :param model_pred: classifier model's classification prediction
    :param target: True target/labels (y_test)
    :return: sensitivity value
    """
    y_pred = model_pred # prediction
    y_true = target # true labels
    #Confusion Matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    TN = float(cm[0,0]) #True Negative
    FP = float(cm[0,1]) #False Positive
    FN = float(cm[1,0]) #False Negative
    TP = float(cm[1,1]) #True Positive

    #sensitivity calculation
    final_senstivity = TP/(TP + FN)
    return final_senstivity


def specificity(model_pred, target):
    """
    This function takes two parameters, numpy array of the model's classification prediction and true target/labels.
    Given these inputs, the function will calculate and return the specificity of the classification as a float.

    :param model_pred: classifier model's classification prediction
    :param target: True target/labels (y_test)
    :return: specificity value
    """
    y_pred = model_pred #prediction
    y_true = target #true labels
    #Confusion Matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    TN = float(cm[0,0]) #True Negative
    FP = float(cm[0,1]) #False Positive
    FN = float(cm[1,0]) #False Negative
    TP = float(cm[1,1]) #True Positive

    #specificity calculation
    N = FP + TN
    TNR = TN/N
    return TNR


def accuracy(model_pred, target):
    """
    This function takes two parameters, numpy array of the model's classification prediction and true target/labels.
    Given these inputs, the function will calculate and return the accuracy value of the classification as a float.

    :param model_pred: classifier model's classification prediction
    :param target: True target/labels (y_test)
    :return: accuracy value
    """
    accuracy = metrics.accuracy_score(target, model_pred)
    return accuracy


def precision(model_pred, target):
    """
    This function takes two parameters, numpy array of the model's classification prediction and true target/labels.
    Given these inputs, the function calculates and returns the precision value of the classification as a float.

    :param model_pred: classifier model's classification prediction
    :param target: True target/labels (y_test)
    :return: precision value
    """
    y_pred = model_pred #prediction
    y_true = target #true labels
    precision_score = metrics.precision_score(y_true, y_pred) #precision calculation
    return precision_score


def recall(model_pred, target):
    """
    This function takes two parameters, numpy array of the model's classification prediction and true target/labels.
    Given these inputs, the function calculates and returns the recall value of the classification as a float.

    :param model_pred: classifier model's classification prediction
    :param target: True target/labels (y_test)
    :return: recall value
    """
    y_pred = model_pred #prediction
    y_true = target #true labels
    recall_score = metrics.recall_score(y_true, y_pred) #recall calculation
    return recall_score


def auc(model, X_test, target):
    """
    This function takes three parameters, the classifier model, testing set samples, and true targets/labels test set.
    Given these inputs, the area under the (ROC) curve will be returned based on the decision function (y_score).

    :param model: fitted classifier model
    :param X_test: testing set samples
    :param target: True target/labels (y_test)
    :return: AUC score
    """
    y_true = target
    y_score = model.decision_function(X_test) #Predict confidence scores
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score) #calculate FPR & TPR
    auc_score = metrics.auc(fpr, tpr) #calculate area under the curve
    return auc_score


def plot_roc(model, X_test, target, n_features):
    """
    This function takes four parameters, the fitted classification model, Testing set samples, target/labels test set,
    and number of features. Given these inputs, matplotlib will be used to plot the ROC curve of the classifier.
    The function will return the figure of the plot.

    :param model: fitted classification model
    :param X_test: Testing set samples
    :param target: true target/labels (y_test)
    :param n_features: int indicating number of features of data set
    :return: Plot of ROC curve
    """
    y_true = target
    y_score = model.decision_function(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)  # calculate FPR & TPR
    auc_score = metrics.auc(fpr, tpr)  # calculate area under the curve

    fig = plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic: \n (n_features selected= %d)' % (n_features))
    return fig


@app.route('/')
def index():
    df=get_data()
    X=df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix() #as_matrix() pandas method to return data as matrix/np array
    y=df.ix[:, df.columns=='class'].as_matrix()
    #scale
    scaler =preprocessing.StandardScaler().fit(X)
    scaled=scaler.transform(X)
    #PCA
    pcomp=decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components= pcomp.transform(scaled)
    var=pcomp.explained_variance_ratio_.sum() #View w/ Debug
    #KMeans
    model = KMeans(init='k-means++', n_clusters=2)
    model.fit(components)
    #Plot
    fig=plt.figure()
    plt.scatter(components[:,0], components[:,1], c=model.labels_)
    centers = plt.plot(
        [model.cluster_centers_[0,0], model.cluster_centers_[1,0]],
        [model.cluster_centers_[1,0], model.cluster_centers_[1,1]],
        'kx', c='Green'
    )
    #Increse size of center points
    plt.setp(centers, ms=11.0)
    plt.setp(centers, mew=1.8)
    #Plot axes adjustments
    axes =plt.gca()
    axes.set_xlim([-7.5, 3])
    axes.set_ylim([-2, 5])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clustering of PCs ({:.2f}% Var. Explained'.format(
        var * 100
    ))
    #Save fig
    fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cluster.png')
    fig.savefig(fig_path)
    # return render_template('index.html', fig=fig_path)#render name of html file
    return render_template('index.html',
                           fig=url_for('static',
                                       filename='tmp/cluster.png'))


@app.route('/d3')
def d3():
    df = get_data()
    X = df.ix[:, (df.columns != 'class') & (df.columns != 'code')].as_matrix()
    y = df.ix[:, df.columns == 'class'].as_matrix()
    # Scale
    scaler = preprocessing.StandardScaler().fit(X)
    scaled = scaler.transform(X)
    # PCA
    pcomp = decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components = pcomp.transform(scaled)
    var = pcomp.explained_variance_ratio_.sum() # View explained var w/ debug
    # Kmeans
    model = KMeans(init='k-means++', n_clusters=2)
    model.fit(components)
    # Generate CSV
    cluster_data = pd.DataFrame({'pc1': components[:, 0],
                                 'pc2': components[:, 1],
                                 'labels': model.labels_})
    csv_path = os.path.join(get_abs_path(), 'static', 'tmp', 'kmeans.csv')
    cluster_data.to_csv(csv_path)
    return render_template('d3.html',
                           data_file=url_for('static',
                                             filename='tmp/kmeans.csv'))


#Bonus: D3.js chart -- Grouped Bar Chart
@app.route('/bar')
def bar():
    df = get_data() #obtain DataFrame of data from CSV
    data = df.ix[:, df.columns != 'code'].as_matrix() #convert DataFrame to Numpy array
    #Filter data by class labels
    cls = data[:, -1]
    class_0 = data[cls==0]
    class_1 = data[cls==1]
    #Calculate Average for each Descriptor -- Grouped by classification
    avg_0 = np.average(class_0, axis=0)
    avg_1 = np.average(class_1, axis=0)
    avg_data = np.vstack((avg_0, avg_1))
    avg_data = avg_data.transpose() #tranpose data
    #Isolate Features from Outcomes
    X = avg_data[:-1,:] #features
    y = avg_data[-1,:] #outcomes
    #Create array for plot labelling purposes
    descriptors = np.array(['clump_thickness', 'size_uniformity', 'shape_uniformity',
               'adhesion', 'cell_size', 'bare_nuclei', 'bland_chromatin',
               'normal_nuclei', 'mitosis']).transpose()
    #Create pandas DataFrame from Numpy array
    class_df = pd.DataFrame({'benign': X[:, 0],
                             'malignant': X[:, 1],
                             'descriptors': descriptors})

    bar_path = os.path.join(get_abs_path(), 'static', 'tmp', 'breast_cancer_bar.csv') #save new CSV file at bar_path
    class_df.to_csv(bar_path, index=False) #DataFrame to CSV file in static/tmp
    return render_template('bar.html', d_file=url_for('static',
                                               filename='tmp/breast_cancer_bar.csv'))


@app.route('/prediction')
def prediction():
    #Obtain data
    pd_df = get_data() #obtain DataFrame of data from CSV
    data = get_numpy(pd_df) #convert DataFrame to Numpy array

    #Partition Data into Train-Test sets (70/30)
    X_train, X_test, y_train, y_test = partition(data)
    #Scale/Normalize Data
    scaled_X_train, scaled_X_test = scale(X_train, X_test)
    #Feature Selection - n_features=6
    select_X_train, select_X_test, score, pval = feature_select(scaled_X_train, scaled_X_test, y_train, n_feat=6)

    #Classifier Model
    best_est, y_pred, best_params, score = gridsearch(select_X_train, select_X_test, y_train)
    model = best_est
    # model, y_pred = clf(select_X_train, select_X_test, y_train) #convenience clf instantiation -- after grid search
    #Metrics
    # acc = accuracy(y_pred, y_test)
    # sens = sensitivity(y_pred, y_test)
    # spec = specificity(y_pred, y_test)
    # prec = precision(y_pred, y_test)
    # rec = recall(y_pred, y_test)
    # area = auc(model, select_X_test, y_test)
    # print acc, sens, spec, prec, rec, area

    #Pass Variable to prediction_confusion_matrix -- flask.session
    # cm=metrics.confusion_matrix(y_test, y_pred)
    # cm_dict = {'fp': cm[0,1], 'tp': cm[1,1], 'fn': cm[1,0], 'tn': cm[0,0]}
    # session['cm'] = cm_dict

    #Plot ROC Curve
    fig=plot_roc(model, select_X_test, y_test, 6)

    # Save fig
    fig_path = os.path.join(get_abs_path(), 'static', 'tmp', 'roc.png') #path to save fig
    fig.savefig(fig_path)
    return render_template('prediction.html',
                           fig=url_for('static',
                                       filename='tmp/roc.png'))


#new API endpoint
@app.route('/api/v1/prediction_confusion_matrix')
def prediction_confusion_matrix():
    #Obtain data
    pd_df = get_data() #DataFrame
    data = get_numpy(pd_df) #convert to Numpy array

    #Partition Data into Train-Test sets (70/30)
    X_train, X_test, y_train, y_test = partition(data)
    #Scale/Normalize Data
    scaled_X_train, scaled_X_test = scale(X_train, X_test)
    #Feature Selection - n_features=6
    select_X_train, select_X_test, score, pval = feature_select(scaled_X_train, scaled_X_test, y_train, n_feat=6)

    #Classifier Model
    best_est, y_pred, best_params, score = gridsearch(select_X_train, select_X_test, y_train)
    model = best_est

    #Confusion Matrix
    cm=metrics.confusion_matrix(y_test, y_pred) #confusion matrix
    cm_dict = {'fp': cm[0,1], 'tp': cm[1,1], 'fn': cm[1,0], 'tn': cm[0,0]} #dictionary of confusion matrix key-value pairs
    # cm_dict = session.pop('cm') #retrieve passed variables between routes
    model_cm = {'logistic regression': cm_dict} #key-value pair between model and dict of confusion matrix -- to JSON
    return jsonify(model_cm)


@app.route('/head') #head - url
def head(): #head - function to access url
    df = get_data().head() #head - dataframe
    data = json.loads(df.to_json()) #exports data frame as json string --> load/parsed json into python object (dict or lsit)
    return jsonify(data)

# app.secret_key = 'secret!key' #flask/session