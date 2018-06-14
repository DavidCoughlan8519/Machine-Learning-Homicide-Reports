
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import sklearn.gaussian_process as gp
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model
import numpy as np
from sklearn import model_selection
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr






def runClassifiersCV(data, target):
    
    print ("\nThe following are the initial accuracies using CV 10")
    dTree = tree.DecisionTreeClassifier()
    scores = model_selection.cross_val_score(dTree, data, target, cv=10)
    print ("Tree : ", scores.mean())
    
    #rbfSvm = SVC()
    #scores = model_selection.cross_val_score(rbfSvm, data, target, cv=10)
    #print ("SVM : ", scores.mean())
    
    nearestN = KNeighborsClassifier()
    scores = model_selection.cross_val_score(nearestN, data, target, cv=10)
    print ("NNeighbour : ", scores.mean())
    
    randomForest = RandomForestClassifier()
    scores = model_selection.cross_val_score(randomForest, data, target, cv=10)
    print ("RForest : ",scores.mean())
    
    nBayes = naive_bayes.GaussianNB()
    scores = model_selection.cross_val_score(nBayes, data, target, cv=10)
    print ("Naive Bayes : ",scores.mean())

    logR = linear_model.LogisticRegression()
    scores = model_selection.cross_val_score(logR, data, target, cv=10)
    print ("Log R : ",scores.mean())
    

def standardizer (data,target):
    # Standardize the data
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.8, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)
    return X_train_scaled,X_test_scaled

def runModelSelectionRandomForest(data, target):
    param_grid = [ {'n_estimators': list(range(10, 400, 30)),  'criterion':["gini", "entropy"], "max_features":["auto", "log2", "sqrt"] }  ]
    
    clf = GridSearchCV(RandomForestClassifier(random_state=10), param_grid, cv=10)
    
    clf.fit(data, target)
    
    print("\n Best parameters set found on development set:")
    
    print(clf.best_params_ , "with a score of ", clf.best_score_)

    return clf.best_estimator_


def runModelSelectionNaiveBayes(data, target):
    param_grid = [{
        'vect__max_features': (5000, 10000), # how many features to vectorize
        'vect__ngram_range': ((1,1), (1, 2)),  # unigrams or bigrams
        'clf__alpha': (0.00001, 0.000001), # smoothing parameter for classfier
    }]
    clf = GridSearchCV(naive_bayes.GaussianNB(),param_grid, cv=10)

    results = model_selection.cross_val_score(clf, data, target, cv=10)
    clf.fit(data, target)

    print("\n Best parameters set found on development set:")

    print(clf.best_params_, "with a score of ", clf.best_score_)

    return clf.best_estimator_

def runModelSelectionRandomForestRegression(X_train,X_test,y_train,y_test):
    clf = RandomForestRegressor(n_estimators=150, min_samples_split=2)
    clf.fit(X_train, y_train)
    print("Random Forest Regression:",clf.predict(X_test))


    #predicted_train = clf.predict(X_train)
    #predicted_test = clf.predict(X_test)
    #test_score = r2_score(y_test, predicted_test)
    #spearman = spearmanr(y_test, predicted_test)
    #pearson = pearsonr(y_test, predicted_test)
    #print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
    #print(f'Test data R-2 score: {test_score:>5.3}')
    #print(f'Test data Spearman correlation: {spearman[0]:.3}')
    #print(f'Test data Pearson correlation: {pearson[0]:.3}')

    #-----------------------------------------------------------------------------------------------
    #print("Random Forest Regression: ", accuracy_score(y_test, regressor.predict(X_test) * 100, "%"))
    # overall accuracy
    #acc = clf.score(X_test,y_test)
    #acc=   clf.predict(y_test)

    # get roc/auc info
    #Y_score = clf.decision_function(X_test)
    #fpr = dict()
    #tpr = dict()
    #fpr, tpr, _ = roc_curve(y_test, Y_score)

    #roc_auc = dict()
    #roc_auc = auc(fpr, tpr)

    # make the plot
    #plt.figure(figsize=(10,10))
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlim([-0.05, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.grid(True)
   # plt.plot(fpr, tpr, label='AUC = {0}'.format(roc_auc))
    #plt.legend(loc="lower right", shadow=True, fancybox =True)


def runModelSelectionKNN(data, target):
    param_grid = [ {'n_neighbors': list(range(1, 30, 2)),  'p':[1, 2, 3, 4, 5] , "weights":["uniform", "distance"]}]
      
    
    clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)
    
    clf.fit(data, target)
    
    print("\n Best parameters set found on development set:")
    
    print(clf.best_params_ , "with a score of ", clf.best_score_)
    
    return clf.best_estimator_


def runModelSelectionLR(data, target):
    clf =GridSearchCV(cv=10,
                 estimator=linear_model.LogisticRegression(C=1.0, intercept_scaling=1,
                                              dual=False, fit_intercept=True, penalty='l2', tol=0.0001),
                 param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})

    clf.fit(data, target)

    print("\n Best parameters set found on logistic regression set:")

    print(clf.best_params_, "with a score of ", clf.best_score_)

    return clf.best_estimator_

def runModelSelectionSVC(data, target):
    knn = SVC()
    scores = model_selection.cross_val_score(knn, data, target, cv=10)
    print ("Knn Scores",scores.mean())
    
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1, 1]
    kernel = ['linear', 'rbf']
    
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernel}
    
    clf = GridSearchCV(SVC(), param_grid, cv=10)
    
    clf.fit(data, target)
    
    print("\n Best parameters set found on SVC development set:")
    
    print(clf.best_params_ , "with a score of ", clf.best_score_)
    
    return clf.best_estimator_


def graph_weapons_gender(df):
    threshold = 0.05
    weaponCountM = df[df['Perpetrator_Sex'] == 'Male']['Weapon'].value_counts(normalize=True).sort_values()
    weaponCountF = df[df['Perpetrator_Sex'] == 'Female']['Weapon'].value_counts(normalize=True).sort_values()
    weaponCountM['Other'] = weaponCountM[weaponCountM < threshold].sum()
    weaponCountF['Other'] = weaponCountF[weaponCountF < threshold].sum()
    plt.figure()
    plt.pie(weaponCountM[weaponCountM > threshold].values, labels=weaponCountM[weaponCountM > threshold].index)
    plt.title('Weapons used by male Perpetrators')
    plt.show()
    plt.figure()
    plt.pie(weaponCountF[weaponCountF > threshold].values, labels=weaponCountF[weaponCountF > threshold].index)
    plt.title('Weapons used by female Perpetrators')
    plt.show()



def performPreprocessing(homicide):
    # Relevant data is only where the crime is murder or manslaughter
    homicide = homicide.groupby('Crime_Type').get_group('Murder or Manslaughter')
    graph_weapons_gender(homicide)

    # remove features
    #homicide = homicide.drop(['Agency_Name'], axis=1)
    homicide = homicide.drop(['Perpetrator_Ethnicity'], axis=1)
    homicide = homicide.drop(['Victim_Ethnicity'], axis=1)
    homicide = homicide.drop(['Victim_Count'], axis=1)
    homicide = homicide.drop(['Perpetrator_Count'], axis=1)
    homicide = homicide.drop(['Relationship'], axis=1)
    homicide = homicide.drop(['Incident'], axis=1)
    homicide = homicide.drop(['Record_ID'], axis=1)
    #homicide = homicide.drop(['Agency_Code'], axis=1)
    homicide = homicide.drop(['Crime_Solved'], axis=1)
    homicide = homicide.drop(['Record_Source'], axis=1)
    homicide = homicide.drop(['Agency_Type'], axis=1)

    homicide['Perpetrator_Age'] = homicide.query("Perpetrator_Age != 0")

    homicide = homicide.replace('Unknown', np.nan)
    homicide = homicide.dropna()

    imputer = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
    imputer.fit(homicide[['Perpetrator_Sex']])
    #homicide['Perpetrator Sex'] = imputer.transform(homicide[['Perpetrator Sex']])
 
    #USE LABEL ENCODER FOR RACES
    #label_encoder = preprocessing.LabelEncoder()
    #homicide['Agency_Name'] = label_encoder.fit_transform(homicide['Agency_Name'])
    #print(homicide['Agency_Name'])
    label_encoder = preprocessing.LabelEncoder()
    homicide['Perpetrator_Sex'] = label_encoder.fit_transform(homicide['Perpetrator_Sex'])
    label_encoder = preprocessing.LabelEncoder()
    homicide['Perpetrator_Race'] = label_encoder.fit_transform(homicide['Perpetrator_Race'])
    label_encoder = preprocessing.LabelEncoder()
    homicide['Victim_Race'] = label_encoder.fit_transform(homicide['Victim_Race'])
    label_encoder = preprocessing.LabelEncoder()
    homicide['Victim_Sex'] = label_encoder.fit_transform(homicide['Victim_Sex'])
    label_encoder = preprocessing.LabelEncoder()
    homicide['Crime_Type'] = label_encoder.fit_transform(homicide['Crime_Type'])
    label_encoder = preprocessing.LabelEncoder()
    homicide['Month'] = label_encoder.fit_transform(homicide['Month'])
    #-----------------------------------------------------------------------------------
    label_encoder = preprocessing.LabelEncoder()
    homicide['Agency_Name'] = label_encoder.fit_transform(homicide['Agency_Name'])
    label_encoder = preprocessing.LabelEncoder()
    homicide['Agency_Code'] = label_encoder.fit_transform(homicide['Agency_Code'])



    #label_encoder = preprocessing.LabelEncoder()
   # homicide['Agency_Type'] = label_encoder.fit_transform(homicide['Agency_Type'])
    #print(homicide['Agency_Type'])
    label_encoder = preprocessing.LabelEncoder()
    homicide['City'] = label_encoder.fit_transform(homicide['City'])
    label_encoder = preprocessing.LabelEncoder()
    homicide['State'] = label_encoder.fit_transform(homicide['State'])


    #titanic['Embarked_Numeric'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    #titanic = titanic.drop(['Embarked'], axis=1)
    #homicide = pd.get_dummies(homicide, columns=["Embarked"])


    #scalingObj = preprocessing.MinMaxScaler()
   # homicide[["Age", "SibSp", "Parch", "Fare", "Pclass"]]= scalingObj.fit_transform(homicide[["Age", "SibSp", "Parch", "Fare", "Pclass"]])


    return homicide

def correlation_matrix(df):
    X, y = make_classification(n_samples=500, n_features=8, n_informative=6,
                               n_redundant=0, n_repeated=2, n_classes=2, random_state=0,
                               shuffle=False)
    df = pd.DataFrame(X)
    corrResults = df.corr()
    sns.heatmap(corrResults)
    plt.show()

def correlation_matrix2(df):
    X, y = make_classification(n_samples=500, n_features=8, n_informative=6,
                               n_redundant=0, n_repeated=2, n_classes=2, random_state=0,
                               shuffle=False)
    df = pd.DataFrame(X)
    corrResults = df.corr()
    sns.heatmap(corrResults)
    plt.show()
    sns.jointplot(df.iloc[:, 2].values, df.iloc[:, 6].values)
    plt.show()

def outliers(df):
    X = df.data
    y = df.target
    sns.boxplot(x=X[:, 1])
    plt.show()


def performManualKFold(data, target):

    kf = model_selection.KFold(n_splits=10)

    misclassificationSurvived = []
    misclassificationDeaths = []

    for train_fold_i, test_fold_i in kf.split(data):
        clf = RandomForestClassifier()
        clf.fit( data[train_fold_i], target[train_fold_i] )
        #clf.fit(data.iloc[train_index],target.iloc[train_index])
        testFeatures = data[test_fold_i]
        testLabels = target[test_fold_i]
        
        results= clf.predict(testFeatures)

        misclassifiedResults = results[ results != testLabels]
        #Calcualte Number of deaths as a fraction of total deaths

        totalFirearns =  len(testLabels[testLabels==1])
        totalNotFirearms =    len(testLabels[testLabels==0])

        firearmMisclassRate = len(misclassifiedResults[misclassifiedResults == 1])/totalFirearns
        otherMisclassRate = len(misclassifiedResults[misclassifiedResults == 0])/totalNotFirearms
       
        misclassificationSurvived.append(firearmMisclassRate)
        misclassificationDeaths.append(otherMisclassRate)

    print ("Average misclassification on FIREARMS = ", np.mean(misclassificationSurvived))
    print ("Average misclassification on NON-FIREARMS = ", np.mean(misclassificationDeaths))



def main():

    # Open the training dataset as a dataframe and perform preprocessing
    homicide_train = pd.read_csv("database.csv",low_memory=False)  #,encoding="utf-8"
    feature_train=performPreprocessing(homicide_train)


    #outliers(feature_train)
    feature_train['Weapon'] = feature_train['Weapon'].apply(lambda x: 'Gun' if x in ['Rifle', 'Firearm', 'Shotgun', 'Handgun', 'Gun'] else 'Other')

    label_encoder = preprocessing.LabelEncoder()
    feature_train['Weapon'] = label_encoder.fit_transform(feature_train['Weapon'])
    feature_train = feature_train.apply(label_encoder.fit_transform)

    #Standardize
    #scalingObj = preprocessing.StandardScaler()
    #standardizedData = scalingObj.fit_transform(feature_train)
    #feature_train = pd.DataFrame(standardizedData, columns=feature_train.columns)

    # Split the training dataset into features and classes
    label_train = feature_train["Weapon"]
    feature_train= feature_train.drop(["Weapon"], axis= 1)
    #Split into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(feature_train, label_train, random_state=100)
    #feature_test =  performManualKFold(feature_train.values, label_train.values)


    #Run the classifier


    correlation_matrix(feature_train)
    #correlation_matrix2(feature_train)
    #feature_test=perform_kfold(feature_train,label_train)
    #print(feature_test)


    #runClassifiersCV(feature_train, label_train)
    #runClassifiersCV(X_train,y_train)
    
    #bestModel = runModelSelectionSVC(feature_train, label_train)
    #bestModel = runModelSelectionKNN(feature_train, label_train)
    #bestModel = runModelSelectionRandomForest(feature_train, label_train)
    #bestModel = runModelSelectionNaiveBayes(feature_train,label_train)
    #bestModel = runModelSelectionLR(feature_train,label_train)
    bestModel = runModelSelectionRandomForest(feature_train, label_train)

    #tried for increase in accuracy but didnt get it to run properly
    #bestModel = runModelSelectionRandomForestRegression(X_train,X_test,y_train,y_test)

    #results = bestModel.predict(feature_train)


    #print(results)
    
    #Uncomment below if you want to perform manual K Fold Cross Validation
    #performManualKFold(feature_train.values, label_train.values)
    
    #resultSeries = pd.Series(data = results, name = 'Weapon', dtype='int64')
main()



