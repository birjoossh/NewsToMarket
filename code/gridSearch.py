from sklearn import naive_bayes, metrics, linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
 
from sklearn.model_selection import GridSearchCV
from sklearn import svm
 
svmparameters = {'kernel':('linear', 'rbf','poly','sigmoid'), 'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100],'degree':[2,3,4,5],'shrinking':[True,False],'random_state':[None,2017]}
svr = svm.SVC()
clf_GS_svm = GridSearchCV(svr, svmparameters)
clf_GS_svm.fit(x_train_tfidf,y_train)
#sorted(clf_GS_svm.cv_results_.keys())
GS_svm_pred = clf_GS_svm.predict(x_test_tfidf)
GS_svm_accuracy = metrics.accuracy_score(y_test,GS_svm_pred)
print ('Accuracy score for clf_GS_svm: ',GS_svm_accuracy)
print ('Best params for clf_GS_svm: ', clf_GS_svm.best_params_ )
#print ('Detailed GS_SVM results: ', clf_GS_svm.cv_results_ )
 
from sklearn.ensemble import BaggingClassifier
 
Baggingparameters = {'n_jobs':[2],'n_estimators':[10,100,200,400,800,1600,1200], 'max_features':[10,15,20,25], 'bootstrap':[True],
                     'oob_score':[True,False],'random_state':[None,2017]}
Bgc = BaggingClassifier()
clf_GS_BG = GridSearchCV(Bgc, Baggingparameters)
clf_GS_BG.fit(x_train_tfidf,y_train)
#sorted(clf.cv_results_.keys())
GridPredBag = clf_GS_BG.predict(x_test_tfidf)
GS_BG_accuracy = metrics.accuracy_score(y_test,GridPredBag)
print ('Accuracy score for clf_GS_BG: ',GS_BG_accuracy)
print ('Best params for clf_GS_BG: ', clf_GS_BG.best_params_ )
#print ('Detailed clf_GS_BG results: ', clf_GS_BG.cv_results_ )
#print ('OOB score from clf_GS_BG : ', clf_GS_BG.oob_score_ )
#print ('OOB decision function from clf_GS_BG : ', clf_GS_BG.oob_decision_function_ )
 
from sklearn.ensemble import RandomForestClassifier
 
RFparameters = {'n_jobs':[2],'criterion':('gini','entropy'), 'n_estimators':[10,100,200,400,800,1600,1200], 'max_features':[10,15,20,25],
                'bootstrap':[True],'oob_score':[True,False],'random_state':[None,2017]}
RF_GS = RandomForestClassifier()
clf_GS_RF = GridSearchCV(RF_GS, RFparameters)
clf_GS_RF.fit(x_train_tfidf,y_train)
sorted(clf_GS_RF.cv_results_.keys())
GridPredRF = clf_GS_RF.predict(x_test_tfidf)
GS_RF_accuracy = metrics.accuracy_score(y_test,GridPredBag)
print ('Accuracy score for clf_GS_RF: ',GS_RF_accuracy)
print ('Best params for clf_GS_RF: ', clf_GS_RF.best_params_ )
print ('Detailed clf_GS_RF results: ', clf_GS_RF.cv_results_ )
#print ('OOB score from clf_GS_RF : ', clf_GS_RF.oob_score_ )
#print ('OOB decision function from clf_GS_RF : ', RF_GS.oob_decision_function_ )
print ('Feature importances from clf_GS_RF : ', RF_GS.feature_importances_ )
RF_GS.decision_path(x_train_tfidf)
print (clf_GS_RF.best_params_)
print (clf_GS_RF.best_score_)
print (clf_GS_RF.grid_scores_)
#clf_GS_RF.error_score
print(metrics.confusion_matrix( y_test, GridPredRF ))
#plt.plot( , clf_GS_RF)
 
for e in df['param_max_features'].value_counts().index:
    print( e)
    estimators = df.param_n_estimators[(df.param_criterion == 'entropy') & (df.param_oob_score == True)]
    mean_score = df.mean_test_score[(df.param_criterion == 'entropy') & (df.param_oob_score == True)]
    plt.plot( estimators[df['param_max_features']==e],
             mean_score[df['param_max_features']==e], label=e )
plt.ylabel( "Mean Score" )
plt.xlabel( "n_estimators" )
plt.title( " Best Mean Score vs estimators for diff max features" )
plt.legend()
plt.show()
