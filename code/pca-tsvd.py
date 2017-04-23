import pandas as pd
import re
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt


## NLTK provides a list of wtopwords; read it and keep it handy
stopwords_list = stopwords.words('english')

## PorterStemmer Algo will be used to stem the words i.e reduce inflection . eg. walk and walked will have stem as 'walk'
stemmer = PorterStemmer()

# # This transform function is used to preprocess the data at the time of reading the file itself. This is much more effieicnt than doing the preprocessing after reading the entire file
def transform( val ):
    val = re.sub( "^b[\"\']|[\"\']", "", str( val ).lower() )
    val_new = ""
    for word in re.split( "[\t\s]+", val ):
        stemmed_word = stemmer.stem( word )
        if stemmed_word in stopwords_list: continue
        val_new = val_new + " " + stemmed_word
    return val_new

# # Create a map of column id and converter function which will get applied to that col
c = {}
for i in range( 2, 27 ):
    c[i] = lambda x : transform( x )


dirs = "C:/DATA/SMU/_CLASSES/B9/pdataset/"
df = pd.read_csv( dirs + "Combined_News_DJIA.csv", sep=",", converters=c )
df['combined_news'] = df.apply( lambda row: " ".join( map( lambda x: str(x), row[2:] ) ), axis=1 )
df.to_csv( dirs + "cleaned_combined_News_DJIA.csv",sep=",", doublequote=True, index=False )







import pandas as pd
import re
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt


dirs = "C:/DATA/SMU/_CLASSES/B9/pdataset/"
df2 = pd.read_csv( dirs + "cleaned_combined_News_DJIA.csv", sep="," )
df2['word_count'] = df2['combined_news'].apply( lambda x: len( re.split( "[\t\s]+", x.strip( "\n" ) )))


plt.subplot(1,2,1)
sns.distplot( df2[(df2.Label==1)]['word_count'], bins=20, hist=True, kde=True, rug=True, norm_hist=True, label="Label=1" )
plt.xticks( rotation='vertical')
plt.legend()
plt.subplot(1, 2, 2)
sns.distplot( df2[(df2.Label==0)]['word_count'], bins=20, hist=True, kde=True, rug=True, norm_hist=True, label="Label=0" )
plt.legend()
plt.xticks( rotation='vertical')
plt.show()


# # ## Mean word count is centered around 300. Label = 0 seems slightly right skewed
sns.boxplot( x='Label', y='word_count', data=df2)
plt.show()



# # ## More variation of the word counts can be seen for label = 1
# # # Lets check the distribution of shared words each day
df2['shared_word_count'] = df2['combined_news'].apply( lambda x : np.count_nonzero(
                                                                pd.Series( re.split( "[\t\s]+", x ) )
                                                                    .value_counts().values != 1
                                                      )
                                 )

plt.subplot( 1, 2, 1)
sns.distplot( df2[(df2.Label==0)]['shared_word_count'], bins=20, hist=True, norm_hist=True, label="Label=0" )
plt.xticks( rotation='vertical' )
plt.legend()
plt.subplot( 1, 2, 2)
sns.distplot( df2[(df2.Label==1)]['shared_word_count'], bins=20, kde=True, hist=True, norm_hist=True, label="Label=1" )
plt.xticks( rotation='vertical' )
plt.legend()
plt.show()



# ## Label = 0 shows more density of shared words than Label = 1; will see if this cane be used as a predictive feature
from nltk.tokenize import word_tokenize
from nltk import pos_tag

word_tokens = df2['combined_news'].apply( lambda x: word_tokenize( x ) )
word_tokens.shape
len( word_tokens[0] )

word_tokens_tagged = word_tokens.apply( lambda x : pos_tag( x ) )
word_tokens_tagged[:10]

pos_map_1 = {}
for row in word_tokens_tagged[ df2['Label'] ==1 ]:
    for t in row:
        if not re.search( "\w+", t[1] ): continue
        pos_map_1.setdefault( t[1], 0 )
        pos_map_1[ t[1] ] += 1

pos_map_0 = {}
for row in word_tokens_tagged[ df2['Label'] ==0 ]:
    for t in row:
        if not re.search( "\w+", t[1] ): continue
        pos_map_0.setdefault( t[1], 0 )
        pos_map_0[ t[1] ] += 1


plt.figure( figsize=( 10, 10 ) )
plt.subplot( 2,1,1)
sns.barplot( x = list( pos_map_1.keys() ), y = list( pos_map_1.values() ))
plt.subplot( 2,1,2)
sns.barplot( x = list( pos_map_0.keys() ), y = list( pos_map_0.values() ))
plt.show()


# # ## Comparing the pos counts for both Label equal to 1 and 0 , we almost similar distribution. Not sure if including POS tags is going to improve the predictive power in any way but would be iteresting to see especially dominant POS like NN and JJ
df2['word_tokens'] = word_tokens


# # # Lets now try to build models on the data so far; Lets see how it goes
#
# # ## Before we start with model building lets split the data between train and test
#
from sklearn.feature_extraction.text import TfidfVectorizer

min( df2['Date'] ), max( df2['Date'])

train = df2[df2['Date']<'2015-10-01'].index
test = df2[df2['Date']>'2015-10-01'].index

# train = df2[df2['Date']<'2013-10-04'].index
# test = df2[df2['Date']>'2013-10-04'].index

# train = df2[df2['Date']<'2014-12-30'].index
# test = df2[df2['Date']>'2014-12-30'].index

print(len(train))
print(len(test))

x_train = df2['combined_news'].iloc[train]
x_test = df2['combined_news'].iloc[test]
y_train = df2['Label'].iloc[train]
y_test = df2['Label'].iloc[test]

min( x_train.index),  max( x_train.index)


#
vectorizer = TfidfVectorizer( ngram_range=(1,1), stop_words='english', norm='l2' )
# vectorizer_2 = TfidfVectorizer( ngram_range=(2,2), stop_words='english', norm='l2' )

x_train_tfidf = vectorizer.fit_transform( x_train )
x_test_tfidf = vectorizer.transform( x_test )

len(train)
x_train_tfidf.shape


# ## Every date has a total of 300 words but the vocab is 25021 words long. Too Sparse!!!
# NN = Noun
# JJ = Adjective

pos_holder = np.zeros( word_tokens_tagged.shape[0] )
for i, items in enumerate( word_tokens_tagged ):
    pos_holder[i] = 0
    for t in items:
        if t[1] == 'NN' or t[1] == 'JJ':
            pos_holder[i] += 1
pos_holder = pos_holder[:,np.newaxis]

x_train_extended = np.hstack( (x_train_tfidf.toarray(), pos_holder[train]) )
x_test_extended = np.hstack( (x_test_tfidf.toarray(), pos_holder[test]) )






from sklearn import naive_bayes, metrics, linear_model, neighbors, svm, tree, ensemble, neural_network

# In[ ]:
from sklearn.metrics import roc_curve
import matplotlib.patches as mpatches

Classifiers = [
    # naive_bayes.MultinomialNB(alpha = 0.1),

    linear_model.LogisticRegression(C=1,solver='liblinear',max_iter=100),   # chosen
    linear_model.LogisticRegression(C=1,solver='liblinear',max_iter=200),   # chosen

    neighbors.KNeighborsClassifier(3),

    svm.SVC(kernel="rbf", C=10, probability=True),
    svm.SVC(kernel="rbf", C=1, probability=True),
    svm.SVC(kernel="rbf", C=0.1, probability=True),
    svm.SVC(kernel="rbf", C=0.025, probability=True),

    tree.DecisionTreeClassifier(),

    ensemble.RandomForestClassifier(n_estimators=200),  # chosen

    ensemble.AdaBoostClassifier(n_estimators=100),
    ensemble.AdaBoostClassifier(n_estimators=100, algorithm='SAMME.R'),
    ensemble.AdaBoostClassifier(n_estimators=100, algorithm='SAMME.R', learning_rate=1.2),
    ensemble.AdaBoostClassifier(n_estimators=200),
    ensemble.AdaBoostClassifier(n_estimators=200, algorithm='SAMME.R'),
    ensemble.AdaBoostClassifier(n_estimators=200, algorithm='SAMME.R', learning_rate=1.2),

    ensemble.BaggingClassifier(n_estimators=10),
    ensemble.BaggingClassifier(n_estimators=10, bootstrap=False),
    ensemble.BaggingClassifier(n_estimators=20),
    ensemble.BaggingClassifier(n_estimators=20, bootstrap=False),
    ensemble.BaggingClassifier(n_estimators=50),
    ensemble.BaggingClassifier(n_estimators=50, bootstrap=False),
    ensemble.BaggingClassifier(n_estimators=100),
    ensemble.BaggingClassifier(n_estimators=100, bootstrap=False),

    naive_bayes.GaussianNB(),
    naive_bayes.GaussianNB(priors=None),

    neural_network.MLPClassifier()
]



def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def run_model( x_train, y_train, x_test, y_test ):
    plt.figure( figsize=(8,8))
    Accuracy=[]
    Model=[]
    F1=[]
    Precision=[]
    Recall=[]
    lw = 2
    patches = []
    for classifier in Classifiers:
        i = 1
        try:
            fit = classifier.fit( x_train, y_train)
            y_pred = fit.predict( x_test )
        except:
            fit = classifier.fit( x_train.toarray(), y_train)
            y_pred = fit.predict( x_test.toarray() )

        Accuracy.append( metrics.accuracy_score( y_test, y_pred ) )
        F1.append( metrics.f1_score( y_test, y_pred ) )
        Precision.append( metrics.precision_score( y_test, y_pred ) )
        Recall.append( metrics.recall_score( y_test, y_pred ) )
        # AUC.append( metrics.auc( y_test, y_pred ) )
        Model.append( classifier.__class__.__name__ )
        print( metrics.confusion_matrix( y_test, y_pred ) )
        fpr, tpr, _ = roc_curve( y_test, y_pred )
        #plt.subplot( 2, 4, i )
        cl = np.random.rand(3,1)
        plt.plot( fpr, tpr, label=classifier.__class__.__name__, c=cl)
        i += 1
        patches.append( mpatches.Patch( color=cl, label=str( classifier.__class__.__name__ ) + " : " +
                                                               str( metrics.accuracy_score( y_test, y_pred ) )))

    m = pd.DataFrame( list( zip( Model, Accuracy, F1, Precision, Recall ) ), columns=['Model','Accuracy', 'F1', 'Precision','Recall'] )
    print_full( m )
    plt.legend( loc='best' )
    plt.legend(  handles=patches, loc='best')
    plt.plot( [ 0,1],[0,1], color='navy', lw=lw, linestyle='--')
    plt.show()
    #  print( metrics.confusion_matrix( y_test, y_pred ) )









# PCA #
from sklearn import decomposition

set_ = [3,5,10,20,50,100,150,200,250,300,500]

for i in set_:
    pca = decomposition.PCA(n_components = i)
    fit_train_transform_result = pca.fit_transform(x_train_tfidf.toarray())
    fit_test_transform_result = pca.transform(x_test_tfidf.toarray())
    fit_train_transform_result.shape
    fit_test_transform_result.shape
    run_model(fit_train_transform_result, y_train, fit_test_transform_result, y_test)

    pca = decomposition.PCA(n_components = i)
    fit_train_transform_result = pca.fit_transform(x_train_extended)
    fit_test_transform_result = pca.transform(x_test_extended)
    fit_train_transform_result.shape
    fit_test_transform_result.shape
    run_model(fit_train_transform_result, y_train, fit_test_transform_result, y_test)



# trunc SVD #
from sklearn import decomposition, metrics, model_selection
import numpy as np

ks = [10, 20, 30, 50, 75, 100, 150, 250, 300]
ttrain = [x_train_tfidf, x_train_extended]
ttest = [x_test_tfidf, x_test_extended]

for tt in range(2):
    for kk in ks:
        tSVD = decomposition.TruncatedSVD(n_components = kk, random_state = 2017)
        xt_train = tSVD.fit_transform(ttrain[tt])
        xt_test = tSVD.transform(ttest[tt])

        run_model(xt_train, y_train, xt_test, y_test)
