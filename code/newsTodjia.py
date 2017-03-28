from pprint import pprint


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

"""
read data
prepare corpus

apply classification models

perform sentiment

reapply models


"""

doc_df = pd.read_csv( "/home/brij/smu/MachineLearning/project/NewsToMarket/data/Combined_News_DJIA.csv",
                      sep=",", encoding="ascii" )

pprint( doc_df.iloc[:5] )
"""
vectorizer = CountVectorizer( input='filename', encoding='utf-8', analyzer='word',
                              stop_words='english', lowercase=True, 
                              max_df=0.7, min_df=0.1 )

dictionary = vectorizer.fit( [  ] )

pprint( dictionary )
"""

def NewsToMarket:
    def init():
        self.file = "./data/Combined_News_DJIA.csv"
        self.df = pd.DataFrame( data=self.file, sep=","  )

        self.df2['Label'] = self.df['Label']

    def strip_binary_tags( value ):
            value = re.sub( "^b[\"\']|[\"\']","", str( value ) )

                
                    return value.lower()

    def preprocessing( self, df ):
        columns = self.df.columns
        self.df.apply( lambda row : row.apply( lambda col: self.strip_binary_tags( col ) ) )
