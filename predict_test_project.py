import jieba.analyse
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Small self-made script for example test on real data
from small_test_example import test_list

# Real data example test output
text_for_test = '''
Review_ID: {}\n
Predict_Star: {}\n
Real_Star: {}
'''

# Choose 20 keywords with the highest TF/IDF weights
def clean(text):
    clean_text = ' '.join(jieba.analyse.extract_tags(text, topK=20, 
                                                     withWeight=False, 
                                                     allowPOS=()))
    return clean_text

shop_review_data = pd.read_csv('jd_reviews.csv')

# Drop null values from DataFrame
shop_review_data = shop_review_data.dropna()

new_clean_review = []

for text in shop_review_data['review']:    
    clean_text = clean(text)
    new_clean_review.append(clean_text)        
   
new_data = {'review': new_clean_review, 'stars': shop_review_data.stars}

# Create new DataFrame with clean data
shop_new_data = pd.DataFrame(data=new_data)

X_train, X_test, y_train, y_test = train_test_split(
        shop_new_data['review'], shop_new_data['stars'], random_state=0)

vect = CountVectorizer()
train_data = vect.fit_transform(X_train)
test_data = vect.transform(X_test)

nb = MultinomialNB(alpha=0.1)
nb.fit(train_data, y_train)
preds = nb.predict(test_data)

# Model accuracy score output
'''
print(metrics.accuracy_score(y_test, preds))
'''

# BASICALLY EXAMPLE FOR REAL DATA
'''
for review_id, item in enumerate(test_list, 1):
    clean_item = clean(item[0])
    pred = vect.transform([clean_item])
    print(text_for_test.format(review_id, nb.predict(pred), item[1]))
    print('*' * 20)   
'''      

# OUTPUT

'''  
In [1]: runfile('predict_test_project.py')

Review_ID: 1

Predict_Star: [ 1.]

Real_Star: 1

********************

Review_ID: 2

Predict_Star: [ 1.]

Real_Star: 1

********************

Review_ID: 3

Predict_Star: [ 5.]

Real_Star: 3

********************

Review_ID: 4

Predict_Star: [ 4.]

Real_Star: 3

********************

Review_ID: 5

Predict_Star: [ 4.]

Real_Star: 5

********************

Review_ID: 6

Predict_Star: [ 1.]

Real_Star: 5

********************

In [2]:
    
'''    
    



