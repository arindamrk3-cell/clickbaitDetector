import pandas as pd
import pickle
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from explain import get_explanation
nltk.download('stopwords')
print("stopwords downloaded!")
from nltk.corpus import stopwords

df=pd.read_csv("../dataset/final_dataset.csv")
df=df.dropna()
x=df["text"]
y=df["label"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)
vectorizer=TfidfVectorizer(
    stop_words='english',
    max_features=15000,
    ngram_range=(1,2),
    #min_df=2
)
x_train_vect=vectorizer.fit_transform(x_train)
x_test_vect=vectorizer.transform(x_test)

model=LogisticRegression(
    max_iter=3000,
    class_weight='balanced',
    C=2
)
model.fit(x_train_vect,y_train)
y_pred=model.predict(x_test_vect)

print("accuracy:",accuracy_score(y_test,y_pred))
print("classification report:",classification_report(y_test,y_pred))

pickle.dump(model,open("model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

print("model saved Successfully!")