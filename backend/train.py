import pandas as pd
import pickle
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix as cm
from sklearn.svm import LinearSVC
nltk.download('stopwords')
print("stopwords downloaded!")
from nltk.corpus import stopwords

df_old=pd.read_csv("../dataset/final_dataset.csv")
df_bolly=pd.read_csv("../dataset/BollyBAIT_dataset.csv")
df_old = df_old.rename(columns={"text": "combined"})
for col in [
    "Misleading_Video",
    "False_Promises",
    "Exaggerated_Video",
    "Spam_Content",
    "Exploits_Curiosity_Gap"
]:
    df_bolly[col] = df_bolly[col].apply(
        lambda x: col if str(x).strip().lower() == "yes" else ""
    )
df_bolly["combined"] = (
    df_bolly["Title"].fillna("") + " " +
    df_bolly["Channel_Title"].fillna("") + " " +
    df_bolly["Misleading_Video"].fillna("") + " " +
    df_bolly["False_Promises"].fillna("") + " " +
    df_bolly["Exaggerated_Video"].fillna("") + " " +
    df_bolly["Spam_Content"].fillna("") + " " +
    df_bolly["Exploits_Curiosity_Gap"].fillna("")
)
df_bolly["label"] = df_bolly["Label"].apply(
    lambda x: 1 if str(x).lower() == "clickbait" else 0
)
df_bolly = df_bolly[["combined", "label"]]
df_final = pd.concat([df_old, df_bolly], ignore_index=True)
df_final = df_final.drop_duplicates()
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9$ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
df_final["combined"] = df_final["combined"].apply(clean_text)

from sklearn.utils import resample
df_majority = df_final[df_final.label == 0]
df_minority = df_final[df_final.label == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)
df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42)

x=df_balanced["combined"]
y=df_balanced["label"]
print(df_balanced["label"].value_counts())
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)
vectorizer=TfidfVectorizer(
    stop_words='english',
    max_features=15000,
    ngram_range=(1,2),
    min_df=2
)
x_train_vect=vectorizer.fit_transform(x_train)
x_test_vect=vectorizer.transform(x_test)
model=LinearSVC(
    class_weight='balanced'
)
model.fit(x_train_vect,y_train)
y_pred=model.predict(x_test_vect)

print("accuracy:",accuracy_score(y_test,y_pred))
print("confusion matrix:\n",cm(y_test,y_pred))
print("classification report:",classification_report(y_test,y_pred))

pickle.dump(model,open("model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

print("model saved Successfully!")