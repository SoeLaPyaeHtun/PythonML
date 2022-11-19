import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(docs):
    cleansed = []
    punc = str.maketrans('', '', string.punctuation)
    for doc in docs:
        doc_no_punc = doc.translate(punc) 
        words = doc_no_punc.lower().split()
        words = [lemmatizer.lemmatize(word, 'v') for word in words if word not in stop_words]
        cleansed.append(' '.join(words)) 
    return cleansed

lemmatizer = WordNetLemmatizer() 
stop_words = stopwords.words('english')

docs = [
'John has some cats.', 
'Cats, being cats, eat fish.', 
'I ate a big fish'
]
query = ['cats and fish']
docs_clean = preprocess(docs) 
query_clean = preprocess(query)
# compute normalized TF-IDF
tfidf = TfidfVectorizer() 
tfidf.fit(docs_clean)
fv_corpus = tfidf.transform(docs_clean).toarray() 
fv_query = tfidf.transform(query_clean).toarray()

print(fv_corpus)
print(fv_query)

print(fv_corpus.shape)
print(fv_query.shape)

fv = pd.DataFrame(data=fv_query, index=['query string'],
columns=tfidf.get_feature_names()) 
print(fv, '\n')
#compute cosine similarity
similarity = cosine_similarity(fv_query, fv_corpus)
cs = pd.DataFrame(data=similarity, index=['cosine similarity'],
columns=['doc1', 'doc2', 'doc3']) 
print(cs)