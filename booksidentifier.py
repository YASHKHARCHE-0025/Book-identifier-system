
import pandas as pd  
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches


dataset = pd.read_csv(r"E:\deep learning\book prediction\web scraping books data\dataset of books\dataset of books.csv")
dataset['combined'] = dataset['Book name'] + " " + dataset['Author'] 


def remove_punctuations(book_name):
    updated_name = " "
    for i in book_name:
        if i not in punctuation:
            updated_name+=i
    return updated_name.lower()

dataset['combined'] = dataset['combined'].apply(remove_punctuations)


vector = TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=5000)
vector.fit(dataset['combined'])
data_vector = vector.transform(dataset['combined'])

model = NearestNeighbors(n_neighbors=5,metric="cosine")
model.fit(data_vector)



book_name_user = input("Enter Book Name : ")
book_name_user = remove_punctuations(book_name_user)

match = get_close_matches(book_name_user, dataset['Book name'], n=1,cutoff=0.6)

if match:
    book_name_user = match[0]


book_name_vector = vector.transform([book_name_user])
distance , index = model.kneighbors(book_name_vector,n_neighbors=5)

print(dataset.iloc[index[0][0]])






