
import requests
from bs4 import BeautifulSoup
from string import punctuation
import pandas as pd 


url = "https://libraryof1000books.wordpress.com/the-list-of-1000-books/"
html_code = requests.get(url)

html_code.status_code

soup = BeautifulSoup(html_code.content,"html.parser")


book_name = []
author_name = []

for i in soup.find_all(class_="entry-content"):
    for index, name in enumerate(i.find_all("li")):
        if index < 1000:
            try:
                text = name.text.replace("–", "-")
                
                book_name.append(text.split("-")[0].strip())
                author_name.append(text.split("-")[1].split("(")[0].strip())
                
            except:
                print(index)


dataset = pd.DataFrame({"Book name":book_name,"Author":author_name})


dataset.to_csv(r"E:\deep learning\book prediction\web scraping books data\dataset of books.csv",index=False)





