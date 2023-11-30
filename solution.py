# %% [markdown]
# ## Importing required libraries

# %%
import pandas as pd
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.tokenize import sent_tokenize
from nltk import tokenize

# %%
df = pd.read_excel("Input.xlsx")
df

# %%
df.head()

# %% [markdown]
# ## Iterating through links in URL columns

# %%
df['URL']

# %%
url = 'https://insights.blackcoffer.com/rise-of-telemedicine-and-its-impact-on-livelihood-by-2040-3-2/'
r = requests.get(url)
print(r.content)

# %%
soup = BeautifulSoup(r.content, 'html5lib')
print(soup.prettify())

# %%
article_content = []

classcontent = soup.find("div",attrs={'class':"td-post-content tagdiv-type"})

contents = classcontent.find_all('p')
ans = ''
for para in contents:
    para_text = para.text
    ans+= para_text

ans
article_content.append(ans)

# %%
article_content

# %%

soup_content = []
def create_soup_content(df):
    for link in df['URL']:
        r = requests.get(link)
        soup_content.append(r.content)
        


# %%
def create_article_content(soup_content):
    for content in soup_content:
        soup = BeautifulSoup(content, 'html5lib')
        classcontent = soup.find("div",attrs={'class':"td-post-content tagdiv-type"})
        if classcontent:
            contents = classcontent.find_all('p')
            ans = ''
            for para in contents:
                para_text = para.text
                ans+= para_text
            article_content.append(ans)
        else:
            new_classcontent = soup.find_all('p')
            ans = ''
            for para in new_classcontent:
                para_text = para.text
                ans+= para_text
            article_content.append(ans)


# %%
article_title_content = []

# %%
def create_article_title_content(soup_content):
    for content in soup_content:
        soup = BeautifulSoup(content, 'html5lib')
        new_classcontent = soup.find_all('h1')
        if new_classcontent:
            ans = ''
            for para in new_classcontent:
                para_text = para.text
                ans+= para_text
            article_title_content.append(ans)
        else:
            article_title_content.append("No title available")

# %%
create_soup_content(df)

# %%
article_content = []

# %%
create_article_content(soup_content)

# %%
create_article_title_content(soup_content)

# %%
article_title_content.__len__()

# %%
df.shape

# %% [markdown]
# ## Generated new dataset to save the content 

# %%
df = df.assign(article_content = article_content)

# %%
df = df.assign(article_title = article_title_content)

# %%
df

# %%
df.to_csv('data.csv') ##saving the useful data to save time to process


# %%
df= pd.read_csv('data.csv')

# %% [markdown]
# ## getting stopwords from files

# %%
with open('StopWords/StopWords_Auditor.txt','r') as f:
    data = f.read()

# %%
import chardet
with open('StopWords/StopWords_Currencies.txt', 'rb') as file:
    result = chardet.detect(file.read())

encoding = result['encoding']
print(result)

# %%
with open('StopWords/StopWords_Currencies.txt','r',encoding='ISO-8859-1') as f:
    data1 = f.read()

# %%
with open('StopWords/StopWords_DatesandNumbers.txt','r') as f:
    data2 = f.read()

# %%
with open('StopWords/StopWords_Generic.txt','r') as f:
    data3 = f.read()

with open('StopWords/StopWords_GenericLong.txt','r') as f:
    data4 = f.read()

with open('StopWords/StopWords_Geographic.txt','r') as f:
    data5 = f.read()

with open('StopWords/StopWords_Names.txt','r') as f:
    data6 = f.read()

# %%
print(type(data))

# %%
stop_words = set()

def creating_custom_stopWords(data):
    # stop_words.update(word.lower().split())
    stop_words.update(data.lower().split())
    # print(data.lower().split())


# %%
data.lower().split()

# %%
creating_custom_stopWords(data=data)
creating_custom_stopWords(data=data1)
creating_custom_stopWords(data=data2)
creating_custom_stopWords(data=data3)
creating_custom_stopWords(data=data4)
creating_custom_stopWords(data=data5)
creating_custom_stopWords(data=data6)

# %%
stop_words.__len__()# All stop words are in the set now

# %% [markdown]
# ## Removing the stopwords from articles

# %%
def remove_stopwords(text, stop_words):
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

df['article_content'] = df['article_content'].apply(lambda x: remove_stopwords(x, stop_words))
df['article_title'] = df['article_title'].apply(lambda x: remove_stopwords(x, stop_words))

# %%
df['article_title'][0]

# %% [markdown]
# ## creating a dictionary for positive and negative words

# %%
with open('MasterDictionary/positive-words.txt','r') as f:
    pos_data = f.read()

# %%
with open('MasterDictionary/negative-words.txt','rb') as f:
    result = chardet.detect(f.read())
result

# %%
with open('MasterDictionary/negative-words.txt','r', encoding = 'ISO-8859-1') as f:
    neg_data = f.read()

# %%
print(pos_data)

# %%
postive_words = []
negative_words = []



# %%
postive_words = pos_data.lower().split()

negative_words = neg_data.lower().split()

# %%
print(negative_words.__len__())

# %%
postive_words

# %%


def calculate_positive_score(text):
    lowercase_text = text.lower()
    positive_count = sum(word in lowercase_text for word in postive_words)
    
    return positive_count

# %%
def calculate_negative_score(text):
    lowercase_text = text.lower()
    negative_score = sum(word in lowercase_text for word in negative_words)
    
    return negative_score * -1

# %%
df['positive_scores'] = df['article_content'].apply(lambda x: calculate_positive_score(x))

# %%
df['negative_scores'] = df['article_content'].apply(lambda x: calculate_negative_score(x))

# %%
df.columns

# %%
df = df.drop('Unnamed: 0',axis=1)

# %%
def calculate_polarity_score(data):
    diff = data['positive_scores'] - (data['negative_scores']* -1)
    addition = (data['positive_scores'] + (data['negative_scores']* -1)) + 0.000001
    return diff / addition

df['polarity_scores'] = df.apply(calculate_polarity_score,axis=1)
    

# %%
def calculate_subjectivity_score(data):
    addition = data['positive_scores'] + (data['negative_scores'] *-1)
    lengthofContent = data['article_content'].__len__() + 0.000001
    return addition / lengthofContent

df['subjectivity_score'] = df.apply(calculate_subjectivity_score,axis=1)    

# %%
df['article_content'][0].__len__()

# %%
def avg_sentence_length(data):
    sentences = sent_tokenize(data)
    length_sentences = sentences.__len__()
    num_words = data.__len__()
    return num_words / length_sentences

df['avg_sentence_length'] = df['article_content'].apply(avg_sentence_length)

# %%
punctuation_marks = ['.',',','?','!','-','?','<','>','/','\\','"']
def remove_punctuation_marks(data,punctuation_marks):
    data = data.split()
    filtered_data = [d for d in data if d not in punctuation_marks]
    return ' '.join(filtered_data)

df['article_content'] = df['article_content'].apply(lambda x: remove_punctuation_marks(x,punctuation_marks))
df['article_title'] = df['article_title'].apply(lambda x: remove_punctuation_marks(x,punctuation_marks))

# %%
def complex_word_count(data):
    tokens = data.split()
    complexWord = 0
    
    for word in tokens:
        vowels=0
        if word.endswith(('es','ed')):
            pass
        else:
            for w in word:
                if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                    vowels += 1
            if(vowels > 2):
                complexWord += 1
    return complexWord

# %%
df['complex_word_count'] = df['article_content'].apply(complex_word_count)

# %%
def calculate_percentage_complex_word(data):
    count_words = data['article_content'].__len__()
    return data['complex_word_count'] / count_words
df['percentage_complex_word'] = df.apply(calculate_percentage_complex_word,axis=1)

# %%
def calculate_fog_index(data):
    avg_sentence_length = data['avg_sentence_length']
    percentage_complex_word = data['percentage_complex_word']
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_word)
    return fog_index
df['fog_index'] = df.apply(calculate_fog_index,axis=1)

# %%
df['word_count'] = df['article_content'].apply(lambda x: len(x))
    

# %%
syllable = ['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u', 'ed', 'es', 'ing', 'er']
def syllable_per_word(data, syllable):
    data = data.lower()
    syllable_count = 0
    for s in syllable:
        if s in data:
            syllable_count += data.count(s)
    return syllable_count
    

df['syllable_per_word'] = df['article_content'].apply(lambda x: syllable_per_word(x,syllable))
    

# %%
personal_pronoun_list = ["I", "we", "We", "my", "My", "ours", "our", "Ours", "Our", "us", 'Us']
def count_personal_pronoun(data,personal_pronoun_list):
    data = data.lower()
    pronoun_count = 0
    for p in personal_pronoun_list:
        if p in data:
            pronoun_count += data.count(p)
    return pronoun_count

df['personal_pronoun'] = df['article_content'].apply(lambda x: count_personal_pronoun(x,personal_pronoun_list))

# %%
def average_word_length(text):
    words = text.split()
    total_length = sum(len(word) for word in words)
    word_count = len(words)
    return total_length // word_count if word_count > 0 else 0


df['average_word_length'] = df['article_content'].apply(average_word_length)

# %%
df

# %%
df.to_excel('Output Data Structure.xlsx')

# %%



