
import pickle
import re

import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import streamlit as st

# Load pickled files & data
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('tfidf_transformer.pkl', 'rb') as f:
    tfidf_transformer = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Cleaning data:
stop_words = set(stopwords.words('english'))
new_stop_words = ["fig", "figure", "image", "sample", "using",
                  "show", "result", "large",
                  "also", "one", "two", "three",
                  "four", "five", "seven", "eight", "nine"]
stop_words = list(stop_words.union(new_stop_words))

def preprocess_text(txt):
    # Lower case
    txt = txt.lower()
    # Remove HTML tags
    txt = re.sub(r"<.*?>", " ", txt)
    # Remove special characters and digits
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    # tokenization
    txt = nltk.word_tokenize(txt)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    txt = [word for word in txt if word not in stop_words]
    # Remove words less than three letters
    txt = [word for word in txt if len(word) >= 3]
    # Lemmatize
    lmtr = WordNetLemmatizer()
    txt = [lmtr.lemmatize(word) for word in txt]

    return " ".join(txt)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

# def add_bg_from_url():
#                 st.markdown(
#                     f"""
#                      <style>
#                      .stApp {{
#                          background-image: url("https://www.google.com/url?sa=i&url=https%3A%2F%2Fstock.adobe.com%2Fimages%2Flearn-french-guide-text-background-word-cloud-concept%2F165306740&psig=AOvVaw3dS12Ke9Tp47w8OBoLvlzt&ust=1710614465108000&source=images&cd=vfe&opi=89978449&ved=2ahUKEwjo8cvN9faEAxWCa2wGHUi8AdkQjRx6BAgAEBc");
#                          background-attachment: fixed;
#                          background-size: cover
#                      }}
#                      </style>
#                      """,
#                     unsafe_allow_html=True
#                 )

def main():
    st.title('Keyword Extractor')

    menu = ['Home', 'Extract Keywords', 'Search Keywords']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Home')
        st.write('Welcome to Keyword Extractor. Use the sidebar to navigate.')

    elif choice == 'Extract Keywords':
        st.subheader('Extract Keywords')
        document = st.file_uploader("Upload document", type=['txt'])
        if document is not None:
            text = document.read().decode('utf-8', errors='ignore')
            preprocessed_text = preprocess_text(text)
            tf_idf_vector = tfidf_transformer.transform(cv.transform([preprocessed_text]))
            sorted_items = sort_coo(tf_idf_vector.tocoo())
            keywords = extract_topn_from_vector(feature_names, sorted_items, 20)
            st.write("Top 20 keywords:")
            st.write(keywords)

    elif choice == 'Search Keywords':
        st.subheader('Search Keywords')
        search_query = st.text_input("Enter keyword to search:")
        if search_query:
            keywords = []
            for keyword in feature_names:
                if search_query.lower() in keyword.lower():
                    keywords.append(keyword)
                    if len(keywords) == 20:  # Limit to 20 keywords
                        break
            st.write("Matching keywords:")
            st.write(keywords)





if __name__ == '__main__':
    main()
    
