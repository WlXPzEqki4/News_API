# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # ---------------------------------------------------------------
# # 1) HELPER FUNCTIONS
# # ---------------------------------------------------------------
# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """
#     Fetch articles from NewsAPI /v2/everything endpoint.
#     Returns list of article dicts or raises an exception on error.
#     """
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date: params['from'] = from_date
#     if to_date: params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']

# def clean_and_combine_text(title, description, content):
#     """
#     Combine and clean text: lowercase, remove punctuation, etc.
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}"
#     combined = combined.lower()
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined).strip()
#     return combined

# def compute_word_frequency(text_list):
#     """
#     Given a list of cleaned strings, create a global frequency map of words.
#     Returns a dict { 'word': count, ... }
#     """
#     freq = {}
#     for txt in text_list:
#         words = txt.split(' ')
#         for w in words:
#             if w:
#                 freq[w] = freq.get(w, 0) + 1
#     return freq

# def analyze_sentiment(text):
#     """
#     Use TextBlob's sentiment. Returns a dict with polarity (-1 to +1) and subjectivity.
#     """
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)

# # ---------------------------------------------------------------
# # 2) STREAMLIT APP
# # ---------------------------------------------------------------
# def main():
#     st.title("News Analysis with Streamlit")

#     # Sidebar inputs
#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input("Enter your NewsAPI key", value="", type="password")
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])
#     from_date = st.sidebar.date_input("From Date", None)
#     to_date = st.sidebar.date_input("To Date", None)

#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return

#         # Convert date objects to string (YYYY-MM-DD) if set
#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                     return

#                 # Build a DataFrame
#                 df = pd.DataFrame(articles)
#                 # Some fields: 'source' (dict), 'title', 'description', 'url', 'publishedAt', 'content', etc.
#                 # Extract 'source.name'
#                 df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))
#                 # Clean text
#                 df['cleanedText'] = df.apply(
#                     lambda row: clean_and_combine_text(row.title, row.description, row.content), 
#                     axis=1
#                 )
#                 # Sentiment
#                 df['sentiment'] = df['cleanedText'].apply(lambda txt: analyze_sentiment(txt))
#                 df['polarity'] = df['sentiment'].apply(lambda s: s.polarity)
#                 df['subjectivity'] = df['sentiment'].apply(lambda s: s.subjectivity)

#                 st.success(f"Fetched {len(df)} articles.")
                
#                 # Display the DataFrame
#                 st.subheader("Articles Data")
#                 st.dataframe(df[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#                 # Word frequency
#                 wordFreq = compute_word_frequency(df['cleanedText'].tolist())

#                 # Let's show top 20 words as a bar chart
#                 st.subheader("Top Words (Frequency)")
#                 freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#                 topN = 20
#                 top_words = freq_items[:topN]
#                 words, counts = zip(*top_words)
#                 freq_df = pd.DataFrame({'word': words, 'count': counts})
#                 st.bar_chart(freq_df.set_index('word'))

#                 # Optional: Show a word cloud
#                 st.subheader("Word Cloud")
#                 # Build one big text chunk for the word cloud
#                 all_text = ' '.join(df['cleanedText'].tolist())
#                 wc = WordCloud(width=600, height=400, background_color='white').generate(all_text)

#                 fig, ax = plt.subplots(figsize=(6,4))
#                 ax.imshow(wc, interpolation='bilinear')
#                 ax.axis('off')
#                 st.pyplot(fig)

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

# if __name__ == "__main__":
#     main()






















# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # ---------------------------------------------------------------
# # 1) HELPER FUNCTIONS
# # ---------------------------------------------------------------
# def validate_api_key(api_key):
#     """
#     Attempt a quick call (e.g. top-headlines) to confirm the key is valid.
#     Returns True if valid, raises an exception otherwise.
#     """
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}

#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")

# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """
#     Fetch articles from NewsAPI /v2/everything endpoint.
#     Returns list of article dicts or raises an exception on error.
#     """
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']

# def clean_and_combine_text(title, description, content):
#     """
#     Combine and clean text: lowercase, remove punctuation, etc.
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}"
#     combined = combined.lower()
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined).strip()
#     return combined

# def compute_word_frequency(text_list):
#     """
#     Given a list of cleaned strings, create a global frequency map of words.
#     Returns a dict { 'word': count, ... }
#     """
#     freq = {}
#     for txt in text_list:
#         words = txt.split(' ')
#         for w in words:
#             if w:
#                 freq[w] = freq.get(w, 0) + 1
#     return freq

# def analyze_sentiment(text):
#     """
#     Use TextBlob's sentiment. Returns a namedtuple with polarity (-1 to +1) and subjectivity.
#     """
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)

# def create_wordcloud(all_text):
#     """
#     Given one big string of text, create a WordCloud image (using the wordcloud library).
#     Returns a matplotlib figure for display in Streamlit.
#     """
#     wc = WordCloud(width=600, height=400, background_color='white').generate(all_text)
#     fig, ax = plt.subplots(figsize=(6,4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig

# # ---------------------------------------------------------------
# # 2) STREAMLIT APP
# # ---------------------------------------------------------------
# def main():
#     st.title("News Analysis with Streamlit")

#     # Use Session State to store validation status
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False

#     # Sidebar inputs
#     st.sidebar.header("NewsAPI Settings")

#     # 1) Pre-filled key (dev only)
#     # Note the default value is your dev key. Type is password so it doesn't show plainly.
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # dev only, not for production
#         type="password"
#     )

#     # 2) Validate Key button
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     valid = validate_api_key(api_key)
#                     if valid:
#                         st.sidebar.success("API key is valid!")
#                         st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # 3) Search parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])
#     from_date = st.sidebar.date_input("From Date", None)
#     to_date = st.sidebar.date_input("To Date", None)

#     # "Fetch News" button
#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return

#         # Check if key validated
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate before fetching news.")
#             return

#         # Convert date objects to string (YYYY-MM-DD) if set
#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                     return

#                 # Build a DataFrame
#                 df = pd.DataFrame(articles)
#                 # Some fields: 'source' (dict), 'title', 'description', 'url', 'publishedAt', 'content', etc.
#                 # Extract 'source.name'
#                 df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))
#                 # Clean text
#                 df['cleanedText'] = df.apply(
#                     lambda row: clean_and_combine_text(row.title, row.description, row.content), 
#                     axis=1
#                 )
#                 # Sentiment (TextBlob)
#                 df['sentiment'] = df['cleanedText'].apply(lambda txt: analyze_sentiment(txt))
#                 df['polarity'] = df['sentiment'].apply(lambda s: s.polarity)
#                 df['subjectivity'] = df['sentiment'].apply(lambda s: s.subjectivity)

#                 st.success(f"Fetched {len(df)} articles.")
                
#                 # Display the DataFrame
#                 st.subheader("Articles Data")
#                 st.dataframe(df[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#                 # Word frequency
#                 wordFreq = compute_word_frequency(df['cleanedText'].tolist())

#                 # Show top words as a bar chart
#                 st.subheader("Top Words (Frequency)")
#                 freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#                 topN = 20
#                 top_words = freq_items[:topN]
#                 words, counts = zip(*top_words)
#                 freq_df = pd.DataFrame({'word': words, 'count': counts})
#                 st.bar_chart(freq_df.set_index('word'))

#                 # Word Cloud
#                 st.subheader("Word Cloud")
#                 all_text = ' '.join(df['cleanedText'].tolist())
#                 fig = create_wordcloud(all_text)
#                 st.pyplot(fig)

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")


# if __name__ == "__main__":
#     main()




































# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")

# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']

# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined

# def apply_custom_stopwords(text, custom_stopwords):
#     """
#     Remove any words that appear in custom_stopwords set.
#     """
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in custom_stopwords]
#     return ' '.join(filtered)

# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment

# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq

# def create_wordcloud(all_text):
#     """Generate a word cloud image from the given text."""
#     wc = WordCloud(width=600, height=400, background_color='white',stopwords=set()).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6,4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Custom Stopwords")

#     # Store multiple states:
#     #  - Whether the API key is validated
#     #  - The actual article DataFrame
#     #  - The custom stopwords
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # Sidebar for API setup
#     st.sidebar.markdown("News Analysis")

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # Pre-filled for dev
#         type="password"
#     )

#     # Validate key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Sidebar: Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])


#     # Approach A: Checkbox to enable date filtering
#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)

#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None


#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return

#         if not st.session_state.api_key_validated:
#             st.error("API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # More thorough cleaning
#                     df['cleanedText'] = df.apply(lambda row:
#                         clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     # -------------------------
#     # MAIN AREA: if we have data, show custom stopwords UI & analysis
#     # -------------------------
#     df = st.session_state.articles_df
#     if not df.empty:
#         st.subheader("Custom Stopwords")
#         st.write("""
#             You can add words to remove from the corpus. 
#             The frequency bar chart, word cloud, and sentiment will update accordingly.
#         """)
        
#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", "")
#         if st.button("Add Word to Remove"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())
#                 st.experimental_rerun()

#         # Display and manage existing stopwords
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'"):
#                     st.session_state.custom_stopwords.remove(w)
#                     st.experimental_rerun()
#         else:
#             st.info("No custom stopwords yet.")

#         # -------------------------
#         # Apply stopwords, then analyze
#         # -------------------------
#         df['finalText'] = df['cleanedText'].apply(
#             lambda txt: apply_custom_stopwords(txt, st.session_state.custom_stopwords)
#         )

#         # Sentiment
#         df['sentiment'] = df['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df['polarity'] = df['sentiment'].apply(lambda s: s.polarity)
#         df['subjectivity'] = df['sentiment'].apply(lambda s: s.subjectivity)

#         # Show data
#         st.subheader("Articles Table")
#         st.dataframe(df[[
#             'title','publication','author','publishedAt','description','polarity','subjectivity'
#         ]])

#         # Word Frequency
#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing custom stopwords!")

#         # Now add the table for user sorting:
#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)  # user can click columns to sort

#         # Word Cloud
#         st.subheader("Word Cloud")
#         all_text = ' '.join(df['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text)
#             st.pyplot(fig)
#         else:
#             st.write("No text available to generate a word cloud after stopword removal.")

# if __name__ == "__main__":
#     main()



















# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")

# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']

# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined

# def apply_custom_stopwords(text, custom_stopwords):
#     """
#     Remove any words that appear in custom_stopwords set.
#     """
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in custom_stopwords]
#     return ' '.join(filtered)

# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)

# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq

# def create_wordcloud(all_text):
#     """
#     Generate a word cloud image from the given text.
#     stopwords=set() ensures it doesn't remove words like 'a' by default.
#     """
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=set()).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6,4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig

# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Custom Stopwords")

#     # -- STATE INITIALIZATION --
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     # Large heading with faint divider
#     st.sidebar.markdown("# News Analysis\n---")

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # Pre-filled for dev
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     # Optional Date Range
#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Single "Fetch News" button
#     if st.sidebar.button("Fetch News"):
#         # Basic checks
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         # Convert dates to string if set
#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough Cleaning
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     # -------------------------
#     # MAIN AREA - STOPWORDS AND ANALYSIS
#     # -------------------------
#     df = st.session_state.articles_df
#     if not df.empty:
#         st.subheader("Custom Stopwords")
#         st.write("""
#             You can add words to remove from the corpus. 
#             The frequency bar chart, word cloud, and sentiment will update accordingly.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", "")
#         if st.button("Add Word to Remove"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())
#                 st.experimental_rerun()

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'"):
#                     st.session_state.custom_stopwords.remove(w)
#                     st.experimental_rerun()
#         else:
#             st.info("No custom stopwords yet.")

#         # Apply stopwords -> finalText
#         df['finalText'] = df['cleanedText'].apply(
#             lambda txt: apply_custom_stopwords(txt, st.session_state.custom_stopwords)
#         )

#         # Sentiment
#         df['sentiment'] = df['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df['polarity'] = df['sentiment'].apply(lambda s: s.polarity)
#         df['subjectivity'] = df['sentiment'].apply(lambda s: s.subjectivity)

#         # Show articles table
#         st.subheader("Articles Table")
#         st.dataframe(df[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)  # sort desc by count
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             # Build a DataFrame for the bar chart
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             # Sort freq_df by 'count' descending so the bar chart shows highest -> lowest
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing custom stopwords!")

#         # Sortable table of the entire frequency set
#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)  # user can click columns to sort

#         # Word Cloud
#         st.subheader("Word Cloud")
#         all_text = ' '.join(df['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text)
#             st.pyplot(fig)
#         else:
#             st.write("No text available to generate a word cloud after stopword removal.")

# if __name__ == "__main__":
#     main()




















# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS  # We'll use STOPWORDS here
# import matplotlib.pyplot as plt

# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")

# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']

# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined

# def apply_custom_stopwords(text, custom_stopwords):
#     """
#     Remove any words that appear in custom_stopwords set.
#     """
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in custom_stopwords]
#     return ' '.join(filtered)

# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)

# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq

# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6,4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig

# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Custom Stopwords")

#     # State initialization
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # Dev only
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     # Optional Date Range
#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Single "Fetch News" button
#     if st.sidebar.button("Fetch News"):
#         # Basic checks
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough Cleaning
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     # Retrieve
#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # -------------
#     # TABS
#     # -------------
#     tab1, tab2 = st.tabs(["Custom Stopwords", "Built-In Stopwords"])

#     # -------------------------
#     # TAB 1: CUSTOM STOPWORDS
#     # -------------------------
#     with tab1:
#         st.subheader("Custom Stopwords")
#         st.write("""
#             You can add words to remove from the corpus. 
#             The frequency bar chart, word cloud, and sentiment will update accordingly.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", "")
#         if st.button("Add Word to Remove"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())
#                 st.experimental_rerun()

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'"):
#                     st.session_state.custom_stopwords.remove(w)
#                     st.experimental_rerun()
#         else:
#             st.info("No custom stopwords yet.")

#         # Apply custom stopwords -> finalText1
#         df1 = df.copy()
#         df1['finalText'] = df1['cleanedText'].apply(
#             lambda txt: apply_custom_stopwords(txt, st.session_state.custom_stopwords)
#         )

#         # Sentiment
#         df1['sentiment'] = df1['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df1['polarity'] = df1['sentiment'].apply(lambda s: s.polarity)
#         df1['subjectivity'] = df1['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader("Articles Table (Custom Stopwords)")
#         st.dataframe(df1[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader("Top Words (Frequency) - Custom Stopwords")
#         wordFreq = compute_word_frequency(df1['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)  # desc
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing custom stopwords!")

#         # Full frequency table
#         st.subheader("Word Frequency Table (Custom Stopwords)")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader("Word Cloud (Custom Stopwords)")
#         all_text = ' '.join(df1['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after custom stopwords removal.")

#     # -------------------------
#     # TAB 2: BUILT-IN STOPWORDS
#     # -------------------------
#     with tab2:
#         st.subheader("Built-In Stopwords (WordCloud's Default Set)")
#         st.write("""
#             This tab removes a standard list of English stopwords 
#             (like 'the', 'and', 'to', etc.) from WordCloud's built-in set.
#         """)

#         # We'll create a separate DataFrame so we don't clash with the custom approach
#         df2 = df.copy()
#         # Step: remove built-in stopwords from each cleanedText
#         builtin_stopwords = set(STOPWORDS)  # WordCloud's built-in set

#         def apply_built_in_stopwords(txt):
#             tokens = txt.split()
#             filtered = [w for w in tokens if w not in builtin_stopwords]
#             return ' '.join(filtered)

#         df2['finalText'] = df2['cleanedText'].apply(apply_built_in_stopwords)

#         # Sentiment on finalText
#         df2['sentiment'] = df2['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df2['polarity'] = df2['sentiment'].apply(lambda s: s.polarity)
#         df2['subjectivity'] = df2['sentiment'].apply(lambda s: s.subjectivity)

#         st.subheader("Articles Table (Built-In Stopwords)")
#         st.dataframe(df2[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word frequency
#         st.subheader("Top Words (Frequency) - Built-In Stopwords")
#         wordFreq2 = compute_word_frequency(df2['finalText'])
#         freq_items2 = sorted(wordFreq2.items(), key=lambda x: x[1], reverse=True)
#         topN2 = 50
#         top_words2 = freq_items2[:topN2]

#         if top_words2:
#             words2, counts2 = zip(*top_words2)
#             freq_df2 = pd.DataFrame({'word': words2, 'count': counts2})
#             freq_df2 = freq_df2.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df2.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing built-in stopwords!")

#         st.subheader("Word Frequency Table (Built-In Stopwords)")
#         freq_df2_all = pd.DataFrame(freq_items2, columns=["Word", "Count"])
#         st.dataframe(freq_df2_all)

#         # Word cloud
#         st.subheader("Word Cloud (Built-In Stopwords)")
#         all_text2 = ' '.join(df2['finalText'].tolist())
#         if all_text2.strip():
#             # Also pass builtin_stopwords if you want them removed again,
#             # but we already removed them from finalText, so it's not necessary
#             fig2 = create_wordcloud(all_text2, stopwords=set())
#             st.pyplot(fig2)
#         else:
#             st.write("No text available for word cloud after built-in stopwords removal.")


# if __name__ == "__main__":
#     main()






















# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt


# import spacy
# import nltk
# # For bigrams
# from nltk.util import ngrams

# # Initialize spaCy model once
# nlp = spacy.load("en_core_web_sm")






# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")

# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']

# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined

# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)

# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq

# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6,4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig

# def apply_stopwords_union(text, custom_stopwords):
#     """
#     Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords.
#     """
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# def lemmatise_text_spacy(txt):
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)

# def generate_bigrams(txt):
#     tokens = txt.split()
#     bigram_tuples = list(ngrams(tokens, 2))
#     # "python code" -> "python_code"
#     bigram_strings = ["_".join(pair) for pair in bigram_tuples]
#     return " ".join(bigram_strings)














# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Built-In + Custom Stopwords in Both Tabs")

#     # State initialization
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     # We track custom stopwords in session_state
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # Dev only
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     # Optional Date Range
#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Single "Fetch News" button
#     if st.sidebar.button("Fetch News"):
#         # Basic checks
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough Cleaning
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     # Retrieve
#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # -------------
#     # TABS
#     # -------------
#     # Both tabs remove built-in + custom stopwords. 
#     # But the user wants "the exact same approach" in each.
#     # We'll just replicate the add/remove stopwords UI in each tab, 
#     # but we STILL do the union with built-in behind the scenes.

#     tab1, tab2 = st.tabs(["Stopwords Tab 1", "Stopwords Tab 2"])

#     # Common function to handle the entire flow: 
#     #   1) Show user-driven stopword UI
#     #   2) Apply union of built-in + custom
#     #   3) Show freq bar chart, table, word cloud
#     def render_stopwords_tab(tab_label):
#         st.subheader(f"{tab_label}: Manage Custom Stopwords")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", key=f"new_word_{tab_label}")
#         if st.button("Add Word to Remove", key=f"add_btn_{tab_label}"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())
#                 st.experimental_rerun()

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_{tab_label}"):
#                     st.session_state.custom_stopwords.remove(w)
#                     st.experimental_rerun()
#         else:
#             st.info("No custom stopwords yet.")

#         # Combine built-in + custom 
#         # Then apply them to each article's cleanedText
#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader(f"Articles Table - {tab_label}")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader(f"Top Words (Frequency) - {tab_label}")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full frequency table
#         st.subheader(f"Word Frequency Table - {tab_label}")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader(f"Word Cloud - {tab_label}")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             # We already removed built-in + custom from finalText, 
#             # so no additional set needed in create_wordcloud.
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords.")


#         # -------------- ADVANCED LEMMAS & BIGRAMS --------------
#         st.subheader(f"Advanced: Lemmas & Bigrams - {tab_label}")

#         # 1) We'll copy the DataFrame again so we can apply further transformations
#         df_advanced = df_tab.copy()

#         # 2) Lemmatisation
#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)

#         # 3) Bigrams
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(generate_bigrams)

#         # 4) Lemma frequency
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])  # Or build a chart, etc.

#         # 5) Bigram frequency
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         # 6) Lemma Word Cloud (Optional)
#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # 7) Bigram Word Cloud (Optional)
#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_bigrams = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_bigrams)


#     # Render tab1 
#     with tab1:
#         render_stopwords_tab("Tab 1")

#     # Render tab2 
#     with tab2:
#         render_stopwords_tab("Tab 2")

# if __name__ == "__main__":
#     main()










# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# # For bigrams
# from nltk.util import ngrams

# # Initialize spaCy model once
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """
#     Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords.
#     """
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# def lemmatise_text_spacy(txt):
#     """
#     Lemmatise the text using spaCy,
#     e.g. 'running' -> 'run', 'mice' -> 'mouse'.
#     """
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_bigrams(txt):
#     """
#     Convert text into bigrams,
#     e.g. 'python code' -> 'python_code'.
#     """
#     tokens = txt.split()
#     bigram_tuples = list(ngrams(tokens, 2))
#     bigram_strings = ["_".join(pair) for pair in bigram_tuples]
#     return " ".join(bigram_strings)


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Built-In + Custom Stopwords in Both Tabs")

#     # State initialization
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     # We track custom stopwords in session_state
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # Dev only
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     # Optional Date Range
#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Single "Fetch News" button
#     if st.sidebar.button("Fetch News"):
#         # Basic checks
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough Cleaning
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     # Retrieve
#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # -------------
#     # TABS
#     # -------------
#     tab1, tab2 = st.tabs(["Stopwords Tab 1", "Stopwords Tab 2"])

#     # Common function to handle the entire flow:
#     #   1) Show user-driven stopword UI
#     #   2) Apply union of built-in + custom
#     #   3) Show freq bar chart, table, word cloud
#     #   4) Then advanced Lemmas & Bigrams
#     def render_stopwords_tab(tab_label):
#         st.subheader(f"{tab_label}: Manage Custom Stopwords")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", key=f"new_word_{tab_label}")
#         if st.button("Add Word to Remove", key=f"add_btn_{tab_label}"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())
#                 st.experimental_rerun()

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_{tab_label}"):
#                     st.session_state.custom_stopwords.remove(w)
#                     st.experimental_rerun()
#         else:
#             st.info("No custom stopwords yet.")

#         # Combine built-in + custom
#         # Then apply them to each article's cleanedText
#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader(f"Articles Table - {tab_label}")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader(f"Top Words (Frequency) - {tab_label}")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full frequency table
#         st.subheader(f"Word Frequency Table - {tab_label}")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader(f"Word Cloud - {tab_label}")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             # We already removed built-in + custom from finalText,
#             # so no additional set needed in create_wordcloud.
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # -----------------------------------------------------------------
#         # ADVANCED LEMMAS & BIGRAMS
#         # -----------------------------------------------------------------
#         st.subheader(f"Advanced: Lemmas & Bigrams - {tab_label}")

#         # Copy the DataFrame again so we can apply transformations
#         df_advanced = df_tab.copy()

#         # 1) Lemmatisation
#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)

#         # 2) Bigrams
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(generate_bigrams)

#         # Lemma frequency
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])  # Or build a chart, table, etc.

#         # Bigram frequency
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         # Lemma Word Cloud
#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # Bigram Word Cloud
#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_bigrams = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_bigrams)


#     # Render tab1
#     with tab1:
#         render_stopwords_tab("Tab 1")

#     # Render tab2
#     with tab2:
#         render_stopwords_tab("Tab 2")


# if __name__ == "__main__":
#     main()































# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# # For n-grams
# from nltk.util import ngrams

# # Initialize spaCy model once
# nlp = spacy.load("en_core_web_sm")

# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """
#     Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords.
#     """
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# def lemmatise_text_spacy(txt):
#     """
#     Lemmatise the text using spaCy.
#     E.g. 'running' -> 'run', 'mice' -> 'mouse', etc.
#     """
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """
#     Convert text into n-grams (bigrams if n=2, trigrams if n=3, etc.).
#     e.g. tokens ['machine','learning'] -> 'machine_learning'
#     """
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Built-In + Custom Stopwords in Both Tabs")

#     # State initialization
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # Dev usage only
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     # Optional Date Range
#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Single "Fetch News" button
#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough Cleaning
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     # Retrieve
#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # -------------
#     # TABS
#     # -------------
#     tab1, tab2 = st.tabs(["Stopwords Tab 1", "Stopwords Tab 2"])

#     # Common function to handle the entire flow
#     def render_stopwords_tab(tab_label):
#         st.subheader(f"{tab_label}: Manage Custom Stopwords")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", key=f"new_word_{tab_label}")
#         if st.button("Add Word to Remove", key=f"add_btn_{tab_label}"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())
#                 st.experimental_rerun()

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_{tab_label}"):
#                     st.session_state.custom_stopwords.remove(w)
#                     st.experimental_rerun()
#         else:
#             st.info("No custom stopwords yet.")

#         # Combine built-in + custom
#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader(f"Articles Table - {tab_label}")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader(f"Top Words (Frequency) - {tab_label}")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full frequency table
#         st.subheader(f"Word Frequency Table - {tab_label}")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader(f"Word Cloud - {tab_label}")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # -----------------------------------------------------------------
#         # ADVANCED LEMMAS, BIGRAMS, TRIGRAMS
#         # -----------------------------------------------------------------
#         st.subheader(f"Advanced: Lemmas, Bigrams, Trigrams - {tab_label}")

#         df_advanced = df_tab.copy()

#         # ---- LEMMAS ----
#         st.markdown("### Lemmas")
#         st.write("""\**Description**:  
#         Lemmas unify different forms of a word. For example, 'running', 'runs', and 'ran' all become 'run'. 
#         This means we count root forms rather than each variant.

#         **Example** (raw text → lemma text):
#         ```json
#         {
#         "raw": "Running daily helps runners",
#         "lemmas": "run daily help runner"
#         }
#         In a frequency table or word cloud, you'll see 'run' and 'runner' more consistently, rather than each form separately. """)

#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)

#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])

#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)



#         # ---- BIGRAMS ----
#         st.markdown("### Bigrams")
#         st.write("""\Description:
#         Bigrams are pairs of consecutive words, e.g. 'machine learning', 'data science', displayed as 'machine_learning', etc.

#         Example: {
#         "tokens": ["machine", "learning", "works"],
#         "bigrams": ["machine_learning", "learning_works"]
#         }
#         In the frequency table or word cloud, you'll see these pairs as single tokens like 'machine_learning'. """)


#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)

#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         # ---- TRIGRAMS ----
#         st.markdown("### Trigrams")
#         st.write("""\Description:
#         Trigrams are triplets of consecutive words, e.g. 'new york city' becomes 'new_york_city'.

#         Example:
#         {
#         "tokens": ["the", "quick", "brown", "fox"],
#         "trigrams": ["the_quick_brown", "quick_brown_fox"]
#         }
#         So you'll see tokens like 'the_quick_brown' in frequency tables/word clouds if it appears often enough. """)

#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)

#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])

#         st.write("#### Trigram Word Cloud")
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # Render tab1
#     with tab1:
#         render_stopwords_tab("Tab 1")

#     # Render tab2
#     with tab2:
#         render_stopwords_tab("Tab 2")

# # if name == "main": main()

# if __name__ == "__main__":
#     main()

























# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # Initialize spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """
#     Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords.
#     """
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """
#     Lemmatise the text using spaCy.
#     e.g. 'running' -> 'run', 'mice' -> 'mouse', etc.
#     """
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """
#     Convert text into n-grams (bigrams if n=2, trigrams if n=3).
#     e.g. tokens ['machine','learning'] -> 'machine_learning'
#     """
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# # Named Entity Extraction on the original text
# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing),
#     so we don't lose capitalization & advanced signals.
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     results = []
#     for ent in doc.ents:
#         results.append((ent.text, ent.label_))
#     return results


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Additional NER Tab")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # Dev usage only
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough cleaning for freq analysis & word clouds
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # Create 3 tabs now:
#     tab1, tab2, tab3 = st.tabs(["Stopwords Tab 1", "Stopwords Tab 2", "NER Tab"])

#     # -- Tab 1 & 2: replicate your previous pipeline
#     def render_stopwords_tab(tab_label):
#         st.subheader(f"{tab_label}: Manage Custom Stopwords")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", key=f"new_word_{tab_label}")
#         if st.button("Add Word to Remove", key=f"add_btn_{tab_label}"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())
#                 st.experimental_rerun()

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_{tab_label}"):
#                     st.session_state.custom_stopwords.remove(w)
#                     st.experimental_rerun()
#         else:
#             st.info("No custom stopwords yet.")

#         # Combine built-in + custom
#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader(f"Articles Table - {tab_label}")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader(f"Top Words (Frequency) - {tab_label}")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full frequency table
#         st.subheader(f"Word Frequency Table - {tab_label}")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader(f"Word Cloud - {tab_label}")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # -----------------------------------------------------------------
#         # ADVANCED: LEMMAS, BIGRAMS, TRIGRAMS
#         # -----------------------------------------------------------------
#         st.subheader(f"Advanced: Lemmas, Bigrams, Trigrams - {tab_label}")

#         df_advanced = df_tab.copy()

#         # Lemmas
#         st.markdown("### Lemmas")
#         st.write("""\
# **Description**:  
# Lemmas unify different forms of a word (e.g., 'running', 'runs', 'ran' → 'run').  
# We do this with spaCy on the finalText after stopwords are removed.
# """)

#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])

#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # Bigrams
#         st.markdown("### Bigrams")
#         st.write("""\
# Bigrams: pairs of consecutive tokens.  
# E.g. 'machine learning' → 'machine_learning'.
# """)
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         # Trigrams
#         st.markdown("### Trigrams")
#         st.write("""\
# Trigrams: triplets of consecutive tokens.  
# E.g. 'new york city' → 'new_york_city'.
# """)
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])

#         st.write("#### Trigram Word Cloud")
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # -- Render Tab 1 & 2
#     with tab1:
#         render_stopwords_tab("Tab 1")

#     with tab2:
#         render_stopwords_tab("Tab 2")

#     # ------------------------------------------------------------------
#     # 3) TAB 3: Named Entity Recognition (NER)
#     # ------------------------------------------------------------------
#     with tab3:
#         st.subheader("Named Entity Recognition (NER)")
#         st.write("""
#         **Explanation**:
#         - We parse the **original** article text (title + description + content) to preserve case.
#         - Use spaCy to detect entities like PERSON, ORG, GPE, etc.
#         - We'll show a table and bar chart of the top entities in the corpus.
#         """)

#         # We'll collect entity frequencies across all articles
#         entity_counts = {}  # {(ent_text, ent_label): count}

#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         # Sort by frequency descending
#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         # Show top 30 in a table
#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]

#         # Each row: [Entity Text, Label, Count]
#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])

#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         # Build a bar chart
#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]

#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values("count", ascending=False)
#             st.bar_chart(chart_df.set_index("entity"))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text only
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             # repeat the entity text 'count' times
#             # replacing spaces with underscores
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")


# if __name__ == "__main__":
#     main()




























# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # Initialize spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """
#     Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords.
#     """
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """
#     Lemmatise the text using spaCy.
#     e.g. 'running' -> 'run', 'mice' -> 'mouse', etc.
#     """
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """
#     Convert text into n-grams (bigrams if n=2, trigrams if n=3).
#     e.g. tokens ['machine','learning'] -> 'machine_learning'
#     """
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# # Named Entity Extraction on the original text
# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     results = []
#     for ent in doc.ents:
#         results.append((ent.text, ent.label_))
#     return results


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with 1 Stopwords Tab + NER Tab")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # Dev usage only
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough cleaning for freq analysis & word clouds
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # Create only 2 tabs now:
#     tab_stopwords, tab_ner = st.tabs(["Stopwords & Advanced", "NER Tab"])

#     # ------------------------------------------------------------------
#     # TAB 1: Stopwords & Advanced (lemmas, bigrams, trigrams)
#     # ------------------------------------------------------------------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())
#                 st.experimental_rerun()

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#                     st.experimental_rerun()
#         else:
#             st.info("No custom stopwords yet.")

#         # Combine built-in + custom
#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full frequency table
#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # -----------------------------------------------------------------
#         # ADVANCED: LEMMAS, BIGRAMS, TRIGRAMS
#         # -----------------------------------------------------------------
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")

#         df_advanced = df_tab.copy()

#         # Lemmas
#         st.markdown("### Lemmas")
#         st.write("""\
# **Description**:  
# Lemmas unify different forms of a word. e.g. 'running', 'ran' → 'run' (SpaCy-based).
# """)

#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])

#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # Bigrams
#         st.markdown("### Bigrams")
#         st.write("""Pairs of consecutive tokens. e.g. 'machine learning' → 'machine_learning'.""")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         # Trigrams
#         st.markdown("### Trigrams")
#         st.write("""Triplets of consecutive tokens. e.g. 'new york city' → 'new_york_city'.""")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])

#         st.write("#### Trigram Word Cloud")
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # ------------------------------------------------------------------
#     # TAB 2: Named Entity Recognition (NER)
#     # ------------------------------------------------------------------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         st.write("""
#         **Explanation**:
#         - We parse the **original** article text (title + description + content) to preserve case.
#         - Use spaCy to detect entities like PERSON, ORG, GPE, etc.
#         - We'll show a table and bar chart of the top entities in the corpus.
#         """)

#         # We'll collect entity frequencies across all articles
#         entity_counts = {}  # {(ent_text, ent_label): count}

#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         # Sort by frequency descending
#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         # Show top 30 in a table
#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]

#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])

#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         # Build a bar chart
#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]

#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values("count", ascending=False)
#             st.bar_chart(chart_df.set_index("entity"))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")


# if __name__ == "__main__":
#     main()
















# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # Initialize spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """
#     Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords.
#     """
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """
#     Lemmatise the text using spaCy.
#     e.g. 'running' -> 'run', 'mice' -> 'mouse', etc.
#     """
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """
#     Convert text into n-grams (bigrams if n=2, trigrams if n=3).
#     e.g. tokens ['machine','learning'] -> 'machine_learning'
#     """
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# # Named Entity Extraction on the original text
# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     results = []
#     for ent in doc.ents:
#         results.append((ent.text, ent.label_))
#     return results


# # Topic Modelling (LDA) Helper
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     """
#     Train an LDA model using Gensim on the given list of documents (already pre-processed).
#     Returns the trained LDA model plus the dictionary/corpus used.
#     """
#     # Split each doc into tokens
#     tokens_list = [doc.split() for doc in docs if doc.strip()]

#     # Create a dictionary from the tokenised documents
#     dictionary = Dictionary(tokens_list)
#     # Create a bag-of-words corpus
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

#     # Train LDA model
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

#     return lda_model, corpus, dictionary


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Stopwords, NER and Topic Modelling")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # dev only, not for production
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough cleaning for freq analysis & word clouds
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # Create 3 tabs: Stopwords & Advanced, NER Tab, Topic Modelling
#     tab_stopwords, tab_ner, tab_topics = st.tabs(["Stopwords & Advanced", "NER Tab", "Topic Modelling"])

#     # ------------------------------------------------------------------
#     # TAB 1: Stopwords & Advanced (lemmas, bigrams, trigrams)
#     # ------------------------------------------------------------------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())
#                 st.experimental_rerun()

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#                     st.experimental_rerun()
#         else:
#             st.info("No custom stopwords yet.")

#         # Combine built-in + custom
#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full frequency table
#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # -----------------------------------------------------------------
#         # ADVANCED: LEMMAS, BIGRAMS, TRIGRAMS
#         # -----------------------------------------------------------------
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")

#         df_advanced = df_tab.copy()

#         # Lemmas
#         st.markdown("### Lemmas")
#         st.write("""\
# **Description**:  
# Lemmas unify different forms of a word. e.g. 'running', 'ran' → 'run' (SpaCy-based).
# """)

#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])

#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # Bigrams
#         st.markdown("### Bigrams")
#         st.write("""Pairs of consecutive tokens. e.g. 'machine learning' → 'machine_learning'.""")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         # Trigrams
#         st.markdown("### Trigrams")
#         st.write("""Triplets of consecutive tokens. e.g. 'new york city' → 'new_york_city'.""")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])

#         st.write("#### Trigram Word Cloud")
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # ------------------------------------------------------------------
#     # TAB 2: Named Entity Recognition (NER)
#     # ------------------------------------------------------------------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         st.write("""
#         **Explanation**:
#         - We parse the **original** article text (title + description + content) to preserve case.
#         - Use spaCy to detect entities like PERSON, ORG, GPE, etc.
#         - We'll show a table and bar chart of the top entities in the corpus.
#         """)

#         # We'll collect entity frequencies across all articles
#         entity_counts = {}  # {(ent_text, ent_label): count}

#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         # Sort by frequency descending
#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         # Show top 30 in a table
#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]

#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])

#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         # Build a bar chart
#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]

#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values("count", ascending=False)
#             st.bar_chart(chart_df.set_index("entity"))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")

#     # ------------------------------------------------------------------
#     # TAB 3: Topic Modelling (LDA)
#     # ------------------------------------------------------------------
#     with tab_topics:
#         st.subheader("Topic Modelling (LDA)")
#         st.write("""
#         **Discover major themes or topics in the corpus** using Latent Dirichlet Allocation (LDA).
#         """)

#         # We'll use the finalText (stopwords removed) for topic modelling
#         df_for_topics = df_tab.copy()

#         # User inputs for LDA
#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words to Display per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     # Prepare documents
#                     docs = df_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )

#                     st.success("LDA Topic Modelling complete!")

#                     # Display the topics
#                     st.write("### Discovered Topics:")
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                 except Exception as ex:
#                     st.error(f"Error running LDA: {ex}")


# if __name__ == "__main__":
#     main()



























































# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # For interactive network visualisation
# from pyvis.network import Network
# import streamlit.components.v1 as components

# # Initialize spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """
#     Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords.
#     """
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """
#     Lemmatise the text using spaCy.
#     e.g. 'running' -> 'run', 'mice' -> 'mouse', etc.
#     """
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """
#     Convert text into n-grams (bigrams if n=2, trigrams if n=3).
#     e.g. tokens ['machine','learning'] -> 'machine_learning'
#     """
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# # Named Entity Extraction on the original text
# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     results = []
#     for ent in doc.ents:
#         results.append((ent.text, ent.label_))
#     return results


# # Topic Modelling (LDA) Helpers
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     """
#     Train an LDA model using Gensim on the given list of documents (already pre-processed).
#     Returns the trained LDA model plus the dictionary/corpus used.
#     """
#     # Split each doc into tokens
#     tokens_list = [doc.split() for doc in docs if doc.strip()]

#     # Create a dictionary from the tokenised documents
#     dictionary = Dictionary(tokens_list)
#     # Create a bag-of-words corpus
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

#     # Train LDA model
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

#     return lda_model, corpus, dictionary


# # def create_topic_pyvis_network(topic_id, topic_terms):
# #     """
# #     Build a PyVis Network for a single LDA topic, linking the 'Topic' node
# #     to each top word in that topic. Node sizes reflect word importance.
# #     Returns the PyVis Network object.
# #     """
# #     net = Network(height="400px", width="100%", directed=False)
    
# #     # Central node (the topic itself)
# #     center_node_id = f"Topic_{topic_id}"
# #     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")
    
# #     # Connect top words to the topic node
# #     # 'topic_terms' is a list of (word, weight) pairs
# #     for (term, weight) in topic_terms:
# #         # Scale the node size by weight, e.g., 3000 factor
# #         # and clamp or offset so it's not too small or too large
# #         size = 10 + (weight * 3000)
# #         net.add_node(term, label=term, size=size, color="#1f77b4")
# #         net.add_edge(center_node_id, term, value=weight, title=f"Weight: {weight:.5f}")
    
# #     return net

# # def display_pyvis_network(net, topic_id):
# #     """
# #     Shows a PyVis Network in Streamlit by writing it to HTML, then rendering.
# #     """
# #     html_filename = f"topic_network_{topic_id}.html"
# #     net.show(html_filename)
# #     # Read the HTML file
# #     with open(html_filename, "r", encoding="utf-8") as f:
# #         html_content = f.read()
# #     # Display as HTML in Streamlit
# #     components.html(html_content, height=500, scrolling=True)





# # def create_topic_pyvis_network(topic_id, topic_terms):
# #     """
# #     Build a PyVis Network for a single LDA topic, linking the 'Topic' node
# #     to each top word in that topic. Node sizes reflect word importance.
# #     Returns the PyVis Network object.
# #     """
# #     net = Network(height="400px", width="100%", directed=False)
    
# #     # Central node (the topic itself)
# #     center_node_id = f"Topic_{topic_id}"
# #     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")
    
# #     # Connect top words to the topic node
# #     for (term, weight) in topic_terms:
# #         # Scale the node size by weight
# #         size = 10 + (weight * 3000)  
# #         net.add_node(term, label=term, size=size, color="#1f77b4")
# #         net.add_edge(center_node_id, term, value=weight, title=f"Weight: {weight:.5f}")
    
# #     return net







# def create_topic_pyvis_network(topic_id, topic_terms):
#     """
#     Build a PyVis Network for a single LDA topic, linking the 'Topic' node
#     to each top word in that topic. Node sizes reflect word importance.
#     """
#     from pyvis.network import Network

#     net = Network(height="400px", width="100%", directed=False)
    
#     # Central node (the topic itself)
#     center_node_id = f"Topic_{topic_id}"
#     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")
    
#     # Connect top words to the topic node
#     for (term, weight) in topic_terms:
#         # Convert float32 -> Python float
#         weight_val = float(weight)
#         # Scale node size
#         size = 10 + (weight_val * 3000.0)

#         net.add_node(term, label=term, size=size, color="#1f77b4")
#         net.add_edge(
#             center_node_id, 
#             term, 
#             value=weight_val, 
#             title=f"Weight: {weight_val:.5f}"
#         )
    
#     return net















# def display_pyvis_network(net, topic_id):
#     """
#     Shows a PyVis Network in Streamlit by writing it to HTML, then embedding that HTML.
#     """
#     html_filename = f"topic_network_{topic_id}.html"
#     # Instead of net.show(), we call net.write_html() to avoid returning None.
#     net.write_html(html_filename)

#     # Now read the HTML file
#     with open(html_filename, "r", encoding="utf-8") as f:
#         html_content = f.read()
#     # Render the HTML in Streamlit
#     import streamlit.components.v1 as components
#     components.html(html_content, height=500, scrolling=True)





# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Stopwords, NER and Topic Modelling (Now with Networks!)")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # dev only, not for production
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough cleaning for freq analysis & word clouds
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # Create 3 tabs: Stopwords & Advanced, NER Tab, Topic Modelling
#     tab_stopwords, tab_ner, tab_topics = st.tabs(["Stopwords & Advanced", "NER Tab", "Topic Modelling"])

#     # ------------------------------------------------------------------
#     # TAB 1: Stopwords & Advanced (lemmas, bigrams, trigrams)
#     # ------------------------------------------------------------------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())
#                 st.experimental_rerun()

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#                     st.experimental_rerun()
#         else:
#             st.info("No custom stopwords yet.")

#         # Combine built-in + custom
#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full frequency table
#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # -----------------------------------------------------------------
#         # ADVANCED: LEMMAS, BIGRAMS, TRIGRAMS
#         # -----------------------------------------------------------------
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")

#         df_advanced = df_tab.copy()

#         # Lemmas
#         st.markdown("### Lemmas")
#         st.write("""\
# **Description**:  
# Lemmas unify different forms of a word. e.g. 'running', 'ran' → 'run' (SpaCy-based).
# """)

#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])

#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # Bigrams
#         st.markdown("### Bigrams")
#         st.write("""Pairs of consecutive tokens. e.g. 'machine learning' → 'machine_learning'.""")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         # Trigrams
#         st.markdown("### Trigrams")
#         st.write("""Triplets of consecutive tokens. e.g. 'new york city' → 'new_york_city'.""")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])

#         st.write("#### Trigram Word Cloud")
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # ------------------------------------------------------------------
#     # TAB 2: Named Entity Recognition (NER)
#     # ------------------------------------------------------------------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         st.write("""
#         **Explanation**:
#         - We parse the **original** article text (title + description + content) to preserve case.
#         - Use spaCy to detect entities like PERSON, ORG, GPE, etc.
#         - We'll show a table and bar chart of the top entities in the corpus.
#         """)

#         # We'll collect entity frequencies across all articles
#         entity_counts = {}  # {(ent_text, ent_label): count}

#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         # Sort by frequency descending
#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         # Show top 30 in a table
#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]

#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])

#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         # Build a bar chart
#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]

#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values("count", ascending=False)
#             st.bar_chart(chart_df.set_index("entity"))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")

#     # ------------------------------------------------------------------
#     # TAB 3: Topic Modelling (LDA) with Interactive Networks
#     # ------------------------------------------------------------------
#     with tab_topics:
#         st.subheader("Topic Modelling (LDA) + Interactive Networks")
#         st.write("""
#         **Discover major themes (topics) in the corpus** using Latent Dirichlet Allocation (LDA).
        
#         For each topic, we also generate a **network graph** showing:
#         - A central node representing the topic,
#         - Edges linking to its top words (sized according to importance).
#         """)

#         # We'll use the finalText (stopwords removed) for topic modelling
#         df_for_topics = df_tab.copy()

#         # User inputs for LDA
#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words to Display per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     # Prepare documents
#                     docs = df_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )

#                     st.success("LDA Topic Modelling complete!")

#                     # Display the topics
#                     st.write("### Discovered Topics:")
#                     # show_topics returns a list of (topic_id, list_of_(word, weight)) if formatted=False
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
                    
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                     # Display interactive network for each topic
#                     st.write("### Interactive Topic Networks")
#                     st.info("Below are network graphs representing each topic and its top words.")

#                     for i, topic in topics:
#                         st.subheader(f"Topic {i} Network")
#                         # 'topic' is a list of (term, weight)
#                         # Create the network
#                         net = create_topic_pyvis_network(i, topic)
#                         # Render it in Streamlit
#                         display_pyvis_network(net, i)

#                 except Exception as ex:
#                     st.error(f"Error running LDA or building networks: {ex}")


# if __name__ == "__main__":
#     main()




























# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # For interactive network visualisation
# from pyvis.network import Network
# import streamlit.components.v1 as components

# # Initialize spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """
#     Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords.
#     """
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """
#     Lemmatise the text using spaCy.
#     e.g. 'running' -> 'run', 'mice' -> 'mouse', etc.
#     """
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """
#     Convert text into n-grams (bigrams if n=2, trigrams if n=3).
#     e.g. tokens ['machine','learning'] -> 'machine_learning'
#     """
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# # Named Entity Extraction on the original text
# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     results = []
#     for ent in doc.ents:
#         results.append((ent.text, ent.label_))
#     return results


# # Topic Modelling (LDA) Helpers
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     """
#     Train an LDA model using Gensim on the given list of documents (already pre-processed).
#     Returns the trained LDA model plus the dictionary/corpus used.
#     """
#     # Split each doc into tokens
#     tokens_list = [doc.split() for doc in docs if doc.strip()]

#     # Create a dictionary from the tokenised documents
#     dictionary = Dictionary(tokens_list)
#     # Create a bag-of-words corpus
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

#     # Train LDA model
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

#     return lda_model, corpus, dictionary


# def create_topic_pyvis_network(topic_id, topic_terms):
#     """
#     Build a PyVis Network for a single LDA topic, linking the 'Topic' node
#     to each top word in that topic. Node sizes reflect word importance.
#     """
#     net = Network(height="400px", width="100%", directed=False)
    
#     # Central node (the topic itself)
#     center_node_id = f"Topic_{topic_id}"
#     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")
    
#     # Connect top words to the topic node, cast weights to float so they're JSON-serialisable
#     for (term, weight) in topic_terms:
#         weight_val = float(weight)  # ensure it's a native float
#         size = 10 + (weight_val * 3000.0)
#         net.add_node(term, label=term, size=size, color="#1f77b4")
#         net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    
#     return net


# def display_pyvis_network(net, topic_id):
#     """
#     Shows a PyVis Network in Streamlit by writing it to HTML, then embedding that HTML.
#     """
#     html_filename = f"topic_network_{topic_id}.html"
#     net.write_html(html_filename)  # do not use net.show(), as it can return None

#     # Now read the HTML file
#     with open(html_filename, "r", encoding="utf-8") as f:
#         html_content = f.read()
#     # Render the HTML in Streamlit
#     components.html(html_content, height=500, scrolling=True)


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Stopwords, NER, Topic Modelling + Reset Button")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     # RESET BUTTON
#     if st.sidebar.button("Reset Data & Analyses"):
#         # Preserve whether API key was validated or not
#         was_valid = st.session_state.api_key_validated
#         # Clear session state
#         st.session_state.clear()
#         # Restore validation status
#         st.session_state.api_key_validated = was_valid
#         st.experimental_rerun()

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # dev only, not for production
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Fetch News
#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough cleaning for freq analysis & word clouds
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # Create 3 tabs: Stopwords & Advanced, NER Tab, Topic Modelling
#     tab_stopwords, tab_ner, tab_topics = st.tabs(["Stopwords & Advanced", "NER Tab", "Topic Modelling"])

#     # ------------------------------------------------------------------
#     # TAB 1: Stopwords & Advanced (lemmas, bigrams, trigrams)
#     # ------------------------------------------------------------------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         # Note: We do NOT use st.experimental_rerun here, so we don't force a tab reset.
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())
#             # The UI will re-run automatically; we won't lose tab selection

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#                     # Again, rely on automatic re-run from the button click
#         else:
#             st.info("No custom stopwords yet.")

#         # Combine built-in + custom
#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full frequency table
#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # -----------------------------------------------------------------
#         # ADVANCED: LEMMAS, BIGRAMS, TRIGRAMS
#         # -----------------------------------------------------------------
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")

#         df_advanced = df_tab.copy()

#         # Lemmas
#         st.markdown("### Lemmas")
#         st.write("""\
# **Description**:  
# Lemmas unify different forms of a word. e.g. 'running', 'ran' → 'run' (SpaCy-based).
# """)

#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])

#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # Bigrams
#         st.markdown("### Bigrams")
#         st.write("""Pairs of consecutive tokens. e.g. 'machine learning' → 'machine_learning'.""")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         # Trigrams
#         st.markdown("### Trigrams")
#         st.write("""Triplets of consecutive tokens. e.g. 'new york city' → 'new_york_city'.""")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])

#         st.write("#### Trigram Word Cloud")
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # ------------------------------------------------------------------
#     # TAB 2: Named Entity Recognition (NER)
#     # ------------------------------------------------------------------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         st.write("""
#         **Explanation**:
#         - We parse the **original** article text (title + description + content) to preserve case.
#         - Use spaCy to detect entities like PERSON, ORG, GPE, etc.
#         - We'll show a table and bar chart of the top entities in the corpus.
#         """)

#         # We'll collect entity frequencies across all articles
#         entity_counts = {}  # {(ent_text, ent_label): count}

#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         # Sort by frequency descending
#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         # Show top 30 in a table
#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]

#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])

#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         # Build a bar chart
#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]

#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values("count", ascending=False)
#             st.bar_chart(chart_df.set_index("entity"))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")

#     # ------------------------------------------------------------------
#     # TAB 3: Topic Modelling (LDA) with Interactive Networks
#     # ------------------------------------------------------------------
#     with tab_topics:
#         st.subheader("Topic Modelling (LDA) + Interactive Networks")
#         st.write("""
#         **Discover major themes (topics) in the corpus** using Latent Dirichlet Allocation (LDA).
        
#         For each topic, we also generate a **network graph** showing:
#         - A central node representing the topic,
#         - Edges linking to its top words (sized according to importance).
#         """)

#         # We'll use the finalText (stopwords removed) for topic modelling
#         df_for_topics = df_tab.copy()

#         # User inputs for LDA
#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words to Display per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     # Prepare documents
#                     docs = df_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )

#                     st.success("LDA Topic Modelling complete!")

#                     # Display the topics
#                     st.write("### Discovered Topics:")
#                     # show_topics returns list of (topic_id, list_of_(word, weight)) if formatted=False
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
                    
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                     # Display interactive network for each topic
#                     st.write("### Interactive Topic Networks")
#                     st.info("Below are network graphs representing each topic and its top words.")

#                     for i, topic in topics:
#                         st.subheader(f"Topic {i} Network")
#                         net = create_topic_pyvis_network(i, topic)
#                         display_pyvis_network(net, i)

#                 except Exception as ex:
#                     st.error(f"Error running LDA or building networks: {ex}")


# if __name__ == "__main__":
#     main()
















# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # For interactive network visualisation
# from pyvis.network import Network
# import streamlit.components.v1 as components

# # Initialize spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation or non-word chars
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """
#     Given a series of strings, build a frequency dict {word: count, ...}.
#     """
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """
#     Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords.
#     """
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """
#     Lemmatise the text using spaCy.
#     e.g. 'running' -> 'run', 'mice' -> 'mouse', etc.
#     """
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """
#     Convert text into n-grams (bigrams if n=2, trigrams if n=3).
#     e.g. tokens ['machine','learning'] -> 'machine_learning'
#     """
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# # Named Entity Extraction on the original text
# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     results = []
#     for ent in doc.ents:
#         results.append((ent.text, ent.label_))
#     return results


# # Topic Modelling (LDA) Helpers
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     """
#     Train an LDA model using Gensim on the given list of documents (already pre-processed).
#     Returns the trained LDA model plus the dictionary/corpus used.
#     """
#     # Split each doc into tokens
#     tokens_list = [doc.split() for doc in docs if doc.strip()]

#     # Create a dictionary from the tokenised documents
#     dictionary = Dictionary(tokens_list)
#     # Create a bag-of-words corpus
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

#     # Train LDA model
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

#     return lda_model, corpus, dictionary


# def create_topic_pyvis_network(topic_id, topic_terms):
#     """
#     Build a PyVis Network for a single LDA topic, linking the 'Topic' node
#     to each top word in that topic. Node sizes reflect word importance.
#     """
#     net = Network(height="400px", width="100%", directed=False)
    
#     # Central node (the topic itself)
#     center_node_id = f"Topic_{topic_id}"
#     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")
    
#     # Connect top words to the topic node, cast weights to float so they're JSON-serialisable
#     for (term, weight) in topic_terms:
#         weight_val = float(weight)  # ensure it's a native float
#         size = 10 + (weight_val * 3000.0)
#         net.add_node(term, label=term, size=size, color="#1f77b4")
#         net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    
#     return net


# def display_pyvis_network(net, topic_id):
#     """
#     Shows a PyVis Network in Streamlit by writing it to HTML, then embedding that HTML.
#     """
#     html_filename = f"topic_network_{topic_id}.html"
#     net.write_html(html_filename)  # do not use net.show(), as it can return None

#     # Now read the HTML file
#     with open(html_filename, "r", encoding="utf-8") as f:
#         html_content = f.read()
#     # Render the HTML in Streamlit
#     components.html(html_content, height=500, scrolling=True)


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Stopwords, NER, Topic Modelling + Reset Button")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()
#     if 'active_tab' not in st.session_state:
#         st.session_state.active_tab = "Stopwords & Advanced"

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     # RESET BUTTON
#     if st.sidebar.button("Reset Data & Analyses"):
#         # Preserve whether API key was validated or not
#         was_valid = st.session_state.api_key_validated
#         st.session_state.clear()
#         # Restore validation status
#         st.session_state.api_key_validated = was_valid
#         # Default tab again
#         st.session_state.active_tab = "Stopwords & Advanced"
#         st.experimental_rerun()

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # dev only, not for production
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough cleaning for freq analysis & word clouds
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     # -------------------------
#     # OUR CUSTOM TABBING SYSTEM
#     # -------------------------
#     tabs = ["Stopwords & Advanced", "NER Tab", "Topic Modelling"]
#     # Use a radio button to mimic tab selection
#     selected = st.radio("Navigation", tabs, index=tabs.index(st.session_state.active_tab))
#     st.session_state.active_tab = selected  # store which tab is active

#     # We'll retrieve df from session_state after fetch
#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # Pre-processed for stopword removal etc. We'll re-use in multiple sections.
#     df_tab = df.copy()
#     def remove_all_stopwords(txt):
#         return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#     df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#     # We can share df_tab with all sections as needed.

#     # ==================================================================
#     # STOPWORDS & ADVANCED
#     # ==================================================================
#     if selected == "Stopwords & Advanced":
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#         else:
#             st.info("No custom stopwords yet.")

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full frequency table
#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # -----------------------------------------------------------------
#         # ADVANCED: LEMMAS, BIGRAMS, TRIGRAMS
#         # -----------------------------------------------------------------
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")

#         df_advanced = df_tab.copy()

#         # Lemmas
#         st.markdown("### Lemmas")
#         st.write("""\
# **Description**:  
# Lemmas unify different forms of a word. e.g. 'running', 'ran' → 'run' (SpaCy-based).
# """)

#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])

#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # Bigrams
#         st.markdown("### Bigrams")
#         st.write("""Pairs of consecutive tokens. e.g. 'machine learning' → 'machine_learning'.""")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         # Trigrams
#         st.markdown("### Trigrams")
#         st.write("""Triplets of consecutive tokens. e.g. 'new york city' → 'new_york_city'.""")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])

#         st.write("#### Trigram Word Cloud")
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # ==================================================================
#     # NER TAB
#     # ==================================================================
#     elif selected == "NER Tab":
#         st.subheader("Named Entity Recognition (NER)")
#         st.write("""
#         **Explanation**:
#         - We parse the **original** article text (title + description + content) to preserve case.
#         - Use spaCy to detect entities like PERSON, ORG, GPE, etc.
#         - We'll show a table and bar chart of the top entities in the corpus.
#         """)

#         # We'll collect entity frequencies across all articles
#         entity_counts = {}  # {(ent_text, ent_label): count}

#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         # Sort by frequency descending
#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         # Show top 30 in a table
#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]

#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])

#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         # Build a bar chart
#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]

#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values("count", ascending=False)
#             st.bar_chart(chart_df.set_index("entity"))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")

#     # ==================================================================
#     # TOPIC MODELLING TAB
#     # ==================================================================
#     elif selected == "Topic Modelling":
#         st.subheader("Topic Modelling (LDA) + Interactive Networks")
#         st.write("""
#         **Discover major themes (topics) in the corpus** using Latent Dirichlet Allocation (LDA).
        
#         For each topic, we also generate a **network graph** showing:
#         - A central node representing the topic,
#         - Edges linking to its top words (sized according to importance).
#         """)

#         df_for_topics = df_tab.copy()

#         # User inputs for LDA
#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words to Display per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     # Prepare documents
#                     docs = df_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )

#                     st.success("LDA Topic Modelling complete!")

#                     # Display the topics
#                     st.write("### Discovered Topics:")
#                     # show_topics returns list of (topic_id, list_of_(word, weight)) if formatted=False
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
                    
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                     # Display interactive network for each topic
#                     st.write("### Interactive Topic Networks")
#                     st.info("Below are network graphs representing each topic and its top words.")

#                     for i, topic in topics:
#                         st.subheader(f"Topic {i} Network")
#                         net = create_topic_pyvis_network(i, topic)
#                         display_pyvis_network(net, i)

#                 except Exception as ex:
#                     st.error(f"Error running LDA or building networks: {ex}")


# if __name__ == "__main__":
#     main()


















# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # For interactive network visualisation
# from pyvis.network import Network
# import streamlit.components.v1 as components

# # Initialise spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """Given a series of strings, build a frequency dict {word: count, ...}."""
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords."""
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """Lemmatise the text using spaCy (e.g., 'running' -> 'run')."""
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """
#     Convert text into n-grams (bigrams if n=2, trigrams if n=3).
#     e.g. tokens ['machine','learning'] -> 'machine_learning'
#     """
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     return [(ent.text, ent.label_) for ent in doc.ents]


# # Topic Modelling (LDA) Helpers
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     """
#     Train an LDA model using Gensim on the given list of documents (already pre-processed).
#     Returns the trained LDA model plus the dictionary/corpus used.
#     """
#     # Split each doc into tokens
#     tokens_list = [doc.split() for doc in docs if doc.strip()]

#     # Create a dictionary from the tokenised documents
#     dictionary = Dictionary(tokens_list)
#     # Create a bag-of-words corpus
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

#     # Train LDA model
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

#     return lda_model, corpus, dictionary


# def create_topic_pyvis_network(topic_id, topic_terms):
#     """
#     Build a PyVis Network for a single LDA topic, linking the 'Topic' node
#     to each top word in that topic. Node sizes reflect word importance.
#     """
#     net = Network(height="400px", width="100%", directed=False)
    
#     # Central node (the topic itself)
#     center_node_id = f"Topic_{topic_id}"
#     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")
    
#     # Connect top words to the topic node, cast weights to float so they're JSON-serialisable
#     for (term, weight) in topic_terms:
#         weight_val = float(weight)  # ensure it's a native float
#         size = 10 + (weight_val * 3000.0)
#         net.add_node(term, label=term, size=size, color="#1f77b4")
#         net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    
#     return net


# def display_pyvis_network(net, topic_id):
#     """
#     Shows a PyVis Network in Streamlit by writing it to HTML, then embedding that HTML.
#     """
#     html_filename = f"topic_network_{topic_id}.html"
#     net.write_html(html_filename)  # do not use net.show() (it can return None)

#     # Now read the HTML file
#     with open(html_filename, "r", encoding="utf-8") as f:
#         html_content = f.read()
#     # Render the HTML in Streamlit
#     components.html(html_content, height=500, scrolling=True)


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     st.title("News Analysis with Stopwords, NER, Topic Modelling + Reset Button")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # SIDEBAR
#     st.sidebar.markdown("# News Analysis\n---")

#     # RESET BUTTON
#     if st.sidebar.button("Reset Data & Analyses"):
#         # Preserve whether API key was validated or not
#         was_valid = st.session_state.api_key_validated
#         st.session_state.clear()
#         # Restore validation status
#         st.session_state.api_key_validated = was_valid
#         st.experimental_rerun()  # Re-run app to clear data

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # dev only, not for production
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Fetch News
#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough cleaning for freq analysis & word clouds
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # --------------------------------
#     # TABS
#     # --------------------------------
#     tab_stopwords, tab_ner, tab_topics = st.tabs(["Stopwords & Advanced", "NER Tab", "Topic Modelling"])

#     # --------------- TAB 1: STOPWORDS & ADVANCED ---------------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#         else:
#             st.info("No custom stopwords yet.")

#         # Create finalText (with all stopwords removed)
#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full frequency table
#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # ------------------- Advanced: Lemmas, Bigrams, Trigrams -------------------
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")

#         df_advanced = df_tab.copy()

#         # Lemmas
#         st.markdown("### Lemmas")
#         st.write("""\
#         **Description**:  
#         Lemmas unify different forms of a word. e.g. 'running', 'ran' → 'run' (SpaCy-based).
#         """)

#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])

#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # Bigrams
#         st.markdown("### Bigrams")
#         st.write("""Pairs of consecutive tokens. e.g. 'machine learning' → 'machine_learning'.""")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         # Trigrams
#         st.markdown("### Trigrams")
#         st.write("""Triplets of consecutive tokens. e.g. 'new york city' → 'new_york_city'.""")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])

#         st.write("#### Trigram Word Cloud")
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)


#     # --------------- TAB 2: NER ---------------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         st.write("""
#         **Explanation**:
#         - We parse the **original** article text (title + description + content) to preserve case.
#         - Use spaCy to detect entities like PERSON, ORG, GPE, etc.
#         - We'll show a table and bar chart of the top entities in the corpus.
#         """)

#         # We'll collect entity frequencies across all articles
#         entity_counts = {}  # {(ent_text, ent_label): count}
#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         # Sort by frequency descending
#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         # Show top 30 in a table
#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]

#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])

#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         # Build a bar chart
#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]

#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values("count", ascending=False)
#             st.bar_chart(chart_df.set_index("entity"))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")


#     # --------------- TAB 3: TOPIC MODELLING ---------------
#     with tab_topics:
#         st.subheader("Topic Modelling (LDA) + Interactive Networks")
#         st.write("""
#         **Discover major themes (topics) in the corpus** using Latent Dirichlet Allocation (LDA).
        
#         For each topic, we also generate a **network graph** showing:
#         - A central node representing the topic,
#         - Edges linking to its top words (sized according to importance).
#         """)

#         # We'll re-use the same finalText from the Stopwords tab
#         df_tab_for_topics = df.copy()
#         df_tab_for_topics['finalText'] = df_tab_for_topics['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         # LDA settings
#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words to Display per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     # Prepare documents
#                     docs = df_tab_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )

#                     st.success("LDA Topic Modelling complete!")

#                     # Display the topics
#                     st.write("### Discovered Topics:")
#                     # show_topics returns list of (topic_id, list_of_(word, weight)) if formatted=False
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
                    
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                     # Display interactive network for each topic
#                     st.write("### Interactive Topic Networks")
#                     st.info("Below are network graphs representing each topic and its top words.")

#                     for i, topic in topics:
#                         st.subheader(f"Topic {i} Network")
#                         net = create_topic_pyvis_network(i, topic)
#                         display_pyvis_network(net, i)

#                 except Exception as ex:
#                     st.error(f"Error running LDA or building networks: {ex}")


# if __name__ == "__main__":
#     main()



































# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # For interactive network visualisation
# from pyvis.network import Network
# import streamlit.components.v1 as components

# # Initialise spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """Given a series of strings, build a frequency dict {word: count, ...}."""
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords."""
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """Lemmatise text using spaCy, e.g. 'running' -> 'run'."""
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """
#     Convert text into n-grams (bigrams if n=2, trigrams if n=3).
#     e.g. tokens ['machine','learning'] -> 'machine_learning'
#     """
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     return [(ent.text, ent.label_) for ent in doc.ents]


# # Topic Modelling (LDA) Helpers
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     """
#     Train an LDA model using Gensim on the given list of documents (already pre-processed).
#     Returns the trained LDA model plus the dictionary/corpus used.
#     """
#     # Split each doc into tokens
#     tokens_list = [doc.split() for doc in docs if doc.strip()]

#     # Create a dictionary from the tokenised documents
#     dictionary = Dictionary(tokens_list)
#     # Create a bag-of-words corpus
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

#     # Train LDA model
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

#     return lda_model, corpus, dictionary


# def create_topic_pyvis_network(topic_id, topic_terms):
#     """
#     Build a PyVis Network for a single LDA topic, linking the 'Topic' node
#     to each top word in that topic. Node sizes reflect word importance.
#     """
#     # We make it taller to minimise white space below the graph
#     # We'll also define some styling so labels appear in the centre of each node.
#     net = Network(height="600px", width="100%", directed=False)

#     # Add custom config to centre labels on nodes, remove extra smoothing
#     net.set_options("""
#     var options = {
#       "nodes": {
#         "font": {
#           "size": 16,
#           "align": "center"
#         },
#         "shape": "circle"
#       },
#       "edges": {
#         "smooth": false,
#         "color": {
#           "inherit": false
#         }
#       },
#       "physics": {
#         "enabled": true,
#         "stabilization": {
#           "enabled": true,
#           "iterations": 100
#         }
#       },
#       "interaction": {
#         "dragNodes": true
#       }
#     }
#     """)

#     # Central node (the topic itself)
#     center_node_id = f"Topic_{topic_id}"
#     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")
    
#     # Connect top words to the topic node
#     for (term, weight) in topic_terms:
#         weight_val = float(weight)  # ensure it's a native float for JSON
#         size = 10 + (weight_val * 3000.0)
#         net.add_node(term, label=term, size=size, color="#1f77b4")
#         net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    
#     return net


# def display_pyvis_network(net, topic_id):
#     """
#     Shows a PyVis Network in Streamlit by writing it to HTML, then embedding that HTML.
#     """
#     html_filename = f"topic_network_{topic_id}.html"
#     net.write_html(html_filename)

#     # Now read the HTML file
#     with open(html_filename, "r", encoding="utf-8") as f:
#         html_content = f.read()

#     # Render the HTML in Streamlit
#     # We remove 'scrolling=True' to minimise extra white space
#     components.html(html_content, height=600, scrolling=False)


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     # Inject custom CSS at the top for dark green buttons
#     st.markdown("""
#     <style>
#     div.stButton > button {
#         background-color: #006400 !important;
#         color: white !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     st.title("News Analysis with Prettier UI (Stopwords, NER, Topic Modelling)")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     # RESET BUTTON
#     if st.sidebar.button("Reset Data & Analyses"):
#         # Preserve whether API key was validated or not
#         was_valid = st.session_state.api_key_validated
#         st.session_state.clear()
#         st.session_state.api_key_validated = was_valid
#         st.experimental_rerun()

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="791a2ea85e1b470c8620abf40b88dcac",  # dev only, not for production
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Fetch News
#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough cleaning for freq analysis & word clouds
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # --------------------------------
#     # TABS
#     # --------------------------------
#     tab_stopwords, tab_ner, tab_topics = st.tabs(["Stopwords & Advanced", "NER Tab", "Topic Modelling"])

#     # --------------- TAB 1: STOPWORDS & ADVANCED ---------------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         # Add new stopword
#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#         else:
#             st.info("No custom stopwords yet.")

#         # Create finalText (with all stopwords removed)
#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         # Articles Table
#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full frequency table
#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # ------------------- Advanced: Lemmas, Bigrams, Trigrams -------------------
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")

#         df_advanced = df_tab.copy()

#         # Lemmas
#         st.markdown("### Lemmas")
#         st.write("""\
#         **Description**:  
#         Lemmas unify different forms of a word. e.g. 'running', 'ran' → 'run' (SpaCy-based).
#         """)

#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])

#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # Bigrams
#         st.markdown("### Bigrams")
#         st.write("""Pairs of consecutive tokens. e.g. 'machine learning' → 'machine_learning'.""")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         # Trigrams
#         st.markdown("### Trigrams")
#         st.write("""Triplets of consecutive tokens. e.g. 'new york city' → 'new_york_city'.""")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])

#         st.write("#### Trigram Word Cloud")
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)


#     # --------------- TAB 2: NER ---------------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         st.write("""
#         **Explanation**:
#         - We parse the **original** article text (title + description + content) to preserve case.
#         - Use spaCy to detect entities like PERSON, ORG, GPE, etc.
#         - We'll show a table and bar chart of the top entities in the corpus.
#         """)

#         # We'll collect entity frequencies across all articles
#         entity_counts = {}
#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         # Sort by frequency descending
#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         # Show top 30 in a table
#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]

#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])

#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         # Bar chart
#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]

#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values("count", ascending=False)
#             st.bar_chart(chart_df.set_index("entity"))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")


#     # --------------- TAB 3: TOPIC MODELLING ---------------
#     with tab_topics:
#         st.subheader("Topic Modelling (LDA) + Interactive Networks")
#         st.write("""
#         **Discover major themes (topics) in the corpus** using Latent Dirichlet Allocation (LDA).
        
#         For each topic, we also generate a **network graph** showing:
#         - A central node representing the topic,
#         - Edges linking to its top words (sized according to importance),
#         - Labels are now centred on each node,
#         - Extra space is minimised.
#         """)

#         # We'll re-use finalText from the Stopwords tab
#         df_tab_for_topics = df.copy()
#         df_tab_for_topics['finalText'] = df_tab_for_topics['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         # LDA settings
#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     docs = df_tab_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )

#                     st.success("LDA Topic Modelling complete!")

#                     # Display the topics
#                     st.write("### Discovered Topics:")
#                     # show_topics returns a list of (topic_id, [(word, weight), ...]) if formatted=False
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
                    
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                     # Display interactive network for each topic
#                     st.write("### Interactive Topic Networks")
#                     st.info("Below are network graphs for each topic. Labels are centred, and we remove extra scroll space.")

#                     for i, topic in topics:
#                         st.subheader(f"Topic {i} Network")
#                         net = create_topic_pyvis_network(i, topic)
#                         display_pyvis_network(net, i)

#                 except Exception as ex:
#                     st.error(f"Error running LDA or building networks: {ex}")


# if __name__ == "__main__":
#     main()


































# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # For interactive network visualisation
# from pyvis.network import Network
# import streamlit.components.v1 as components

# # Initialise spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")

# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """Given a series of strings, build a frequency dict {word: count, ...}."""
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords."""
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """Lemmatise text using spaCy, e.g. 'running' -> 'run'."""
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """
#     Convert text into n-grams (bigrams if n=2, trigrams if n=3).
#     e.g. tokens ['machine','learning'] -> 'machine_learning'
#     """
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     return [(ent.text, ent.label_) for ent in doc.ents]


# # Topic Modelling (LDA) Helpers
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     """
#     Train an LDA model using Gensim on the given list of documents (already pre-processed).
#     Returns the trained LDA model plus the dictionary/corpus used.
#     """
#     # Split each doc into tokens
#     tokens_list = [doc.split() for doc in docs if doc.strip()]

#     # Create a dictionary from the tokenised documents
#     dictionary = Dictionary(tokens_list)
#     # Create a bag-of-words corpus
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

#     # Train LDA model
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

#     return lda_model, corpus, dictionary


# def create_topic_pyvis_network(topic_id, topic_terms):
#     """
#     Build a PyVis Network for a single LDA topic, linking the 'Topic' node
#     to each top word in that topic. Node sizes reflect word importance.
#     """
#     # Increase height to reduce bottom whitespace
#     net = Network(height="600px", width="100%", directed=False)

#     # Custom config: centre labels, remove extra smoothing
#     net.set_options("""
#     var options = {
#       "nodes": {
#         "font": {
#           "size": 16,
#           "align": "center"
#         },
#         "shape": "circle"
#       },
#       "edges": {
#         "smooth": false,
#         "color": {
#           "inherit": false
#         }
#       },
#       "physics": {
#         "enabled": true,
#         "stabilization": {
#           "enabled": true,
#           "iterations": 100
#         }
#       },
#       "interaction": {
#         "dragNodes": true
#       }
#     }
#     """)

#     # Central topic node
#     center_node_id = f"Topic_{topic_id}"
#     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")
    
#     # Connect top words to the topic node
#     for (term, weight) in topic_terms:
#         weight_val = float(weight)
#         size = 10 + (weight_val * 3000.0)
#         net.add_node(term, label=term, size=size, color="#1f77b4")
#         net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    
#     return net


# def display_pyvis_network(net, topic_id):
#     """
#     Shows a PyVis Network in Streamlit by writing it to HTML, then embedding that HTML.
#     """
#     html_filename = f"topic_network_{topic_id}.html"
#     net.write_html(html_filename)

#     # Read the HTML file
#     with open(html_filename, "r", encoding="utf-8") as f:
#         html_content = f.read()

#     # Render the HTML in Streamlit
#     components.html(html_content, height=600, scrolling=False)


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     # -- 1) Dark green buttons
#     st.markdown("""
#     <style>
#     div.stButton > button {
#         background-color: #006400 !important;
#         color: white !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     # -- 2) Move sidebar content higher by reducing top padding
#     st.markdown("""
#     <style>
#     /* Force the sidebar content to start closer to the top */
#     section[data-testid="stSidebar"] .css-1d391kg {
#         padding-top: 1rem !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     # Title
#     st.title("News Analysis with Elevated Sidebar")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # ---------------
#     # SIDEBAR
#     # ---------------
#     st.sidebar.markdown("# News Analysis\n---")

#     # RESET BUTTON
#     if st.sidebar.button("Reset Data & Analyses"):
#         was_valid = st.session_state.api_key_validated
#         st.session_state.clear()
#         st.session_state.api_key_validated = was_valid
#         st.experimental_rerun()

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="a2b4f531cea743a1b10d0aad86bd44f5",  # dev only, not for production
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Fetch News
#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough cleaning for freq analysis & word clouds
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     # If no articles, do nothing further
#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # ---------------
#     # MAIN TABS
#     # ---------------
#     tab_stopwords, tab_ner, tab_topics = st.tabs(["Stopwords & Advanced", "NER Tab", "Topic Modelling"])

#     # ----------- TAB 1: STOPWORDS & ADVANCED -----------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())

#         # Display existing stopwords & remove
#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#         else:
#             st.info("No custom stopwords yet.")

#         # finalText with all stopwords removed
#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full freq table
#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # ---------- Lemmas, Bigrams, Trigrams ----------
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")

#         df_advanced = df_tab.copy()

#         # Lemmas
#         st.markdown("### Lemmas")
#         st.write("""\
#         **Description**:  
#         Lemmas unify different forms of a word. e.g. 'running', 'ran' → 'run' (SpaCy-based).
#         """)
#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])

#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # Bigrams
#         st.markdown("### Bigrams")
#         st.write("""Pairs of consecutive tokens. e.g. 'machine learning' → 'machine_learning'.""")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         # Trigrams
#         st.markdown("### Trigrams")
#         st.write("""Triplets of consecutive tokens. e.g. 'new york city' → 'new_york_city'.""")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])

#         st.write("#### Trigram Word Cloud")
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)


#     # ----------- TAB 2: NER -----------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         st.write("""
#         **Explanation**:
#         - We parse the **original** article text (title + description + content) to preserve case.
#         - Use spaCy to detect entities like PERSON, ORG, GPE, etc.
#         - We'll show a table and bar chart of the top entities in the corpus.
#         """)

#         entity_counts = {}
#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         # Sort by frequency desc
#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         # Top 30
#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]

#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])

#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         # Bar chart
#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]

#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values("count", ascending=False)
#             st.bar_chart(chart_df.set_index("entity"))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")


#     # ----------- TAB 3: TOPIC MODELLING -----------
#     with tab_topics:
#         st.subheader("Topic Modelling (LDA) + Interactive Networks")
#         st.write("""
#         **Discover major themes (topics) in the corpus** using Latent Dirichlet Allocation (LDA).
        
#         For each topic, we also generate a **network graph** showing:
#         - A central node representing the topic,
#         - Edges linking to its top words (sized according to importance),
#         - Centred labels,
#         - Minimal extra whitespace at the bottom.
#         """)

#         df_tab_for_topics = df.copy()
#         df_tab_for_topics['finalText'] = df_tab_for_topics['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     docs = df_tab_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )

#                     st.success("LDA Topic Modelling complete!")

#                     # Display the topics
#                     st.write("### Discovered Topics:")
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
                    
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                     # Interactive networks
#                     st.write("### Interactive Topic Networks")
#                     st.info("Below are network graphs for each topic with centred labels and minimal whitespace.")

#                     for i, topic in topics:
#                         st.subheader(f"Topic {i} Network")
#                         net = create_topic_pyvis_network(i, topic)
#                         display_pyvis_network(net, i)

#                 except Exception as ex:
#                     st.error(f"Error running LDA or building networks: {ex}")


# if __name__ == "__main__":
#     main()




































# import streamlit as st
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # For interactive network visualisation
# from pyvis.network import Network
# import streamlit.components.v1 as components

# # For keyword extraction and clustering
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import numpy as np

# # Initialise spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """Given a series of strings, build a frequency dict {word: count, ...}."""
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords."""
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """Lemmatise text using spaCy, e.g. 'running' -> 'run'."""
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """
#     Convert text into n-grams (bigrams if n=2, trigrams if n=3).
#     e.g. tokens ['machine','learning'] -> 'machine_learning'
#     """
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     return [(ent.text, ent.label_) for ent in doc.ents]


# # -----------------------------
# # Topic Modelling (LDA)
# # -----------------------------
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     """
#     Train an LDA model using Gensim on the given list of documents (already pre-processed).
#     Returns the trained LDA model plus the dictionary/corpus used.
#     """
#     # Split each doc into tokens
#     tokens_list = [doc.split() for doc in docs if doc.strip()]

#     # Create a dictionary from the tokenised documents
#     dictionary = Dictionary(tokens_list)
#     # Create a bag-of-words corpus
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

#     # Train LDA model
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

#     return lda_model, corpus, dictionary


# def create_topic_pyvis_network(topic_id, topic_terms):
#     """
#     Build a PyVis Network for a single LDA topic, linking the 'Topic' node
#     to each top word in that topic. Node sizes reflect word importance.
#     """
#     net = Network(height="600px", width="100%", directed=False)

#     # Custom config: centre labels, remove extra smoothing
#     net.set_options("""
#     var options = {
#       "nodes": {
#         "font": {
#           "size": 16,
#           "align": "center"
#         },
#         "shape": "circle"
#       },
#       "edges": {
#         "smooth": false,
#         "color": {
#           "inherit": false
#         }
#       },
#       "physics": {
#         "enabled": true,
#         "stabilization": {
#           "enabled": true,
#           "iterations": 100
#         }
#       },
#       "interaction": {
#         "dragNodes": true
#       }
#     }
#     """)

#     # Central topic node
#     center_node_id = f"Topic_{topic_id}"
#     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")
    
#     # Connect top words to the topic node
#     for (term, weight) in topic_terms:
#         weight_val = float(weight)  # ensure JSON-serialisable
#         size = 10 + (weight_val * 3000.0)
#         net.add_node(term, label=term, size=size, color="#1f77b4")
#         net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    
#     return net


# def display_pyvis_network(net, topic_id):
#     """
#     Shows a PyVis Network in Streamlit by writing it to HTML, then embedding that HTML.
#     """
#     html_filename = f"topic_network_{topic_id}.html"
#     net.write_html(html_filename)

#     with open(html_filename, "r", encoding="utf-8") as f:
#         html_content = f.read()
#     components.html(html_content, height=600, scrolling=False)


# # -----------------------------
# # Keyword Extraction (TF-IDF)
# # -----------------------------
# def extract_keywords_tfidf(docs, top_n=10):
#     """
#     Extract top_n keywords by average TF-IDF across the entire corpus.
#     Each doc is a string (already stopword/lemmatised if needed).
#     """
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)

#     # Average tf-idf for each term across all docs
#     avg_tfidf = np.mean(X.toarray(), axis=0)
#     feature_names = vectorizer.get_feature_names_out()

#     # Sort features by descending average tf-idf
#     sorted_indices = np.argsort(avg_tfidf)[::-1]  # highest first
#     top_indices = sorted_indices[:top_n]

#     top_keywords = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
#     return top_keywords


# # -----------------------------
# # Clustering (K-Means)
# # -----------------------------
# def cluster_documents_kmeans(docs, num_clusters=5):
#     """
#     Perform K-Means clustering on the final text docs (already processed).
#     Return cluster labels plus the cluster model and feature names.
#     """
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)

#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(X)

#     labels = kmeans.labels_
#     feature_names = vectorizer.get_feature_names_out()
#     return labels, kmeans, vectorizer, X


# def get_top_terms_per_cluster(kmeans, vectorizer, num_terms=10):
#     """
#     Return a dictionary of {cluster_id: [(term, weight), ...], ...}
#     capturing the top terms for each cluster centroid.
#     """
#     centroids = kmeans.cluster_centers_
#     feature_names = vectorizer.get_feature_names_out()
#     results = {}

#     for cluster_id, centroid in enumerate(centroids):
#         # Sort by centroid weight
#         sorted_indices = np.argsort(centroid)[::-1]
#         top_features = [(feature_names[i], centroid[i]) for i in sorted_indices[:num_terms]]
#         results[cluster_id] = top_features

#     return results


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     # -- 1) Dark green buttons
#     st.markdown("""
#     <style>
#     div.stButton > button {
#         background-color: #006400 !important;
#         color: white !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     # -- 2) Move sidebar content higher (reduce top padding)
#     st.markdown("""
#     <style>
#     section[data-testid="stSidebar"] .css-1d391kg {
#         padding-top: 1rem !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     st.title("News Analysis Extended (Stopwords, NER, Topic Modelling, Keyword Extraction, Clustering)")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     # RESET BUTTON
#     if st.sidebar.button("Reset Data & Analyses"):
#         was_valid = st.session_state.api_key_validated
#         st.session_state.clear()
#         st.session_state.api_key_validated = was_valid
#         st.experimental_rerun()

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input(
#         "Enter your NewsAPI key",
#         value="a2b4f531cea743a1b10d0aad86bd44f5",
#         type="password"
#     )

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Fetch News
#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))

#                     # Thorough cleaning for freq analysis & word clouds
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )

#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     # If no articles, do nothing further
#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # -------------------------------------------------------------
#     # TABS
#     # -------------------------------------------------------------
#     tab_stopwords, tab_ner, tab_topics, tab_keywords, tab_clustering = st.tabs([
#         "Stopwords & Advanced", "NER Tab", "Topic Modelling",
#         "Keyword Extraction", "Clustering & Classification"
#     ])

#     # -------------------------------------------------------------
#     # TAB 1: Stopwords & Advanced
#     # -------------------------------------------------------------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         st.write("""
#             We remove built-in English stopwords by default 
#             plus any custom words you specify below.
#         """)

#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())

#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#         else:
#             st.info("No custom stopwords yet.")

#         df_tab = df.copy()
#         def remove_all_stopwords(txt):
#             return apply_stopwords_union(txt, st.session_state.custom_stopwords)
#         df_tab['finalText'] = df_tab['cleanedText'].apply(remove_all_stopwords)

#         # Sentiment
#         df_tab['sentiment'] = df_tab['finalText'].apply(lambda txt: analyze_sentiment(txt))
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         # Word Frequency
#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         # Full freq table
#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         # Word Cloud
#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # ---------- Lemmas, Bigrams, Trigrams ----------
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")

#         df_advanced = df_tab.copy()

#         # Lemmas
#         st.markdown("### Lemmas")
#         st.write("""\
#         **Description**:  
#         Lemmas unify different forms of a word. e.g. 'running', 'ran' → 'run'.
#         """)
#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])

#         st.write("#### Lemma Word Cloud")
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         # Bigrams
#         st.markdown("### Bigrams")
#         st.write("""Pairs of consecutive tokens. e.g. 'machine learning' → 'machine_learning'.""")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])

#         st.write("#### Bigram Word Cloud")
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         # Trigrams
#         st.markdown("### Trigrams")
#         st.write("""Triplets of consecutive tokens. e.g. 'new york city' → 'new_york_city'.""")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])

#         st.write("#### Trigram Word Cloud")
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # -------------------------------------------------------------
#     # TAB 2: NER (Named Entity Recognition)
#     # -------------------------------------------------------------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         st.write("""
#         **Explanation**:
#         - We parse the **original** article text (title + description + content) to preserve case.
#         - Use spaCy to detect entities like PERSON, ORG, GPE, etc.
#         - We'll show a table and bar chart of the top entities in the corpus.
#         """)

#         entity_counts = {}
#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]

#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])

#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         # Bar chart
#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]

#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values(by='count', ascending=False)
#             st.bar_chart(chart_df.set_index('entity'))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")

#     # -------------------------------------------------------------
#     # TAB 3: Topic Modelling (LDA)
#     # -------------------------------------------------------------
#     with tab_topics:
#         st.subheader("Topic Modelling (LDA) + Interactive Networks")
#         st.write("""
#         **Discover major themes (topics)** in the stopword-removed text using Latent Dirichlet Allocation (LDA).
        
#         We also generate **interactive PyVis** graphs linking each topic to its top words.
#         """)

#         df_tab_for_topics = df.copy()
#         df_tab_for_topics['finalText'] = df_tab_for_topics['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     docs = df_tab_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )

#                     st.success("LDA Topic Modelling complete!")

#                     # Display the topics
#                     st.write("### Discovered Topics:")
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
                    
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                     # Interactive networks
#                     st.write("### Interactive Topic Networks")
#                     for i, topic in topics:
#                         st.subheader(f"Topic {i} Network")
#                         net = create_topic_pyvis_network(i, topic)
#                         display_pyvis_network(net, i)

#                 except Exception as ex:
#                     st.error(f"Error running LDA or building networks: {ex}")

#     # -------------------------------------------------------------
#     # TAB 4: Keyword Extraction
#     # -------------------------------------------------------------
#     with tab_keywords:
#         st.subheader("Keyword Extraction (TF-IDF)")
#         st.write("""
#         **Identify the most important words** across the entire corpus 
#         (already stopword-removed), using TF-IDF. We average TF-IDF scores 
#         per feature across all documents, then pick the top N.
#         """)

#         # Reuse the finalText from the stopwords tab
#         df_tab_for_keywords = df.copy()
#         df_tab_for_keywords['finalText'] = df_tab_for_keywords['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         topN_keywords = st.number_input("Number of Keywords to Extract", min_value=5, max_value=50, value=10, step=1)

#         if st.button("Run Keyword Extraction"):
#             with st.spinner("Extracting keywords..."):
#                 try:
#                     docs = df_tab_for_keywords['finalText'].tolist()
#                     top_keywords = extract_keywords_tfidf(docs, top_n=topN_keywords)

#                     st.success("Keyword extraction complete!")
#                     st.write("### Top Keywords (Across Entire Corpus by Avg TF-IDF)")
#                     # top_keywords is list of (term, score)
#                     # Sort descending by score
#                     # They may already be in descending order, but we'll confirm
#                     top_keywords_sorted = sorted(top_keywords, key=lambda x: x[1], reverse=True)
#                     df_kw = pd.DataFrame(top_keywords_sorted, columns=["Keyword", "TF-IDF Score"])
#                     st.dataframe(df_kw)

#                     # Optional: bar chart
#                     st.write("#### TF-IDF Bar Chart")
#                     if not df_kw.empty:
#                         chart_df = df_kw.set_index("Keyword")
#                         st.bar_chart(chart_df)

#                 except Exception as ex:
#                     st.error(f"Error extracting keywords: {ex}")

#     # -------------------------------------------------------------
#     # TAB 5: Clustering & Classification
#     # -------------------------------------------------------------
#     with tab_clustering:
#         st.subheader("Clustering & Classification (K-Means Demo)")
#         st.write("""
#         **Clustering**: We group articles by their content similarity 
#         (using TF-IDF vectors of the stopword-removed text).  
#         **Classification**: Typically requires labelled data, so here we provide a placeholder if you had labels.
#         """)

#         # Reuse finalText from the stopwords tab
#         df_tab_for_clustering = df.copy()
#         df_tab_for_clustering['finalText'] = df_tab_for_clustering['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         num_clusters = st.number_input("Number of Clusters (K-Means)", min_value=2, max_value=10, value=3, step=1)
#         show_top_cluster_terms = st.checkbox("Show top terms in each cluster?", value=True)

#         if st.button("Run Clustering"):
#             with st.spinner("Running K-Means clustering..."):
#                 try:
#                     docs = df_tab_for_clustering['finalText'].tolist()
#                     labels, kmeans_model, vectorizer, X = cluster_documents_kmeans(docs, num_clusters=num_clusters)

#                     st.success("K-Means Clustering complete!")
#                     # Show cluster assignments
#                     df_cluster = df_tab_for_clustering[['title', 'publication', 'finalText']].copy()
#                     df_cluster['cluster'] = labels
#                     st.write("### Documents & Their Assigned Clusters")
#                     st.dataframe(df_cluster)

#                     # Show top terms per cluster centroid if requested
#                     if show_top_cluster_terms:
#                         st.write("### Top Terms by Cluster Centroid")
#                         cluster_top_terms = get_top_terms_per_cluster(kmeans_model, vectorizer, num_terms=10)
#                         for cid, terms in cluster_top_terms.items():
#                             st.markdown(f"**Cluster {cid}**")
#                             top_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in terms])
#                             st.write(top_str)

#                 except Exception as ex:
#                     st.error(f"Error clustering: {ex}")

#         # Placeholder for classification
#         st.write("---")
#         st.write("## Classification (Placeholder)")
#         st.info("""
#         Typically, you need labelled data (e.g. categories for each article). 
#         Then you'd train a supervised model like Logistic Regression or Naive Bayes. 
#         Since we have no labels here, we cannot demonstrate classification directly.
#         """)


# if __name__ == "__main__":
#     main()














# import streamlit as st
# import colorsys
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # For interactive network visualisation
# from pyvis.network import Network
# import streamlit.components.v1 as components

# # For TF-IDF, clustering
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import numpy as np

# # For sentiment visualisation
# import plotly.express as px

# # Initialise spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """Given a series of strings, build a frequency dict {word: count, ...}."""
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords."""
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """Lemmatise text using spaCy, e.g. 'running' -> 'run'."""
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """Convert text into n-grams (bigrams if n=2, trigrams if n=3)."""
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     return [(ent.text, ent.label_) for ent in doc.ents]


# # -----------------------------
# # Topic Modelling (LDA)
# # -----------------------------
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     from gensim.corpora import Dictionary
#     from gensim.models.ldamodel import LdaModel

#     tokens_list = [doc.split() for doc in docs if doc.strip()]

#     dictionary = Dictionary(tokens_list)
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
#     return lda_model, corpus, dictionary


# def create_topic_pyvis_network(topic_id, topic_terms):
#     net = Network(height="600px", width="100%", directed=False)
#     net.set_options("""
#     var options = {
#       "nodes": {
#         "font": {
#           "size": 16,
#           "align": "center"
#         },
#         "shape": "circle"
#       },
#       "edges": {
#         "smooth": false,
#         "color": {
#           "inherit": false
#         }
#       },
#       "physics": {
#         "enabled": true,
#         "stabilization": {
#           "enabled": true,
#           "iterations": 100
#         }
#       },
#       "interaction": {
#         "dragNodes": true
#       }
#     }
#     """)

#     center_node_id = f"Topic_{topic_id}"
#     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")

#     for (term, weight) in topic_terms:
#         weight_val = float(weight)
#         size = 10 + (weight_val * 3000.0)
#         net.add_node(term, label=term, size=size, color="#1f77b4")
#         net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    
#     return net


# def display_pyvis_network(net, topic_id):
#     html_filename = f"topic_network_{topic_id}.html"
#     net.write_html(html_filename)
#     with open(html_filename, "r", encoding="utf-8") as f:
#         html_content = f.read()
#     components.html(html_content, height=600, scrolling=False)


# # -----------------------------
# # Keyword Extraction (TF-IDF)
# # -----------------------------
# def extract_keywords_tfidf(docs, top_n=10):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)

#     avg_tfidf = np.mean(X.toarray(), axis=0)
#     feature_names = vectorizer.get_feature_names_out()
#     sorted_indices = np.argsort(avg_tfidf)[::-1]
#     top_indices = sorted_indices[:top_n]
#     top_keywords = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
#     return top_keywords


# # -----------------------------
# # Clustering (K-Means)
# # -----------------------------
# def cluster_documents_kmeans(docs, num_clusters=5):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)

#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(X)

#     labels = kmeans.labels_
#     feature_names = vectorizer.get_feature_names_out()
#     return labels, kmeans, vectorizer, X


# def get_top_terms_per_cluster(kmeans, vectorizer, num_terms=10):
#     centroids = kmeans.cluster_centers_
#     feature_names = vectorizer.get_feature_names_out()
#     results = {}

#     for cluster_id, centroid in enumerate(centroids):
#         sorted_indices = np.argsort(centroid)[::-1]
#         top_features = [(feature_names[i], centroid[i]) for i in sorted_indices[:num_terms]]
#         results[cluster_id] = top_features
#     return results


# # -----------------------------
# # Co-occurrence Networks
# # -----------------------------
# def build_word_cooccurrence(docs, min_freq=2):
#     from collections import defaultdict
#     cooc = defaultdict(int)

#     for doc in docs:
#         tokens = set(doc.split())  # unique words
#         sorted_tokens = sorted(tokens)
#         n = len(sorted_tokens)
#         for i in range(n):
#             for j in range(i+1, n):
#                 w1 = sorted_tokens[i]
#                 w2 = sorted_tokens[j]
#                 cooc[(w1, w2)] += 1

#     cooc_filtered = {pair: count for pair, count in cooc.items() if count >= min_freq}
#     return cooc_filtered


# def build_entity_cooccurrence(df, min_freq=2):
#     from collections import defaultdict
#     cooc = defaultdict(int)

#     for idx, row in df.iterrows():
#         raw_title = row.title or ''
#         raw_desc = row.description or ''
#         raw_cont = row.content or ''
#         doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")

#         unique_ents = set([ent.text for ent in doc.ents])
#         sorted_ents = sorted(unique_ents)
#         n = len(sorted_ents)
#         for i in range(n):
#             for j in range(i+1, n):
#                 e1 = sorted_ents[i]
#                 e2 = sorted_ents[j]
#                 cooc[(e1, e2)] += 1

#     cooc_filtered = {pair: count for pair, count in cooc.items() if count >= min_freq}
#     return cooc_filtered


# def create_cooccurrence_pyvis_network(cooccurrence_dict, label_prefix=""):
#     net = Network(height="600px", width="100%", directed=False)
#     net.set_options("""
#     var options = {
#       "nodes": {
#         "font": {
#           "size": 16,
#           "align": "center"
#         },
#         "shape": "circle"
#       },
#       "edges": {
#         "smooth": false,
#         "color": {
#           "inherit": false
#         }
#       },
#       "physics": {
#         "enabled": true,
#         "stabilization": {
#           "enabled": true,
#           "iterations": 100
#         }
#       },
#       "interaction": {
#         "dragNodes": true
#       }
#     }
#     """)

#     for (item1, item2), freq in cooccurrence_dict.items():
#         if not net.get_node(item1):
#             net.add_node(item1, label=f"{label_prefix}{item1}")
#         if not net.get_node(item2):
#             net.add_node(item2, label=f"{label_prefix}{item2}")
#         net.add_edge(item1, item2, value=freq, title=f"Co-occur in {freq} docs")

#     return net


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     # Inject styling for dark green buttons, higher sidebar
#     st.markdown("""
#     <style>
#     div.stButton > button {
#         background-color: #006400 !important;
#         color: white !important;
#     }
#     section[data-testid="stSidebar"] .css-1d391kg {
#         padding-top: 1rem !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     st.title("News Analysis Extended (Fixes for Co-occurrence & Sentiment Scatter)")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     # RESET BUTTON
#     if st.sidebar.button("Reset Data & Analyses"):
#         was_valid = st.session_state.api_key_validated
#         st.session_state.clear()
#         st.session_state.api_key_validated = was_valid
#         st.experimental_rerun()

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input("Enter your NewsAPI key", value="a2b4f531cea743a1b10d0aad86bd44f5", type="password")

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Fetch News
#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )
#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # TABS
#     tab_stopwords, tab_ner, tab_topics, tab_keywords, tab_clustering, tab_sentviz, tab_network = st.tabs([
#         "Stopwords & Advanced", 
#         "NER Tab", 
#         "Topic Modelling",
#         "Keyword Extraction", 
#         "Clustering & Classification",
#         "Sentiment Visualisation",
#         "Co-occurrence Networks"
#     ])

#     # -------------------------------------------------------------
#     # TAB 1: Stopwords & Advanced
#     # -------------------------------------------------------------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())

#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#         else:
#             st.info("No custom stopwords yet.")

#         df_tab = df.copy()
#         df_tab['finalText'] = df_tab['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )
#         df_tab['sentiment'] = df_tab['finalText'].apply(analyze_sentiment)
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # Lemmas, Bigrams, Trigrams
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")
#         df_advanced = df_tab.copy()
#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)

#         st.markdown("### Lemmas")
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         st.markdown("### Bigrams")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         st.markdown("### Trigrams")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # -------------------------------------------------------------
#     # TAB 2: NER
#     # -------------------------------------------------------------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         entity_counts = {}
#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]
#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])
#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]
#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values(by='count', ascending=False)
#             st.bar_chart(chart_df.set_index('entity'))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")

#     # -------------------------------------------------------------
#     # TAB 3: Topic Modelling
#     # -------------------------------------------------------------
#     with tab_topics:
#         st.subheader("Topic Modelling (LDA) + Interactive Networks")
#         df_tab_for_topics = df.copy()
#         df_tab_for_topics['finalText'] = df_tab_for_topics['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     docs = df_tab_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )
#                     st.success("LDA Topic Modelling complete!")

#                     st.write("### Discovered Topics:")
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                     st.write("### Interactive Topic Networks")
#                     for i, topic in topics:
#                         st.subheader(f"Topic {i} Network")
#                         net = create_topic_pyvis_network(i, topic)
#                         display_pyvis_network(net, i)

#                 except Exception as ex:
#                     st.error(f"Error running LDA or building networks: {ex}")

#     # -------------------------------------------------------------
#     # TAB 4: Keyword Extraction (TF-IDF)
#     # -------------------------------------------------------------
#     with tab_keywords:
#         st.subheader("Keyword Extraction (TF-IDF)")
#         df_tab_for_keywords = df.copy()
#         df_tab_for_keywords['finalText'] = df_tab_for_keywords['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         topN_keywords = st.number_input("Number of Keywords to Extract", min_value=5, max_value=50, value=10, step=1)

#         if st.button("Run Keyword Extraction"):
#             with st.spinner("Extracting keywords..."):
#                 try:
#                     docs = df_tab_for_keywords['finalText'].tolist()
#                     if not docs:
#                         st.warning("No documents found to analyse.")
#                     else:
#                         top_keywords = extract_keywords_tfidf(docs, top_n=topN_keywords)
#                         st.success("Keyword extraction complete!")
#                         st.write("### Top Keywords (by TF-IDF)")
#                         top_keywords_sorted = sorted(top_keywords, key=lambda x: x[1], reverse=True)
#                         df_kw = pd.DataFrame(top_keywords_sorted, columns=["Keyword", "TF-IDF Score"])
#                         st.dataframe(df_kw)

#                         st.write("#### TF-IDF Bar Chart")
#                         if not df_kw.empty:
#                             chart_df = df_kw.set_index("Keyword")
#                             st.bar_chart(chart_df)

#                 except Exception as ex:
#                     st.error(f"Error extracting keywords: {ex}")

#     # -------------------------------------------------------------
#     # TAB 5: Clustering & Classification
#     # -------------------------------------------------------------
#     with tab_clustering:
#         st.subheader("Clustering & Classification (K-Means Demo)")
#         df_tab_for_clustering = df.copy()
#         df_tab_for_clustering['finalText'] = df_tab_for_clustering['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         num_clusters = st.number_input("Number of Clusters (K-Means)", min_value=2, max_value=10, value=3, step=1)
#         show_top_cluster_terms = st.checkbox("Show top terms in each cluster?", value=True)

#         if st.button("Run Clustering"):
#             with st.spinner("Running K-Means clustering..."):
#                 try:
#                     docs = df_tab_for_clustering['finalText'].tolist()
#                     if not docs:
#                         st.warning("No documents found to cluster.")
#                     else:
#                         labels, kmeans_model, vectorizer, X = cluster_documents_kmeans(docs, num_clusters=num_clusters)
#                         st.success("K-Means Clustering complete!")

#                         df_cluster = df_tab_for_clustering[['title', 'publication', 'finalText']].copy()
#                         df_cluster['cluster'] = labels
#                         st.write("### Documents & Their Assigned Clusters")
#                         st.dataframe(df_cluster)

#                         if show_top_cluster_terms:
#                             st.write("### Top Terms by Cluster Centroid")
#                             cluster_top_terms = get_top_terms_per_cluster(kmeans_model, vectorizer, num_terms=10)
#                             for cid, terms in cluster_top_terms.items():
#                                 st.markdown(f"**Cluster {cid}**")
#                                 top_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in terms])
#                                 st.write(top_str)

#                 except Exception as ex:
#                     st.error(f"Error clustering: {ex}")

#         st.write("---")
#         st.write("## Classification (Placeholder)")
#         st.info("""Typically, you'd need labelled data to train a supervised model. 
#                 This is just a placeholder for a future extension.""")

#     # -------------------------------------------------------------
#     # TAB 6: Sentiment Visualisation
#     # -------------------------------------------------------------


#     def compute_color_for_polarity(p):
#         """
#         Given a polarity p in [-1, 1], return a background colour (hex) that
#         interpolates from red (negative) to green (positive).
#         We skip p == 0 by returning None (meaning no highlight).
#         The absolute value |p| determines how strong the colour is.

#         For p>0, we blend from a light green to a darker green.
#         For p<0, we blend from a light red to a darker red.
#         """
#         if p == 0:
#             return None  # no highlight
#         # Clip just in case
#         p = max(-1, min(1, p))

#         # For negative: scale from p = -1 => darkest red, p = 0 => none
#         # For positive: scale from p = 0 => none, p = +1 => darkest green
#         # We'll define two separate scales:
#         # Negative = Red range, Positive = Green range

#         # Easiest is to treat them separately:
#         if p < 0:
#             # p in [-1,0)
#             # We map -1 => 1.0 intensity, 0 => 0.0 intensity
#             intensity = abs(p)  # 0..1
#             # We'll pick a base color around HSL(0°,100%,someLight)
#             # but let's do a simple mix: lighter red (#ffc9c9) to darker red (#ff0000)
#             # We'll do linear interpolation on each channel
#             # lighter red = (255,201,201), darker red = (255,0,0)
#             r1, g1, b1 = 255, 201, 201
#             r2, g2, b2 = 255,   0,   0
#             r = int(r1 + (r2-r1)*intensity)
#             g = int(g1 + (g2-g1)*intensity)
#             b = int(b1 + (b2-b1)*intensity)
#             return f"#{r:02x}{g:02x}{b:02x}"
#         else:
#             # p in (0,1]
#             intensity = p  # 0..1
#             # from lighter green (#c8f7c5) to darker (#00b000)
#             r1, g1, b1 = 200, 247, 197  # c8 f7 c5
#             r2, g2, b2 = 0,   176, 0
#             r = int(r1 + (r2-r1)*intensity)
#             g = int(g1 + (g2-g1)*intensity)
#             b = int(b1 + (b2-b1)*intensity)
#             return f"#{r:02x}{g:02x}{b:02x}"


#     def compute_color_for_subjectivity(s):
#         """
#         Given a subjectivity s in [0,1], return a background colour (hex)
#         that interpolates from none (s=0 => no highlight) to dark blue (s=1).
#         """
#         if s == 0:
#             return None
#         # s in (0,1]
#         # from a light blue (#d5f3fe) to a darker blue (#0077be)
#         r1, g1, b1 = 213, 243, 254  # d5 f3 fe
#         r2, g2, b2 = 0,   119, 190  # 00 77 be
#         r = int(r1 + (r2-r1)*s)
#         g = int(g1 + (g2-g1)*s)
#         b = int(b1 + (b2-b1)*s)
#         return f"#{r:02x}{g:02x}{b:02x}"


#     def highlight_word_polarity(word):
#         """
#         Compute the polarity of a single word using TextBlob.
#         If polarity != 0, return a <span> with background color scaled by polarity.
#         Otherwise, return a plain <span> with no highlight.
#         """
#         from textblob import TextBlob
#         p = TextBlob(word).sentiment.polarity
#         col = compute_color_for_polarity(p)
#         if col is None:
#             return f"<span style='margin:2px;'>{word}</span>"
#         else:
#             return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"


#     def highlight_word_subjectivity(word):
#         """
#         Compute the subjectivity of a single word using TextBlob.
#         If subjectivity != 0, return a <span> with background color scaled by subjectivity.
#         Otherwise, return a plain <span> with no highlight.
#         """
#         from textblob import TextBlob
#         s = TextBlob(word).sentiment.subjectivity
#         col = compute_color_for_subjectivity(s)
#         if col is None:
#             return f"<span style='margin:2px;'>{word}</span>"
#         else:
#             return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"


#     def highlight_text_polarity(full_text):
#         """
#         For each word in full_text, highlight by polarity intensity.
#         """
#         words = full_text.split()
#         highlighted = [highlight_word_polarity(w) for w in words]
#         return " ".join(highlighted)


#     def highlight_text_subjectivity(full_text):
#         """
#         For each word in full_text, highlight by subjectivity intensity.
#         """
#         words = full_text.split()
#         highlighted = [highlight_word_subjectivity(w) for w in words]
#         return " ".join(highlighted)

#     # Now, replace your "Sentiment Visualisation" tab block:
#     with tab_sentviz:
#         st.subheader("Per-Article Sentiment Explanation")

#         # Copy dataframe so we don't modify the original
#         df_tab_sent = df.copy()
#         # Create the finalText after removing stopwords
#         df_tab_sent['finalText'] = df_tab_sent['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         # Also compute polarity/subjectivity for the entire finalText
#         df_tab_sent['sentiment'] = df_tab_sent['finalText'].apply(analyze_sentiment)
#         df_tab_sent['polarity'] = df_tab_sent['sentiment'].apply(lambda s: s.polarity)
#         df_tab_sent['subjectivity'] = df_tab_sent['sentiment'].apply(lambda s: s.subjectivity)

#         # Let user pick an article index
#         st.write("Select an article to view its metadata and highlighted text sentiment.")
#         if df_tab_sent.empty:
#             st.warning("No articles to display. Please fetch some articles first.")
#         else:
#             # Provide an index-based dropdown
#             article_indices = df_tab_sent.index.tolist()
#             chosen_idx = st.selectbox("Choose article index:", article_indices)

#             row = df_tab_sent.loc[chosen_idx]

#             st.write("### Article Metadata")
#             details = {
#                 "Title": row.get('title', 'N/A'),
#                 "Publication": row.get('publication', 'N/A'),
#                 "Published": row.get('publishedAt', 'N/A'),
#                 "Polarity": round(row.get('polarity', 0), 3),
#                 "Subjectivity": round(row.get('subjectivity', 0), 3)
#             }
#             meta_df = pd.DataFrame([details])
#             st.table(meta_df)

#             final_text = row['finalText'] or ""

#             st.write("### Polarity Highlighter (Word-Level)")
#             pol_html = highlight_text_polarity(final_text)
#             st.markdown(pol_html, unsafe_allow_html=True)

#             st.write("### Subjectivity Highlighter (Word-Level)")
#             subj_html = highlight_text_subjectivity(final_text)
#             st.markdown(subj_html, unsafe_allow_html=True)


#     # -------------------------------------------------------------
#     # TAB 7: Co-occurrence Networks
#     # -------------------------------------------------------------


#     def create_cooccurrence_pyvis_network(cooccurrence_dict, label_prefix=""):
#         """
#         Build a PyVis network from a co-occurrence dict { (item1, item2): freq, ... }.
#         Each unique item gets a numeric ID (completely decoupled from the word),
#         so we never risk collisions like 'access' or 'title' with PyVis internals.
#         """
#         from pyvis.network import Network

#         net = Network(height="600px", width="100%", directed=False)
#         net.set_options("""
#         var options = {
#         "nodes": {
#             "font": {
#             "size": 16,
#             "align": "center"
#             },
#             "shape": "circle"
#         },
#         "edges": {
#             "smooth": false,
#             "color": {
#             "inherit": false
#             }
#         },
#         "physics": {
#             "enabled": true,
#             "stabilization": {
#             "enabled": true,
#             "iterations": 100
#             }
#         },
#         "interaction": {
#             "dragNodes": true
#         }
#         }
#         """)

#         # We'll store a map from the item (word/entity) -> integer ID
#         node_id_map = {}
#         next_id = 0  # We'll assign IDs 0,1,2,3,...

#         def get_node_id(item):
#             nonlocal next_id
#             if item not in node_id_map:
#                 node_id_map[item] = next_id
#                 next_id += 1
#             return node_id_map[item]

#         for (item1, item2), freq in cooccurrence_dict.items():
#             # Retrieve numeric IDs
#             id1 = get_node_id(item1)
#             id2 = get_node_id(item2)

#             # If the node is new, create it with the real text as label
#             if not net.get_node(id1):
#                 net.add_node(id1, label=f"{label_prefix}{item1}")
#             if not net.get_node(id2):
#                 net.add_node(id2, label=f"{label_prefix}{item2}")

#             # Add edge
#             net.add_edge(
#                 source=id1,
#                 to=id2,
#                 value=freq,
#                 title=f"Co-occurrence: {freq}"
#             )

#         return net


#     with tab_network:
#         st.subheader("Co-occurrence Network Analysis (Words or Entities)")
#         st.write("""
#         Build a larger co-occurrence network of words or named entities 
#         to visualise relationships among frequently co-occurring terms/people/places.
#         """)

#         cooc_type = st.radio("Choose Co-occurrence Type", ["Words", "Entities"])
#         min_freq = st.slider("Minimum co-occurrence frequency", min_value=1, max_value=10, value=2, step=1)

#         if st.button("Build Co-occurrence Network"):
#             with st.spinner("Building network..."):
#                 try:
#                     if cooc_type == "Words":
#                         # Stopword-removed text
#                         df_cooc = df.copy()
#                         df_cooc['finalText'] = df_cooc['cleanedText'].apply(
#                             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#                         )
#                         docs = df_cooc['finalText'].tolist()
#                         cooc_dict = build_word_cooccurrence(docs, min_freq=min_freq)

#                         if not cooc_dict:
#                             st.warning(f"No word pairs found with co-occurrence >= {min_freq}.")
#                         else:
#                             # Now we call the updated function:
#                             net = create_cooccurrence_pyvis_network(cooc_dict)
#                             st.info(f"Found {len(cooc_dict)} co-occurring word pairs (freq >= {min_freq}).")

#                             html_filename = f"word_cooc_{min_freq}.html"
#                             net.write_html(html_filename)
#                             with open(html_filename, "r", encoding="utf-8") as f:
#                                 html_content = f.read()
#                             components.html(html_content, height=600, scrolling=False)

#                     else:
#                         # Entities
#                         cooc_dict = build_entity_cooccurrence(df, min_freq=min_freq)
#                         if not cooc_dict:
#                             st.warning(f"No entity pairs found with co-occurrence >= {min_freq}.")
#                         else:
#                             net = create_cooccurrence_pyvis_network(cooc_dict)
#                             st.info(f"Found {len(cooc_dict)} co-occurring entity pairs (freq >= {min_freq}).")

#                             html_filename = f"entity_cooc_{min_freq}.html"
#                             net.write_html(html_filename)
#                             with open(html_filename, "r", encoding="utf-8") as f:
#                                 html_content = f.read()
#                             components.html(html_content, height=600, scrolling=False)

#                 except Exception as ex:
#                     st.error(f"Error building co-occurrence network: {ex}")


# if __name__ == "__main__":
#     main()





































# import streamlit as st
# import colorsys
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # For interactive network visualisation
# from pyvis.network import Network
# import streamlit.components.v1 as components

# # For TF-IDF, clustering
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.cluster import KMeans
# import numpy as np
# from sklearn.decomposition import LatentDirichletAllocation

# # For sentiment visualisation
# import plotly.express as px

# # Initialise spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """Given a series of strings, build a frequency dict {word: count, ...}."""
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords."""
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """Lemmatise text using spaCy, e.g. 'running' -> 'run'."""
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """Convert text into n-grams (bigrams if n=2, trigrams if n=3)."""
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     return [(ent.text, ent.label_) for ent in doc.ents]


# # -----------------------------
# # Topic Modelling (LDA)
# # -----------------------------
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     from gensim.corpora import Dictionary
#     from gensim.models.ldamodel import LdaModel

#     tokens_list = [doc.split() for doc in docs if doc.strip()]

#     dictionary = Dictionary(tokens_list)
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
#     return lda_model, corpus, dictionary


# def create_topic_pyvis_network(topic_id, topic_terms):
#     net = Network(height="600px", width="100%", directed=False)
#     net.set_options("""
#     var options = {
#       "nodes": {
#         "font": {
#           "size": 16,
#           "align": "center"
#         },
#         "shape": "circle"
#       },
#       "edges": {
#         "smooth": false,
#         "color": {
#           "inherit": false
#         }
#       },
#       "physics": {
#         "enabled": true,
#         "stabilization": {
#           "enabled": true,
#           "iterations": 100
#         }
#       },
#       "interaction": {
#         "dragNodes": true
#       }
#     }
#     """)

#     center_node_id = f"Topic_{topic_id}"
#     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")

#     for (term, weight) in topic_terms:
#         weight_val = float(weight)
#         size = 10 + (weight_val * 3000.0)
#         net.add_node(term, label=term, size=size, color="#1f77b4")
#         net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    
#     return net


# def display_pyvis_network(net, topic_id):
#     html_filename = f"topic_network_{topic_id}.html"
#     net.write_html(html_filename)
#     with open(html_filename, "r", encoding="utf-8") as f:
#         html_content = f.read()
#     components.html(html_content, height=600, scrolling=False)


# # -----------------------------
# # Keyword Extraction (TF-IDF)
# # -----------------------------
# def extract_keywords_tfidf(docs, top_n=10):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)

#     avg_tfidf = np.mean(X.toarray(), axis=0)
#     feature_names = vectorizer.get_feature_names_out()
#     sorted_indices = np.argsort(avg_tfidf)[::-1]
#     top_indices = sorted_indices[:top_n]
#     top_keywords = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
#     return top_keywords


# # -----------------------------
# # Clustering (K-Means)
# # -----------------------------
# def cluster_documents_kmeans(docs, num_clusters=5):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)

#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(X)

#     labels = kmeans.labels_
#     feature_names = vectorizer.get_feature_names_out()
#     return labels, kmeans, vectorizer, X


# def get_top_terms_per_cluster(kmeans, vectorizer, num_terms=10):
#     centroids = kmeans.cluster_centers_
#     feature_names = vectorizer.get_feature_names_out()
#     results = {}

#     for cluster_id, centroid in enumerate(centroids):
#         sorted_indices = np.argsort(centroid)[::-1]
#         top_features = [(feature_names[i], centroid[i]) for i in sorted_indices[:num_terms]]
#         results[cluster_id] = top_features
#     return results


# # -----------------------------
# # Co-occurrence Networks
# # -----------------------------
# def build_word_cooccurrence(docs, min_freq=2):
#     from collections import defaultdict
#     cooc = defaultdict(int)

#     for doc in docs:
#         tokens = set(doc.split())  # unique words
#         sorted_tokens = sorted(tokens)
#         n = len(sorted_tokens)
#         for i in range(n):
#             for j in range(i+1, n):
#                 w1 = sorted_tokens[i]
#                 w2 = sorted_tokens[j]
#                 cooc[(w1, w2)] += 1

#     cooc_filtered = {pair: count for pair, count in cooc.items() if count >= min_freq}
#     return cooc_filtered


# def build_entity_cooccurrence(df, min_freq=2):
#     from collections import defaultdict
#     cooc = defaultdict(int)

#     for idx, row in df.iterrows():
#         raw_title = row.title or ''
#         raw_desc = row.description or ''
#         raw_cont = row.content or ''
#         doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")

#         unique_ents = set([ent.text for ent in doc.ents])
#         sorted_ents = sorted(unique_ents)
#         n = len(sorted_ents)
#         for i in range(n):
#             for j in range(i+1, n):
#                 e1 = sorted_ents[i]
#                 e2 = sorted_ents[j]
#                 cooc[(e1, e2)] += 1

#     cooc_filtered = {pair: count for pair, count in cooc.items() if count >= min_freq}
#     return cooc_filtered


# def create_cooccurrence_pyvis_network(cooccurrence_dict, label_prefix=""):
#     net = Network(height="600px", width="100%", directed=False)
#     net.set_options("""
#     var options = {
#       "nodes": {
#         "font": {
#           "size": 16,
#           "align": "center"
#         },
#         "shape": "circle"
#       },
#       "edges": {
#         "smooth": false,
#         "color": {
#           "inherit": false
#         }
#       },
#       "physics": {
#         "enabled": true,
#         "stabilization": {
#           "enabled": true,
#           "iterations": 100
#         }
#       },
#       "interaction": {
#         "dragNodes": true
#       }
#     }
#     """)

#     for (item1, item2), freq in cooccurrence_dict.items():
#         if not net.get_node(item1):
#             net.add_node(item1, label=f"{label_prefix}{item1}")
#         if not net.get_node(item2):
#             net.add_node(item2, label=f"{label_prefix}{item2}")
#         net.add_edge(item1, item2, value=freq, title=f"Co-occur in {freq} docs")

#     return net


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     # Inject styling for dark green buttons, higher sidebar
#     st.markdown("""
#     <style>
#     div.stButton > button {
#         background-color: #006400 !important;
#         color: white !important;
#     }
#     section[data-testid="stSidebar"] .css-1d391kg {
#         padding-top: 1rem !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     st.title("News Analysis Extended (Fixes for Co-occurrence & Sentiment Scatter)")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     # RESET BUTTON
#     if st.sidebar.button("Reset Data & Analyses"):
#         was_valid = st.session_state.api_key_validated
#         st.session_state.clear()
#         st.session_state.api_key_validated = was_valid
#         st.experimental_rerun()

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input("Enter your NewsAPI key", value="a2b4f531cea743a1b10d0aad86bd44f5", type="password")

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Fetch News
#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )
#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # TABS
#     tab_stopwords, tab_ner, tab_topics, tab_keywords, tab_clustering, tab_sentviz, tab_network = st.tabs([
#         "Stopwords & Advanced", 
#         "NER Tab", 
#         "Topic Modelling",
#         "Keyword Extraction", 
#         "Clustering & Classification",
#         "Sentiment Visualisation",
#         "Detailed Topics & Clusters"
#     ])

#     # -------------------------------------------------------------
#     # TAB 1: Stopwords & Advanced
#     # -------------------------------------------------------------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())

#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#         else:
#             st.info("No custom stopwords yet.")

#         df_tab = df.copy()
#         df_tab['finalText'] = df_tab['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )
#         df_tab['sentiment'] = df_tab['finalText'].apply(analyze_sentiment)
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # Lemmas, Bigrams, Trigrams
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")
#         df_advanced = df_tab.copy()
#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)

#         st.markdown("### Lemmas")
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         st.markdown("### Bigrams")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         st.markdown("### Trigrams")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # -------------------------------------------------------------
#     # TAB 2: NER
#     # -------------------------------------------------------------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         entity_counts = {}
#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]
#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])
#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]
#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values(by='count', ascending=False)
#             st.bar_chart(chart_df.set_index('entity'))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")

#     # -------------------------------------------------------------
#     # TAB 3: Topic Modelling
#     # -------------------------------------------------------------
#     with tab_topics:
#         st.subheader("Topic Modelling (LDA) + Interactive Networks")
#         df_tab_for_topics = df.copy()
#         df_tab_for_topics['finalText'] = df_tab_for_topics['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     docs = df_tab_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )
#                     st.success("LDA Topic Modelling complete!")

#                     st.write("### Discovered Topics:")
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                     st.write("### Interactive Topic Networks")
#                     for i, topic in topics:
#                         st.subheader(f"Topic {i} Network")
#                         net = create_topic_pyvis_network(i, topic)
#                         display_pyvis_network(net, i)

#                 except Exception as ex:
#                     st.error(f"Error running LDA or building networks: {ex}")

#     # -------------------------------------------------------------
#     # TAB 4: Keyword Extraction (TF-IDF)
#     # -------------------------------------------------------------
#     with tab_keywords:
#         st.subheader("Keyword Extraction (TF-IDF)")
#         df_tab_for_keywords = df.copy()
#         df_tab_for_keywords['finalText'] = df_tab_for_keywords['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         topN_keywords = st.number_input("Number of Keywords to Extract", min_value=5, max_value=50, value=10, step=1)

#         if st.button("Run Keyword Extraction"):
#             with st.spinner("Extracting keywords..."):
#                 try:
#                     docs = df_tab_for_keywords['finalText'].tolist()
#                     if not docs:
#                         st.warning("No documents found to analyse.")
#                     else:
#                         top_keywords = extract_keywords_tfidf(docs, top_n=topN_keywords)
#                         st.success("Keyword extraction complete!")
#                         st.write("### Top Keywords (by TF-IDF)")
#                         top_keywords_sorted = sorted(top_keywords, key=lambda x: x[1], reverse=True)
#                         df_kw = pd.DataFrame(top_keywords_sorted, columns=["Keyword", "TF-IDF Score"])
#                         st.dataframe(df_kw)

#                         st.write("#### TF-IDF Bar Chart")
#                         if not df_kw.empty:
#                             chart_df = df_kw.set_index("Keyword")
#                             st.bar_chart(chart_df)

#                 except Exception as ex:
#                     st.error(f"Error extracting keywords: {ex}")

#     # -------------------------------------------------------------
#     # TAB 5: Clustering & Classification
#     # -------------------------------------------------------------
#     with tab_clustering:
#         st.subheader("Clustering & Classification (K-Means Demo)")
#         df_tab_for_clustering = df.copy()
#         df_tab_for_clustering['finalText'] = df_tab_for_clustering['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         num_clusters = st.number_input("Number of Clusters (K-Means)", min_value=2, max_value=10, value=3, step=1)
#         show_top_cluster_terms = st.checkbox("Show top terms in each cluster?", value=True)

#         if st.button("Run Clustering"):
#             with st.spinner("Running K-Means clustering..."):
#                 try:
#                     docs = df_tab_for_clustering['finalText'].tolist()
#                     if not docs:
#                         st.warning("No documents found to cluster.")
#                     else:
#                         labels, kmeans_model, vectorizer, X = cluster_documents_kmeans(docs, num_clusters=num_clusters)
#                         st.success("K-Means Clustering complete!")

#                         df_cluster = df_tab_for_clustering[['title', 'publication', 'finalText']].copy()
#                         df_cluster['cluster'] = labels
#                         st.write("### Documents & Their Assigned Clusters")
#                         st.dataframe(df_cluster)

#                         if show_top_cluster_terms:
#                             st.write("### Top Terms by Cluster Centroid")
#                             cluster_top_terms = get_top_terms_per_cluster(kmeans_model, vectorizer, num_terms=10)
#                             for cid, terms in cluster_top_terms.items():
#                                 st.markdown(f"**Cluster {cid}**")
#                                 top_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in terms])
#                                 st.write(top_str)

#                 except Exception as ex:
#                     st.error(f"Error clustering: {ex}")

#         st.write("---")
#         st.write("## Classification (Placeholder)")
#         st.info("""Typically, you'd need labelled data to train a supervised model. 
#                 This is just a placeholder for a future extension.""")

#     # -------------------------------------------------------------
#     # TAB 6: Sentiment Visualisation
#     # -------------------------------------------------------------


#     def compute_color_for_polarity(p):
#         """
#         Given a polarity p in [-1, 1], return a background colour (hex) that
#         interpolates from red (negative) to green (positive).
#         We skip p == 0 by returning None (meaning no highlight).
#         The absolute value |p| determines how strong the colour is.

#         For p>0, we blend from a light green to a darker green.
#         For p<0, we blend from a light red to a darker red.
#         """
#         if p == 0:
#             return None  # no highlight
#         # Clip just in case
#         p = max(-1, min(1, p))

#         # For negative: scale from p = -1 => darkest red, p = 0 => none
#         # For positive: scale from p = 0 => none, p = +1 => darkest green
#         # We'll define two separate scales:
#         # Negative = Red range, Positive = Green range

#         # Easiest is to treat them separately:
#         if p < 0:
#             # p in [-1,0)
#             # We map -1 => 1.0 intensity, 0 => 0.0 intensity
#             intensity = abs(p)  # 0..1
#             # We'll pick a base color around HSL(0°,100%,someLight)
#             # but let's do a simple mix: lighter red (#ffc9c9) to darker red (#ff0000)
#             # We'll do linear interpolation on each channel
#             # lighter red = (255,201,201), darker red = (255,0,0)
#             r1, g1, b1 = 255, 201, 201
#             r2, g2, b2 = 255,   0,   0
#             r = int(r1 + (r2-r1)*intensity)
#             g = int(g1 + (g2-g1)*intensity)
#             b = int(b1 + (b2-b1)*intensity)
#             return f"#{r:02x}{g:02x}{b:02x}"
#         else:
#             # p in (0,1]
#             intensity = p  # 0..1
#             # from lighter green (#c8f7c5) to darker (#00b000)
#             r1, g1, b1 = 200, 247, 197  # c8 f7 c5
#             r2, g2, b2 = 0,   176, 0
#             r = int(r1 + (r2-r1)*intensity)
#             g = int(g1 + (g2-g1)*intensity)
#             b = int(b1 + (b2-b1)*intensity)
#             return f"#{r:02x}{g:02x}{b:02x}"


#     def compute_color_for_subjectivity(s):
#         """
#         Given a subjectivity s in [0,1], return a background colour (hex)
#         that interpolates from none (s=0 => no highlight) to dark blue (s=1).
#         """
#         if s == 0:
#             return None
#         # s in (0,1]
#         # from a light blue (#d5f3fe) to a darker blue (#0077be)
#         r1, g1, b1 = 213, 243, 254  # d5 f3 fe
#         r2, g2, b2 = 0,   119, 190  # 00 77 be
#         r = int(r1 + (r2-r1)*s)
#         g = int(g1 + (g2-g1)*s)
#         b = int(b1 + (b2-b1)*s)
#         return f"#{r:02x}{g:02x}{b:02x}"


#     def highlight_word_polarity(word):
#         """
#         Compute the polarity of a single word using TextBlob.
#         If polarity != 0, return a <span> with background color scaled by polarity.
#         Otherwise, return a plain <span> with no highlight.
#         """
#         from textblob import TextBlob
#         p = TextBlob(word).sentiment.polarity
#         col = compute_color_for_polarity(p)
#         if col is None:
#             return f"<span style='margin:2px;'>{word}</span>"
#         else:
#             return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"


#     def highlight_word_subjectivity(word):
#         """
#         Compute the subjectivity of a single word using TextBlob.
#         If subjectivity != 0, return a <span> with background color scaled by subjectivity.
#         Otherwise, return a plain <span> with no highlight.
#         """
#         from textblob import TextBlob
#         s = TextBlob(word).sentiment.subjectivity
#         col = compute_color_for_subjectivity(s)
#         if col is None:
#             return f"<span style='margin:2px;'>{word}</span>"
#         else:
#             return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"


#     def highlight_text_polarity(full_text):
#         """
#         For each word in full_text, highlight by polarity intensity.
#         """
#         words = full_text.split()
#         highlighted = [highlight_word_polarity(w) for w in words]
#         return " ".join(highlighted)


#     def highlight_text_subjectivity(full_text):
#         """
#         For each word in full_text, highlight by subjectivity intensity.
#         """
#         words = full_text.split()
#         highlighted = [highlight_word_subjectivity(w) for w in words]
#         return " ".join(highlighted)

#     # Now, replace your "Sentiment Visualisation" tab block:
#     with tab_sentviz:
#         st.subheader("Per-Article Sentiment Explanation")

#         # Copy dataframe so we don't modify the original
#         df_tab_sent = df.copy()
#         # Create the finalText after removing stopwords
#         df_tab_sent['finalText'] = df_tab_sent['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         # Also compute polarity/subjectivity for the entire finalText
#         df_tab_sent['sentiment'] = df_tab_sent['finalText'].apply(analyze_sentiment)
#         df_tab_sent['polarity'] = df_tab_sent['sentiment'].apply(lambda s: s.polarity)
#         df_tab_sent['subjectivity'] = df_tab_sent['sentiment'].apply(lambda s: s.subjectivity)

#         # Let user pick an article index
#         st.write("Select an article to view its metadata and highlighted text sentiment.")
#         if df_tab_sent.empty:
#             st.warning("No articles to display. Please fetch some articles first.")
#         else:
#             # Provide an index-based dropdown
#             article_indices = df_tab_sent.index.tolist()
#             chosen_idx = st.selectbox("Choose article index:", article_indices)

#             row = df_tab_sent.loc[chosen_idx]

#             st.write("### Article Metadata")
#             details = {
#                 "Title": row.get('title', 'N/A'),
#                 "Publication": row.get('publication', 'N/A'),
#                 "Published": row.get('publishedAt', 'N/A'),
#                 "Polarity": round(row.get('polarity', 0), 3),
#                 "Subjectivity": round(row.get('subjectivity', 0), 3)
#             }
#             meta_df = pd.DataFrame([details])
#             st.table(meta_df)

#             final_text = row['finalText'] or ""

#             st.write("### Polarity Highlighter (Word-Level)")
#             pol_html = highlight_text_polarity(final_text)
#             st.markdown(pol_html, unsafe_allow_html=True)

#             st.write("### Subjectivity Highlighter (Word-Level)")
#             subj_html = highlight_text_subjectivity(final_text)
#             st.markdown(subj_html, unsafe_allow_html=True)


#     # -------------------------------------------------------------
#     # TAB 7: Co-occurrence Networks
#     # -------------------------------------------------------------


#     def display_top_words_for_lda(lda_model, feature_names, n_top_words=10):
#         """
#         Return a dict: {topic_index: [word1, word2, ...], ...} 
#         showing top words for each topic.
#         """
#         results = {}
#         for topic_idx, topic in enumerate(lda_model.components_):
#             # topic is an array of shape [n_features], with raw counts
#             top_indices = topic.argsort()[:-n_top_words - 1:-1]  # highest values first
#             top_words = [feature_names[i] for i in top_indices]
#             results[topic_idx] = top_words
#         return results

#     def run_sklearn_lda_topic_modelling(docs, n_topics=5, n_top_words=10, max_iter=10):
#         """
#         Use scikit-learn's LDA on a corpus of docs (list of strings).
#         We'll use CountVectorizer -> LatentDirichletAllocation.
#         Return the fitted LDA model, the doc-topic matrix, and the vectorizer feature names.
#         """
#         # Convert docs to a count matrix
#         vectorizer = CountVectorizer(stop_words=None)
#         X = vectorizer.fit_transform(docs)
#         feature_names = vectorizer.get_feature_names_out()

#         # Fit LDA
#         lda = LatentDirichletAllocation(
#             n_components=n_topics, 
#             max_iter=max_iter, 
#             random_state=42
#         )
#         lda.fit(X)

#         # doc_topic_matrix: shape [n_docs, n_topics]
#         doc_topic_matrix = lda.transform(X)
#         return lda, doc_topic_matrix, feature_names


#     def run_kmeans_clustering(docs, n_clusters=5, n_top_words=10):
#         """
#         Use K-Means on TF-IDF vectors of the docs.
#         Return the KMeans model, doc labels, feature names, and the TF-IDF matrix.
#         """
#         vectorizer = TfidfVectorizer(stop_words=None)
#         X = vectorizer.fit_transform(docs)
#         feature_names = vectorizer.get_feature_names_out()

#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         kmeans.fit(X)
#         labels = kmeans.labels_

#         return kmeans, labels, feature_names, X


#     def get_top_terms_per_cluster(kmeans_model, feature_names, n_top_words=10):
#         """
#         For each cluster centroid, find the top terms by magnitude.
#         Return a dict {cluster_id: [ (term, weight), ... ], ... }
#         """
#         centroids = kmeans_model.cluster_centers_
#         results = {}
#         for cluster_id, centroid_vector in enumerate(centroids):
#             top_indices = centroid_vector.argsort()[::-1][:n_top_words]
#             top_items = [(feature_names[i], centroid_vector[i]) for i in top_indices]
#             results[cluster_id] = top_items
#         return results


#     # ----------------------------------------------------------------------------
#     # Now the Streamlit tab (or page) for "Detailed Topics & Clusters".
#     # ----------------------------------------------------------------------------
#     # Suppose you have a tab: 
#     #   with tab_topics_clusters:
#     #       ...
#     # If you want to replace co-occurrence, rename accordingly.
#     # We'll show it as a new tab called "Detailed Topics & Clusters":

#     with tab_topics_clusters:
#         # Put your code for "Detailed Topic Discovery & Clustering" here
#         st.subheader("Detailed Topic Discovery & Clustering (scikit-learn)")


#         st.write("""
#         This tab demonstrates **two** methods:
#         1. **LDA (Latent Dirichlet Allocation)** for topic modelling, 
#         using **CountVectorizer**.
#         2. **K-Means Clustering** of the same text, using **TF-IDF**.
#         """)

#         # We'll reuse the same final text from your stopword cleaning step
#         # For example:
#         df_for_analysis = df.copy()
#         df_for_analysis['finalText'] = df_for_analysis['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         docs = df_for_analysis['finalText'].tolist()
#         if not docs:
#             st.warning("No documents available. Please fetch articles and ensure there's text.")
#         else:
#             # Let user pick parameters
#             num_topics = st.number_input("Number of LDA Topics", 2, 20, 5, 1)
#             top_words_lda = st.number_input("Top Words per Topic (LDA)", 5, 30, 10, 1)
#             lda_max_iter = st.slider("LDA Max Iterations", 5, 100, 10, 5)

#             num_clusters = st.number_input("Number of K-Means Clusters", 2, 20, 5, 1)
#             top_words_kmeans = st.number_input("Top Words per Cluster (K-Means)", 5, 30, 10, 1)

#             if st.button("Run Analysis"):
#                 with st.spinner("Performing LDA and K-Means..."):
#                     # 1) LDA
#                     lda_model, doc_topic_matrix, lda_feature_names = run_sklearn_lda_topic_modelling(
#                         docs, 
#                         n_topics=num_topics, 
#                         n_top_words=top_words_lda, 
#                         max_iter=lda_max_iter
#                     )

#                     # Show top words for each topic
#                     st.write("## LDA Topics & Top Words")
#                     topic_top_words = display_top_words_for_lda(lda_model, lda_feature_names, n_top_words=top_words_lda)
#                     for topic_idx, words in topic_top_words.items():
#                         st.markdown(f"**Topic {topic_idx}**: {', '.join(words)}")

#                     # Show doc-topic distribution (which topic each doc is strongest in)
#                     # doc_topic_matrix shape [n_docs, n_topics]
#                     # We'll find the topic with the highest probability for each doc
#                     dominant_topics = doc_topic_matrix.argmax(axis=1)
#                     # Add to DF
#                     df_for_analysis['LDA_Topic'] = dominant_topics
#                     st.write("### Document → Topic Mapping (LDA)")
#                     st.dataframe(df_for_analysis[['title','publication','finalText','LDA_Topic']])

#                     # 2) K-Means
#                     kmeans_model, labels, kmeans_feature_names, X = run_kmeans_clustering(
#                         docs, 
#                         n_clusters=num_clusters, 
#                         n_top_words=top_words_kmeans
#                     )

#                     df_for_analysis['KMeans_Cluster'] = labels
#                     st.write("## K-Means Clusters")
#                     st.write("### Document → Cluster Mapping")
#                     st.dataframe(df_for_analysis[['title','publication','finalText','KMeans_Cluster']])

#                     # Show top terms per cluster
#                     st.write("### Top Terms in Each K-Means Cluster")
#                     cluster_top_terms = get_top_terms_per_cluster(kmeans_model, kmeans_feature_names, n_top_words=top_words_kmeans)
#                     for cid, term_list in cluster_top_terms.items():
#                         # Each item is (term, weight)
#                         top_words_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in term_list])
#                         st.markdown(f"**Cluster {cid}**: {top_words_str}")

#                 st.success("Analysis complete!")

# if __name__ == "__main__":
#     main()



















# import streamlit as st
# import colorsys
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # For interactive network visualisation
# from pyvis.network import Network
# import streamlit.components.v1 as components

# # For TF-IDF, clustering
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.cluster import KMeans
# import numpy as np
# from sklearn.decomposition import LatentDirichletAllocation

# # For sentiment visualisation
# import plotly.express as px

# # Initialise spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """Given a series of strings, build a frequency dict {word: count, ...}."""
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords."""
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """Lemmatise text using spaCy, e.g. 'running' -> 'run'."""
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """Convert text into n-grams (bigrams if n=2, trigrams if n=3)."""
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     return [(ent.text, ent.label_) for ent in doc.ents]


# # -----------------------------
# # Topic Modelling (LDA)
# # -----------------------------
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     from gensim.corpora import Dictionary
#     from gensim.models.ldamodel import LdaModel

#     tokens_list = [doc.split() for doc in docs if doc.strip()]

#     dictionary = Dictionary(tokens_list)
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
#     return lda_model, corpus, dictionary


# def create_topic_pyvis_network(topic_id, topic_terms):
#     net = Network(height="600px", width="100%", directed=False)
#     net.set_options("""
#     var options = {
#       "nodes": {
#         "font": {
#           "size": 16,
#           "align": "center"
#         },
#         "shape": "circle"
#       },
#       "edges": {
#         "smooth": false,
#         "color": {
#           "inherit": false
#         }
#       },
#       "physics": {
#         "enabled": true,
#         "stabilization": {
#           "enabled": true,
#           "iterations": 100
#         }
#       },
#       "interaction": {
#         "dragNodes": true
#       }
#     }
#     """)

#     center_node_id = f"Topic_{topic_id}"
#     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")

#     for (term, weight) in topic_terms:
#         weight_val = float(weight)
#         size = 10 + (weight_val * 3000.0)
#         net.add_node(term, label=term, size=size, color="#1f77b4")
#         net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    
#     return net


# def display_pyvis_network(net, topic_id):
#     html_filename = f"topic_network_{topic_id}.html"
#     net.write_html(html_filename)
#     with open(html_filename, "r", encoding="utf-8") as f:
#         html_content = f.read()
#     components.html(html_content, height=600, scrolling=False)


# # -----------------------------
# # Keyword Extraction (TF-IDF)
# # -----------------------------
# def extract_keywords_tfidf(docs, top_n=10):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)

#     avg_tfidf = np.mean(X.toarray(), axis=0)
#     feature_names = vectorizer.get_feature_names_out()
#     sorted_indices = np.argsort(avg_tfidf)[::-1]
#     top_indices = sorted_indices[:top_n]
#     top_keywords = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
#     return top_keywords


# # -----------------------------
# # Clustering (K-Means)
# # -----------------------------
# def cluster_documents_kmeans(docs, num_clusters=5):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)

#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(X)

#     labels = kmeans.labels_
#     feature_names = vectorizer.get_feature_names_out()
#     return labels, kmeans, vectorizer, X


# def get_top_terms_per_cluster(kmeans, vectorizer, num_terms=10):
#     centroids = kmeans.cluster_centers_
#     feature_names = vectorizer.get_feature_names_out()
#     results = {}

#     for cluster_id, centroid in enumerate(centroids):
#         sorted_indices = np.argsort(centroid)[::-1]
#         top_features = [(feature_names[i], centroid[i]) for i in sorted_indices[:num_terms]]
#         results[cluster_id] = top_features
#     return results


# # (Co-occurrence definitions remain but are no longer used in any tab.)
# # def build_word_cooccurrence(...) etc.


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     # Inject styling for dark green buttons, higher sidebar
#     st.markdown("""
#     <style>
#     div.stButton > button {
#         background-color: #006400 !important;
#         color: white !important;
#     }
#     section[data-testid="stSidebar"] .css-1d391kg {
#         padding-top: 1rem !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     st.title("News Analysis Extended (Fixes for Co-occurrence & Sentiment Scatter)")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     # RESET BUTTON
#     if st.sidebar.button("Reset Data & Analyses"):
#         was_valid = st.session_state.api_key_validated
#         st.session_state.clear()
#         st.session_state.api_key_validated = was_valid
#         st.experimental_rerun()

#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input("Enter your NewsAPI key", value="a2b4f531cea743a1b10d0aad86bd44f5", type="password")

#     # Validate Key
#     if st.sidebar.button("Validate Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("API key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Fetch News
#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your API key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )
#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")

#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # TABS: EXACTLY SEVEN variables, matching your 7 labels
#     tab_stopwords, tab_ner, tab_topics, tab_keywords, tab_clustering, tab_sentviz, tab_topics_clusters = st.tabs([
#         "Stopwords & Advanced", 
#         "NER Tab", 
#         "Topic Modelling",
#         "Keyword Extraction", 
#         "Clustering & Classification",
#         "Sentiment Visualisation",
#         "Detailed Topics & Clusters"
#     ])

#     # -------------------------------------------------------------
#     # TAB 1: Stopwords & Advanced
#     # -------------------------------------------------------------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())

#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#         else:
#             st.info("No custom stopwords yet.")

#         df_tab = df.copy()
#         df_tab['finalText'] = df_tab['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )
#         df_tab['sentiment'] = df_tab['finalText'].apply(analyze_sentiment)
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # Lemmas, Bigrams, Trigrams
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")
#         df_advanced = df_tab.copy()
#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)

#         st.markdown("### Lemmas")
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         st.markdown("### Bigrams")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         st.markdown("### Trigrams")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # -------------------------------------------------------------
#     # TAB 2: NER
#     # -------------------------------------------------------------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         entity_counts = {}
#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]
#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])
#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]
#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values(by='count', ascending=False)
#             st.bar_chart(chart_df.set_index('entity'))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")

#     # -------------------------------------------------------------
#     # TAB 3: Topic Modelling
#     # -------------------------------------------------------------
#     with tab_topics:
#         st.subheader("Topic Modelling (LDA) + Interactive Networks")
#         df_tab_for_topics = df.copy()
#         df_tab_for_topics['finalText'] = df_tab_for_topics['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     docs = df_tab_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )
#                     st.success("LDA Topic Modelling complete!")

#                     st.write("### Discovered Topics:")
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                     st.write("### Interactive Topic Networks")
#                     for i, topic in topics:
#                         st.subheader(f"Topic {i} Network")
#                         net = create_topic_pyvis_network(i, topic)
#                         display_pyvis_network(net, i)

#                 except Exception as ex:
#                     st.error(f"Error running LDA or building networks: {ex}")

#     # -------------------------------------------------------------
#     # TAB 4: Keyword Extraction (TF-IDF)
#     # -------------------------------------------------------------
#     with tab_keywords:
#         st.subheader("Keyword Extraction (TF-IDF)")
#         df_tab_for_keywords = df.copy()
#         df_tab_for_keywords['finalText'] = df_tab_for_keywords['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         topN_keywords = st.number_input("Number of Keywords to Extract", min_value=5, max_value=50, value=10, step=1)

#         if st.button("Run Keyword Extraction"):
#             with st.spinner("Extracting keywords..."):
#                 try:
#                     docs = df_tab_for_keywords['finalText'].tolist()
#                     if not docs:
#                         st.warning("No documents found to analyse.")
#                     else:
#                         top_keywords = extract_keywords_tfidf(docs, top_n=topN_keywords)
#                         st.success("Keyword extraction complete!")
#                         st.write("### Top Keywords (by TF-IDF)")
#                         top_keywords_sorted = sorted(top_keywords, key=lambda x: x[1], reverse=True)
#                         df_kw = pd.DataFrame(top_keywords_sorted, columns=["Keyword", "TF-IDF Score"])
#                         st.dataframe(df_kw)

#                         st.write("#### TF-IDF Bar Chart")
#                         if not df_kw.empty:
#                             chart_df = df_kw.set_index("Keyword")
#                             st.bar_chart(chart_df)

#                 except Exception as ex:
#                     st.error(f"Error extracting keywords: {ex}")

#     # -------------------------------------------------------------
#     # TAB 5: Clustering & Classification
#     # -------------------------------------------------------------
#     with tab_clustering:
#         st.subheader("Clustering & Classification (K-Means Demo)")
#         df_tab_for_clustering = df.copy()
#         df_tab_for_clustering['finalText'] = df_tab_for_clustering['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         num_clusters = st.number_input("Number of Clusters (K-Means)", min_value=2, max_value=10, value=3, step=1)
#         show_top_cluster_terms = st.checkbox("Show top terms in each cluster?", value=True)

#         if st.button("Run Clustering"):
#             with st.spinner("Running K-Means clustering..."):
#                 try:
#                     docs = df_tab_for_clustering['finalText'].tolist()
#                     if not docs:
#                         st.warning("No documents found to cluster.")
#                     else:
#                         labels, kmeans_model, vectorizer, X = cluster_documents_kmeans(docs, num_clusters=num_clusters)
#                         st.success("K-Means Clustering complete!")

#                         df_cluster = df_tab_for_clustering[['title', 'publication', 'finalText']].copy()
#                         df_cluster['cluster'] = labels
#                         st.write("### Documents & Their Assigned Clusters")
#                         st.dataframe(df_cluster)

#                         if show_top_cluster_terms:
#                             st.write("### Top Terms by Cluster Centroid")
#                             cluster_top_terms = get_top_terms_per_cluster(kmeans_model, vectorizer, num_terms=10)
#                             for cid, terms in cluster_top_terms.items():
#                                 st.markdown(f"**Cluster {cid}**")
#                                 top_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in terms])
#                                 st.write(top_str)

#                 except Exception as ex:
#                     st.error(f"Error clustering: {ex}")

#         st.write("---")
#         st.write("## Classification (Placeholder)")
#         st.info("""Typically, you'd need labelled data to train a supervised model. 
#                 This is just a placeholder for a future extension.""")

#     # -------------------------------------------------------------
#     # TAB 6: Sentiment Visualisation
#     # -------------------------------------------------------------

#     def compute_color_for_polarity(p):
#         """
#         ...
#         (Same code you provided, unchanged)
#         """
#         if p == 0:
#             return None
#         p = max(-1, min(1, p))
#         if p < 0:
#             intensity = abs(p)
#             r1, g1, b1 = 255, 201, 201
#             r2, g2, b2 = 255,   0,   0
#             r = int(r1 + (r2-r1)*intensity)
#             g = int(g1 + (g2-g1)*intensity)
#             b = int(b1 + (b2-b1)*intensity)
#             return f"#{r:02x}{g:02x}{b:02x}"
#         else:
#             intensity = p
#             r1, g1, b1 = 200, 247, 197
#             r2, g2, b2 = 0,   176, 0
#             r = int(r1 + (r2-r1)*intensity)
#             g = int(g1 + (g2-g1)*intensity)
#             b = int(b1 + (b2-b1)*intensity)
#             return f"#{r:02x}{g:02x}{b:02x}"

#     def compute_color_for_subjectivity(s):
#         """
#         ...
#         (Same code you provided, unchanged)
#         """
#         if s == 0:
#             return None
#         r1, g1, b1 = 213, 243, 254
#         r2, g2, b2 = 0,   119, 190
#         r = int(r1 + (r2-r1)*s)
#         g = int(g1 + (g2-g1)*s)
#         b = int(b1 + (b2-b1)*s)
#         return f"#{r:02x}{g:02x}{b:02x}"

#     def highlight_word_polarity(word):
#         from textblob import TextBlob
#         p = TextBlob(word).sentiment.polarity
#         col = compute_color_for_polarity(p)
#         if col is None:
#             return f"<span style='margin:2px;'>{word}</span>"
#         else:
#             return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"

#     def highlight_word_subjectivity(word):
#         from textblob import TextBlob
#         s = TextBlob(word).sentiment.subjectivity
#         col = compute_color_for_subjectivity(s)
#         if col is None:
#             return f"<span style='margin:2px;'>{word}</span>"
#         else:
#             return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"

#     def highlight_text_polarity(full_text):
#         words = full_text.split()
#         highlighted = [highlight_word_polarity(w) for w in words]
#         return " ".join(highlighted)

#     def highlight_text_subjectivity(full_text):
#         words = full_text.split()
#         highlighted = [highlight_word_subjectivity(w) for w in words]
#         return " ".join(highlighted)

#     with tab_sentviz:
#         st.subheader("Per-Article Sentiment Explanation")
#         df_tab_sent = df.copy()
#         df_tab_sent['finalText'] = df_tab_sent['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         df_tab_sent['sentiment'] = df_tab_sent['finalText'].apply(analyze_sentiment)
#         df_tab_sent['polarity'] = df_tab_sent['sentiment'].apply(lambda s: s.polarity)
#         df_tab_sent['subjectivity'] = df_tab_sent['sentiment'].apply(lambda s: s.subjectivity)

#         if df_tab_sent.empty:
#             st.warning("No articles to display. Please fetch some articles first.")
#         else:
#             article_indices = df_tab_sent.index.tolist()
#             chosen_idx = st.selectbox("Choose article index:", article_indices)

#             row = df_tab_sent.loc[chosen_idx]

#             st.write("### Article Metadata")
#             details = {
#                 "Title": row.get('title', 'N/A'),
#                 "Publication": row.get('publication', 'N/A'),
#                 "Published": row.get('publishedAt', 'N/A'),
#                 "Polarity": round(row.get('polarity', 0), 3),
#                 "Subjectivity": round(row.get('subjectivity', 0), 3)
#             }
#             meta_df = pd.DataFrame([details])
#             st.table(meta_df)

#             final_text = row['finalText'] or ""

#             st.write("### Polarity Highlighter (Word-Level)")
#             pol_html = highlight_text_polarity(final_text)
#             st.markdown(pol_html, unsafe_allow_html=True)

#             st.write("### Subjectivity Highlighter (Word-Level)")
#             subj_html = highlight_text_subjectivity(final_text)
#             st.markdown(subj_html, unsafe_allow_html=True)

#     # -------------------------------------------------------------
#     # TAB 7: Detailed Topics & Clusters
#     # -------------------------------------------------------------

#     def display_top_words_for_lda(lda_model, feature_names, n_top_words=10):
#         results = {}
#         for topic_idx, topic in enumerate(lda_model.components_):
#             top_indices = topic.argsort()[:-n_top_words - 1:-1]
#             top_words = [feature_names[i] for i in top_indices]
#             results[topic_idx] = top_words
#         return results

#     def run_sklearn_lda_topic_modelling(docs, n_topics=5, n_top_words=10, max_iter=10):
#         vectorizer = CountVectorizer(stop_words=None)
#         X = vectorizer.fit_transform(docs)
#         feature_names = vectorizer.get_feature_names_out()

#         lda = LatentDirichletAllocation(
#             n_components=n_topics, 
#             max_iter=max_iter, 
#             random_state=42
#         )
#         lda.fit(X)
#         doc_topic_matrix = lda.transform(X)
#         return lda, doc_topic_matrix, feature_names

#     def run_kmeans_clustering(docs, n_clusters=5, n_top_words=10):
#         vectorizer = TfidfVectorizer(stop_words=None)
#         X = vectorizer.fit_transform(docs)
#         feature_names = vectorizer.get_feature_names_out()

#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         kmeans.fit(X)
#         labels = kmeans.labels_

#         return kmeans, labels, feature_names, X

#     def get_top_terms_per_cluster(kmeans_model, feature_names, n_top_words=10):
#         centroids = kmeans_model.cluster_centers_
#         results = {}
#         for cluster_id, centroid_vector in enumerate(centroids):
#             top_indices = centroid_vector.argsort()[::-1][:n_top_words]
#             top_items = [(feature_names[i], centroid_vector[i]) for i in top_indices]
#             results[cluster_id] = top_items
#         return results

#     with tab_topics_clusters:
#         st.subheader("Detailed Topic Discovery & Clustering (scikit-learn)")

#         st.write("""
#         This tab demonstrates **two** methods:
#         1. **LDA (Latent Dirichlet Allocation)** for topic modelling, 
#            using **CountVectorizer**.
#         2. **K-Means Clustering** of the same text, using **TF-IDF**.
#         """)

#         df_for_analysis = df.copy()
#         df_for_analysis['finalText'] = df_for_analysis['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         docs = df_for_analysis['finalText'].tolist()
#         if not docs:
#             st.warning("No documents available. Please fetch articles and ensure there's text.")
#         else:
#             num_topics = st.number_input("Number of LDA Topics", 2, 20, 5, 1)
#             top_words_lda = st.number_input("Top Words per Topic (LDA)", 5, 30, 10, 1)
#             lda_max_iter = st.slider("LDA Max Iterations", 5, 100, 10, 5)

#             num_clusters = st.number_input("Number of K-Means Clusters", 2, 20, 5, 1)
#             top_words_kmeans = st.number_input("Top Words per Cluster (K-Means)", 5, 30, 10, 1)

#             if st.button("Run Analysis"):
#                 with st.spinner("Performing LDA and K-Means..."):
#                     # 1) LDA
#                     lda_model, doc_topic_matrix, lda_feature_names = run_sklearn_lda_topic_modelling(
#                         docs, 
#                         n_topics=num_topics, 
#                         n_top_words=top_words_lda, 
#                         max_iter=lda_max_iter
#                     )

#                     st.write("## LDA Topics & Top Words")
#                     topic_top_words = display_top_words_for_lda(lda_model, lda_feature_names, n_top_words=top_words_lda)
#                     for topic_idx, words in topic_top_words.items():
#                         st.markdown(f"**Topic {topic_idx}**: {', '.join(words)}")

#                     dominant_topics = doc_topic_matrix.argmax(axis=1)
#                     df_for_analysis['LDA_Topic'] = dominant_topics
#                     st.write("### Document → Topic Mapping (LDA)")
#                     st.dataframe(df_for_analysis[['title','publication','finalText','LDA_Topic']])

#                     # 2) K-Means
#                     kmeans_model, labels, kmeans_feature_names, X = run_kmeans_clustering(
#                         docs, 
#                         n_clusters=num_clusters, 
#                         n_top_words=top_words_kmeans
#                     )

#                     df_for_analysis['KMeans_Cluster'] = labels
#                     st.write("## K-Means Clusters")
#                     st.write("### Document → Cluster Mapping")
#                     st.dataframe(df_for_analysis[['title','publication','finalText','KMeans_Cluster']])

#                     st.write("### Top Terms in Each K-Means Cluster")
#                     cluster_top_terms = get_top_terms_per_cluster(kmeans_model, kmeans_feature_names, n_top_words=top_words_kmeans)
#                     for cid, term_list in cluster_top_terms.items():
#                         top_words_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in term_list])
#                         st.markdown(f"**Cluster {cid}**: {top_words_str}")

#                 st.success("Analysis complete!")


# if __name__ == "__main__":
#     main()














# import streamlit as st
# import colorsys
# import requests
# import pandas as pd
# import re
# from textblob import TextBlob
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import spacy
# import nltk
# from nltk.util import ngrams  # for bigrams/trigrams

# # For topic modelling
# from gensim.corpora import Dictionary
# from gensim.models.ldamodel import LdaModel

# # For interactive network visualisation
# from pyvis.network import Network
# import streamlit.components.v1 as components

# # For TF-IDF, clustering
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.cluster import KMeans
# import numpy as np
# from sklearn.decomposition import LatentDirichletAllocation

# # For sentiment visualisation
# import plotly.express as px

# # For calling ChatGPT
# import openai

# # Initialise spaCy model (for NER and lemmatisation)
# nlp = spacy.load("en_core_web_sm")


# # -----------------------------
# # 1) Helper Functions
# # -----------------------------
# def validate_api_key(api_key):
#     """Ping NewsAPI with a small request to confirm if the key is valid."""
#     url = 'https://newsapi.org/v2/top-headlines'
#     params = {'country': 'us', 'pageSize': 1}
#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code == 200 and data.get('status') == 'ok':
#         return True
#     else:
#         msg = data.get('message', 'Unknown error')
#         raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


# def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
#     """Fetch articles from NewsAPI /v2/everything."""
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': query,
#         'language': language,
#         'sortBy': sort_by
#     }
#     if from_date:
#         params['from'] = from_date
#     if to_date:
#         params['to'] = to_date

#     headers = {'Authorization': f'Bearer {api_key}'}
#     resp = requests.get(url, headers=headers, params=params)
#     data = resp.json()

#     if resp.status_code != 200 or data.get('status') != 'ok':
#         raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
#     return data['articles']


# def clean_text_more_thoroughly(title, description, content):
#     """
#     Combine title/description/content, then:
#     - lowercase
#     - remove punctuation
#     - remove digits
#     - collapse multiple spaces
#     """
#     combined = f"{title or ''} {description or ''} {content or ''}".lower()
#     # remove digits
#     combined = re.sub(r'\d+', '', combined)
#     # remove punctuation
#     combined = re.sub(r'[^\w\s]', '', combined)
#     # collapse multiple spaces
#     combined = re.sub(r'\s+', ' ', combined)
#     # strip leading/trailing
#     combined = combined.strip()
#     return combined


# def analyze_sentiment(text):
#     """Use TextBlob sentiment (polarity between -1 and +1, subjectivity between 0 and 1)."""
#     blob = TextBlob(text)
#     return blob.sentiment  # NamedTuple(polarity, subjectivity)


# def compute_word_frequency(text_series):
#     """Given a series of strings, build a frequency dict {word: count, ...}."""
#     freq = {}
#     for txt in text_series:
#         for w in txt.split():
#             freq[w] = freq.get(w, 0) + 1
#     return freq


# def create_wordcloud(all_text, stopwords=None):
#     """
#     Generate a word cloud image from the given text.
#     If stopwords is None, we won't remove anything; if it's a set, we pass it to WordCloud.
#     """
#     if stopwords is None:
#         stopwords = set()
#     wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.imshow(wc, interpolation='bilinear')
#     ax.axis('off')
#     return fig


# def apply_stopwords_union(text, custom_stopwords):
#     """Remove words that are either in WordCloud's STOPWORDS or in custom_stopwords."""
#     combined_stopwords = STOPWORDS.union(custom_stopwords)
#     tokens = text.split()
#     filtered = [w for w in tokens if w not in combined_stopwords]
#     return ' '.join(filtered)


# # Lemmatisation & N-Grams
# def lemmatise_text_spacy(txt):
#     """Lemmatise text using spaCy, e.g. 'running' -> 'run'."""
#     doc = nlp(txt)
#     lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
#     return " ".join(lemmas)


# def generate_ngrams(txt, n=2):
#     """Convert text into n-grams (bigrams if n=2, trigrams if n=3)."""
#     tokens = txt.split()
#     ngram_tuples = list(ngrams(tokens, n))
#     ngram_strings = ["_".join(pair) for pair in ngram_tuples]
#     return " ".join(ngram_strings)


# def extract_entities_spacy(title, description, content):
#     """
#     Run spaCy NER on the *original* raw text (without lowercasing).
#     Returns a list of (ent_text, ent_label).
#     """
#     raw_text = f"{title or ''} {description or ''} {content or ''}"
#     doc = nlp(raw_text)
#     return [(ent.text, ent.label_) for ent in doc.ents]


# # -----------------------------
# # Topic Modelling (LDA)
# # -----------------------------
# def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
#     tokens_list = [doc.split() for doc in docs if doc.strip()]
#     dictionary = Dictionary(tokens_list)
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
#     return lda_model, corpus, dictionary


# def create_topic_pyvis_network(topic_id, topic_terms):
#     net = Network(height="600px", width="100%", directed=False)
#     net.set_options("""
#     var options = {
#       "nodes": {
#         "font": {
#           "size": 16,
#           "align": "center"
#         },
#         "shape": "circle"
#       },
#       "edges": {
#         "smooth": false,
#         "color": {
#           "inherit": false
#         }
#       },
#       "physics": {
#         "enabled": true,
#         "stabilization": {
#           "enabled": true,
#           "iterations": 100
#         }
#       },
#       "interaction": {
#         "dragNodes": true
#       }
#     }
#     """)

#     center_node_id = f"Topic_{topic_id}"
#     net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")

#     for (term, weight) in topic_terms:
#         weight_val = float(weight)
#         size = 10 + (weight_val * 3000.0)
#         net.add_node(term, label=term, size=size, color="#1f77b4")
#         net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    
#     return net


# def display_pyvis_network(net, topic_id):
#     html_filename = f"topic_network_{topic_id}.html"
#     net.write_html(html_filename)
#     with open(html_filename, "r", encoding="utf-8") as f:
#         html_content = f.read()
#     components.html(html_content, height=600, scrolling=False)


# # -----------------------------
# # Keyword Extraction (TF-IDF)
# # -----------------------------
# def extract_keywords_tfidf(docs, top_n=10):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)
#     avg_tfidf = np.mean(X.toarray(), axis=0)
#     feature_names = vectorizer.get_feature_names_out()
#     sorted_indices = np.argsort(avg_tfidf)[::-1]
#     top_indices = sorted_indices[:top_n]
#     top_keywords = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
#     return top_keywords


# # -----------------------------
# # Clustering (K-Means)
# # -----------------------------
# def cluster_documents_kmeans(docs, num_clusters=5):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(X)
#     labels = kmeans.labels_
#     feature_names = vectorizer.get_feature_names_out()
#     return labels, kmeans, vectorizer, X


# def get_top_terms_per_cluster(kmeans, vectorizer, num_terms=10):
#     centroids = kmeans.cluster_centers_
#     feature_names = vectorizer.get_feature_names_out()
#     results = {}
#     for cluster_id, centroid in enumerate(centroids):
#         sorted_indices = np.argsort(centroid)[::-1]
#         top_features = [(feature_names[i], centroid[i]) for i in sorted_indices[:num_terms]]
#         results[cluster_id] = top_features
#     return results


# # For sentiment visualisation
# import openai


# # -----------------------------
# # 2) Streamlit Main App
# # -----------------------------
# def main():
#     # Inject styling for dark green buttons, higher sidebar
#     st.markdown("""
#     <style>
#     div.stButton > button {
#         background-color: #006400 !important;
#         color: white !important;
#     }
#     section[data-testid="stSidebar"] .css-1d391kg {
#         padding-top: 1rem !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     st.title("News Analysis Extended (With ChatGPT LLM)")

#     # Session state
#     if 'api_key_validated' not in st.session_state:
#         st.session_state.api_key_validated = False
#     if 'articles_df' not in st.session_state:
#         st.session_state.articles_df = pd.DataFrame()
#     if 'custom_stopwords' not in st.session_state:
#         st.session_state.custom_stopwords = set()
#     if 'llm_api_key' not in st.session_state:
#         st.session_state.llm_api_key = None
#     if 'llm_key_validated' not in st.session_state:
#         st.session_state.llm_key_validated = False

#     # -------------------------
#     # SIDEBAR
#     # -------------------------
#     st.sidebar.markdown("# News Analysis\n---")

#     # RESET BUTTON
#     if st.sidebar.button("Reset Data & Analyses"):
#         was_valid_news = st.session_state.api_key_validated
#         was_valid_llm = st.session_state.llm_key_validated
#         st.session_state.clear()
#         st.session_state.api_key_validated = was_valid_news
#         st.session_state.llm_key_validated = was_valid_llm
#         st.experimental_rerun()

#     # -- NewsAPI Settings
#     st.sidebar.header("NewsAPI Settings")
#     api_key = st.sidebar.text_input("Enter your NewsAPI key", value="a2b4f531cea743a1b10d0aad86bd44f5", type="password")

#     if st.sidebar.button("Validate NewsAPI Key"):
#         if not api_key:
#             st.sidebar.error("Please provide a NewsAPI key.")
#         else:
#             with st.spinner("Validating NewsAPI key..."):
#                 try:
#                     validate_api_key(api_key)
#                     st.sidebar.success("NewsAPI key is valid!")
#                     st.session_state.api_key_validated = True
#                 except Exception as e:
#                     st.session_state.api_key_validated = False
#                     st.sidebar.error(f"Key invalid or error occurred: {e}")

#     st.sidebar.markdown("---")

#     # -- ChatGPT (LLM) Settings
#     st.sidebar.header("ChatGPT Settings")
#     llm_api_key = st.sidebar.text_input("Enter your ChatGPT API key", type="password")
#     if st.sidebar.button("Validate LLM Key"):
#         if not llm_api_key:
#             st.sidebar.error("Please provide a ChatGPT API key.")
#         else:
#             # We'll do a simple test call or set it in session state
#             # For now, assume it's valid if not empty
#             st.session_state.llm_api_key = llm_api_key
#             st.session_state.llm_key_validated = True
#             st.sidebar.success("LLM key stored. Ready to generate narratives!")

#     st.sidebar.markdown("---")

#     # Search Parameters
#     query = st.sidebar.text_input("Search Query", value="Python")
#     language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
#     sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

#     enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
#     if enable_date_filter:
#         from_date = st.sidebar.date_input("From Date")
#         to_date = st.sidebar.date_input("To Date")
#     else:
#         from_date = None
#         to_date = None

#     # Fetch News
#     if st.sidebar.button("Fetch News"):
#         if not api_key:
#             st.error("Please provide a NewsAPI key.")
#             return
#         if not st.session_state.api_key_validated:
#             st.error("Your NewsAPI key is not validated. Please validate it before fetching.")
#             return

#         from_date_str = from_date.isoformat() if from_date else None
#         to_date_str = to_date.isoformat() if to_date else None

#         with st.spinner("Fetching articles..."):
#             try:
#                 articles = fetch_articles(api_key, query, language, sort_by, from_date_str, to_date_str)
#                 if not articles:
#                     st.warning("No articles found. Try a different query or date range.")
#                 else:
#                     df = pd.DataFrame(articles)
#                     df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))
#                     df['cleanedText'] = df.apply(
#                         lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
#                         axis=1
#                     )
#                     st.session_state.articles_df = df
#                     st.success(f"Fetched {len(df)} articles.")
#             except Exception as e:
#                 st.error(f"Error fetching or processing results: {e}")

#     # If no articles, do nothing further
#     df = st.session_state.articles_df
#     if df.empty:
#         st.info("No articles fetched yet. Please fetch news to proceed.")
#         return

#     # TABS: we define EIGHT variables now, to incorporate the new "Narratives (LLM)" tab
#     tab_stopwords, tab_ner, tab_topics, tab_keywords, tab_clustering, tab_sentviz, tab_topics_clusters, tab_narratives = st.tabs([
#         "Stopwords & Advanced", 
#         "NER Tab", 
#         "Topic Modelling",
#         "Keyword Extraction", 
#         "Clustering & Classification",
#         "Sentiment Visualisation",
#         "Detailed Topics & Clusters",
#         "Narratives (LLM)"
#     ])

#     # -------------------------------------------------------------
#     # TAB 1: Stopwords & Advanced
#     # -------------------------------------------------------------
#     with tab_stopwords:
#         st.subheader("Stopwords: Manage Built-In & Custom")
#         new_word = st.text_input("Add a word to remove", key="new_word_tab1")
#         if st.button("Add Word to Remove", key="add_btn_tab1"):
#             if new_word.strip():
#                 st.session_state.custom_stopwords.add(new_word.strip().lower())

#         if st.session_state.custom_stopwords:
#             st.write("#### Currently Removed (Custom) Words")
#             remove_list = sorted(st.session_state.custom_stopwords)
#             for w in remove_list:
#                 col1, col2 = st.columns([4,1])
#                 col1.write(w)
#                 if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
#                     st.session_state.custom_stopwords.remove(w)
#         else:
#             st.info("No custom stopwords yet.")

#         df_tab = df.copy()
#         df_tab['finalText'] = df_tab['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )
#         df_tab['sentiment'] = df_tab['finalText'].apply(analyze_sentiment)
#         df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
#         df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

#         st.subheader("Articles Table")
#         st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

#         st.subheader("Top Words (Frequency)")
#         wordFreq = compute_word_frequency(df_tab['finalText'])
#         freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
#         topN = 50
#         top_words = freq_items[:topN]

#         if top_words:
#             words, counts = zip(*top_words)
#             freq_df = pd.DataFrame({'word': words, 'count': counts})
#             freq_df = freq_df.sort_values(by='count', ascending=False)
#             st.bar_chart(freq_df.set_index('word'))
#         else:
#             st.write("No words left in the corpus after removing all stopwords!")

#         st.subheader("Word Frequency Table")
#         freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
#         st.dataframe(freq_df_all)

#         st.subheader("Word Cloud")
#         all_text = ' '.join(df_tab['finalText'].tolist())
#         if all_text.strip():
#             fig = create_wordcloud(all_text, stopwords=set())
#             st.pyplot(fig)
#         else:
#             st.write("No text available for word cloud after removing all stopwords!")

#         # Lemmas, Bigrams, Trigrams
#         st.subheader("Advanced: Lemmas, Bigrams, Trigrams")
#         df_advanced = df_tab.copy()
#         df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)

#         st.markdown("### Lemmas")
#         lemma_freq = compute_word_frequency(df_advanced['lemmas'])
#         lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Lemma Frequency (Top 20)")
#         st.write(lemma_items[:20])
#         all_lemmas = " ".join(df_advanced['lemmas'])
#         fig_lem = create_wordcloud(all_lemmas, stopwords=set())
#         st.pyplot(fig_lem)

#         st.markdown("### Bigrams")
#         df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
#         bigram_freq = compute_word_frequency(df_advanced['bigrams'])
#         bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Bigram Frequency (Top 20)")
#         st.write(bigram_items[:20])
#         all_bigrams = " ".join(df_advanced['bigrams'])
#         fig_big = create_wordcloud(all_bigrams, stopwords=set())
#         st.pyplot(fig_big)

#         st.markdown("### Trigrams")
#         df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
#         trigram_freq = compute_word_frequency(df_advanced['trigrams'])
#         trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
#         st.write("#### Trigram Frequency (Top 20)")
#         st.write(trigram_items[:20])
#         all_trigrams = " ".join(df_advanced['trigrams'])
#         fig_tri = create_wordcloud(all_trigrams, stopwords=set())
#         st.pyplot(fig_tri)

#     # -------------------------------------------------------------
#     # TAB 2: NER
#     # -------------------------------------------------------------
#     with tab_ner:
#         st.subheader("Named Entity Recognition (NER)")
#         entity_counts = {}
#         for idx, row in df.iterrows():
#             raw_title = row.title or ''
#             raw_desc = row.description or ''
#             raw_cont = row.content or ''
#             doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
#             for ent in doc.ents:
#                 key = (ent.text, ent.label_)
#                 entity_counts[key] = entity_counts.get(key, 0) + 1

#         sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#         st.write(f"**Total unique entities found**: {len(sorted_entities)}")

#         topN_ents = 30
#         top_entities = sorted_entities[:topN_ents]
#         rows = []
#         for (text_label, count) in top_entities:
#             (ent_text, ent_label) = text_label
#             rows.append([ent_text, ent_label, count])
#         df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])
#         st.write("### Top Entities (Table)")
#         st.dataframe(df_ents)

#         combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
#         combined_counts = [c for t, c in top_entities]
#         st.write("### Top Entities (Bar Chart)")
#         if combined_keys:
#             chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
#             chart_df = chart_df.sort_values(by='count', ascending=False)
#             st.bar_chart(chart_df.set_index('entity'))
#         else:
#             st.info("No entities found to display.")

#         # Word Cloud of entity text
#         all_ents_text = []
#         for (ent_text, ent_label), count in top_entities:
#             ent_underscored = ent_text.replace(" ", "_")
#             all_ents_text.extend([ent_underscored] * count)

#         if all_ents_text:
#             st.write("### Word Cloud of Entity Text")
#             entity_text_big_string = " ".join(all_ents_text)
#             fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
#             st.pyplot(fig_ent_wc)
#         else:
#             st.info("No entity text available for word cloud.")

#     # -------------------------------------------------------------
#     # TAB 3: Topic Modelling
#     # -------------------------------------------------------------
#     with tab_topics:
#         st.subheader("Topic Modelling (LDA) + Interactive Networks")
#         df_tab_for_topics = df.copy()
#         df_tab_for_topics['finalText'] = df_tab_for_topics['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
#         num_words = st.number_input("Number of Top Words per Topic", min_value=5, max_value=30, value=10, step=1)
#         passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

#         if st.button("Run Topic Modelling"):
#             with st.spinner("Running LDA..."):
#                 try:
#                     docs = df_tab_for_topics['finalText'].tolist()
#                     lda_model, corpus, dictionary = run_lda_topic_model(
#                         docs, 
#                         num_topics=num_topics, 
#                         passes=passes, 
#                         num_words=num_words
#                     )
#                     st.success("LDA Topic Modelling complete!")

#                     st.write("### Discovered Topics:")
#                     topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
#                     for i, topic in topics:
#                         st.write(f"**Topic {i}**")
#                         topic_terms = [term for term, _ in topic]
#                         st.write(", ".join(topic_terms))
#                         st.write("---")

#                     st.write("### Interactive Topic Networks")
#                     for i, topic in topics:
#                         st.subheader(f"Topic {i} Network")
#                         net = create_topic_pyvis_network(i, topic)
#                         display_pyvis_network(net, i)

#                 except Exception as ex:
#                     st.error(f"Error running LDA or building networks: {ex}")

#     # -------------------------------------------------------------
#     # TAB 4: Keyword Extraction (TF-IDF)
#     # -------------------------------------------------------------
#     with tab_keywords:
#         st.subheader("Keyword Extraction (TF-IDF)")
#         df_tab_for_keywords = df.copy()
#         df_tab_for_keywords['finalText'] = df_tab_for_keywords['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         topN_keywords = st.number_input("Number of Keywords to Extract", min_value=5, max_value=50, value=10, step=1)

#         if st.button("Run Keyword Extraction"):
#             with st.spinner("Extracting keywords..."):
#                 try:
#                     docs = df_tab_for_keywords['finalText'].tolist()
#                     if not docs:
#                         st.warning("No documents found to analyse.")
#                     else:
#                         top_keywords = extract_keywords_tfidf(docs, top_n=topN_keywords)
#                         st.success("Keyword extraction complete!")
#                         st.write("### Top Keywords (by TF-IDF)")
#                         top_keywords_sorted = sorted(top_keywords, key=lambda x: x[1], reverse=True)
#                         df_kw = pd.DataFrame(top_keywords_sorted, columns=["Keyword", "TF-IDF Score"])
#                         st.dataframe(df_kw)

#                         st.write("#### TF-IDF Bar Chart")
#                         if not df_kw.empty:
#                             chart_df = df_kw.set_index("Keyword")
#                             st.bar_chart(chart_df)

#                 except Exception as ex:
#                     st.error(f"Error extracting keywords: {ex}")

#     # -------------------------------------------------------------
#     # TAB 5: Clustering & Classification
#     # -------------------------------------------------------------
#     with tab_clustering:
#         st.subheader("Clustering & Classification (K-Means Demo)")
#         df_tab_for_clustering = df.copy()
#         df_tab_for_clustering['finalText'] = df_tab_for_clustering['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         num_clusters = st.number_input("Number of Clusters (K-Means)", min_value=2, max_value=10, value=3, step=1)
#         show_top_cluster_terms = st.checkbox("Show top terms in each cluster?", value=True)

#         if st.button("Run Clustering"):
#             with st.spinner("Running K-Means clustering..."):
#                 try:
#                     docs = df_tab_for_clustering['finalText'].tolist()
#                     if not docs:
#                         st.warning("No documents found to cluster.")
#                     else:
#                         labels, kmeans_model, vectorizer, X = cluster_documents_kmeans(docs, num_clusters=num_clusters)
#                         st.success("K-Means Clustering complete!")

#                         df_cluster = df_tab_for_clustering[['title', 'publication', 'finalText']].copy()
#                         df_cluster['cluster'] = labels
#                         st.write("### Documents & Their Assigned Clusters")
#                         st.dataframe(df_cluster)

#                         if show_top_cluster_terms:
#                             st.write("### Top Terms by Cluster Centroid")
#                             cluster_top_terms = get_top_terms_per_cluster(kmeans_model, vectorizer, num_terms=10)
#                             for cid, terms in cluster_top_terms.items():
#                                 st.markdown(f"**Cluster {cid}**")
#                                 top_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in terms])
#                                 st.write(top_str)
#                 except Exception as ex:
#                     st.error(f"Error clustering: {ex}")

#         st.write("---")
#         st.write("## Classification (Placeholder)")
#         st.info("""Typically, you'd need labelled data to train a supervised model. 
#                 This is just a placeholder for a future extension.""")

#     # -------------------------------------------------------------
#     # TAB 6: Sentiment Visualisation
#     # -------------------------------------------------------------
#     def compute_color_for_polarity(p):
#         if p == 0:
#             return None
#         p = max(-1, min(1, p))
#         if p < 0:
#             intensity = abs(p)
#             r1, g1, b1 = 255, 201, 201
#             r2, g2, b2 = 255,   0,   0
#             r = int(r1 + (r2-r1)*intensity)
#             g = int(g1 + (g2-g1)*intensity)
#             b = int(b1 + (b2-b1)*intensity)
#             return f"#{r:02x}{g:02x}{b:02x}"
#         else:
#             intensity = p
#             r1, g1, b1 = 200, 247, 197
#             r2, g2, b2 = 0,   176, 0
#             r = int(r1 + (r2-r1)*intensity)
#             g = int(g1 + (g2-g1)*intensity)
#             b = int(b1 + (b2-b1)*intensity)
#             return f"#{r:02x}{g:02x}{b:02x}"

#     def compute_color_for_subjectivity(s):
#         if s == 0:
#             return None
#         r1, g1, b1 = 213, 243, 254
#         r2, g2, b2 = 0,   119, 190
#         r = int(r1 + (r2-r1)*s)
#         g = int(g1 + (g2-g1)*s)
#         b = int(b1 + (b2-b1)*s)
#         return f"#{r:02x}{g:02x}{b:02x}"

#     def highlight_word_polarity(word):
#         from textblob import TextBlob
#         p = TextBlob(word).sentiment.polarity
#         col = compute_color_for_polarity(p)
#         if col is None:
#             return f"<span style='margin:2px;'>{word}</span>"
#         else:
#             return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"

#     def highlight_word_subjectivity(word):
#         from textblob import TextBlob
#         s = TextBlob(word).sentiment.subjectivity
#         col = compute_color_for_subjectivity(s)
#         if col is None:
#             return f"<span style='margin:2px;'>{word}</span>"
#         else:
#             return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"

#     def highlight_text_polarity(full_text):
#         words = full_text.split()
#         highlighted = [highlight_word_polarity(w) for w in words]
#         return " ".join(highlighted)

#     def highlight_text_subjectivity(full_text):
#         words = full_text.split()
#         highlighted = [highlight_word_subjectivity(w) for w in words]
#         return " ".join(highlighted)

#     with tab_sentviz:
#         st.subheader("Per-Article Sentiment Explanation")
#         df_tab_sent = df.copy()
#         df_tab_sent['finalText'] = df_tab_sent['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         df_tab_sent['sentiment'] = df_tab_sent['finalText'].apply(analyze_sentiment)
#         df_tab_sent['polarity'] = df_tab_sent['sentiment'].apply(lambda s: s.polarity)
#         df_tab_sent['subjectivity'] = df_tab_sent['sentiment'].apply(lambda s: s.subjectivity)

#         if df_tab_sent.empty:
#             st.warning("No articles to display. Please fetch some articles first.")
#         else:
#             article_indices = df_tab_sent.index.tolist()
#             chosen_idx = st.selectbox("Choose article index:", article_indices)

#             row = df_tab_sent.loc[chosen_idx]

#             st.write("### Article Metadata")
#             details = {
#                 "Title": row.get('title', 'N/A'),
#                 "Publication": row.get('publication', 'N/A'),
#                 "Published": row.get('publishedAt', 'N/A'),
#                 "Polarity": round(row.get('polarity', 0), 3),
#                 "Subjectivity": round(row.get('subjectivity', 0), 3)
#             }
#             meta_df = pd.DataFrame([details])
#             st.table(meta_df)

#             final_text = row['finalText'] or ""

#             st.write("### Polarity Highlighter (Word-Level)")
#             pol_html = highlight_text_polarity(final_text)
#             st.markdown(pol_html, unsafe_allow_html=True)

#             st.write("### Subjectivity Highlighter (Word-Level)")
#             subj_html = highlight_text_subjectivity(final_text)
#             st.markdown(subj_html, unsafe_allow_html=True)

#     # -------------------------------------------------------------
#     # TAB 7: Detailed Topics & Clusters
#     # -------------------------------------------------------------
#     def display_top_words_for_lda(lda_model, feature_names, n_top_words=10):
#         results = {}
#         for topic_idx, topic in enumerate(lda_model.components_):
#             top_indices = topic.argsort()[:-n_top_words - 1:-1]
#             top_words = [feature_names[i] for i in top_indices]
#             results[topic_idx] = top_words
#         return results

#     def run_sklearn_lda_topic_modelling(docs, n_topics=5, n_top_words=10, max_iter=10):
#         vectorizer = CountVectorizer(stop_words=None)
#         X = vectorizer.fit_transform(docs)
#         feature_names = vectorizer.get_feature_names_out()

#         lda = LatentDirichletAllocation(
#             n_components=n_topics, 
#             max_iter=max_iter, 
#             random_state=42
#         )
#         lda.fit(X)
#         doc_topic_matrix = lda.transform(X)
#         return lda, doc_topic_matrix, feature_names

#     def run_kmeans_clustering(docs, n_clusters=5, n_top_words=10):
#         vectorizer = TfidfVectorizer(stop_words=None)
#         X = vectorizer.fit_transform(docs)
#         feature_names = vectorizer.get_feature_names_out()

#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         kmeans.fit(X)
#         labels = kmeans.labels_

#         return kmeans, labels, feature_names, X

#     def get_top_terms_per_cluster(kmeans_model, feature_names, n_top_words=10):
#         centroids = kmeans_model.cluster_centers_
#         results = {}
#         for cluster_id, centroid_vector in enumerate(centroids):
#             top_indices = centroid_vector.argsort()[::-1][:n_top_words]
#             top_items = [(feature_names[i], centroid_vector[i]) for i in top_indices]
#             results[cluster_id] = top_items
#         return results

#     with tab_topics_clusters:
#         st.subheader("Detailed Topic Discovery & Clustering (scikit-learn)")
#         st.write("""
#         This tab demonstrates **two** methods:
#         1. **LDA (Latent Dirichlet Allocation)** for topic modelling, 
#            using **CountVectorizer**.
#         2. **K-Means Clustering** of the same text, using **TF-IDF**.
#         """)

#         df_for_analysis = df.copy()
#         df_for_analysis['finalText'] = df_for_analysis['cleanedText'].apply(
#             lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#         )

#         docs = df_for_analysis['finalText'].tolist()
#         if not docs:
#             st.warning("No documents available. Please fetch articles and ensure there's text.")
#         else:
#             num_topics = st.number_input("Number of LDA Topics", 2, 20, 5, 1)
#             top_words_lda = st.number_input("Top Words per Topic (LDA)", 5, 30, 10, 1)
#             lda_max_iter = st.slider("LDA Max Iterations", 5, 100, 10, 5)

#             num_clusters = st.number_input("Number of K-Means Clusters", 2, 20, 5, 1)
#             top_words_kmeans = st.number_input("Top Words per Cluster (K-Means)", 5, 30, 10, 1)

#             if st.button("Run Analysis"):
#                 with st.spinner("Performing LDA and K-Means..."):
#                     # 1) LDA
#                     lda_model, doc_topic_matrix, lda_feature_names = run_sklearn_lda_topic_modelling(
#                         docs, 
#                         n_topics=num_topics, 
#                         n_top_words=top_words_lda, 
#                         max_iter=lda_max_iter
#                     )

#                     st.write("## LDA Topics & Top Words")
#                     topic_top_words = display_top_words_for_lda(lda_model, lda_feature_names, n_top_words=top_words_lda)
#                     for topic_idx, words in topic_top_words.items():
#                         st.markdown(f"**Topic {topic_idx}**: {', '.join(words)}")

#                     dominant_topics = doc_topic_matrix.argmax(axis=1)
#                     df_for_analysis['LDA_Topic'] = dominant_topics
#                     st.write("### Document → Topic Mapping (LDA)")
#                     st.dataframe(df_for_analysis[['title','publication','finalText','LDA_Topic']])

#                     # 2) K-Means
#                     kmeans_model, labels, kmeans_feature_names, X = run_kmeans_clustering(
#                         docs, 
#                         n_clusters=num_clusters, 
#                         n_top_words=top_words_kmeans
#                     )
#                     df_for_analysis['KMeans_Cluster'] = labels

#                     st.write("## K-Means Clusters")
#                     st.write("### Document → Cluster Mapping")
#                     st.dataframe(df_for_analysis[['title','publication','finalText','KMeans_Cluster']])

#                     st.write("### Top Terms in Each K-Means Cluster")
#                     cluster_top_terms = get_top_terms_per_cluster(kmeans_model, kmeans_feature_names, n_top_words=top_words_kmeans)
#                     for cid, term_list in cluster_top_terms.items():
#                         top_words_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in term_list])
#                         st.markdown(f"**Cluster {cid}**: {top_words_str}")

#                 st.success("Analysis complete!")

#     # -------------------------------------------------------------
#     # TAB 8: Narratives (LLM)
#     # -------------------------------------------------------------
#     import openai


#     with tab_narratives:
#         st.subheader("Narratives (LLM)")

#         st.write("""
#         This tab uses our existing articles and final cleaned text, 
#         then sends a prompt to GPT-4 to summarise key narratives 
#         across the corpus.
#         """)

#         df_llm = st.session_state.articles_df.copy()
#         if df_llm.empty:
#             st.warning("No articles found. Please fetch news first.")
#         else:
#             if not st.session_state.llm_key_validated or not st.session_state.llm_api_key:
#                 st.warning("Please provide and validate your ChatGPT API key in the sidebar.")
#             else:
#                 df_llm['finalText'] = df_llm['cleanedText'].apply(
#                     lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
#                 )

#                 max_articles = min(5, len(df_llm))
#                 snippet_text = ""
#                 for i in range(max_articles):
#                     row = df_llm.iloc[i]
#                     snippet_text += f"\n--- ARTICLE {i} ---\n"
#                     snippet_text += f"Title: {row.get('title','N/A')}\n"
#                     snippet_text += f"Publication: {row.get('publication','N/A')}\n"
#                     cleaned = row.get('finalText','')
#                     snippet_text += f"Text excerpt: {cleaned[:500]}...\n"

#                 st.write("### Example Data for LLM Prompt")
#                 st.text(snippet_text)

#                 if st.button("Generate Narratives"):
#                     with st.spinner("Contacting ChatGPT..."):
#                         try:
#                             openai.api_key = st.session_state.llm_api_key
                            
#                             system_msg = (
#                                 "You are a helpful data analyst. We have done some text processing and summarised articles below. "
#                                 "Please provide a cohesive narrative that highlights main topics or stories across these articles."
#                             )
#                             user_msg = f"ARTICLES:\n{snippet_text}\n\nINSTRUCTION:\nPlease give a concise narrative summary."

#                             # NEW usage in openai>=1.0.0
#                             response = openai.Chat.create(
#                                 model="gpt-4",
#                                 messages=[
#                                     {"role": "system", "content": system_msg},
#                                     {"role": "user", "content": user_msg}
#                                 ],
#                                 max_tokens=800,
#                                 temperature=0.7
#                             )
#                             result = response.choices[0].message.content

#                             st.write("### LLM Narrative Summary")
#                             st.write(result)

#                         except Exception as ex:
#                             st.error(f"LLM Error: {ex}")



# if __name__ == "__main__":
#     main()


















import streamlit as st
import colorsys
import requests
import pandas as pd
import re
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

import spacy
import nltk
from nltk.util import ngrams  # for bigrams/trigrams

# For topic modelling
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

# For interactive network visualisation
from pyvis.network import Network
import streamlit.components.v1 as components

# For TF-IDF, clustering, LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

# For ChatGPT calls
import openai

# Load spaCy
nlp = spacy.load("en_core_web_sm")


# 1) Helper Functions
def validate_api_key(api_key):
    url = 'https://newsapi.org/v2/top-headlines'
    params = {'country': 'us', 'pageSize': 1}
    headers = {'Authorization': f'Bearer {api_key}'}
    resp = requests.get(url, headers=headers, params=params)
    data = resp.json()

    if resp.status_code == 200 and data.get('status') == 'ok':
        return True
    else:
        msg = data.get('message', 'Unknown error')
        raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'language': language,
        'sortBy': sort_by
    }
    if from_date:
        params['from'] = from_date
    if to_date:
        params['to'] = to_date

    headers = {'Authorization': f'Bearer {api_key}'}
    resp = requests.get(url, headers=headers, params=params)
    data = resp.json()

    if resp.status_code != 200 or data.get('status') != 'ok':
        raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
    return data['articles']


def clean_text_more_thoroughly(title, description, content):
    combined = f"{title or ''} {description or ''} {content or ''}".lower()
    combined = re.sub(r'\d+', '', combined)
    combined = re.sub(r'[^\w\s]', '', combined)
    combined = re.sub(r'\s+', ' ', combined)
    combined = combined.strip()
    return combined


def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment


def compute_word_frequency(text_series):
    freq = {}
    for txt in text_series:
        for w in txt.split():
            freq[w] = freq.get(w, 0) + 1
    return freq


def create_wordcloud(all_text, stopwords=None):
    if stopwords is None:
        stopwords = set()
    wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig


def apply_stopwords_union(text, custom_stopwords):
    combined_stopwords = STOPWORDS.union(custom_stopwords)
    tokens = text.split()
    filtered = [w for w in tokens if w not in combined_stopwords]
    return ' '.join(filtered)


def lemmatise_text_spacy(txt):
    doc = nlp(txt)
    lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    return " ".join(lemmas)


def generate_ngrams(txt, n=2):
    tokens = txt.split()
    ngram_tuples = list(ngrams(tokens, n))
    ngram_strings = ["_".join(pair) for pair in ngram_tuples]
    return " ".join(ngram_strings)


def extract_entities_spacy(title, description, content):
    raw_text = f"{title or ''} {description or ''} {content or ''}"
    doc = nlp(raw_text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# Gensim LDA
def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
    tokens_list = [doc.split() for doc in docs if doc.strip()]
    dictionary = Dictionary(tokens_list)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
    return lda_model, corpus, dictionary


def create_topic_pyvis_network(topic_id, topic_terms):
    net = Network(height="600px", width="100%", directed=False)
    net.set_options("""
    var options = {
      "nodes": {
        "font": {"size": 16, "align": "center"},
        "shape": "circle"
      },
      "edges": {
        "smooth": false,
        "color": {"inherit": false}
      },
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "iterations": 100}
      },
      "interaction": {"dragNodes": true}
    }
    """)

    center_node_id = f"Topic_{topic_id}"
    net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")

    for (term, weight) in topic_terms:
        weight_val = float(weight)
        size = 10 + (weight_val * 3000.0)
        net.add_node(term, label=term, size=size, color="#1f77b4")
        net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    return net


def display_pyvis_network(net, topic_id):
    html_filename = f"topic_network_{topic_id}.html"
    net.write_html(html_filename)
    with open(html_filename, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=False)


def extract_keywords_tfidf(docs, top_n=10):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    avg_tfidf = np.mean(X.toarray(), axis=0)
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = np.argsort(avg_tfidf)[::-1]
    top_indices = sorted_indices[:top_n]
    top_keywords = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
    return top_keywords


# def cluster_documents_kmeans(docs, num_clusters=5):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(X)
#     labels = kmeans.labels_
#     # feature_names = vectorizer.get_feature_names_out()
#     # return labels, kmeans, vectorizer, X
#     return kmeans, vectorizer, X


def cluster_documents_kmeans(docs, num_clusters=5):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Return the model, the vectorizer object, and X if needed
    return kmeans, vectorizer, X, labels







# def get_top_terms_per_cluster(kmeans, vectorizer, n_top_words=10):
#     centroids = kmeans.cluster_centers_
#     feature_names = vectorizer.get_feature_names_out()
#     results = {}

#     for cluster_id, centroid in enumerate(centroids):
#         sorted_indices = np.argsort(centroid)[::-1]
#         top_features = [
#             (feature_names[i], centroid[i]) 
#             for i in sorted_indices[:n_top_words]
#         ]
#         results[cluster_id] = top_features
    
#     return results







def get_top_terms_per_cluster(kmeans, vectorizer, num_terms=10):
    centroids = kmeans.cluster_centers_
    # Now we call the vectorizer method to get real feature names
    feature_names = vectorizer.get_feature_names_out()

    results = {}
    for cluster_id, centroid in enumerate(centroids):
        sorted_indices = np.argsort(centroid)[::-1]
        top_features = [
            (feature_names[i], centroid[i])
            for i in sorted_indices[:num_terms]
        ]
        results[cluster_id] = top_features
    return results








# Additional scikit-based LDA code
def display_top_words_for_lda(lda_model, feature_names, n_top_words=10):
    results = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        results[topic_idx] = top_words
    return results

def run_sklearn_lda_topic_modelling(docs, n_topics=5, n_top_words=10, max_iter=10):
    vectorizer = CountVectorizer(stop_words=None)
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter, random_state=42)
    lda.fit(X)
    doc_topic_matrix = lda.transform(X)
    return lda, doc_topic_matrix, feature_names

def run_kmeans_clustering_sklearn(docs, n_clusters=5, n_top_words=10):
    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    return kmeans, labels, feature_names, X


# 2) Streamlit Main App
def main():
    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #006400 !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] .css-1d391kg {
        padding-top: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("News Analysis Extended + Detailed LLM Narratives")

    if 'api_key_validated' not in st.session_state:
        st.session_state.api_key_validated = False
    if 'articles_df' not in st.session_state:
        st.session_state.articles_df = pd.DataFrame()
    if 'custom_stopwords' not in st.session_state:
        st.session_state.custom_stopwords = set()

    # We'll track the LLM key as well
    if 'llm_api_key' not in st.session_state:
        st.session_state.llm_api_key = None
    if 'llm_key_validated' not in st.session_state:
        st.session_state.llm_key_validated = False

    # We'll also store the "Detailed Topics & Clusters" results so that
    # the new "Narratives" tab can access them.
    if 'df_for_thematic' not in st.session_state:
        st.session_state.df_for_thematic = pd.DataFrame()
    if 'lda_topic_top_words' not in st.session_state:
        st.session_state.lda_topic_top_words = {}
    if 'cluster_top_terms' not in st.session_state:
        st.session_state.cluster_top_terms = {}
    if 'topic_assignments' not in st.session_state:
        st.session_state.topic_assignments = []
    if 'cluster_assignments' not in st.session_state:
        st.session_state.cluster_assignments = []

    # SIDEBAR
    st.sidebar.markdown("# News Analysis\n---")

    if st.sidebar.button("Reset Data & Analyses"):
        st.session_state.clear()
        st.experimental_rerun()

    st.sidebar.header("NewsAPI Settings")
    news_api_key = st.sidebar.text_input("Enter your NewsAPI key", type="password", value="YOUR_NEWS_API_KEY_HERE")
    if st.sidebar.button("Validate NewsAPI Key"):
        if not news_api_key:
            st.sidebar.error("Please provide a NewsAPI key.")
        else:
            with st.spinner("Validating NewsAPI key..."):
                try:
                    validate_api_key(news_api_key)
                    st.session_state.api_key_validated = True
                    st.sidebar.success("NewsAPI key is valid!")
                except Exception as e:
                    st.session_state.api_key_validated = False
                    st.sidebar.error(f"Key invalid or error occurred: {e}")

    st.sidebar.markdown("---")

    # ChatGPT Key
    st.sidebar.header("LLM Settings (ChatGPT)")
    llm_key = st.sidebar.text_input("Enter your ChatGPT API key", type="password")
    if st.sidebar.button("Validate ChatGPT Key"):
        if not llm_key:
            st.sidebar.error("Please provide a ChatGPT API key.")
        else:
            st.session_state.llm_api_key = llm_key
            st.session_state.llm_key_validated = True
            st.sidebar.success("LLM key stored. Ready for narratives!")

    st.sidebar.markdown("---")

    # Search Parameters
    query = st.sidebar.text_input("Search Query", value="Python")
    language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
    sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])
    enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
    if enable_date_filter:
        from_date = st.sidebar.date_input("From Date")
        to_date = st.sidebar.date_input("To Date")
    else:
        from_date = None
        to_date = None

    # Fetch
    if st.sidebar.button("Fetch News"):
        if not news_api_key:
            st.error("Please provide a NewsAPI key.")
            return
        if not st.session_state.api_key_validated:
            st.error("Your NewsAPI key is not validated. Please validate it before fetching.")
            return
        from_date_str = from_date.isoformat() if from_date else None
        to_date_str = to_date.isoformat() if to_date else None
        with st.spinner("Fetching articles..."):
            try:
                articles = fetch_articles(news_api_key, query, language, sort_by, from_date_str, to_date_str)
                if not articles:
                    st.warning("No articles found. Try a different query or date range.")
                else:
                    df = pd.DataFrame(articles)
                    df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))
                    df['cleanedText'] = df.apply(
                        lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
                        axis=1
                    )
                    st.session_state.articles_df = df
                    st.success(f"Fetched {len(df)} articles.")
            except Exception as e:
                st.error(f"Error fetching or processing results: {e}")

    df = st.session_state.articles_df
    if df.empty:
        st.info("No articles fetched yet. Please fetch news to proceed.")
        return

    # TABS
    tab_stopwords, tab_ner, tab_topics, tab_keywords, tab_clustering, tab_sentviz, tab_topics_clusters, tab_narratives = st.tabs([
        "Stopwords & Advanced", 
        "NER Tab", 
        "Topic Modelling",
        "Keyword Extraction", 
        "Clustering & Classification",
        "Sentiment Visualisation",
        "Detailed Topics & Clusters",
        "Narratives (LLM)"
    ])

    # Tab 1: Stopwords & Advanced
    with tab_stopwords:
        st.subheader("Stopwords: Manage Built-In & Custom")
        new_word = st.text_input("Add a word to remove", key="new_word_tab1")
        if st.button("Add Word to Remove", key="add_btn_tab1"):
            if new_word.strip():
                st.session_state.custom_stopwords.add(new_word.strip().lower())

        if st.session_state.custom_stopwords:
            st.write("#### Currently Removed (Custom) Words")
            remove_list = sorted(st.session_state.custom_stopwords)
            for w in remove_list:
                col1, col2 = st.columns([4,1])
                col1.write(w)
                if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
                    st.session_state.custom_stopwords.remove(w)
        else:
            st.info("No custom stopwords yet.")

        df_tab = df.copy()
        df_tab['finalText'] = df_tab['cleanedText'].apply(lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords))
        df_tab['sentiment'] = df_tab['finalText'].apply(analyze_sentiment)
        df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
        df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

        st.subheader("Articles Table")
        st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

        st.subheader("Top Words (Frequency)")
        wordFreq = compute_word_frequency(df_tab['finalText'])
        freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
        topN = 50
        top_words = freq_items[:topN]

        if top_words:
            words, counts = zip(*top_words)
            freq_df = pd.DataFrame({'word': words, 'count': counts})
            freq_df = freq_df.sort_values(by='count', ascending=False)
            st.bar_chart(freq_df.set_index('word'))
        else:
            st.write("No words left after removing all stopwords!")

        st.subheader("Word Frequency Table")
        freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
        st.dataframe(freq_df_all)

        st.subheader("Word Cloud")
        all_text = ' '.join(df_tab['finalText'].tolist())
        if all_text.strip():
            fig = create_wordcloud(all_text, stopwords=set())
            st.pyplot(fig)
        else:
            st.write("No text available for word cloud after removing stopwords!")

        # Lemmas, Bigrams, Trigrams
        st.subheader("Advanced: Lemmas, Bigrams, Trigrams")
        df_advanced = df_tab.copy()
        df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)

        st.markdown("### Lemmas")
        lemma_freq = compute_word_frequency(df_advanced['lemmas'])
        lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
        st.write("#### Lemma Frequency (Top 20)")
        st.write(lemma_items[:20])
        all_lemmas = " ".join(df_advanced['lemmas'])
        fig_lem = create_wordcloud(all_lemmas, stopwords=set())
        st.pyplot(fig_lem)

        st.markdown("### Bigrams")
        df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
        bigram_freq = compute_word_frequency(df_advanced['bigrams'])
        bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
        st.write("#### Bigram Frequency (Top 20)")
        st.write(bigram_items[:20])
        all_bigrams = " ".join(df_advanced['bigrams'])
        fig_big = create_wordcloud(all_bigrams, stopwords=set())
        st.pyplot(fig_big)

        st.markdown("### Trigrams")
        df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
        trigram_freq = compute_word_frequency(df_advanced['trigrams'])
        trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
        st.write("#### Trigram Frequency (Top 20)")
        st.write(trigram_items[:20])
        all_trigrams = " ".join(df_advanced['trigrams'])
        fig_tri = create_wordcloud(all_trigrams, stopwords=set())
        st.pyplot(fig_tri)

    # Tab 2: NER
    with tab_ner:
        st.subheader("Named Entity Recognition (NER)")
        entity_counts = {}
        for idx, row in df.iterrows():
            raw_title = row.title or ''
            raw_desc = row.description or ''
            raw_cont = row.content or ''
            doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
            for ent in doc.ents:
                key = (ent.text, ent.label_)
                entity_counts[key] = entity_counts.get(key, 0) + 1

        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        st.write(f"**Total unique entities found**: {len(sorted_entities)}")

        topN_ents = 30
        top_entities = sorted_entities[:topN_ents]
        rows = []
        for (text_label, count) in top_entities:
            (ent_text, ent_label) = text_label
            rows.append([ent_text, ent_label, count])
        df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])
        st.write("### Top Entities (Table)")
        st.dataframe(df_ents)

        combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
        combined_counts = [c for t, c in top_entities]
        st.write("### Top Entities (Bar Chart)")
        if combined_keys:
            chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
            chart_df = chart_df.sort_values(by='count', ascending=False)
            st.bar_chart(chart_df.set_index('entity'))
        else:
            st.info("No entities found to display.")

        # Word Cloud of entity text
        all_ents_text = []
        for (ent_text, ent_label), count in top_entities:
            ent_underscored = ent_text.replace(" ", "_")
            all_ents_text.extend([ent_underscored] * count)

        if all_ents_text:
            st.write("### Word Cloud of Entity Text")
            entity_text_big_string = " ".join(all_ents_text)
            fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
            st.pyplot(fig_ent_wc)
        else:
            st.info("No entity text available for word cloud.")

    # Tab 3: Gensim Topic Modelling
    with tab_topics:
        st.subheader("Topic Modelling (Gensim LDA) + Interactive Networks")
        df_tab_for_topics = df.copy()
        df_tab_for_topics['finalText'] = df_tab_for_topics['cleanedText'].apply(
            lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
        )

        num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
        num_words = st.number_input("Number of Top Words per Topic", min_value=5, max_value=30, value=10, step=1)
        passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

        if st.button("Run Topic Modelling (Gensim)"):
            with st.spinner("Running LDA..."):
                try:
                    docs = df_tab_for_topics['finalText'].tolist()
                    lda_model, corpus, dictionary = run_lda_topic_model(
                        docs, 
                        num_topics=num_topics, 
                        passes=passes, 
                        num_words=num_words
                    )
                    st.success("Gensim LDA complete!")

                    st.write("### Discovered Topics:")
                    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
                    for i, topic in topics:
                        st.write(f"**Topic {i}**")
                        topic_terms = [term for term, _ in topic]
                        st.write(", ".join(topic_terms))
                        st.write("---")

                    st.write("### Interactive Topic Networks")
                    for i, topic in topics:
                        st.subheader(f"Topic {i} Network")
                        net = create_topic_pyvis_network(i, topic)
                        display_pyvis_network(net, i)
                except Exception as ex:
                    st.error(f"Error running Gensim LDA: {ex}")

    # Tab 4: Keyword Extraction (TF-IDF)
    with tab_keywords:
        st.subheader("Keyword Extraction (TF-IDF)")
        df_tab_for_keywords = df.copy()
        df_tab_for_keywords['finalText'] = df_tab_for_keywords['cleanedText'].apply(
            lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
        )

        topN_keywords = st.number_input("Number of Keywords to Extract", min_value=5, max_value=50, value=10, step=1)

        if st.button("Run Keyword Extraction"):
            with st.spinner("Extracting keywords..."):
                try:
                    docs = df_tab_for_keywords['finalText'].tolist()
                    if not docs:
                        st.warning("No documents found.")
                    else:
                        top_keywords = extract_keywords_tfidf(docs, top_n=topN_keywords)
                        st.success("Keyword extraction complete!")
                        st.write("### Top Keywords (by TF-IDF)")
                        top_keywords_sorted = sorted(top_keywords, key=lambda x: x[1], reverse=True)
                        df_kw = pd.DataFrame(top_keywords_sorted, columns=["Keyword", "TF-IDF Score"])
                        st.dataframe(df_kw)

                        st.write("#### TF-IDF Bar Chart")
                        if not df_kw.empty:
                            chart_df = df_kw.set_index("Keyword")
                            st.bar_chart(chart_df)
                except Exception as ex:
                    st.error(f"Error extracting keywords: {ex}")

    # Tab 5: Clustering & Classification
    with tab_clustering:
        st.subheader("Clustering & Classification (K-Means Demo)")
        df_tab_for_clustering = df.copy()
        df_tab_for_clustering['finalText'] = df_tab_for_clustering['cleanedText'].apply(
            lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
        )

        num_clusters = st.number_input("Number of Clusters (K-Means)", min_value=2, max_value=10, value=3, step=1)
        show_top_cluster_terms = st.checkbox("Show top terms in each cluster?", value=True)

        if st.button("Run Clustering"):
            with st.spinner("Running K-Means clustering..."):
                try:
                    docs = df_tab_for_clustering['finalText'].tolist()
                    if not docs:
                        st.warning("No documents found to cluster.")
                    else:
                        labels, kmeans_model, vectorizer, X = cluster_documents_kmeans(docs, num_clusters=num_clusters)
                        st.success("K-Means Clustering complete!")

                        df_cluster = df_tab_for_clustering[['title', 'publication', 'finalText']].copy()
                        df_cluster['cluster'] = labels
                        st.write("### Documents & Their Assigned Clusters")
                        st.dataframe(df_cluster)

                        if show_top_cluster_terms:
                            st.write("### Top Terms by Cluster Centroid")
                            cluster_top_terms = get_top_terms_per_cluster(kmeans_model, vectorizer, num_terms=10)
                            for cid, terms in cluster_top_terms.items():
                                st.markdown(f"**Cluster {cid}**")
                                top_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in terms])
                                st.write(top_str)
                except Exception as ex:
                    st.error(f"Error clustering: {ex}")

        st.write("---")
        st.write("## Classification (Placeholder)")
        st.info("""Typically, you'd need labelled data to train a supervised model. 
                This is just a placeholder for a future extension.""")

    # Tab 6: Sentiment Visualisation
    def compute_color_for_polarity(p):
        if p == 0:
            return None
        p = max(-1, min(1, p))
        if p < 0:
            intensity = abs(p)
            r1, g1, b1 = 255, 201, 201
            r2, g2, b2 = 255,   0,   0
            r = int(r1 + (r2-r1)*intensity)
            g = int(g1 + (g2-g1)*intensity)
            b = int(b1 + (b2-b1)*intensity)
            return f"#{r:02x}{g:02x}{b:02x}"
        else:
            intensity = p
            r1, g1, b1 = 200, 247, 197
            r2, g2, b2 = 0,   176, 0
            r = int(r1 + (r2-r1)*intensity)
            g = int(g1 + (g2-g1)*intensity)
            b = int(b1 + (b2-b1)*intensity)
            return f"#{r:02x}{g:02x}{b:02x}"

    def compute_color_for_subjectivity(s):
        if s == 0:
            return None
        r1, g1, b1 = 213, 243, 254
        r2, g2, b2 = 0,   119, 190
        r = int(r1 + (r2-r1)*s)
        g = int(g1 + (g2-g1)*s)
        b = int(b1 + (b2-b1)*s)
        return f"#{r:02x}{g:02x}{b:02x}"

    def highlight_word_polarity(word):
        from textblob import TextBlob
        p = TextBlob(word).sentiment.polarity
        col = compute_color_for_polarity(p)
        if col is None:
            return f"<span style='margin:2px;'>{word}</span>"
        else:
            return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"

    def highlight_word_subjectivity(word):
        from textblob import TextBlob
        s = TextBlob(word).sentiment.subjectivity
        col = compute_color_for_subjectivity(s)
        if col is None:
            return f"<span style='margin:2px;'>{word}</span>"
        else:
            return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"

    def highlight_text_polarity(full_text):
        words = full_text.split()
        highlighted = [highlight_word_polarity(w) for w in words]
        return " ".join(highlighted)

    def highlight_text_subjectivity(full_text):
        words = full_text.split()
        highlighted = [highlight_word_subjectivity(w) for w in words]
        return " ".join(highlighted)

    with tab_sentviz:
        st.subheader("Per-Article Sentiment Explanation")
        df_tab_sent = df.copy()
        df_tab_sent['finalText'] = df_tab_sent['cleanedText'].apply(
            lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
        )

        df_tab_sent['sentiment'] = df_tab_sent['finalText'].apply(analyze_sentiment)
        df_tab_sent['polarity'] = df_tab_sent['sentiment'].apply(lambda s: s.polarity)
        df_tab_sent['subjectivity'] = df_tab_sent['sentiment'].apply(lambda s: s.subjectivity)

        if df_tab_sent.empty:
            st.warning("No articles to display. Please fetch some articles first.")
        else:
            article_indices = df_tab_sent.index.tolist()
            chosen_idx = st.selectbox("Choose article index:", article_indices)

            row = df_tab_sent.loc[chosen_idx]

            st.write("### Article Metadata")
            details = {
                "Title": row.get('title', 'N/A'),
                "Publication": row.get('publication', 'N/A'),
                "Published": row.get('publishedAt', 'N/A'),
                "Polarity": round(row.get('polarity', 0), 3),
                "Subjectivity": round(row.get('subjectivity', 0), 3)
            }
            meta_df = pd.DataFrame([details])
            st.table(meta_df)

            final_text = row['finalText'] or ""

            st.write("### Polarity Highlighter (Word-Level)")
            pol_html = highlight_text_polarity(final_text)
            st.markdown(pol_html, unsafe_allow_html=True)

            st.write("### Subjectivity Highlighter (Word-Level)")
            subj_html = highlight_text_subjectivity(final_text)
            st.markdown(subj_html, unsafe_allow_html=True)

    # Tab 7: Detailed Topics & Clusters (scikit-learn approach)
    with tab_topics_clusters:
        st.subheader("Detailed Topic Discovery & Clustering (scikit-learn)")

        st.write("""
        This tab performs scikit-learn LDA and K-Means on the same final text, 
        storing the results so the next tab can build a structured LLM prompt.
        """)

        df_for_analysis = df.copy()
        df_for_analysis['finalText'] = df_for_analysis['cleanedText'].apply(
            lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
        )

        docs = df_for_analysis['finalText'].tolist()
        if not docs:
            st.warning("No documents available. Please fetch or clean your articles.")
        else:
            num_topics = st.number_input("Number of LDA Topics", 2, 20, 5, 1)
            top_words_lda = st.number_input("Top Words per Topic (LDA)", 5, 30, 10, 1)
            lda_max_iter = st.slider("LDA Max Iterations", 5, 100, 10, 5)

            num_clusters = st.number_input("Number of K-Means Clusters", 2, 20, 5, 1)
            top_words_kmeans = st.number_input("Top Words per Cluster (K-Means)", 5, 30, 10, 1)

            if st.button("Run Analysis (Store Results)"):
                with st.spinner("Performing LDA and K-Means..."):
                    # LDA
                    lda_model, doc_topic_matrix, lda_feature_names = run_sklearn_lda_topic_modelling(
                        docs, 
                        n_topics=num_topics, 
                        n_top_words=top_words_lda, 
                        max_iter=lda_max_iter
                    )
                    topic_top_words = display_top_words_for_lda(lda_model, lda_feature_names, n_top_words=top_words_lda)

                    # For each doc, find dominant topic
                    dominant_topics = doc_topic_matrix.argmax(axis=1)
                    df_for_analysis['LDA_Topic'] = dominant_topics

                    # K-Means
                    # kmeans_model, labels, kmeans_feature_names, X = run_kmeans_clustering_sklearn(
                    #     docs, 
                    #     n_clusters=num_clusters, 
                    #     n_top_words=top_words_kmeans
                    # )

                    kmeans_model, vectorizer, X, labels = cluster_documents_kmeans(
                        docs, 
                        num_clusters=3
                    )

                    df_for_analysis['KMeans_Cluster'] = labels

                    # cluster_top_terms = get_top_terms_per_cluster(kmeans_model, kmeans_feature_names, n_top_words=top_words_kmeans)
                    
                    cluster_top_terms = get_top_terms_per_cluster(kmeans_model, vectorizer, num_terms=10)

                    st.success("Analysis complete! Storing results for LLM usage...")

                    st.write("### LDA Topics & Top Words")
                    for topic_idx, words in topic_top_words.items():
                        st.markdown(f"**Topic {topic_idx}**: {', '.join(words)}")

                    st.write("### Document → Topic Mapping (LDA)")
                    st.dataframe(df_for_analysis[['title','publication','finalText','LDA_Topic']])

                    st.write("### K-Means Clusters")
                    st.write("#### Document → Cluster Mapping")
                    st.dataframe(df_for_analysis[['title','publication','finalText','KMeans_Cluster']])

                    st.write("#### Top Terms in Each K-Means Cluster")
                    for cid, term_list in cluster_top_terms.items():
                        top_words_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in term_list])
                        st.markdown(f"**Cluster {cid}**: {top_words_str}")

                    # Save to session_state
                    st.session_state.df_for_thematic = df_for_analysis
                    st.session_state.lda_topic_top_words = topic_top_words
                    st.session_state.cluster_top_terms = cluster_top_terms
                    st.session_state.topic_assignments = dominant_topics
                    st.session_state.cluster_assignments = labels


    # # Tab 8: Narratives (LLM)
    # with tab_narratives:
    #     st.subheader("Narratives (LLM) – Thematic Summaries")
    #     st.write("""
    #     This tab sends the raw data **plus** the thematical analyses (topics, clusters, key words) 
    #     to ChatGPT, asking for a structured explanation of key narratives.
    #     """)

    #     if st.session_state.df_for_thematic.empty:
    #         st.warning("No thematical results found. Please run 'Detailed Topics & Clusters' first.")
    #     else:
    #         df_them = st.session_state.df_for_thematic.copy()
    #         if not st.session_state.llm_key_validated or not st.session_state.llm_api_key:
    #             st.warning("Please provide and validate your ChatGPT API key in the sidebar.")
    #         else:
    #             # We'll gather some data from the thematical analysis
    #             # 1) LDA Topics & top words
    #             topic_info_str = "## LDA Topics:\n"
    #             for t_id, words in st.session_state.lda_topic_top_words.items():
    #                 topic_info_str += f"Topic {t_id}: {', '.join(words)}\n"

    #             # 2) K-Means cluster info
    #             cluster_info_str = "## K-Means Clusters:\n"
    #             for c_id, words in st.session_state.cluster_top_terms.items():
    #                 cluster_info_str += f"Cluster {c_id} top words: "
    #                 cluster_info_str += ", ".join([f"{w[0]}({w[1]:.3f})" for w in words])
    #                 cluster_info_str += "\n"

    #             # 3) Document-level assignments
    #             # We'll limit how many docs we pass for demonstration
    #             doc_samples = df_them.head(5)  # first 5 docs
    #             doc_info_str = "## Document Assignments:\n"
    #             for i, row in doc_samples.iterrows():
    #                 doc_info_str += f"\n--- Document idx={i} ---\n"
    #                 doc_info_str += f"Title: {row.get('title','N/A')}\n"
    #                 doc_info_str += f"LDA_Topic: {row.get('LDA_Topic','N/A')}, "
    #                 doc_info_str += f"KMeans_Cluster: {row.get('KMeans_Cluster','N/A')}\n"
    #                 # We'll pass partial text
    #                 text_excerpt = row.get('finalText','')[:300]  # short snippet
    #                 doc_info_str += f"Text excerpt: {text_excerpt}...\n"

    #             st.write("### Data We'll Send to the LLM")
    #             st.text(topic_info_str + "\n\n" + cluster_info_str + "\n\n" + doc_info_str)

    #             st.write("""
    #             Click below to produce a structured explanation of the main narratives 
    #             across these topics and clusters.
    #             """)

    #             if st.button("Generate LLM Narrative"):
    #                 with st.spinner("Asking ChatGPT..."):
    #                     try:
    #                         openai.api_key = st.session_state.llm_api_key
    #                         system_msg = (
    #                             "You are a helpful data analyst. "
    #                             "We have results from a topic/cluster analysis. "
    #                             "Please read the info below and produce a cohesive narrative. "
    #                             "Focus on describing the key themes or storylines that emerge."
    #                         )
    #                         user_msg = (
    #                             f"THEMATIC ANALYSIS DATA:\n\n{topic_info_str}\n\n{cluster_info_str}\n\n{doc_info_str}\n\n"
    #                             "INSTRUCTION:\nPlease consolidate these findings into a structured summary of the key narratives."
    #                         )

    #                         # old method (openai < 1.0.0)
    #                         response = openai.ChatCompletion.create(
    #                             model="gpt-4",
    #                             messages=[
    #                                 {"role": "system", "content": system_msg},
    #                                 {"role": "user", "content": user_msg}
    #                             ],
    #                             max_tokens=1000,
    #                             temperature=0.7
    #                         )
    #                         result = response["choices"][0]["message"]["content"]

    #                         st.write("### LLM Narrative Summary")
    #                         st.write(result)

    #                     except Exception as ex:
    #                         st.error(f"LLM Error: {ex}")

    # -------------------------------------------------------------
    # TAB 8: Narratives (LLM) – Thematic Summaries
    # -------------------------------------------------------------
    with tab_narratives:
        st.subheader("Narratives (LLM) – Thematic Summaries (Detailed)")

        st.write("""
        This tab sends the raw data **plus** the thematical analyses (topics, clusters, key words) 
        to ChatGPT, asking for a **more detailed** explanation of the key narratives.
        
        Additionally, you can **download** the entire LLM prompt in JSON format 
        if you want to inspect it or store it.
        """)

        if st.session_state.df_for_thematic.empty:
            st.warning("No thematical results found. Please run 'Detailed Topics & Clusters' first.")
        else:
            df_them = st.session_state.df_for_thematic.copy()
            # Check if LLM key is valid
            if not st.session_state.llm_key_validated or not st.session_state.llm_api_key:
                st.warning("Please provide and validate your ChatGPT API key in the sidebar.")
            else:
                # 1) LDA topic info
                topic_info_str = "## LDA Topics:\n"
                for t_id, words in st.session_state.lda_topic_top_words.items():
                    topic_info_str += f"Topic {t_id}: {', '.join(words)}\n"

                # 2) K-Means cluster info
                cluster_info_str = "## K-Means Clusters:\n"
                for c_id, words in st.session_state.cluster_top_terms.items():
                    cluster_info_str += f"Cluster {c_id} top words: "
                    cluster_info_str += ", ".join([f"{w[0]}({w[1]:.3f})" for w in words])
                    cluster_info_str += "\n"

                # 3) Document-level assignments (limit to 5 docs for brevity)
                doc_samples = df_them.head(5)
                doc_info_str = "## Document Assignments:\n"
                for i, row in doc_samples.iterrows():
                    doc_info_str += f"\n--- Document idx={i} ---\n"
                    doc_info_str += f"Title: {row.get('title','N/A')}\n"
                    doc_info_str += f"LDA_Topic: {row.get('LDA_Topic','N/A')}, "
                    doc_info_str += f"KMeans_Cluster: {row.get('KMeans_Cluster','N/A')}\n"
                    text_excerpt = row.get('finalText','')[:300]
                    doc_info_str += f"Text excerpt: {text_excerpt}...\n"

                # Show user the data
                st.write("### Data We'll Send to the LLM")
                st.text(topic_info_str + "\n\n" + cluster_info_str + "\n\n" + doc_info_str)

                # Build the final prompt
                system_msg = (
                    "You are a helpful data analyst. "
                    "We have results from a topic/cluster analysis. "
                    "Please read the info below and produce a cohesive narrative. "
                    "Focus on describing the key themes or storylines that emerge, and "
                    "provide a more thorough, in-depth analysis. Elaborate on how "
                    "these topics and clusters interrelate, discussing any nuanced "
                    "differences or patterns you see."
                )

                user_msg = (
                    f"THEMATIC ANALYSIS DATA:\n\n{topic_info_str}\n\n{cluster_info_str}\n\n{doc_info_str}\n\n"
                    "INSTRUCTION:\n"
                    "Please provide a **more detailed** analysis of the key narratives found in these topics/clusters. "
                    "Explain how the articles interrelate or differ, any major themes, perspectives, or storylines, "
                    "and any relevant patterns that stand out. "
                    "Feel free to be as comprehensive as possible."
                )

                # For the download button, we'll package these messages into JSON
                import json
                prompt_payload = {
                    "system": system_msg,
                    "user": user_msg
                }
                prompt_json_str = json.dumps(prompt_payload, indent=2)

                st.download_button(
                    label="Download LLM Prompt as JSON",
                    data=prompt_json_str,
                    file_name="llm_prompt.json",
                    mime="application/json"
                )

                st.write("""
                Press the button below to send the above data to ChatGPT, 
                requesting a more **detailed** thematic analysis.
                """)

                if st.button("Generate Detailed LLM Narrative"):
                    with st.spinner("Asking ChatGPT..."):
                        try:
                            openai.api_key = st.session_state.llm_api_key
                            # Old method (for openai<1.0.0):
                            response = openai.ChatCompletion.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": system_msg},
                                    {"role": "user", "content": user_msg}
                                ],
                                max_tokens=1500,
                                temperature=0.7
                            )
                            result = response["choices"][0]["message"]["content"]

                            st.write("### LLM Narrative Summary (Detailed)")
                            st.write(result)

                        except Exception as ex:
                            st.error(f"LLM Error: {ex}")





















if __name__ == "__main__":
    main()
























