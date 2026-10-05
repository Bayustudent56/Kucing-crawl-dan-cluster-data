import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --- SETTINGS & STOPWORDS ---
nltk.download('stopwords')
list_stopwords = stopwords.words('indonesian')
list_stopwords_en = stopwords.words('english')
list_stopwords.extend(list_stopwords_en)
list_stopwords.extend(['ya', 'yg', 'ga', 'yuk', 'dah', 'ngga', 'engga', 'ygy', 'the', 'of', 'in', 'and', 'to', 'for', 'is'])
stopwords_custom = list_stopwords

# --- HELPER FUNCTIONS ---
def generate_wordcloud(data):
    data_text = ' '.join(data['full_text'].astype(str).tolist())
    wc = WordCloud(stopwords=stopwords_custom, background_color='black', max_words=500, width=800, height=400).generate(data_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def plot_top_words(data):
    text = ' '.join(data['full_text'].astype(str).tolist())
    words = [word for word in text.split() if word not in stopwords_custom]
    word_counts = Counter(words)
    top_words = word_counts.most_common(12)
    
    if top_words:
        words_list, counts = zip(*top_words)
        colors = plt.cm.Paired(range(len(words_list)))
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(words_list, counts, color=colors)
        ax.set_xlabel('Kata')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Kata yang Sering Muncul')
        plt.xticks(rotation=45)
        for bar, num in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, num + 0.1, str(num), fontsize=10, ha='center')
        st.pyplot(fig)

def analyze_token_sentiment(docx, pipeline_model):
    pos_list, neg_list, neu_list = [], [], []
    for word in docx.split():
        prediction = pipeline_model.predict([word])[0]
        if hasattr(pipeline_model, 'predict_proba'):
            proba = pipeline_model.predict_proba([word])
            proba_max = float(proba.max())
        else:
            proba_max = None
            
        if prediction == 'positive':
            pos_list.append([word, proba_max])
        elif prediction == 'negative':
            neg_list.append([word, proba_max])
        else:
            neu_list.append(word)
            
    return {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}

# Load Model Logistic Regression
@st.cache_resource
def load_pipeline():
    return joblib.load('model_lr.joblib')

pipeline = load_pipeline()

# --- MAIN APP ---
def main():
    st.title("Sentiment Analysis NLP App")
    st.subheader("Streamlit Projects")
    
    menu = ["Home", "Dataset & Analysis", "KMeans Clustering"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # 1. MENU HOME
    if choice == "Home":
        st.subheader("Home")
        with st.form("nlpForm"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')
            
        col1, col2 = st.columns(2)
        if submit_button and raw_text:
            with col1:
                st.info("Results")
                sentiment = pipeline.predict([raw_text])[0]
                st.write(f"Predicted Sentiment:")
                if sentiment == "positive":
                    st.markdown("# Positive 😃")
                elif sentiment == "negative":
                    st.markdown("# Negative 😡")
                else:
                    st.markdown("# Neutral 😐")
            with col2:
                st.info("Token Sentiment")
                token_sentiments = analyze_token_sentiment(raw_text, pipeline)
                st.write(token_sentiments)

    # 2. MENU DATASET & ANALYSIS
    elif choice == "Dataset & Analysis":
        st.subheader("Dataset & Sentiment Analysis")
        data = pd.read_csv('kucing_predicted.csv')
        
        st.write("### Dataset")
        st.dataframe(data)
        
        sentiment_counts = data['sentiment'].value_counts()
        st.write("### Distribusi Sentimen")
        st.bar_chart(sentiment_counts)
        
        st.write("#### Jumlah Sentimen")
        st.write(sentiment_counts)
        
        st.write("### Word Cloud")
        generate_wordcloud(data)
        
        st.write("### Most Frequent Words")
        plot_top_words(data)

    # 3. MENU KMEANS CLUSTERING
    else:
        st.subheader("KMeans Clustering Analysis")
        data = pd.read_csv('kucing_clean.csv')
        data = data.fillna("").astype(str)
        text_data = data['full_text'].tolist()
        
        vectorizer = TfidfVectorizer(stop_words=stopwords_custom)
        X = vectorizer.fit_transform(text_data)
        
        # Cari k terbaik berdasarkan Silhouette Score
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
        high_score, true_k = -1, 2
        
        for n_clusters in range_n_clusters:
            if X.shape[0] > n_clusters:
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                cluster_labels = km.fit_predict(X)
                score = silhouette_score(X, cluster_labels)
                if score > high_score:
                    high_score = score
                    true_k = n_clusters
        
        # Fit model terbaik
        model = KMeans(n_clusters=true_k, random_state=42, n_init='auto')
        cluster_labels = model.fit_predict(X)
        
        st.write(f"### Silhouette Score: {high_score:.2f} with {true_k} clusters")
        
        # Display Cluster Terms
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        cluster_info = ""
        for i in range(true_k):
            cluster_info += f"Cluster {i}: " + ", ".join([terms[ind] for ind in order_centroids[i, :10]]) + "...\n\n"
        
        st.write("### Cluster Terms")
        st.text(cluster_info)
        
        # PCA Visualisation
        pca = PCA(n_components=2, random_state=0)
        reduced_features = pca.fit_transform(X.toarray())
        reduced_cluster_centers = pca.transform(model.cluster_centers_)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
        ax.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='red', label='Centroids')
        ax.set_title('KMeans Clustering Visualization (PCA 2D)')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        plt.colorbar(scatter, ax=ax)
        ax.legend()
        st.pyplot(fig)

if __name__ == '__main__':
    main()