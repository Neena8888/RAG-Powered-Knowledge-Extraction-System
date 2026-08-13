"""
Topic Modeling Module using Latent Dirichlet Allocation (LDA).
Extracts dominant latent themes from technical document corpora.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TechnicalTopicModeler:
    """
    Custom LDA Topic Modeler designed for technical knowledge base corpora.
    """

    def __init__(self, num_topics=5, top_words_count=10):
        self.num_topics = num_topics
        self.top_words_count = top_words_count
        self.vectorizer = CountVectorizer(
            max_df=0.85,
            min_df=2,
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10,
            learning_method='online'
        )
        self.topic_labels = {}

    def fit_and_assign_topics(self, text_series):
        """
        Fits LDA model on text series and predicts dominant topic per text.
        """
        clean_docs = text_series.fillna("").astype(str).tolist()
        doc_term_matrix = self.vectorizer.fit_transform(clean_docs)
        
        lda_output = self.lda_model.fit_transform(doc_term_matrix)
        feature_names = self.vectorizer.get_feature_names_out()

        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_features_ind = topic.argsort()[:-self.top_words_count - 1:-1]
            top_words = [feature_names[i] for i in top_features_ind]
            self.topic_labels[topic_idx] = f"Topic_{topic_idx+1}_({', '.join(top_words[:3])})"

        dominant_topics = lda_output.argmax(axis=1)
        predicted_labels = [self.topic_labels[idx] for idx in dominant_topics]

        return predicted_labels, self.topic_labels


def extract_corpus_topics(csv_path="data/processed/clean_dataset.csv"):
    """
    Utility wrapper to run topic modeling on stored CSV dataset.
    """
    df = pd.read_csv(csv_path)
    modeler = TechnicalTopicModeler(num_topics=5)
    
    target_col = 'clean_text' if 'clean_text' in df.columns else df.columns[-1]
    topic_predictions, topic_map = modeler.fit_and_assign_topics(df[target_col])
    
    df['topic_theme'] = topic_predictions
    return df, topic_map


if __name__ == "__main__":
    df_result, themes = extract_corpus_topics()
    print("--- EXTRACTED TOPIC THEMES ---")
    for k, v in themes.items():
        print(f"ID {k}: {v}")