import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

nltk.download('punkt_tab')
nltk.download('stopwords')


class MovieReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert text to token indices
        tokens = self.text_to_sequence(text)

        return {
            'text': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def text_to_sequence(self, text):
        # Convert text to sequence of indices
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum()]
        tokens = tokens[:self.max_len]

        # Pad or truncate to max_len
        sequence = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        sequence += [self.vocab['<PAD>']] * (self.max_len - len(sequence))

        return sequence


def preprocess_data(csv_path, test_size=0.2, max_len=100):
    # Read the CSV file
    df = pd.read_csv(csv_path, sep='\t')

    # Clean text
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text

    df['Phrase'] = df['Phrase'].apply(clean_text)

    # Encode labels
    '''
    label_encoder = LabelEncoder()
    df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])
    '''
    label_encoder = df['Sentiment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['Phrase'], df['Sentiment'],
        test_size=test_size,
        random_state=42
    )

    # Build vocabulary
    all_tokens = [word_tokenize(text.lower()) for text in X_train]
    flat_tokens = [token for sublist in all_tokens for token in sublist if token.isalnum()]

    # Create vocabulary
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1
    }

    # Add tokens to vocabulary
    for token in set(flat_tokens):
        if token not in vocab:
            vocab[token] = len(vocab)

    # Create datasets
    train_dataset = MovieReviewDataset(
        texts=X_train.tolist(),
        labels=y_train.tolist(),
        vocab=vocab,
        max_len=max_len
    )

    test_dataset = MovieReviewDataset(
        texts=X_test.tolist(),
        labels=y_test.tolist(),
        vocab=vocab,
        max_len=max_len
    )

    return train_dataset, test_dataset, vocab, label_encoder
