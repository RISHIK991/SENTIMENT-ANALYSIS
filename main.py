# SENTIMENT-ANALYSIS
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Example dataset (combine sentiment and sarcasm datasets in practice)
data = [
    {"text": "I absolutely love waiting in traffic for hours!", "sentiment": "negative", "sarcasm": 1},
    {"text": "This is the best day of my life!", "sentiment": "positive", "sarcasm": 0},
    {"text": "The weather is okay.", "sentiment": "neutral", "sarcasm": 0},
    {"text": "Oh great, another meeting that could've been an email.", "sentiment": "negative", "sarcasm": 1},
    {"text": "I enjoy doing my taxes.", "sentiment": "positive", "sarcasm": 1},
    {"text": "I am feeling neutral about this situation.", "sentiment": "neutral", "sarcasm": 0},
]

# Extract texts, sentiment, and sarcasm labels
texts = [item["text"] for item in data]
sentiments = [item["sentiment"] for item in data]
sarcasm_labels = [item["sarcasm"] for item in data]

# Convert sentiment labels to numerical format
sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
sentiment_labels = [sentiment_map[sentiment] for sentiment in sentiments]

# Train-test split
X_train, X_test, y_sentiment_train, y_sentiment_test, y_sarcasm_train, y_sarcasm_test = train_test_split(
    texts, sentiment_labels, sarcasm_labels, test_size=0.2, random_state=42
)

# Tokenize and pad sequences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.word_index) + 1
max_length = 50
trunc_type = 'post'
padding_type = 'post'

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# Define the multi-output model
embedding_dim = 64

input_layer = tf.keras.layers.Input(shape=(max_length,))
embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(input_layer)
lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(embedding_layer)

# Output for sentiment classification
sentiment_output = tf.keras.layers.Dense(3, activation='softmax', name="sentiment_output")(lstm_layer)

# Output for sarcasm detection
sarcasm_output = tf.keras.layers.Dense(1, activation='sigmoid', name="sarcasm_output")(lstm_layer)

# Build and compile the model
model = tf.keras.Model(inputs=input_layer, outputs=[sentiment_output, sarcasm_output])

model.compile(
    optimizer='adam',
    loss={
        "sentiment_output": "sparse_categorical_crossentropy",
       



 "sarcasm_output": "binary_crossentropy",    },
  
  metrics={
        "sentiment_output": "accuracy",
     

   "sarcasm_output": "accuracy",
    }
)

# Train the model
history = model.fit(
    X_train_padded,
    {"sentiment_output": np.array(y_sentiment_train), "sarcasm_output": np.array(y_sarcasm_train)},
    epochs=10,
    batch_size=32,
    validation_data=(
        X_test_padded,
        {"sentiment_output": np.array(y_sentiment_test), "sarcasm_output": np.array(y_sarcasm_test)},
    )
)





# Evaluate the model
results = model.evaluate(X_test_padded, {"sentiment_output": np.array(y_sentiment_test), "sarcasm_output": np.array(y_sarcasm_test)})
print(f"Test Results: {results}")

# Test the model with new examples
sample_texts = [
    "Oh, I just love when my phone dies in the middle of a call.",
    "The food here is delicious!",
    "It’s fine, I guess.",]


sample_sequences = tokenizer.texts_to_sequences(sample_texts)
sample_padded = pad_sequences(sample_sequences, maxlen=max_length, padding=padding_type)

predictions = model.predict(sample_padded)

for text, sentiment_pred, sarcasm_pred in zip(sample_texts, predictions[0], predictions[1]):
    sentiment_class = np.argmax(sentiment_pred)
    sarcasm_class = 1 if sarcasm_pred > 0.5 else 0
    sentiment_label = [k for k, v in sentiment_map.items() if v == sentiment_class][0]  
    print(f"Text: {text}")
    print(f"  Sentiment: {sentiment_label}")
    print(f"  Sarcasm Detected: {'Yes' if sarcasm_class == 1 else 'No'}")
