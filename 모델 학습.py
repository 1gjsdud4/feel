import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SimpleRNN
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import tensorflow as tf

# 전역 변수로 정의
label_encoder_emotion = None
label_encoder_situation = None

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    texts = []
    labels_emotion = []
    labels_situation = []

    for dialogue in data:
        content = dialogue['talk']['content']
        text = ' '.join([content[f'HS{i:02d}'] for i in range(1, 4) if f'HS{i:02d}' in content])
        emotion_label = dialogue['profile']['emotion']['type']
        situation_label = dialogue['profile']['emotion']['situation'][0]

        texts.append(text)
        labels_emotion.append(emotion_label)
        labels_situation.append(situation_label)

    return texts, labels_emotion, labels_situation

def encode_labels(labels):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    return labels_encoded, label_encoder

def tokenize_and_pad(texts_train, texts_test, tokenizer):
    sequences_train = tokenizer.texts_to_sequences(texts_train)
    sequences_test = tokenizer.texts_to_sequences(texts_test)

    padded_sequences_train = pad_sequences(sequences_train)
    padded_sequences_test = pad_sequences(sequences_test, maxlen=padded_sequences_train.shape[1])

    return padded_sequences_train, padded_sequences_test

def build_model(input_dim, output_dim, sequence_length):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=32, input_length=sequence_length))
    model.add(LSTM(64))
    model.add(Dense(units=output_dim, activation='softmax'))
    optimizer = Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=8):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history

def plot_loss(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

def save_model(model, model_name):
    model.save(f"{model_name}.h5")
    print(f"Model {model_name} saved successfully.")

def load_model(model_name):
    model = tf.keras.models.load_model(f"{model_name}.h5")
    print(f"Model {model_name}.h5 loaded successfully")
    return model

def evaluate_model(model, X_eval, y_eval):
    loss, accuracy = model.evaluate(X_eval, y_eval)
    print(f"Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

def predict_emotion(model, tokenizer, input_text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequences = pad_sequences(sequences, maxlen=model.input_shape[1])

    # Make predictions
    predictions = model.predict(padded_sequences)

    # Find the index with the highest probability
    predicted_label_index = np.argmax(predictions[0])

    # Get the original label corresponding to the index
    predicted_emotion = label_encoder_emotion.inverse_transform([predicted_label_index])[0]

    # Get the probability of the predicted emotion
    probability = predictions[0][predicted_label_index]

    return predicted_emotion, probability

def predict_situation(model, tokenizer, input_text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequences = pad_sequences(sequences, maxlen=model.input_shape[1])

    # Make predictions
    predictions = model.predict(padded_sequences)

    # Find the index with the highest probability
    predicted_label_index = np.argmax(predictions[0])

    # Get the original label corresponding to the index
    predicted_situation = label_encoder_situation.inverse_transform([predicted_label_index])[0]

    # Get the probability of the predicted situation
    probability = predictions[0][predicted_label_index]

    return predicted_situation, probability

def evaluate_model_with_threshold(model, X_eval, y_eval, threshold=0.5):
    # Make predictions
    predictions = model.predict(X_eval)

    # Convert predicted probabilities to labels based on the threshold
    predicted_labels = (predictions >= threshold).astype(int)

    # Calculate accuracy
    accuracy = np.sum(predicted_labels == y_eval) / len(y_eval)

    # Calculate prediction ratio (percentage of predictions made)
    prediction_ratio = np.sum(predicted_labels) / len(y_eval)

    print(f"Evaluation with Threshold {threshold:.2f} - Accuracy: {accuracy:.4f}, Prediction Ratio: {prediction_ratio:.4f}")


def main(json_path, eval_json_path):

    global label_encoder_emotion, label_encoder_situation

    # Load data
    data = load_data(json_path)

    # Preprocess data
    texts, labels_emotion, labels_situation = preprocess_data(data)

    # Split data into train and test sets
    texts_train, texts_test, labels_emotion_train, labels_emotion_test, labels_situation_train, labels_situation_test = train_test_split(
        texts, labels_emotion, labels_situation, test_size=0.1, random_state=42
    )

    # Encode labels
    labels_emotion_train_encoded, label_encoder_emotion = encode_labels(labels_emotion_train)
    labels_emotion_test_encoded = label_encoder_emotion.transform(labels_emotion_test)

    labels_situation_train_encoded, label_encoder_situation = encode_labels(labels_situation_train)
    labels_situation_test_encoded = label_encoder_situation.transform(labels_situation_test)

    # Tokenize and pad sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts_train)

    padded_sequences_train, padded_sequences_test = tokenize_and_pad(texts_train, texts_test, tokenizer)

    # Build and train emotion prediction model
    emotion_model = build_model(len(tokenizer.word_index) + 1, len(label_encoder_emotion.classes_), padded_sequences_train.shape[1])
    history_emotion = train_model(emotion_model, padded_sequences_train, labels_emotion_train_encoded, padded_sequences_test, labels_emotion_test_encoded)

    # Plot loss for emotion model
    plot_loss(history_emotion)

    # Save emotion model
    save_model(emotion_model, "emotion_model")

    # Build and train situation prediction model
    situation_model = build_model(len(tokenizer.word_index) + 1, len(label_encoder_situation.classes_), padded_sequences_train.shape[1])
    history_situation = train_model(situation_model, padded_sequences_train, labels_situation_train_encoded, padded_sequences_test, labels_situation_test_encoded)

    # Plot loss for situation model
    plot_loss(history_situation)

    # Save situation model
    save_model(situation_model, "situation_model")

    # Load emotion model
    loaded_emotion_model = load_model("emotion_model")

    # Load situation model
    loaded_situation_model = load_model("situation_model")

    # Load evaluation data
    eval_data = load_data(eval_json_path)

    # Preprocess evaluation data
    eval_texts, eval_labels_emotion, eval_labels_situation = preprocess_data(eval_data)

    # Tokenize and pad evaluation sequences
    padded_sequences_eval = tokenizer.texts_to_sequences(eval_texts)
    padded_sequences_eval = pad_sequences(padded_sequences_eval, maxlen=padded_sequences_train.shape[1])

    # Encode evaluation labels
    eval_labels_emotion_encoded = label_encoder_emotion.transform(eval_labels_emotion)
    eval_labels_situation_encoded = label_encoder_situation.transform(eval_labels_situation)

    # Evaluate emotion model
    evaluate_model(loaded_emotion_model, padded_sequences_eval, eval_labels_emotion_encoded)

    # Evaluate situation model
    evaluate_model(loaded_situation_model, padded_sequences_eval, eval_labels_situation_encoded)

    input_text = "일이 너무 많아서 스트레스 받아"
    predicted_emotion, emotion_probability = predict_emotion(loaded_emotion_model, tokenizer, input_text)
    predicted_situation, situation_probability = predict_situation(loaded_situation_model, tokenizer, input_text)


    print(f"Predicted Emotion: {predicted_emotion} (Probability: {emotion_probability:.4f})")
    print(f"Predicted Situation: {predicted_situation} (Probability: {situation_probability:.4f})")

    threshold = 0.7

    # Evaluate emotion model with the chosen threshold
    evaluate_model_with_threshold(loaded_emotion_model, padded_sequences_eval, eval_labels_emotion_encoded, threshold)

    # Evaluate situation model with the chosen threshold
    evaluate_model_with_threshold(loaded_situation_model, padded_sequences_eval, eval_labels_situation_encoded, threshold)

if __name__ == "__main__":
    json_path = '데이터/감성대화말뭉치(최종데이터)_Training.json'
    eval_json_path = '데이터/감성대화말뭉치(최종데이터)_Validation.json'
    main(json_path, eval_json_path)
