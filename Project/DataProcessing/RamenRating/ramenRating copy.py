#%%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import re
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


#%%
def load_data(filepath):
    """Load the dataset-pcb from a CSV file."""
    return pd.read_csv(filepath)


def preprocess_data(data):
    """Preprocess the dataset-pcb."""
    data['Top Ten'] = data['Top Ten'].replace('\n', np.NaN)
    data['isTopTen'] = data['Top Ten'].apply(lambda x: 0 if str(x) == 'nan' else 1)
    data = data.drop(['Top Ten', 'Review #'], axis=1)
    data = data.dropna(axis=0).reset_index(drop=True)
    data = data.drop(data.query("Stars == 'Unrated'").index, axis=0).reset_index(drop=True)
    return data


def process_name(name):
    """Process and stem ramen names."""
    ps = PorterStemmer()
    new_name = name.lower()
    new_name = re.sub(r'[^a-z0-9\s]', '', new_name)
    new_name = re.sub(r'[0-9]+', 'number', new_name)
    new_name = new_name.split(" ")
    new_name = list(map(lambda x: ps.stem(x), new_name))
    new_name = list(map(lambda x: x.strip(), new_name))
    new_name = [word if word != 'flavour' else 'flavor' for word in new_name]
    return [word for word in new_name if word]


def tokenize_names(ramen_names):
    """Tokenize ramen names."""
    # Flatten the list of lists
    flattened_names = [word for sublist in ramen_names for word in sublist]
    tokenizer = Tokenizer(num_words=len(set(flattened_names)))
    tokenizer.fit_on_texts(ramen_names)
    sequences = tokenizer.texts_to_sequences(ramen_names)
    return pad_sequences(sequences, padding='post')


def onehot_encode(df, columns, prefixes):
    """One-hot encode specified columns."""
    df = df.copy()
    for column, prefix in zip(columns, prefixes):
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


def scale_features(features):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(features), columns=features.columns)


def build_model(vocab_length, max_seq_length, embeddings_dim):
    """Build and compile the ANN model."""
    name_inputs = tf.keras.Input(shape=(max_seq_length,), name='name_input')
    batch_norm = tf.keras.layers.BatchNormalization(name='batch_norm')(name_inputs)
    name_embedding = tf.keras.layers.Embedding(input_dim=vocab_length, output_dim=embeddings_dim,
                                               input_length=max_seq_length, name='name_embedding')(batch_norm)
    name_outputs = tf.keras.layers.Flatten(name='name_flatten')(name_embedding)

    other_inputs = tf.keras.Input(shape=(401,), name='other_inputs')
    hidden = tf.keras.layers.Dense(64, activation='relu', name='dense1')(other_inputs)
    other_outputs = tf.keras.layers.Dense(64, activation='relu', name='dense2')(hidden)

    concat = tf.keras.layers.concatenate([name_outputs, other_outputs], name='concatenate')
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(concat)

    model = tf.keras.Model(inputs=[name_inputs, other_inputs], outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(name='prec'),
                           tf.keras.metrics.Recall(name='rec')])
    return model


def plot_history(history):
    """Plot training and validation loss over time."""
    fig = px.line(history.history, y=['loss', 'val_loss'], labels={'x': "Epoch", 'y': "Loss"},
                  title='Training and Validation Loss Over Time')
    fig.show()


# Main script
if __name__ == "__main__":
    data = load_data('../dataset/1.ramen-ratings.csv')
    data = preprocess_data(data)
    ramen_names = data['Variety'].apply(process_name)
    name_features = tokenize_names(ramen_names)
    data = data.drop('Variety', axis=1)
    data = onehot_encode(data, ['Brand', 'Style', 'Country'], ['B', 'S', 'C'])
    labels = data['isTopTen']
    other_features = scale_features(data.drop('isTopTen', axis=1))
    name_features_series = pd.Series(list(name_features), name='Name')
    features = pd.concat([name_features_series, other_features], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=34)

    class_weights = dict(
        enumerate(class_weight.compute_class_weight(class_weight='balanced', classes=y_train.unique(), y=y_train)))

    # Flatten the list of lists in ramen_names
    flattened_names = [word for sublist in ramen_names for word in sublist]
    vocab_length = len(set(flattened_names))
    max_seq_length = max(ramen_names.apply(len))
    embeddings_dim = 64

    model = build_model(vocab_length, max_seq_length, embeddings_dim)

    X_train_1 = np.stack(X_train['Name'].to_numpy())
    X_train_2 = X_train.drop('Name', axis=1)
    X_test_1 = np.stack(X_test['Name'].to_numpy())
    X_test_2 = X_test.drop('Name', axis=1)

    history = model.fit([X_train_1, X_train_2], y_train, validation_split=0.2, class_weight=class_weights,
                        batch_size=64, epochs=100, callbacks=[tf.keras.callbacks.ReduceLROnPlateau()], verbose=1)

    plot_history(history)

    results = model.evaluate([X_test_1, X_test_2], y_test)
    print(f'\n Accuracy: {results[1] :.5f}')
    print(f'      AUC: {results[2] :.5f}')
    print(f'Precision: {results[3] :.5f}')
    print(f'   Recall: {results[4] :.5f}')

    precision = results[3]
    recall = results[4]
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f'F1 Score: {f1_score :.5f}')