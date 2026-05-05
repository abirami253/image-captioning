# ================================
# IMAGE CAPTIONING PROJECT
# ================================

# 1. Install Libraries
!pip install tensorflow keras numpy pandas pillow tqdm

# 2. Imports
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# ================================
# 3. Load VGG16 Model
# ================================
base_model = VGG16()
model_cnn = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

# ================================
# 4. Extract Image Features
# ================================
def extract_features(directory):
    features = {}
    for img_name in tqdm(os.listdir(directory)):
        path = os.path.join(directory, img_name)
        
        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        feature = model_cnn.predict(image, verbose=0)
        img_id = img_name.split('.')[0]
        features[img_id] = feature
        
    return features

# Change folder name here
features = extract_features("Images")

# ================================
# 5. Load Captions
# ================================
def load_captions(file):
    captions = {}
    
    with open(file, 'r') as f:
        for line in f:
            img, caption = line.strip().split(',')
            img_id = img.split('.')[0]

            caption = "startseq " + caption + " endseq"

            if img_id not in captions:
                captions[img_id] = []

            captions[img_id].append(caption)
            
    return captions

captions = load_captions("captions.txt")

# ================================
# 6. Tokenization
# ================================
all_captions = []
for key in captions:
    all_captions.extend(captions[key])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in all_captions)

# ================================
# 7. Create Sequences
# ================================
def create_sequences(tokenizer, max_length, captions, features):
    X1, X2, y = [], [], []

    for key, cap_list in captions.items():
        for cap in cap_list:
            seq = tokenizer.texts_to_sequences([cap])[0]

            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]

                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = np.eye(vocab_size)[out_seq]

                X1.append(features[key][0])
                X2.append(in_seq)
                y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)

X1, X2, y = create_sequences(tokenizer, max_length, captions, features)
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Text input
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# Combine
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()
model.fit([X1, X2], y, epochs=10, batch_size=32)
def generate_caption(model, tokenizer, photo, max_length):
    text = "startseq"

    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break

        if word is None:
            break

        text += " " + word

        if word == "endseq":
            break

    return text

sample_image = list(features.keys())[0]
photo = features[sample_image]

caption = generate_caption(model, tokenizer, photo, max_length)
print("Generated Caption:", caption)
