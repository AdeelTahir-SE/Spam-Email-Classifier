import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

# Load dataset
df = pd.read_csv('dataset/spam.csv')

# Ensure correct types
text = df["text"].astype("string")
label = df["label"]

# Split data
text_train, text_test, label_train, label_test = train_test_split(
    text, label, test_size=0.4, random_state=42
)
text_test, dev_text, label_test, dev_label = train_test_split(
    text_test, label_test, test_size=0.5, random_state=42
)

# Text vectorizer
tokenizer = TextVectorization(
    max_tokens=7000,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int",
    output_sequence_length=15
)
tokenizer.adapt(text_train)

# Build model
model = Sequential([
    tokenizer,
    Embedding(input_dim=7000, output_dim=128),
    LSTM(128),
    Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    text_train, label_train,
    epochs=1,
    validation_data=(dev_text, dev_label)
)

# Evaluate
loss, acc = model.evaluate(text_test, label_test)

# Save model
model.save("model.h5")

# Predictions vs Actual
# ------------------------

pred_probs = model.predict(text_test)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

plt.scatter(range(len(label_test)), label_test, label="Actual", marker='o',color='blue')
plt.scatter(range(len(pred_probs)), pred_probs, label="Predicted probs", marker='x',color='red')
plt.scatter(range(len(pred_labels)), pred_labels, label="Predicted labels", marker='^',color='green')

plt.title("Actual vs Predicted Labels (0 = Ham, 1 = Spam)")
plt.xlabel("Sample Index")
plt.ylabel("Label")
plt.legend()
plt.show()
