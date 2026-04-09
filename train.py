from prepare_data import *
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report, confusion_matrix # Added
import seaborn as sns # Added
import matplotlib.pyplot as plt # Added
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from nets.MLP import mlp
from nets.conv import conv
from random import randint
import numpy as np

# define some constants
N_FRUITS = 15
FRUITS = {
    0: "Apple", 1: "Banana", 2: "Grape", 3: "Pineapple",
    4: "Watermelon", 5: "Strawberry", 6: "Pear", 7: "Blackberry",
    8: "Blueberry", 9: "Broccoli", 10: "Mushroom", 11: "Carrot",
    12: "Peas", 13: "Potato", 14: "Asparagus"
}

# List of class names for reporting
class_names = [FRUITS[i] for i in range(N_FRUITS)]

# number of samples to take in each class
N = 5000
N_EPOCHS = 30

# data files in the same order as defined in FRUITS
files = [
    "apple.npy", "banana.npy", "grapes.npy", "pineapple.npy",
    "watermelon.npy", "strawberry.npy", "pear.npy", "blackberry.npy",
    "blueberry.npy", "broccoli.npy", "mushroom.npy", "carrot.npy",
    "peas.npy", "potato.npy", "asparagus.npy"
]

# Load and process data
print("Loading data...")
fruits = load("data/", files, reshaped=True)
fruits = set_limit(fruits, N)

# Python 3 map fix: convert map object to a numpy array
fruits = np.array(list(map(normalize, fruits)))

# define the labels
labels = make_labels(N_FRUITS, N)

# prepare the data
x_train, x_test, y_train, y_test = tts(fruits, labels, test_size=0.05)

# one hot encoding
Y_train = to_categorical(y_train, N_FRUITS)
Y_test = to_categorical(y_test, N_FRUITS)

# use our custom designed ConvNet model
model = conv(classes=N_FRUITS, input_shape=(28, 28, 1))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

input("Type 'train' and press Enter to start training: ")
print("Training commenced...")

# Added validation_data to see real-time performance during training
model.fit(np.array(x_train), np.array(Y_train), 
          batch_size=32, 
          epochs=N_EPOCHS, 
          validation_data=(np.array(x_test), np.array(Y_test)),
          verbose=1)

print("\n--- Training complete ---")

# --- PERFORMANCE EVALUATION SECTION ---
print("Generating Performance Metrics...")
preds = model.predict(np.array(x_test))
y_pred_classes = np.argmax(preds, axis=1)

# 1. Print Accuracy, Precision, Recall, and F1-Score to console
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# 2. Generate and Save Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix: Neural Sketch Classifier')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')

# Save the plot for your PPT
plt.savefig('confusion_matrix.png')
print("Confusion Matrix saved as 'confusion_matrix.png'")

# Original manual accuracy check (kept for verification)
score = 0
for i in range(len(preds)):
    if np.argmax(preds[i]) == y_test[i]:
        score += 1
print(f"\nFinal Accuracy: {((score + 0.0) / len(preds)) * 100:.2f}%")

# Save model
name = input("> Enter name to save trained model (e.g., my_model): ")
model.save(name + ".h5")
print(f"Model saved as {name}.h5")

def visualize_and_predict():
    n = randint(0, len(x_test))
    pred = FRUITS[np.argmax(model.predict(np.array([x_test[n]])))]
    actual = FRUITS[y_test[n]]
    print(f"\nTesting Random Sample:")
    print(f"Actual: {actual}")
    print(f"Predicted: {pred}")

visualize_and_predict()