# Import necessary libraries
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from imutils import paths
import imutils
import matplotlib.pyplot as plt
import cv2
import os


# Define the quantum model
def create_quantum_model():
    # Define the qubits and readout operators
    qubits = cirq.GridQubit.rect(4, 4) # Creating a 4x4 grid of qubits
    readout_operators = [cirq.Z(qubit) for qubit in qubits] # Measuring along the Z axis for each qubit

    # Define the quantum circuit
    circuit = cirq.Circuit()
    symbols = sympy.symbols('qconv0 qconv1 qdense') # Define trainable parameters
    circuit += cirq.Circuit(
        # Encoding input data into the quantum state
        cirq.ops.X(qubits[0]),
        cirq.ops.X(qubits[5]),
        cirq.ops.X(qubits[10]),
        cirq.ops.H(qubits[0]),
        cirq.ops.H(qubits[5]),
        cirq.ops.H(qubits[10]),
        cirq.ops.RotXGate(symbols[0])(qubits[0]), # Rotation gate with trainable parameter
        cirq.ops.RotXGate(symbols[1])(qubits[5]),
        cirq.ops.RotXGate(symbols[2])(qubits[10]),
        cirq.ops.CZ(qubits[0], qubits[5]), # Controlled-Z gate
        cirq.ops.CZ(qubits[5], qubits[10]),
        # Decoding quantum state to obtain output
        cirq.ops.RotXGate(symbols[0])(qubits[0]),
        cirq.ops.RotXGate(symbols[1])(qubits[5]),
        cirq.ops.RotXGate(symbols[2])(qubits[10]),
        cirq.ops.H(qubits[0]),
        cirq.ops.H(qubits[5]),
        cirq.ops.H(qubits[10]),
        cirq.ops.X(qubits[0]), # Inverse of input encoding
        cirq.ops.X(qubits[5]),
        cirq.ops.X(qubits[10])
    )

    # Define the input and output tensors
    input_tensor = tf.keras.Input(shape=(), dtype=tf.dtypes.string) # Input tensor for quantum data
    output_tensor = tfq.layers.PQC(circuit, readout_operators)(input_tensor) # Output tensor after applying quantum circuit

    # Define the model
    model = tf.keras.Model(inputs=[input_tensor], outputs=[output_tensor])

    return model


# Load the dataset and preprocess images
def load_and_preprocess_data(dataset_path):
    data = []
    labels = []

    # Loop through each image in the dataset
    for imagePath in sorted(list(paths.list_images(dataset_path))):
        # Load and preprocess the image
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        image = imutils.resize(image, width=28) # Resize to 28x28
        image = img_to_array(image) # Convert to numpy array
        data.append(image)

        # Extract label from the image path
        label = imagePath.split(os.path.sep)[-3]
        label = 'smiling' if label == 'positives' else 'not_smiling' # Encode labels
        labels.append(label)

    # Convert data and labels to numpy arrays
    data = np.array(data, dtype='float') / 255.0 # Normalize data
    labels = np.array(labels)

    # Encode labels
    le = LabelEncoder().fit(labels)
    labels = to_categorical(le.transform(labels), 2)

    return data, labels


# Train the quantum model
def train_quantum_model(quantum_model, train_data, train_labels, test_data, test_labels):
    # Compile the model
    quantum_model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss=tf.losses.CategoricalCrossentropy(),
                          metrics=[tf.metrics.CategoricalAccuracy()])

    # Train the model
    history = quantum_model.fit(train_data, train_labels,
                                validation_data=(test_data, test_labels),
                                epochs=15, batch_size=64, verbose=1)

    return history


# Evaluate the quantum model
def evaluate_quantum_model(quantum_model, test_data, test_labels, label_encoder):
    predictions = quantum_model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    class_names = label_encoder.classes_

    # Print classification report
    print(classification_report(true_labels, predicted_labels, target_names=class_names))


# Plot training history
def plot_training_history(history):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, 15), history.history['loss'], label='train_loss')
    plt.plot(np.arange(0, 15), history.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, 15), history.history['categorical_accuracy'], label='accuracy')
    plt.plot(np.arange(0, 15), history.history['val_categorical_accuracy'], label='val_accuracy')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()


# Main function
def main():
    # Prompt the user for dataset path
    dataset_path = input("Enter the path to the input dataset of faces: ")

    # Load and preprocess the data
    data, labels = load_and_preprocess_data(dataset_path)

    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.20, stratify=labels,
                                                                        random_state=42)

    # Initialize the quantum model
    quantum_model = create_quantum_model()

    # Train the quantum model
    print('[INFO] Training quantum model...')
    history = train_quantum_model(quantum_model, train_data, train_labels, test_data, test_labels)

    # Evaluate the quantum model
    print('[INFO] Evaluating quantum model...')
    evaluate_quantum_model(quantum_model, test_data, test_labels, LabelEncoder())

    # Plot the training history
    plot_training_history(history)


# if __name__ == "__main__":
#     main()
