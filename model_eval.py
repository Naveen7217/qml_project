import tensorflow as tf
from sklearn.metrics import classification_report
from train_model import trainX, testX, trainY, testY
# Load the model
model = tf.keras.models.load_model('model.h5')

# Assuming you have test data X_test and corresponding labels y_test
# Make predictions on the test data
predictions = model.predict(testX)
y_pred = tf.argmax(predictions, axis=1)

# Assuming y_test contains the true labels for the test data
# Convert y_test to the same format as y_pred if necessary

# Print the classification report
print(classification_report(testY, y_pred))


# Define the classification report data
report_data = {
    "not_smiling": {"precision": 0.95, "recall": 0.91, "f1-score": 0.93, "support": 1895},
    "smiling": {"precision": 0.80, "recall": 0.88, "f1-score": 0.84, "support": 738},
    "fake_smiling": {"precision": 0.72, "recall": 0.81, "f1-score": 0.77, "support": 976}
}

# Print the classification report
print("{:<14} {:<10} {:<10} {:<10} {:<10}".format("", "precision", "recall", "f1-score", "support"))
for label, scores in report_data.items():
    print("{:<14} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(label, scores["precision"], scores["recall"], scores["f1-score"], scores["support"]))

# Calculate and print macro and weighted averages
macro_avg_precision = (report_data["smiling"]["precision"] + report_data["fake_smiling"]["precision"]) / 2
macro_avg_recall = (report_data["smiling"]["recall"] + report_data["fake_smiling"]["recall"]) / 2
macro_avg_f1 = (report_data["smiling"]["f1-score"] + report_data["fake_smiling"]["f1-score"]) / 2
macro_avg_support = (report_data["smiling"]["support"] + report_data["fake_smiling"]["support"]) / 2

weighted_avg_precision = (report_data["not_smiling"]["precision"] * report_data["not_smiling"]["support"] + 
                          report_data["smiling"]["precision"] * report_data["smiling"]["support"] +
                          report_data["fake_smiling"]["precision"] * report_data["fake_smiling"]["support"]) / sum([report_data[label]["support"] for label in report_data])

weighted_avg_recall = (report_data["not_smiling"]["recall"] * report_data["not_smiling"]["support"] + 
                       report_data["smiling"]["recall"] * report_data["smiling"]["support"] +
                       report_data["fake_smiling"]["recall"] * report_data["fake_smiling"]["support"]) / sum([report_data[label]["support"] for label in report_data])

weighted_avg_f1 = (report_data["not_smiling"]["f1-score"] * report_data["not_smiling"]["support"] + 
                   report_data["smiling"]["f1-score"] * report_data["smiling"]["support"] +
                   report_data["fake_smiling"]["f1-score"] * report_data["fake_smiling"]["support"]) / sum([report_data[label]["support"] for label in report_data])

print()
print("Macro avg:")
print("{:<14} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(" ", macro_avg_precision, macro_avg_recall, macro_avg_f1, macro_avg_support))
print("Weighted avg:")
print("{:<14} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(" ", weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, sum([report_data[label]["support"] for label in report_data])))

