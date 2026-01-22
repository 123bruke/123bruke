import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data
from model import build_model
train_dataset, test_dataset, train_labels, test_labels, normalizer = \
    load_and_preprocess_data('insurance.csv')
model = build_model(normalizer)
model.fit(
    train_dataset,
    train_labels,
    validation_split=0.2,
    epochs=100,
    verbose=0
)
loss, mae = model.evaluate(test_dataset, test_labels)
print(f"Mean Absolute Error: {mae}")
predictions = model.predict(test_dataset).flatten()

plt.scatter(test_labels, predictions)
plt.xlabel("True Expenses")
plt.ylabel("Predicted Expenses")
plt.plot([0, 50000], [0, 50000])
plt.show()
