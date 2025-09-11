import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("pneumonia_model.h5")

# Define the test data generator (rescale like training)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test dataset
test_generator = test_datagen.flow_from_directory(
    "chest_xray/test",  # path to your test folder
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")
