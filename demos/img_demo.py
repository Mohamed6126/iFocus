import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# 1. Load the fine-tuned model
model = load_model('./models/best_mobilenet.keras')

IMAGE_SIZE = (224, 224) 
CLASS_NAMES = ['alert', 'micro-sleep', 'yawning'] 

def predict_and_plot_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize_with_pad(image, 224, 224)

    image = tf.cast(image, tf.float32)
        
    img_batch = np.expand_dims(image, axis=0)
    
    # 4. Predict
    predictions = model.predict(img_batch)
    
    # 5. Parse prediction scores
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    label = CLASS_NAMES[predicted_class_idx]
    
    # 6. Plot the original image with prediction title
    plt.figure(figsize=(6, 6))
    plt.imshow(image)  # Displays original image before [-1, 1] MobileNet scaling
    plt.title(f"Prediction: {label} ({confidence * 100:.2f}%)\n"
                f"All Scores: {dict(zip(CLASS_NAMES, np.round(predictions[0], 3)))}")
    plt.axis('off')
    plt.show()
    
    return label, confidence

# Insert your image here ! 
label, score = predict_and_plot_image('demos/demo_imgs/frame220.jpg')