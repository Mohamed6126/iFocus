import cv2
import matplotlib.pyplot as plt

def plot_landmarks(image_path, landmarks, label):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    for i, (x, y) in enumerate(landmarks):
        plt.scatter(x, y, s=50)
        plt.text(x + 5, y + 5, f"P{i+1}", fontsize=10, color='white')
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()