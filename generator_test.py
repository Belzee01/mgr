from config import id2code
from data_generator import generate_training_set, generate_labels, onehot_to_rgb
import matplotlib.pyplot as plt

from image_preprocessing import mean_filter, gaussian_blur, noisy

images = generate_training_set(10, 224, 224, 3)
labels = generate_labels(10, 224, 224)
label = labels[0]

images[:5] = [noisy(noise_type="speckle", image=image) for image in images[:5]]

for i in range(10):
    f, axarr = plt.subplots(1, 1)

    axarr.imshow(images[i])
    axarr.set_title("original")
    plt.show()
