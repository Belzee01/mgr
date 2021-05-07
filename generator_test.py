from config import id2code
from data_generator import generate_training_set, generate_labels, onehot_to_rgb
import matplotlib.pyplot as plt


images = generate_training_set(1, 224, 224, 3)
labels = generate_labels(1, 224, 224)
print(images)
label = labels[0]

f, axarr = plt.subplots(2, 10)

axarr[1][9].imshow(images[0])
axarr[1][9].set_title("original")

axarr[1][8].imshow(onehot_to_rgb(label, id2code))
axarr[1][8].set_title("truth")

for i in range(label.shape[2]):
    if i > 9:
        axarr[1][i - 10].imshow(label[:, :, i])
        axarr[1][i - 10].set_title("layer " + id2code[i+1])
    else:
        axarr[0][i].imshow(label[:, :, i])
        axarr[0][i].set_title("layer " + id2code[i+1])

plt.show()