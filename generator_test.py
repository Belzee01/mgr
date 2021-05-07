from data_generator import generate_training_set, generate_labels
import matplotlib.pyplot as plt


images = generate_training_set(1, 224, 224, 3)
labels = generate_labels(1, 224, 224)
print(images)
label = labels[0]

f, axarr = plt.subplots(10, 2)

axarr[9][1].imshow(images[0])
axarr[9][1].set_title("original")

for i in range(label.shape[2]):
    if i > 9:
        axarr[i - 10][1].imshow(label[:, :, i])
        axarr[i - 10][1].set_title("layer " + str(i))
    else:
        axarr[i][0].imshow(label[:, :, i])
        axarr[i][0].set_title("layer " + str(i))

plt.show()
