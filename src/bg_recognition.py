import os
import cv2
import fnmatch
import numpy as np

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier

classes_map = {}

def bg_algorithm():
    images, class_images = get_data()
    images = edit_images(images, 50)

    # Provides train/test indices to split data in train/test sets
    k_folder = KFold(5, shuffle=True) 

    best_precision = 0
    best_k = 0

    for i in range(1, 51):
        # train_index: training set indices for that split
        # test_index: testing set indices for that split
        for train_index, test_index in k_folder.split(images):
            images_train = images[train_index]
            images_test = images[test_index]
            class_images_train = class_images[train_index]
            class_images_test = class_images[test_index]

            classifier = BaggingClassifier(GaussianNB())
            classifier.fit(images_train, class_images_train)

            class_images_prediction = classifier.predict(images_test)
            precision = precision_score(class_images_test, class_images_prediction, average='micro')

            if precision > best_precision:
                best_k = i
                best_precision = precision
                best_model = classifier
        
    print("Best model was with n = " + str(best_k) + ": " + str(best_precision))

    test = get_test()
    test = edit_images(test, 50)

    test_predictions = best_model.predict(test)
    generate_file(test_predictions, "final_bg.txt")

def get_data():
    images = []
    class_images = []

    classes = []
    number_files = 0
    number_class = 0

    for directory_path, directories_names, file_names in os.walk("../training/"):
        for directory in directories_names:
            classes.append(directory)
            classes_map[directory] = number_class

            n = len(fnmatch.filter(os.listdir("../training/" + directory), '*.jpg'))
            
            number_files = number_files + n
            number_class = number_class + 1

    print("Number of files in training directory: ", number_files)

    for current_class in classes:
        for i in range(50):
            # IMREAD_COLOR: Specifies to load a color image
            # IMREAD_GRAYSCALE: Specifies to load a grayscale image
            image = cv2.imread('../training/' + current_class + '/' + str(i) + '.jpg', cv2.IMREAD_COLOR)
            
            images.append(np.array(image))
            class_images.append(classes_map[current_class])

    class_images = np.array(class_images)

    return images, class_images

def edit_images(images, size):
    edited_images = []

    for image in images:
        resized_image = resize(image, (size, size), anti_aliasing=True)
        edited_images.append(resized_image.flatten())
    
    edited_images = np.asarray(edited_images)

    return edited_images

def get_test():
    test_images = []

    for directory_path, directories_names, file_names in os.walk("../testing/"):
        for file in file_names:
            image = cv2.imread("../testing/" + file, cv2.IMREAD_COLOR)

            test_images.append(np.array(image))
    
    return test_images

def generate_file(test_predictions, filename):
    f = open(filename, "w")

    image = 0
    
    for directory_path, directories_names, file_names in os.walk("../testing/"):
        for file in file_names:
            predicted_class = list(classes_map.keys())[list(classes_map.values()).index(test_predictions[image])]
            f.write(file + " " + predicted_class + "\n")
            image = image + 1

    f.close()

begin = time.time()
bg_algorithm()
end = time.time()

print(end-begin, "seconds")