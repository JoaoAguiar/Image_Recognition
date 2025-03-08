import os
import cv2
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time

# A map to store class names (folder names) and their corresponding class indices
classes_map = {}

def main():
    """
    Main function to measure execution time and run the AdaBoost algorithm.
    """
    # Record the start time of the algorithm
    start_time = time.time()

    # Run the AdaBoost algorithm
    best_model = run_adaboost_algorithm()

    # Generate predictions for the test data
    test_images = preprocess_images(get_test_data(), 50)
    test_predictions = best_model.predict(test_images)

    # Save predictions to a file
    generate_file(test_predictions, "ab_predictions.txt")

    # Record the end time and compute the total execution time
    end_time = time.time()

    # Print the execution time of the algorithm
    print(f"Execution time: {end_time - start_time:.2f} seconds")

def run_adaboost_algorithm():
    """
    Runs the AdaBoost Classifier with Decision Tree as the base estimator.
    Performs cross-validation to tune the number of estimators and selects the best model based on precision.
    
    Returns:
        AdaBoostClassifier: The best trained AdaBoost model.
    """
    # Load and preprocess training data
    images, class_images = get_data()
    images = preprocess_images(images, 50)  # Resize images to 50x50 and flatten them

    # Set up 5-fold cross-validation
    kfold = KFold(5, shuffle=True, random_state=42)
    best_precision = 0
    best_k = 0
    best_model = None

    # Iterate through different values of 'n_estimators' (1 to 50)
    for i in range(1, 51):
        current_precision = 0
        model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=i, random_state=42)

        # Cross-validation
        for train_index, test_index in kfold.split(images):
            X_train, X_test = [images[i] for i in train_index], [images[i] for i in test_index]
            y_train, y_test = [class_images[i] for i in train_index], [class_images[i] for i in test_index]

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            current_precision += precision_score(y_test, predictions, average='macro')

        # Average precision over the folds
        current_precision /= 5

        # Update the best model if current model is better
        if current_precision > best_precision:
            best_precision = current_precision
            best_k = i
            best_model = model

    print(f"Best model found with n_estimators={best_k} and precision={best_precision:.4f}")
    return best_model

def get_data():
    """
    Loads and labels training images from the '../training/' directory.
    Returns a list of images and their corresponding labels.
    """
    images = []  # List to store image data
    class_images = []  # List to store class labels
    classes = []  # List to store class names
    current_class = 0  # Class index initialization

    # Walk through the training directory to get the classes and map them to numeric labels
    for directory_path, directories_names, _ in os.walk("../training/"):
        for directory in directories_names:
            classes.append(directory)  # Store the class name (folder name)
            classes_map[directory] = current_class  # Map class name to a numeric label
            current_class += 1  # Increment class index

    print(f"Number of classes in training directory: {len(classes)}")

    # Read and store images along with their respective class labels
    for current_class in classes:
        for i in range(50):  # Assumes each class has 50 images
            image_path = f'../training/{current_class}/{i}.jpg'
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if image is not None:
                images.append(image)  # Store image as a list
                class_images.append(classes_map[current_class])  # Store the corresponding class label
            else:
                print(f"Warning: Image not found at {image_path}")

    return images, class_images

def preprocess_images(images, size):
    """
    Resizes and flattens images to make them compatible with machine learning models.
    
    Args:
        images (list): List of images to be processed.
        size (int): The size to which each image should be resized.
    
    Returns:
        list: List of processed images.
    """
    flattened_images = []  # List to store flattened images
    for image in images:
        resized_image = resize(image, (size, size), anti_aliasing=True)  # Resize the image
        flattened_images.append(resized_image.flatten())  # Flatten the image and store it

    return flattened_images

def get_test_data():
    """
    Loads test images from the '../testing/' directory.
    
    Returns:
        list: List of test images.
    """
    test_images = []  # List to store test image data
    for _, _, file_names in os.walk("../testing/"):
        for file in file_names:
            image_path = f'../testing/{file}'
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                test_images.append(np.array(image))  # Store the test image
            else:
                print(f"Warning: Image not found at {image_path}")
    return test_images

def generate_file(test_predictions, filename):
    """
    Saves the predicted class labels for the test images into a file.
    Each line in the file contains the test image filename and the predicted class label.
    
    Args:
        test_predictions (list): List of predicted class labels for the test images.
        filename (str): The name of the file to save the predictions.
    """
    # Construct the full output path by joining the output directory with the desired filename
    output_path = os.path.join("../output", filename)

    # Write predictions to the file in the parent folder
    with open(output_path, "w") as f:
        for i, file in enumerate(os.listdir("../testing/")):
            predicted_class = list(classes_map.keys())[list(classes_map.values()).index(test_predictions[i])]
            f.write(f"{file} {predicted_class}\n")

    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()