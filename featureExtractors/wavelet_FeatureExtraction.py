import os
import cv2
import numpy as np
import pywt
from sklearn import preprocessing
from progress.bar import Bar
import time


def main():
    mainStartTime = time.time()
    trainImagePath = '../images_split/train/'
    valImagePath = '../images_split/val/'
    testImagePath = '../images_split/test/'
    trainFeaturePath = '../features_labels/wavelet/train/'
    valFeaturePath = '../features_labels/wavelet/val/'
    testFeaturePath = '../features_labels/wavelet/test/'

    print(f'[INFO] ========= TRAINING IMAGES ========= ')
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainFeatures = extractWaveletDescriptors(trainImages)
    saveData(trainFeaturePath, trainEncodedLabels, trainFeatures, encoderClasses)

    print(f'[INFO] ======== VALIDATION IMAGES ======== ')
    valImages, valLabels = getData(valImagePath)
    valEncodedLabels, _ = encodeLabels(valLabels)
    valFeatures = extractWaveletDescriptors(valImages)
    saveData(valFeaturePath, valEncodedLabels, valFeatures, encoderClasses)

    print(f'[INFO] =========== TEST IMAGES =========== ')
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, _ = encodeLabels(testLabels)
    testFeatures = extractWaveletDescriptors(testImages)
    saveData(testFeaturePath, testEncodedLabels, testFeatures, encoderClasses)

    elapsedTime = round(time.time() - mainStartTime, 2)
    print(f'[INFO] Code execution time: {elapsedTime}s')


def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        for dirpath , dirnames , filenames in os.walk(path):   
            if (len(filenames)>0): #it's inside a folder with files
                folder_name = os.path.basename(dirpath)
                bar = Bar(f'[INFO] Getting images and labels from {folder_name}',max=len(filenames),suffix='%(index)d/%(max)d Duration:%(elapsed)ds')            
                for index, file in enumerate(filenames):
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath,file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    bar.next()
                bar.finish()
        #print(labels)
        return images, np.array(labels,dtype=object)


def extractWaveletDescriptors(images, target_shape=(224, 224), wavelet='haar'):
    """
    Extracts wavelet descriptors from a list of images.

    Args:
        images (list): A list of NumPy arrays representing images.
        target_shape (tuple, optional): The desired shape (height, width) for
            resizing the images. Defaults to (224, 224).
        wavelet (str, optional): The type of wavelet to use for decomposition.
            Defaults to 'haar'.

    Returns:
        numpy.ndarray: A NumPy array of shape (num_images, descriptor_size)
            containing the extracted wavelet descriptors for each image.
    """

    wavelet_descriptors_list = []
    bar = Bar('[INFO] Extracting Wavelet descriptors...', max=len(images), suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')

    for image in images:
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_resized = cv2.resize(image, target_shape)

        # Perform 2D Discrete Wavelet Transform
        coeffs = pywt.dwt2(image_resized, wavelet)

        # cA - approximation coefficients
        # (cH, cV, cD) - detail coefficients
        # cH - Horizontal detail coefficients
        # cV - Vertical detail coefficients
        # cD - Diagonal detail coefficients
        cA, (cH, cV, cD) = coeffs

        # Concatenate approximation and detail coefficients
        descriptors = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])
        wavelet_descriptors_list.append(descriptors)

        bar.next()
    bar.finish()
    return np.array(wavelet_descriptors_list)

def encodeLabels(labels):
    startTime = time.time()
    print(f'[INFO] Encoding labels to numerical labels')
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Encoding done in {elapsedTime}s')
    return np.array(encoded_labels,dtype=object), encoder.classes_

def saveData(path, labels, features, encoderClasses):
    startTime = time.time()
    print(f'[INFO] Saving data')
    label_filename = f'{labels=}'.split('=')[0] + '.csv'
    feature_filename = f'{features=}'.split('=')[0] + '.csv'
    encoder_filename = f'{encoderClasses=}'.split('=')[0] + '.csv'
    
    np.savetxt(path + label_filename, labels, delimiter=',', fmt='%i')
    np.savetxt(path + feature_filename, features, delimiter=',')
    np.savetxt(path + encoder_filename, encoderClasses, delimiter=',', fmt='%s')
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Saving done in {elapsedTime}s')

if __name__ == "__main__":
    main()