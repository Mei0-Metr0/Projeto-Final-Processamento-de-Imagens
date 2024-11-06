import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics, preprocessing
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time
from datetime import datetime

def print_menu():      
    print(30 * '-' , 'MENU' , 30 * '-')
    print('1. Wavelet')
    print('2. Exit')
    print(67 * '-')

def menu_choice():
    print_menu()
    featureNames = ['wavelet']
    choice = int(input('Escolha o número da opção desejada: '))
    return featureNames[choice-1]

def choose_model():
    print(30 * '-' , 'MODELS' , 30 * '-')
    print('1. Random Forest')
    print('2. SVM')
    print('3. MLP')
    print('4. All Models')
    print(67 * '-')
    choice = int(input('Escolha o número do modelo desejado: '))
    modelNames = ['RandomForest', 'SVM', 'MLP', 'AllModels']
    return modelNames[choice - 1]

def main():
    featureName = menu_choice()
    modelChoice = choose_model()
    trainFeaturePath = f'../features_labels/{featureName}/train/'
    valFeaturePath = f'../features_labels/{featureName}/val/'
    testFeaturePath = f'../features_labels/{featureName}/test/'
    featureFilename = 'features.csv'
    labelFilename = 'labels.csv'
    encoderFilename = 'encoderClasses.csv'
    mainStartTime = time.time()

    trainFeatures = getFeatures(trainFeaturePath, featureFilename)
    trainEncodedLabels = getLabels(trainFeaturePath, labelFilename)
    valFeatures = getFeatures(valFeaturePath, featureFilename)
    valEncodedLabels = getLabels(valFeaturePath, labelFilename)

    if modelChoice == 'AllModels':
        for modelName in ['RandomForest', 'SVM', 'MLP']:
            run_model(trainFeatures, trainEncodedLabels, valFeatures, valEncodedLabels, testFeaturePath, featureFilename, labelFilename, encoderFilename, featureName, modelName)
    else:
        run_model(trainFeatures, trainEncodedLabels, valFeatures, valEncodedLabels, testFeaturePath, featureFilename, labelFilename, encoderFilename, featureName, modelChoice)

    elapsedTime = round(time.time() - mainStartTime, 2)
    print(f'[INFO] Total code execution time: {elapsedTime}s')

def getFeatures(path, filename):
    features = np.loadtxt(path + filename, delimiter=',')
    return features

def getLabels(path, filename):
    encodedLabels = np.loadtxt(path + filename, delimiter=',', dtype=int)
    return encodedLabels

def getEncoderClasses(path, filename):
    encoderClasses = np.loadtxt(path + filename, delimiter=',', dtype=str)
    return encoderClasses

def trainRandomForest(trainData, trainLabels, valData, valLabels):
    print('[INFO] Training the Random Forest model...')
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    startTime = time.time()
    rf_model.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Training done in {elapsedTime}s')

    print(f'[INFO] Validating the Random Forest model...')
    val_predictions = rf_model.predict(valData)
    val_accuracy = metrics.accuracy_score(valLabels, val_predictions) * 100
    print(f'[INFO] Validation Accuracy: {val_accuracy}%')

    return rf_model, val_accuracy

def trainSVM(trainData, trainLabels, valData, valLabels):
    print('[INFO] Training the SVM model...')
    svm_model = svm.SVC(kernel='linear', C=1, random_state=42)
    startTime = time.time()
    svm_model.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Training done in {elapsedTime}s')

    print(f'[INFO] Validating the SVM model...')
    val_predictions = svm_model.predict(valData)
    val_accuracy = metrics.accuracy_score(valLabels, val_predictions) * 100
    print(f'[INFO] Validation Accuracy: {val_accuracy}%')

    return svm_model, val_accuracy

def trainMLP(trainData, trainLabels, valData, valLabels):
    print('[INFO] Training the MLP model...')
    mlp_model = MLPClassifier(hidden_layer_sizes=(500,), alpha=0.001, max_iter=100, early_stopping=True, random_state=1)
    startTime = time.time()
    mlp_model.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Training done in {elapsedTime}s')

    print(f'[INFO] Validating the MLP model...')
    val_predictions = mlp_model.predict(valData)
    val_accuracy = metrics.accuracy_score(valLabels, val_predictions) * 100
    print(f'[INFO] Validation Accuracy: {val_accuracy}%')

    return mlp_model, val_accuracy

def predict(model, testData):
    print('[INFO] Predicting...')
    startTime = time.time()
    predictedLabels = model.predict(testData)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Predicting done in {elapsedTime}s')
    return predictedLabels

def getCurrentFileNameAndDateTime():
    fileName = os.path.basename(__file__).split('.')[0] 
    dateTime = datetime.now().strftime('-%d%m%Y-%H%M')
    return fileName + dateTime

def plotConfusionMatrix(encoderClasses, testEncodedLabels, predictedLabels, featureName, modelChoice):
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = encoderClasses
    test = encoder.inverse_transform(testEncodedLabels)
    pred = encoder.inverse_transform(predictedLabels)
    print(f'[INFO] Plotting confusion matrix and accuracy...')
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics.ConfusionMatrixDisplay.from_predictions(test, pred, ax=ax, colorbar=False, cmap=plt.cm.Greens)
    currentTime = getCurrentFileNameAndDateTime()
    plt.suptitle(f'Confusion Matrix: {featureName}-{modelChoice}-{currentTime}', fontsize=18)
    accuracy = metrics.accuracy_score(testEncodedLabels, predictedLabels) * 100
    plt.title(f'Accuracy: {accuracy}%', fontsize=18, weight='bold')
    filename = f'../results/{featureName}-{modelChoice}-{currentTime}.png'
    plt.savefig(filename, dpi=300)
    print(f'[INFO] Plotting done!')
    print(f'[INFO] Confusion matrix saved as {filename}')
    return accuracy

def run_model(trainFeatures, trainEncodedLabels, valFeatures, valEncodedLabels, testFeaturePath, featureFilename, labelFilename, encoderFilename, featureName, modelName):
    print(f'[INFO] =========== {modelName} TRAINING PHASE ===========')
    if modelName == 'RandomForest':
        model, val_accuracy = trainRandomForest(trainFeatures, trainEncodedLabels, valFeatures, valEncodedLabels)
    elif modelName == 'SVM':
        model, val_accuracy = trainSVM(trainFeatures, trainEncodedLabels, valFeatures, valEncodedLabels)
    elif modelName == 'MLP':
        model, val_accuracy = trainMLP(trainFeatures, trainEncodedLabels, valFeatures, valEncodedLabels)

    print(f'[INFO] =========== {modelName} TEST PHASE ===========')
    testFeatures = getFeatures(testFeaturePath, featureFilename)
    testEncodedLabels = getLabels(testFeaturePath, labelFilename)
    encoderClasses = getEncoderClasses(testFeaturePath, encoderFilename)
    predictedLabels = predict(model, testFeatures)
    test_accuracy = plotConfusionMatrix(encoderClasses, testEncodedLabels, predictedLabels, featureName, modelName)
    
    print(f'[INFO] Validation Accuracy: {val_accuracy}%')
    print(f'[INFO] Test Accuracy: {test_accuracy}%')
    return val_accuracy, test_accuracy

if __name__ == "__main__":
    main()