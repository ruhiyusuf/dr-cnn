import os
import tensorflow as tf
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
from keras import backend as k
from keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from numpy import savetxt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import interp
from itertools import cycle

from sklearn import preprocessing

# reduce/remove the warning messages that are printed during program execution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# global constants
program_description = "This program is part of Ruhi Yusuf's science project submission for Grade 8. This program " \
                      "builds is Convolutional Neural Network (CNN) implementation for detecting degrees of " \
                      "Diabetic Retinopathy (DR) in patients with diabetes. It allows you to train with different " \
                      "CNNs, measure the performance of the network, and predict the degree of DR in an image or " \
                      "a set of test images. "
program_epilog = 'Thank you for using this program. Contact ruhi.yusuf@gmail.com. '

TRAIN_DATA_PATH = 'images/train'
VALIDATE_DATA_PATH = 'images/validate'
TEST_DATA_PATH = 'images/test'
NUM_CLASSES = 5


def recall_m(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + k.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + k.epsilon()))


def multiclass_roc_auc_score(y_true, y_pred, num_classes):

    lw = 2  # plot line width

    lb = preprocessing.LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)

    # compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # save fpr, tpr arrays to CSV files
        savetxt(RESULTS_DIR + '/fpr_' + str(i) + '.csv', fpr[i], delimiter=',')
        savetxt(RESULTS_DIR + '/fpr_' + str(i) + '.csv', tpr[i], delimiter=',')

    # save roc_auc array to CSV
    roc_auc_df = pd.DataFrame(roc_auc, index=[0])
    roc_auc_df.to_csv(RESULTS_DIR + '/roc_auc.csv')

    # compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # compute macro-average ROC curve and ROC area
    # first aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1, figsize=(9, 6.5))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['blue', 'green', 'red', 'cyan', 'black'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Diabetic Retinopathy Classification')
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_DIR + '/roc_curves.pdf')
    plt.show()

    # Zoom in view of the upper left corner.
    plt.figure(2, figsize=(9, 6.5))
    plt.xlim(0, 0.3)
    plt.ylim(0.7, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['blue', 'green', 'red', 'cyan', 'black'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Diabetic Retinopathy Classification')
    plt.legend(loc="lower right")
    plt.savefig(RESULTS_DIR + '/roc_curves_zoomed.pdf')
    plt.show()


def copy_image_files_aptos2019(image_labels_array, src_dir, dest_dir):
    if verbosity > 0:
        print('copying image files...')

    if not os.path.isdir(dest_dir):
        if verbosity > 0:
            print('creating directory:', dest_dir)
        os.mkdir(dest_dir)
    for i in image_labels_array.index:
        diagnosis = image_labels_array.loc[i, 'diagnosis']  # get diagnosis from column 1
        label_dir = dest_dir + '/' + str(diagnosis)
        if not os.path.isdir(label_dir):
            if verbosity > 0:
                print('creating directory:', label_dir)
            os.mkdir(label_dir)
        img_src_file = src_dir + image_labels_array.loc[i, 'id_code'] + '.png'
        img_dest_file = label_dir + '/' + image_labels_array.loc[i, 'id_code'] + '.png'
        if not os.path.isfile(img_dest_file):
            if verbosity > 0:
                print('copying file:', img_src_file)
            shutil.copy(img_src_file, img_dest_file)


def setup_img_dir_aptos2019(data_split_method_name='random'):
    print('setting up image directories...')

    # delete existing directories; start clean
    shutil.rmtree(TEST_DATA_PATH, ignore_errors=True)
    shutil.rmtree(TRAIN_DATA_PATH, ignore_errors=True)

    img_labels_file = 'images/aptos2019-blindness-detection/train.csv'
    img_src_dir = 'images/aptos2019-blindness-detection/train_images/'
    img_labels = pd.read_csv(img_labels_file)

    if data_split_method_name == 'random':
        train_img_labels, test_img_labels = train_test_split(img_labels, test_size=0.2, shuffle=True, random_state=51)
        train_img_labels, validate_img_labels = \
            train_test_split(train_img_labels, test_size=0.2, shuffle=True, random_state=51)

        if verbosity > 1:
            print('copying test images...')
        copy_image_files_aptos2019(test_img_labels, img_src_dir, TEST_DATA_PATH)
        if verbosity > 1:
            print('copying train images...')
        copy_image_files_aptos2019(train_img_labels, img_src_dir, TRAIN_DATA_PATH)
        if verbosity > 1:
            print('copying validate images...')
        copy_image_files_aptos2019(validate_img_labels, img_src_dir, VALIDATE_DATA_PATH)

    print('setting up image directories: done')
    return


def create_model(model_name, width, height):
    print('creating model...')
    input_shape = (width, height, 3)

    if model_name == 'simple_adam':
        model_in_create = Sequential()
        model_in_create.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model_in_create.add(Conv2D(32, (3, 3), activation='relu'))
        model_in_create.add(MaxPooling2D())
        model_in_create.add(Dropout(0.25))

        model_in_create.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model_in_create.add(Conv2D(64, (3, 3), activation='relu'))
        model_in_create.add(MaxPooling2D())
        model_in_create.add(Dropout(0.25))

        model_in_create.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model_in_create.add(Conv2D(64, (3, 3), activation='relu'))
        model_in_create.add(MaxPooling2D())
        model_in_create.add(Dropout(0.25))
        model_in_create.add(Flatten())
        model_in_create.add(Dense(512, activation='relu'))
        model_in_create.add(Dropout(0.5))
        model_in_create.add(Dense(activation='softmax', units=5))

        # compile the model
        adam_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model_in_create.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=["accuracy"])

    elif model_name == 'vgg16_custom':
        model_in_create = Sequential()
        model_in_create.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model_in_create.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model_in_create.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model_in_create.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model_in_create.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model_in_create.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model_in_create.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model_in_create.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model_in_create.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model_in_create.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model_in_create.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model_in_create.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model_in_create.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model_in_create.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model_in_create.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model_in_create.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model_in_create.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model_in_create.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model_in_create.add(Flatten())

        model_in_create.add(Dense(4096, activation='relu'))
        model_in_create.add(Dense(4096, activation='relu'))
        model_in_create.add(Dense(NUM_CLASSES, activation='softmax'))

        # compile the model
        optimizer_adam = Adam(lr=0.001)
        model_in_create.compile(loss='categorical_crossentropy', optimizer=optimizer_adam,
                                metrics=['acc', recall_m, precision_m, f1_m])

    elif model_name == 'vgg16':
        from keras.applications.vgg16 import VGG16
        from keras.models import Model

        model_orig = VGG16(include_top=True, input_shape=input_shape)

        # Add a layer where input is the output of the  second last layer
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(model_orig.layers[-2].output)

        # Then create the corresponding model
        model_in_create = Model(input=model_orig.input, output=x)

        # compile the model
        model_in_create.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.9),
                                metrics=['acc', recall_m, precision_m, f1_m])

    elif model_name == 'inceptionV3':
        from keras.applications.inception_v3 import InceptionV3
        from keras.models import Model

        model_orig = InceptionV3(include_top=True, input_shape=input_shape)

        # Add a layer where input is the output of the  second last layer
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(model_orig.layers[-2].output)

        # Then create the corresponding model
        model_in_create = Model(input=model_orig.input, output=x)

        # compile the model
        model_in_create.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.9),
                                metrics=['acc', recall_m, precision_m, f1_m])

    print('creating model: done')
    return model_in_create


def setup_data_for_training(width, height, batch_size_train=32, batch_size_validate=32):
    if verbosity > 0:
        print('setting up data for training...')
    
    train_datagen = ImageDataGenerator(horizontal_flip=True,
                                       rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DATA_PATH,
        target_size=(width, height),
        batch_size=batch_size_train,
        class_mode='categorical')

    validation_gen = validation_datagen.flow_from_directory(
        VALIDATE_DATA_PATH,
        target_size=(width, height),
        batch_size=batch_size_validate,
        class_mode='categorical',
        shuffle=True)

    if verbosity > 0:
        print('setting up data for training: done')
    return train_gen, validation_gen


def predict_using_test_dataset(model_to_predict, width, height, batch_size_test=32):
    if verbosity > 0:
        print('predicting using test dataset...')

    test_generator = ImageDataGenerator(rescale=1. / 255)
    test_data_generator = test_generator.flow_from_directory(
        TEST_DATA_PATH,
        target_size=(width, height),
        batch_size=batch_size_test,
        class_mode='categorical',
        shuffle=False)
    test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

    # run the model and get most-likely class
    predictions = model_to_predict.predict_generator(test_data_generator, steps=test_steps_per_epoch,
                                                     verbose=verbosity, use_multiprocessing=True)
    savetxt(RESULTS_DIR + '/predictions.csv', predictions, delimiter=',')

    predicted_classes = np.argmax(predictions, axis=1)  # y_pred
    savetxt(RESULTS_DIR + '/predicted_classes_test.csv', predicted_classes, delimiter=',')

    # get ground-truth classes and class labels
    true_classes = test_data_generator.classes  # y_val
    savetxt(RESULTS_DIR + '/true_classes_test.csv', true_classes, delimiter=',')

    class_labels = list(test_data_generator.class_indices.keys())

    # get confusion matrix and print it
    confusion_matrix = metrics.confusion_matrix(true_classes, predicted_classes)
    savetxt(RESULTS_DIR + '/confusion_matrix.csv', confusion_matrix, delimiter=',')
    print('Confusion Matrix:')
    print(confusion_matrix)
    print()

    # get classification report and print it
    report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
    print('Classification Report:')
    print(report)
    print()

    print('predictions.shape:', predictions.shape)
    print('predictions:', predictions)
    print('true_classes', true_classes)
    print('Multiclass ROC AUC Score:')
    multiclass_roc_auc_score(true_classes, predictions, len(class_labels))

    # get accuracy for the full test dataset and print it
    acc = metrics.accuracy_score(true_classes, predicted_classes)
    print('Accuracy = {}'.format(acc))

    if verbosity > 0:
        print('predicting using test dataset: done')


def save_model(model_to_save, model_string):
    if verbosity > 0: 
        print('saving model...')
    model_fname = model_string + '.h5'
    model_to_save.save(model_fname)
    print('saved model', model_fname)


def create_results_directory(args):
    import time
    ts = time.localtime()
    results_dir_name = 'results/' + time.strftime("%Y%m%d_%H%M%S", ts)

    arg_string = ''
    for arg in vars(args):
        arg_string = arg_string + '-' + str(getattr(args, arg))

    results_dir_name += arg_string

    # if 'results' directory is not present then create one
    if not os.path.isdir('results'):
        os.mkdir('results')

    # create directory to save results
    os.mkdir(results_dir_name)

    return results_dir_name


def parse_input_arguments():
    parser = argparse.ArgumentParser(description=program_description, epilog=program_epilog)
    parser.add_argument("action", choices=['all_steps', 'setup_dir', 'train_and_save', 'load_and_test',
                                           'model_summary'],
                        help="this 'action' describe what the program should do; if no additional parameters "
                             "are specified, the 'action' works on default values")
    parser.add_argument("-m", "--model",
                        choices=['simple_adam', 'vgg16_custom', 'vgg16', 'inceptionV3'], default='simple_adam',
                        help="specify which model to use (default: %(default)s)", action="store")
    parser.add_argument("-d", "--data_split_method", choices=['random', 'precise'], default='random',
                        help="specify which model to use (default: %(default)s)", action="store")
    parser.add_argument("-v", "--verbosity", default=0, action="store", type=int,
                        help="input value between 0 and 2; higher the value, more verbose the output "
                             "(default: %(default)s)")

    print(parser.parse_args())

    return parser.parse_args()


def get_image_size(model_type):
    im_width = im_height = 256

    if model_type == 'simple_adam':
        im_width = im_height = 256
    elif model_type == 'vgg16_custom':
        im_width = im_height = 224
    elif model_type == 'vgg16':
        im_width = im_height = 224
    elif model_type == 'inceptionV3':
        im_width = im_height = 299
    return im_width, im_height


# main program starts here

# read command line parameters
args = parse_input_arguments()
RESULTS_DIR = create_results_directory(args)  # first create results directory to save results

verbosity = args.verbosity
action = args.action
model_name_to_use = args.model
data_split_method = args.data_split_method

img_width, img_height = get_image_size(model_name_to_use)

print('results directory: ', RESULTS_DIR)

if action == 'all_steps':
    # perform all steps!

    # setup image directories
    setup_img_dir_aptos2019(data_split_method_name=data_split_method)

    # create model
    model = create_model(model_name=model_name_to_use, width=img_width, height=img_height)

    # print model summary
    model.summary()

    # setup data for training
    train_generator, validation_generator = setup_data_for_training(width=img_width, height=img_height)

    # train model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=10, use_multiprocessing=True,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=verbosity)

    # save the model
    model_description = 'model_' + model_name_to_use
    save_model(model_to_save=model, model_string=model_description)

    # predict values and test the model using the test dataset
    predict_using_test_dataset(model_to_predict=model, width=img_width, height=img_height)

elif action == 'setup_dir':
    # setup image directories
    setup_img_dir_aptos2019(data_split_method_name=data_split_method)

elif action == 'train_and_save':
    # create model
    model = create_model(model_name=model_name_to_use, width=img_width, height=img_height)

    # setup data for training
    train_generator, validation_generator = setup_data_for_training(width=img_width, height=img_height)

    # train model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=verbosity)

    # save the model
    model_description = 'model_' + model_name_to_use
    save_model(model_to_save=model, model_string=model_description)

elif action == 'load_and_test':
    print('loading model:', model_name_to_use)

    model_file_name = 'model_' + model_name_to_use + '.h5'
    if os.path.isfile(model_file_name):
        model = models.load_model(model_file_name,
                                  custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m})

        # predict values and test the model using the test dataset
        predict_using_test_dataset(model_to_predict=model, width=img_width, height=img_height)

    else:
        print('error: model not found:', model_file_name)
        print('check if model file exist in directory and rerun the command')

elif action == 'model_summary':
    print('loading model:', model_name_to_use)

    # create model
    model = create_model(model_name=model_name_to_use, width=img_width, height=img_height)

    model.summary()
