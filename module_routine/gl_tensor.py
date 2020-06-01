import functools
import tensorflow as tf
import pandas as pd
import numpy as np
from utils import util_database
from utils import util_files
from utils.util_files import get_all_times

###################
# LOCAL VARIABLES #
###################
DEVICE_TO_INSPECT = 'A00E01'
# csv path
TRAIN_PATH_CSV = '../resources/gl_train.csv'
TEST_PATH_CSV = '../resources/gl_test.csv'


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,  # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)
    return dataset


def show_batch(dataset):
    for batch, label, in dataset.take(1):
        for key, value in batch.items():
            print('{:20s}:{}'.format(key, value.numpy()))


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


def normalize_numeric_data(data, mean, std):
    # Center de data
    return (data - mean) / std


def build_model():
    model_tf = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model_tf.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )

    return model_tf


############
# DataBase #
############
# Get data from database
print('Getting data from database...')
list_data = util_database.get_instruction_collection(DEVICE_TO_INSPECT)

# Determinate first 70% elements from data
data_size = len(list_data)
first_elements = int(data_size * 0.7)

list_train_data = []
list_test_data = []

# First 70% of elements will add to list_train_data, and the last items will add to list_test_data
for item_data in range(0, data_size):
    if item_data <= first_elements:
        list_train_data.append(list_data[item_data])
    else:
        list_test_data.append(list_data[item_data])

print('Elements to train: {}\nElements to test: {}\n'.format(len(list_train_data), len(list_test_data)))

# Create file csv with data from database
print('Creating csv files...')
# To train
util_files.create_csv(list_train_data, TRAIN_PATH_CSV)
# To test
util_files.create_csv(list_test_data, TEST_PATH_CSV)

##################
# TensorFlow
##################
# Column to use
LABEL_COLUMN = 'status'
LABELS = [0, 1]

# Make numpy values easier to read
np.set_printoptions(precision=3, suppress=True)

# Read CSV data from the file and create dataset
raw_train_dataset = get_dataset(TRAIN_PATH_CSV)
raw_test_dataset = get_dataset(TEST_PATH_CSV)

# Show dataset readable
print('\nRaw train dataset:')
show_batch(raw_train_dataset)
print('\nRaw test dataset:')
show_batch(raw_test_dataset)

# Define general processor model to separate numeric fields to others
NUMERIC_FEATURES = ['delay']

packed_train_data = raw_train_dataset.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_dataset.map(PackNumericFeatures(NUMERIC_FEATURES))

print('\nPacked numeric data Train:')
show_batch(packed_train_data)
print('\nPacked numeric data Test:')
show_batch(packed_test_data)

# Data normalized
print('\nNormalizing data...')
desc = pd.read_csv(TRAIN_PATH_CSV)[NUMERIC_FEATURES].describe()

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]

# Build numerical layer (DON'T DELETE)
numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)

# Categorical data
CATEGORIES = {
    'date': get_all_times(),
    'action': ['TURN_ON', 'TURN_OFF']
}

categorical_columns = []

for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature,
        vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

# Build categorical layer (DON'T DELETE)
categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)

################
# Check Layers #
################
# Create example_batch
CSV_COLUMNS = ['status', 'date', 'delay', 'action']
temp_dataset = get_dataset(TRAIN_PATH_CSV, column_names=CSV_COLUMNS)
example_batch, labels_batch = next(iter(packed_train_data))

print('\nNumerical layer:\n{}'.format(numeric_layer(example_batch).numpy()))
print('\nCategorical layer:\n{}'.format(categorical_layer(example_batch).numpy()[0]))

# Preprocessing
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)
print('\nPreprocessing layer:\n{}'.format(preprocessing_layer(example_batch).numpy()[0]))

# Build model
print('\nBuilding model...')
model = build_model()

# Train model (That point is wrong)
print('\nTraining model...')
train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data, epochs=10)

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {:.2%}%, Test Accuracy {:.2%}%'.format(test_loss, test_accuracy))

predictions = model.predict(test_data)

result_list = []
# Show some results
print('\n======\nResults\n======')
for prediction, status_device, time in zip(predictions, list(test_data)[0][1], list(test_data)[2][0]['date']):
    prediction = tf.sigmoid(prediction).numpy()
    time = str(time)[12:-26]
    status_device = int(status_device)
    if prediction >= 0.75:
        print('Predicted status: {:.2%}'.format(prediction[0]),
              '| Actual outcome: ',
              'TURN_ON' if bool(status_device) else 'TURN_OFF',
              '| Date: {}'.format(time))

        result_list.append({'time': time,
                            'status_device': status_device,
                            'precision': prediction[0]})

for result in result_list:
    print(result)
