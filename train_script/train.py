
import argparse
import os
import pickle

import tensorflow.keras as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# output will be logged, separate output from previous log entries.
print('-'*100)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, 
                        dest='data_path', 
                        default='data', 
                        help='data folder mounting point')

    return parser.parse_args()


if __name__ == '__main__':

    # parse the parameters passed to the this script
    args = parse_args()

    # set data paths
    train_folder = os.path.join(args.data_path, 'train')
    val_folder = os.path.join(args.data_path, 'validation')


    # Create ImageGenerators
    print('Creating train ImageDataGenerator')
    train_generator = ImageDataGenerator(rescale=1/255)\
                            .flow_from_directory(train_folder, 
                                                 batch_size = 32)
    val_generator = ImageDataGenerator(rescale=1/255)\
                            .flow_from_directory(val_folder, 
                                                 batch_size = 32)

    # Build the model
    model = K.models.Sequential()
    model.add(K.layers.Conv2D(32, (2,2), activation='relu'))
    model.add(K.layers.MaxPooling2D(2,2))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(6, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    # fit model and store history
    history = model.fit(train_generator, 
                        validation_data=val_generator,
                        epochs=10)

    print('Saving model history...')
    with open(f'outputs/model.history', 'wb') as f:
        pickle.dump(history.history, f)

    print('Saving model...')
    model.save(f'outputs/')

    print('Done!')
    print('-'*100)
