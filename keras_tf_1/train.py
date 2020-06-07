from pathlib import Path
from keras import callbacks, layers
from keras.applications.densenet import DenseNet121
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


# manually standardize each image with static number from outside
class FixedImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x -= 0.514
        #     0.514 is mean of xray_stats
        if self.featurewise_std_normalization:
            x /= 0.266
        #     0.266 is sd of xray_stats
        return x


def build_model(num_class=2):
    model = Sequential()
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    model.add(base_model)
    model.trainable = True
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_class, activation='softmax'))
    return model


if __name__ == "__main__":
    # dataset paths
    data_path = '../dataset/cat_and_dog'
    train_dir = Path(f'{data_path}/train')
    valid_dir = Path(f'{data_path}/valid')
    train_len = len(list(train_dir.rglob("*.jpg")))
    valid_len = len(list(valid_dir.rglob("*.jpg")))

    train_datagen = FixedImageDataGenerator(rescale=1. / 255,
                                            rotation_range=30,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.05,
                                            horizontal_flip=True,
                                            brightness_range=[0.9, 1.1],
                                            featurewise_center=True,
                                            featurewise_std_normalization=True,
                                            )
    val_datagen = FixedImageDataGenerator(rescale=1. / 255, featurewise_center=True,
                                          featurewise_std_normalization=True, )

    img_width, img_height = 224, 224
    bs = 8
    model_name = 'cat_dog_keras_tf1'

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(img_width, img_height),
        batch_size=bs,
    )

    val_generator = val_datagen.flow_from_directory(
        directory=valid_dir,
        target_size=(img_width, img_height),
        batch_size=bs,
    )

    num_class = len(train_generator.class_indices)
    model = build_model(num_class)

    model.compile(
        loss=BinaryCrossentropy(from_logits=True, label_smoothing=0.2),
        optimizer=Adam(lr=0.0001),
        metrics=['accuracy', AUC(curve="ROC")]
    )

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    model_checkpoint = callbacks.ModelCheckpoint(f'{model_name}.hdf5',
                                                 monitor='val_loss',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=False)

    csv_logger = callbacks.CSVLogger(f'logs_{model_name}')
    callbacks_list = [model_checkpoint, reduce_lr, csv_logger]

    model.fit_generator(train_generator,
                        steps_per_epoch=train_len // bs,
                        epochs=10,
                        validation_data=val_generator,
                        validation_steps=valid_len // bs,
                        callbacks=callbacks_list)
    model_json = model.to_json()
    with open(f'{model_name}.json', "w") as json_file:
        json_file.write(model_json)
