import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from mobilenet_model import build_mobilenet

# paths
train_dir = "datasets/processed/train"
val_dir = "datasets/processed/val"
test_dir = "datasets/processed/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6  # tuned epochs

# data generators (mild augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# build model
model = build_mobilenet()

# callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/mobilenetv2/mobilenet_best.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# evaluate
test_loss, test_acc = model.evaluate(test_gen)
print("Test Accuracy:", test_acc)

# save final model
model.save("models/mobilenetv2/mobilenet_final.keras")