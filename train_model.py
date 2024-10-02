import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load preprocessed data
X = np.load('X_preprocessed.npy')
y = np.load('y_preprocessed.npy')
print("Data loaded. X shape:", X.shape, "y shape:", y.shape)

# Step 2: Define constants
EMBEDDING_DIM = 32
HIDDEN_UNITS = [64, 32]
DROPOUT_RATE = 0.3
L2_REGULARIZATION = 1e-5
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
EPOCHS = 10

# Step 3: Split features into user and ad features
n_user_features = 7  # Adjust based on your actual number of user features
n_ad_features = X.shape[1] - n_user_features

X_user = X[:, :n_user_features]
X_ad = X[:, n_user_features:]
print("Features split. User features:", X_user.shape, "Ad features:", X_ad.shape)

# Step 4: Define input layers
user_input = keras.layers.Input(shape=(n_user_features,), name='user_input')
ad_input = keras.layers.Input(shape=(n_ad_features,), name='ad_input')

# Step 5: Create tower function
def create_tower(input_layer, name_prefix):
    x = input_layer
    for i, units in enumerate(HIDDEN_UNITS):
        x = keras.layers.Dense(units, activation='relu', name=f'{name_prefix}_dense_{i}',
                               kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION))(x)
        x = keras.layers.BatchNormalization(name=f'{name_prefix}_bn_{i}')(x)
        x = keras.layers.Dropout(DROPOUT_RATE, name=f'{name_prefix}_dropout_{i}')(x)
    return x

# Step 6: Build user and ad towers
user_tower = create_tower(user_input, 'user')
ad_tower = create_tower(ad_input, 'ad')

# Step 7: Combine towers
combined = keras.layers.Concatenate()([user_tower, ad_tower])
output = keras.layers.Dense(1, activation='sigmoid', name='output')(combined)

# Step 8: Create model
model = keras.Model(inputs=[user_input, ad_input], outputs=output)

# Step 9: Define custom loss function
def weighted_binary_crossentropy(y_true, y_pred):
    pos_weight = 2.0
    neg_weight = 1.0
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    loss = -((pos_weight * y_true * tf.math.log(y_pred)) + 
             (neg_weight * (1 - y_true) * tf.math.log(1 - y_pred)))
    return tf.reduce_mean(loss)

# Step 10: Compile model
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy, metrics=['accuracy', keras.metrics.AUC()])

# Step 11: Model summary
model.summary()

# Step 12: Train model
history = model.fit(
    [X_user, X_ad], y,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]
)

# Step 13: Save model (Corrected)
model.save('two_tower_nn_model.keras')  # Added .keras extension

# Step 14: Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

print("Model training complete. Model saved as 'two_tower_nn_model.keras'.")
print("Training history plot saved as 'training_history.png'.")