import tensorflow as tf
from tensorflow import keras

def create_user_model():
    inputs = keras.layers.Input(shape=(13,))
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(16, activation='relu')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="user_model")
    return model

def create_ad_model():
    inputs = keras.layers.Input(shape=(13,))
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(16, activation='relu')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="ad_model")
    return model

# Create and compile the models
user_model_a = create_user_model()
user_model_a.compile(optimizer='adam', loss='mse')

user_model_b = create_user_model()
user_model_b.compile(optimizer='adam', loss='mse')

ad_model_a = create_ad_model()
ad_model_a.compile(optimizer='adam', loss='mse')

ad_model_b = create_ad_model()
ad_model_b.compile(optimizer='adam', loss='mse')


# Save the models
user_model_a.save('user_model_a.keras')
ad_model_a.save('ad_model_a.keras')

user_model_b.save('user_model_b.keras')
ad_model_b.save('ad_model_b.keras')

# Print model summaries
print("User Model A Summary:")
user_model_a.summary()

print("User Model B Summary:")
user_model_b.summary()

print("\nAd Model A Summary:")
ad_model_a.summary()

print("\nAd Model B Summary:")
ad_model_b.summary()