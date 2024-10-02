import tensorflow as tf

def create_dummy_model(input_shape, name):
    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(16, activation='relu')(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    return model

# Create and save dummy models
user_model_a = create_dummy_model(13, 'user_model_a')
user_model_a.save('user_model_a.keras')

ad_model_a = create_dummy_model(13, 'ad_model_a')
ad_model_a.save('ad_model_a.keras')

user_model_b = create_dummy_model(13, 'user_model_b')
user_model_b.save('user_model_b.keras')

ad_model_b = create_dummy_model(13, 'ad_model_b')
ad_model_b.save('ad_model_b.keras')

print("Dummy models created and saved.")