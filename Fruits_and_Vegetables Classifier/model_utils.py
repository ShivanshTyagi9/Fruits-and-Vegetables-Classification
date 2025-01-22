import tensorflow as tf
import tensorflow_hub as hub

def create_model(model_url, num_classes, image_shape):
    feature_extractor_layer = hub.KerasLayer(
        model_url,
        trainable=False,
        name="feature_extractor_layer",
        input_shape=image_shape + (3,)
    )

   model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='output_layer')
    ])

    return model