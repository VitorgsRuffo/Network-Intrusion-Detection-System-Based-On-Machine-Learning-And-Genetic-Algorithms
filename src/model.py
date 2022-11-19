import tensorflow as tf
from keras import backend as K
import tensorflow_addons as tfa
from import_data import import_data


def create_and_compile_model(input_shape):
    mlp_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(12, activation='softmax')
        # tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    mlp_model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='categorical_crossentropy',
                metrics=[tfa.metrics.MatthewsCorrelationCoefficient(num_classes=12)] );
    return mlp_model


def train_model(mlp_model, train_x, train_y, epochs, batch_size=64):
    mlp_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
    return mlp_model


def show_model_results(compiled_model, test_x, test_y):
    model_results = compiled_model.evaluate(test_x, test_y, verbose=1)
    # print("===============MODEL RESULTS===============")
    # print('MCC: {:.4f}'.format(model_results[1]))
    return model_results[1]


def evaluate_model(feature_list):
    train_x, test_x, train_y, test_y = import_data(feature_list)
    print(f"\nEvaluating model ->  {train_x.shape[1:]} features...")

    compiled_model = create_and_compile_model(train_x.shape[1:])
    trained_model = train_model(compiled_model,  train_x, train_y, epochs=20)
    result = show_model_results(trained_model, test_x, test_y)
    return result, train_x.shape[1:][0]
