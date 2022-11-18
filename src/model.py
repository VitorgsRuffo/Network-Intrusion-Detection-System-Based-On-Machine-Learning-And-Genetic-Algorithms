
import tensorflow as tf
import numpy as np
from keras import backend as K
import tensorflow_addons as tfa


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

def train_model(mlp_model, x_train, y_train, epochs):
    mlp_model.fit(x_train, y_train, epochs=epochs)
    return mlp_model

def show_model_results(compiled_model, x_test, y_test):
    model_results = compiled_model.evaluate(x_test, y_test)
    print("===============MODEL RESULTS===============")
    print('MCC: {:.4f}'.format(model_results[1]))



def main():
    train_x = np.loadtxt("train_x.txt")
    train_y = np.loadtxt("train_y.txt")

    y_test = np.loadtxt("test_y.txt")
    x_test = np.loadtxt("test_x.txt")

    compiled_model = create_and_compile_model(train_x.shape[1:])
    trained_model = train_model(compiled_model,  train_x, train_y, epochs=10)
    show_model_results(trained_model, y_test=y_test, x_test=x_test)


if __name__ == "__main__":
    main()