from dataset_raw import *
from model_tf import *

from keras.optimizers import Adam


def train():
    x_train, y_train, _ = read_data('./train_data_aug')
    indices = tf.random.shuffle(tf.range(x_train.shape[0]))
    x_train = tf.gather(x_train, indices, axis=0)
    y_train = tf.gather(y_train, indices, axis=0)

    opt = Adam(0.00001)
    model = make_model()
    model.compile(optimizer=opt, metrics=['binary_accuracy',
                                          tf.keras.metrics.Precision(),
                                          tf.keras.metrics.Recall(),
                                          tf.keras.metrics.AUC(curve='PR', from_logits=True)],
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    model.fit(x_train, y_train, batch_size=8, epochs=1500, shuffle=True)
    return model


def test(model):
    x_test, y_test, _ = read_data('./test_data')
    model.evaluate(x_test, y_test, verbose=2)
    model.predict(x_test, verbose=2)


if __name__ == "__main__":
    model = train()
    test(model)
