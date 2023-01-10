from dataset_raw import *
import augmentation
from model_tf import *

from sklearn.model_selection import GroupKFold
from keras.optimizers import Adam


def run():
    augmentation.prepare('./train_data')
    x_trainval, y_trainval, groups = read_data('./train_data_aug')
    indices = tf.random.shuffle(tf.range(x_trainval.shape[0]))
    x_trainval = tf.gather(x_trainval, indices, axis=0)
    y_trainval = tf.gather(y_trainval, indices, axis=0)
    groups = tf.gather(groups, indices, axis=0)

    loss = 0
    acc = 0
    gkf = GroupKFold(n_splits=5)
    for s, (train_idx, val_idx) in enumerate(gkf.split(x_trainval, y_trainval, groups)):
        x_train = tf.gather(x_trainval, train_idx, axis=0)
        y_train = tf.gather(y_trainval, train_idx, axis=0)
        x_val = tf.gather(x_trainval, val_idx, axis=0)
        y_val = tf.gather(y_trainval, val_idx, axis=0)

        opt = Adam(0.00001)
        model = make_model()
        model.compile(optimizer=opt, metrics=['binary_accuracy'],
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

        model.fit(x_train, y_train, batch_size=8,
                  epochs=1300, validation_data=(x_val, y_val),
                  shuffle=True)
        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=1)
        loss += (val_loss - loss) / (s+1)
        acc += (val_acc - acc) / (s+1)
    print(f'k-fold validation loss: {loss}')
    print(f'k-fold validation accuracy: {acc}')


if __name__ == "__main__":
    run()
