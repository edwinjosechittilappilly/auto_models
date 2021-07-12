"""
Author : Edwin  Jose

"""
import sklearn.model_selection

import sklearn.datasets

import sklearn.metrics
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Activation
import keras

import numpy as np
from hyperopt.pyll.base import scope
from hyperopt import fmin, hp, tpe, Trials, space_eval
from sklearn.model_selection import cross_val_score
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss


class AutoLstm():
    def __init__(self, max_iter, max_eval, n_layers, activation, optimizer,
                 loss, n_steps, n_features):
        self.max_iter = max_iter
        self.max_evals = max_evals
        self.n_layers = n_layers
        self.activation = n_layers
        self.optimizer = optimizer
        self.loss = loss
        self.n_steps = n_steps
        self.n_features = n_features
        self.space = {
            "self": self,
            "len_A": hp.choice("len_A", list(np.arange(10, 700))),
            "drop_out_value": hp.uniform("drop_out_value", 0.0001, 0.9)
        }

    def define_model(self, params):
        model_clf = self.make_model(**params)
        return model_clf

    def hyperparameter_tuning(self, X, y, params):
        clf = self.define_model(params)
        history = clf.fit(X,
                          y,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=0,
                          validation_split=0.1,
                          shuffle=False,
                          callbacks=[
                              keras.callbacks.EarlyStopping(monitor="val_loss",
                                                            patience=10,
                                                            mode="min")
                          ])
        try:
            opt_score = history.history["val_loss"][-1]
            return {"loss": opt_score, "status": STATUS_OK}
        except:
            return {"loss": opt_score, "status": "Failed"}

        trials = Trials()

        best = fmin(fn=hyperparameter_tuning,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=20,
                    trials=trials,
                    early_stop_fn=no_progress_loss(iteration_stop_count=10,
                                                   percent_increase=0.0),
                    return_argmin=False)

        print("Best: {}".format(best))

        print("Best: {}".format(best))
        return best

    def make_model(self, len_A, drop_out_value):
        layer = 1
        model = Sequential()
        model.add(
            LSTM(len_A,
                 activation=activation,
                 return_sequences=True,
                 input_shape=(n_steps, n_features)))
        while (layer < n_layers):
            model.add(LSTM(len_A, activation=activation,
                           return_sequences=True))
            layer = layer + 1
        model.add(LSTM(len_A, activation=activation, return_sequences=False))
        model.add(Dropout(drop_out_value))
        model.add(Dense(1))
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def combile_model():
        pass

    def fit():
        pass
