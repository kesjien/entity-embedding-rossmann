import pickle
import numpy
import math

from sklearn.preprocessing import StandardScaler
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers import Merge
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint,Callback
import keras.backend as K

from prepare_nn_features import split_features
class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

class Model(object):

    def __init__(self, train_ratio):
        self.train_ratio = train_ratio
        self.__load_data()
    

    def evaluate(self):
        if self.train_ratio == 1:
            return 0
        total_sqe = 0
        num_real_test = 0
        for record, sales in zip(self.X_val, self.y_val):
            if sales == 0:
                continue
            guessed_sales = self.guess(record)
            sqe = ((sales - guessed_sales) / sales) ** 2
            total_sqe += sqe
            num_real_test += 1
        result = math.sqrt(total_sqe / num_real_test)
        return result

    def __load_data(self):
        f = open('feature_train_data.pickle', 'rb')
        (self.X, self.y) = pickle.load(f)
        self.X = numpy.array(self.X)
        self.y = numpy.array(self.y)
        self.num_records = len(self.X)
        self.train_size = int(self.train_ratio * self.num_records)
        self.test_size = self.num_records - self.train_size
        self.X, self.X_val = self.X[:self.train_size], self.X[self.train_size:]
        self.y, self.y_val = self.y[:self.train_size], self.y[self.train_size:]


class NN_with_EntityEmbedding(Model):

    def __init__(self, train_ratio):
        super().__init__(train_ratio)
        self.build_preprocessor(self.X)
        self.nb_epoch = 1 #20
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        print(self.y)
        self.max_log_y = numpy.max(numpy.log1p(self.y))
        self.min_log_y = numpy.min(numpy.log1p(self.y))
        self.__build_keras_model()
        self.fit()

    def build_preprocessor(self, X):
        X_list = split_features(X)
        # return True
        # Google trend de
        # self.store_index = StandardScaler()
        # self.store_index.fit(X_list[0])
        self.item_index = StandardScaler()
        self.item_index.fit(X_list[1])
        # self.clas = StandardScaler()
        # self.clas.fit(X_list[9])
        self.oil = StandardScaler()
        self.oil.fit(X_list[12])
        # # Google trend state
        # self.gt_state_enc = StandardScaler()
        # self.gt_state_enc.fit(X_list[33])

    def preprocessing(self, X):
        X_list = split_features(X)
        # X_list[0] = self.store_index.transform(X_list[0])
        X_list[1] = self.item_index.transform(X_list[1])
        # X_list[9] = self.clas.transform(X_list[9])
        X_list[12] = self.oil.transform(X_list[12])
        # X_list[33] = self.gt_state_enc.transform(X_list[33])
        # print(X_list)
        return X_list

    def __build_keras_model(self):
        models = []

        model_store = Sequential()
        model_store.add(Embedding(55, 25, input_length=1))
        model_store.add(Reshape(target_shape=(25,)))
        models.append(model_store)

        # model_store_index = Sequential()
        # model_store_index.add(Dense(1000, input_dim=1))
        # models.append(model_store_index)

        model_item_index = Sequential()
        model_item_index.add(Dense(100, input_dim=1))
        models.append(model_item_index)
        # model_item = Sequential()
        # model_item.add(Embedding(18000, 100, input_length=1))
        # model_item.add(Reshape(target_shape=(100,)))
        # models.append(model_item)

        model_dow = Sequential()
        model_dow.add(Embedding(7, 6, input_length=1))
        model_dow.add(Reshape(target_shape=(6,)))
        models.append(model_dow)

        model_promo = Sequential()
        model_promo.add(Dense(1, input_dim=1))
        models.append(model_promo)

        # model_year = Sequential()
        # model_year.add(Dense(2, input_dim=1))
        # # model_year.add(Reshape(target_shape=(1,)))
        # models.append(model_year)

        # model_month = Sequential()
        # model_month.add(Embedding(12, 6, input_length=1))
        # model_month.add(Reshape(target_shape=(6,)))
        # models.append(model_month)

        model_day = Sequential()
        model_day.add(Embedding(31, 10, input_length=1))
        model_day.add(Reshape(target_shape=(10,)))
        models.append(model_day)




        # model_transferred = Sequential()
        # model_transferred.add(Embedding(3, 2, input_length=1))
        # model_transferred.add(Reshape(target_shape=(2,)))
        # models.append(model_transferred)

        # model_transferred = Sequential()
        # model_transferred.add(Dense(1, input_dim=1))
        # # model_transferred.add(Reshape(target_shape=(1,)))
        # models.append(model_transferred)



        model_family = Sequential()
        model_family.add(Embedding(33, 4, input_length=1))
        model_family.add(Reshape(target_shape=(4,)))
        models.append(model_family)

        # model_class = Sequential()
        # model_class.add(Embedding(999, 20, input_length=1))
        # model_class.add(Reshape(target_shape=(20,)))
        # models.append(model_class)
        # model_class = Sequential()
        # model_class.add(Dense(20, input_dim=1))
        # # model_class.add(Reshape(target_shape=(1,)))
        # models.append(model_class)

        model_perishable = Sequential()
        model_perishable.add(Dense(1, input_dim=1))
        # model_perishable.add(Reshape(target_shape=(1,)))
        models.append(model_perishable)

        model_city = Sequential()
        model_city.add(Embedding(22, 4, input_length=1))
        model_city.add(Reshape(target_shape=(4,)))
        models.append(model_city)

        model_cluster = Sequential()
        model_cluster.add(Embedding(18,3, input_length=1))
        model_cluster.add(Reshape(target_shape=(3,)))
        models.append(model_cluster)

        model_typeO = Sequential()
        model_typeO.add(Embedding(5, 2, input_length=1))
        model_typeO.add(Reshape(target_shape=(2,)))
        models.append(model_typeO)

        model_oil = Sequential()
        model_oil.add(Dense(2, input_dim=1))
        models.append(model_oil)

        model_state = Sequential()
        model_state.add(Embedding(16, 4, input_length=1))
        model_state.add(Reshape(target_shape=(4,)))
        models.append(model_state)

        model_isHoliday = Sequential()
        model_isHoliday.add(Dense(2, input_dim=1))
        models.append(model_isHoliday)

        model_isQuake = Sequential()
        model_isQuake.add(Dense(2, input_dim=1))
        models.append(model_isQuake)

        # model_woy = Sequential()
        # model_woy.add(Embedding(55, 20, input_length=1))
        # model_woy.add(Reshape(target_shape=(20,)))
        # models.append(model_woy)

        # model_competyear = Sequential()
        # model_competyear.add(Embedding(18, 4, input_length=1))
        # model_competyear.add(Reshape(dims=(4,)))
        # models.append(model_competyear)

        # model_promotyear = Sequential()
        # model_promotyear.add(Embedding(8, 4, input_length=1))
        # model_promotyear.add(Reshape(dims=(4,)))
        # models.append(model_promotyear)

        # model_germanstate = Sequential()
        # model_germanstate.add(Embedding(12, 6, input_length=1))
        # model_germanstate.add(Reshape(dims=(6,)))
        # models.append(model_germanstate)

        # model_woy = Sequential()
        # model_woy.add(Embedding(53, 2, input_length=1))
        # model_woy.add(Reshape(dims=(2,)))
        # models.append(model_woy)

        # model_temperature = Sequential()
        # model_temperature.add(Dense(3, input_dim=3))
        # models.append(model_temperature)

        # model_humidity = Sequential()
        # model_humidity.add(Dense(3, input_dim=3))
        # models.append(model_humidity)

        # model_wind = Sequential()
        # model_wind.add(Dense(2, input_dim=2))
        # models.append(model_wind)

        # model_cloud = Sequential()
        # model_cloud.add(Dense(1, input_dim=1))
        # models.append(model_cloud)

        # model_weatherevent = Sequential()
        # model_weatherevent.add(Embedding(22, 4, input_length=1))
        # model_weatherevent.add(Reshape(dims=(4,)))
        # models.append(model_weatherevent)

        # model_promo_forward = Sequential()
        # model_promo_forward.add(Embedding(8, 1, input_length=1))
        # model_promo_forward.add(Reshape(dims=(1,)))
        # models.append(model_promo_forward)

        # model_promo_backward = Sequential()
        # model_promo_backward.add(Embedding(8, 1, input_length=1))
        # model_promo_backward.add(Reshape(dims=(1,)))
        # models.append(model_promo_backward)

        # model_stateholiday_forward = Sequential()
        # model_stateholiday_forward.add(Embedding(8, 1, input_length=1))
        # model_stateholiday_forward.add(Reshape(dims=(1,)))
        # models.append(model_stateholiday_forward)

        # model_sateholiday_backward = Sequential()
        # model_sateholiday_backward.add(Embedding(8, 1, input_length=1))
        # model_sateholiday_backward.add(Reshape(dims=(1,)))
        # models.append(model_sateholiday_backward)

        # model_stateholiday_count_forward = Sequential()
        # model_stateholiday_count_forward.add(Embedding(3, 1, input_length=1))
        # model_stateholiday_count_forward.add(Reshape(dims=(1,)))
        # models.append(model_stateholiday_count_forward)

        # model_stateholiday_count_backward = Sequential()
        # model_stateholiday_count_backward.add(Embedding(3, 1, input_length=1))
        # model_stateholiday_count_backward.add(Reshape(dims=(1,)))
        # models.append(model_stateholiday_count_backward)

        # model_schoolholiday_forward = Sequential()
        # model_schoolholiday_forward.add(Embedding(8, 1, input_length=1))
        # model_schoolholiday_forward.add(Reshape(dims=(1,)))
        # models.append(model_schoolholiday_forward)

        # model_schoolholiday_backward = Sequential()
        # model_schoolholiday_backward.add(Embedding(8, 1, input_length=1))
        # model_schoolholiday_backward.add(Reshape(dims=(1,)))
        # models.append(model_schoolholiday_backward)

        # model_googletrend_de = Sequential()
        # model_googletrend_de.add(Dense(1, input_dim=1))
        # models.append(model_googletrend_de)

        # model_googletrend_state = Sequential()
        # model_googletrend_state.add(Dense(1, input_dim=1))
        # models.append(model_googletrend_state)

        # model_weather = Sequential()
        # model_weather.add(Merge([model_temperature, model_humidity, model_wind, model_weatherevent], mode='concat'))
        # model_weather.add(Dense(1))
        # model_weather.add(Activation('relu'))
        # models.append(model_weather)

        # self.model = Sequential()
        # self.model.add(Merge(models, mode='concat'))
        # self.model.add(Dense(80))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(.35))
        # self.model.add(Dense(20))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(.35))
        # self.model.add(Dense(10))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(.35))
        # self.model.add(Dense(5))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(.15))
        # self.model.add(Dense(1))
        # self.model.add(Activation('sigmoid'))


        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        # self.model.add(Dropout(0.2))
        self.model.add(Dense(100, init='uniform'))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.1))
        self.model.add(Dense(50, init='uniform'))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.1))
        self.model.add(Dense(10, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(5, init='uniform'))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.1))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        def mean_pred(y_true, y_pred):
            return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', mean_pred])
        print(self.model.summary())
    def _val_for_fit(self, val):
        val = numpy.log1p(val) / self.max_log_y
        # print(val, 'here')
        return val

    def _val_for_pred(self, val):
        # print(val, 'yo1')
        print(numpy.expm1(val * self.max_log_y), 'yo2')
        # return val
        return numpy.expm1(val * self.max_log_y)
    
    def fit(self):
        early_stopping = EarlyStopping(patience=1000)
        if self.train_ratio < 1:
            c = self.model.fit(self.preprocessing(self.X), self._val_for_fit(self.y),
                           validation_data=(self.preprocessing(self.X_val), self._val_for_fit(self.y_val)),
                           epochs=self.nb_epoch, batch_size=256,
                           # callbacks=[early_stopping]
                           # callbacks=[self.checkpointer],
                           # callbacks=[TestCallback((self.y, self.y_val))]
                           )
            print("Result on validation data: ", self.evaluate())
            print(c.history, 'AJAJAJJAJAJAJJAJA')
            # self.model.load_weights('best_model_weights.hdf5')
        else:
            self.model.fit(self.preprocessing(self.X), self._val_for_fit(self.y),
                           nb_epoch=self.nb_epoch, batch_size=256)

    def guess(self, feature):
        feature = numpy.array(feature).reshape(1, -1)
        # print(self.model.predict(self.preprocessing(feature)),'ccc')
        return self._val_for_pred(self.model.predict(self.preprocessing(feature)))[0][0]