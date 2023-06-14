import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    GRU,
    Activation,
    ActivityRegularization,
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
    Layer,
)
from tensorflow.keras.models import Model, load_model
import numpy as np


def selector_init(shape, dtype=None):
    c = np.zeros(shape)
    c[0] += 1

    return tf.constant(c, dtype=dtype)


@tf.keras.utils.register_keras_serializable()
class Selector(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        for elem in input_shape[1:]:
            assert (
                elem.as_list() == input_shape[0].as_list()
            ), "All inputs must be the same shape."

        self.selectors = self.add_weight(
            name="selectors",
            shape=(len(input_shape),),
            initializer=selector_init,
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        return sum([self.selectors[i] * x[i] for i in range(self.selectors.shape[0])])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config().copy()
        config.update({})
        return config


@tf.keras.utils.register_keras_serializable()
class ResidualTuner(Layer):
    def __init__(self, hidden_units=100, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dense3 = Dense(1)
        self.batchnorm2 = BatchNormalization()
        self.dense2 = Dense(self.hidden_units, activation="relu")
        self.batchnorm1 = BatchNormalization()
        self.dense1 = Dense(self.hidden_units, activation="relu")

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, inp):
        x = self.dense1(inp)
        x = self.batchnorm1(x)
        x = self.dense2(x)
        x = self.batchnorm2(x)
        x = self.dense3(x)
        return x + inp

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({"hidden_units": self.hidden_units})
        return config


@tf.keras.utils.register_keras_serializable()
class SumDiff(Layer):
    def __init__(self, freeze=False, **kwargs):
        super().__init__(**kwargs)
        self.freeze = freeze

    def build(self, input_shape):
        self.b = self.add_weight(
            name="b",
            shape=(1,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=not self.freeze,
        )
        self.w = self.add_weight(
            name="w",
            shape=(1,),
            initializer=tf.keras.initializers.Ones(),
            trainable=not self.freeze,
        )
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        out = tf.reduce_sum(x[0], axis=(1, 2)) - tf.reduce_sum(x[1], axis=(1, 2))
        return self.b + self.w * tf.reshape(out, shape=(-1, 1))

    def compute_output_shape(
        self, input_shape
    ):  # MUST INCLUDE THIS FUNCTION IF CHANGING SHAPE!
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        return tuple((None, 1))

    def freeze(self, unfreeze=False):
        self.freeze = not unfreeze
        self.w.trainable = unfreeze
        self.b.trainable = unfreeze

    def get_config(self):
        config = super().get_config().copy()
        config.update({"freeze": self.freeze})
        return config


@tf.keras.utils.register_keras_serializable()
def binary_KL(y_true, y_pred):
    # return K.mean(K.binary_crossentropy(y_pred, y_true)-K.binary_crossentropy(y_true, y_true), axis=-1)   # this is for the Ubuntu machine in Courant
    return tf.keras.backend.mean(
        tf.keras.backend.binary_crossentropy(y_true, y_pred)
        - tf.keras.backend.binary_crossentropy(y_true, y_true),
        axis=-1,
    )  # this is for Anaconda or Ubuntu on my PC


@tf.keras.utils.register_keras_serializable()
def pos_reg(x, adjacency_left_trim=0, adjacency_right_trim=0):
    l = x.shape[0]
    return tf.reduce_sum(tf.square(x[adjacency_left_trim : l - adjacency_right_trim]))


@tf.keras.utils.register_keras_serializable()
def adj_reg_fo(x, adjacency_left_trim=0, adjacency_right_trim=0):
    l = x.shape[0]
    x_trimmed = x[adjacency_left_trim : l - adjacency_right_trim]
    x_norm = x_trimmed - tf.reduce_mean(x_trimmed, axis=0)
    A = tf.reduce_sum((x_norm[:-1] - x_norm[1:]) ** 2, axis=0)
    B = tf.reduce_sum(x_norm ** 2, axis=0)

    return tf.reduce_mean(A / B)


@tf.keras.utils.register_keras_serializable()
def adj_reg_so(x, adjacency_left_trim=0, adjacency_right_trim=0):
    l = x.shape[0]
    x_trimmed = x[adjacency_left_trim : l - adjacency_right_trim]
    x_norm = x_trimmed - tf.reduce_mean(x_trimmed, axis=0)
    diff_1 = x_norm[:-1] - x_norm[1:]
    diff_2 = diff_1[:-1] - diff_1[1:]
    A = tf.reduce_sum(diff_2 ** 2, axis=0)
    B = tf.reduce_sum(x_norm ** 2, axis=0)

    return tf.reduce_mean(A / B)


@tf.keras.utils.register_keras_serializable()
class MultiRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(
        self,
        position_regularization,
        adjacency_regularization_fo,
        adjacency_regularization_so,
        adjacency_left_trim=0,
        adjacency_right_trim=0,
    ):
        self.position_regularization = position_regularization
        self.adjacency_regularization_fo = adjacency_regularization_fo
        self.adjacency_regularization_so = adjacency_regularization_so
        self.adjacency_left_trim = adjacency_left_trim
        self.adjacency_right_trim = adjacency_right_trim

    def __call__(self, x):
        return (
            self.position_regularization
            * pos_reg(x, self.adjacency_left_trim, self.adjacency_right_trim)
            + self.adjacency_regularization_fo
            * adj_reg_fo(x, self.adjacency_left_trim, self.adjacency_right_trim)
            + self.adjacency_regularization_so
            * adj_reg_so(x, self.adjacency_left_trim, self.adjacency_right_trim)
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "position_regularization": self.position_regularization,
                "adjacency_regularization_fo": self.adjacency_regularization_fo,
                "adjacency_regularization_so": self.adjacency_regularization_so,
                "adjacency_left_trim": self.adjacency_left_trim,
                "adjacency_right_trim": self.adjacency_right_trim,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable()
class RegularizedBiasLayer(Layer):
    def __init__(
        self,
        position_regularization,
        adjacency_regularization_fo,
        adjacency_regularization_so,
        adjacency_left_trim=0,
        adjacency_right_trim=0,
        **kwargs
    ):
        # self.output_dim = output_dim
        super().__init__(**kwargs)
        self.position_regularization = position_regularization
        self.adjacency_regularization_fo = adjacency_regularization_fo
        self.adjacency_regularization_so = adjacency_regularization_so
        self.adjacency_left_trim = adjacency_left_trim
        self.adjacency_right_trim = adjacency_right_trim

    def build(self, input_shape):
        regularizer = MultiRegularizer(
            self.position_regularization,
            self.adjacency_regularization_fo,
            self.adjacency_regularization_so,
            self.adjacency_left_trim,
            self.adjacency_right_trim,
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[1], input_shape[2]),
            initializer="random_normal",
            regularizer=regularizer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return self.kernel + x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "position_regularization": self.position_regularization,
                "adjacency_regularization_fo": self.adjacency_regularization_fo,
                "adjacency_regularization_so": self.adjacency_regularization_so,
                "adjacency_right_trim": self.adjacency_right_trim,
                "adjacency_left_trim": self.adjacency_left_trim,
            }
        )
        return config


def regularized_act(x, act_reg, activation="exponential"):
    if isinstance(activation, str):
        return ActivityRegularization(l1=act_reg)(Activation(activation)(x))
    return ActivityRegularization(l1=act_reg)(activation(x))


def train_model(
    model,
    input_data,
    target_data,
    filename,
    validation_split=0.25,
    epochs=256,
    batch_size=128,
    custom_callbacks=[],
    verbose=1,
):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filename,
        verbose=0,
        save_weights_only=False,
        monitor="val_binary_KL",
        mode="min",
        save_best_only=True,
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_KL",
        min_delta=0,
        patience=10,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )

    history = model.fit(
        input_data,
        target_data,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split,
        callbacks=[model_checkpoint_callback, early_stopping_callback]
        + custom_callbacks,
    )

    return history


def get_model(
    input_length=90,
    randomized_region=(10, 80),
    num_filters=20,
    num_structure_filters=8,
    filter_width=6,
    structure_filter_width=30,
    dropout_rate=0.01,
    activity_regularization=0.0,
    tune_energy=True,
    position_regularization=2.5e-5,
    adjacency_regularization=0.0,
    adjacency_regularization_so=0.0,
    position_regularization_structure=2.5e-5,
    adjacency_regularization_structure=0.0,
    adjacency_regularization_so_structure=0.0,
    energy_activation="softplus",
):

    ###################
    ## Define layers ##
    ###################

    # Sequence layers
    qc_skip = Conv1D(filters=num_filters, kernel_size=filter_width, name="qc_skip")
    qc_incl = Conv1D(filters=num_filters, kernel_size=filter_width, name="qc_incl")

    # bias layers
    seq_left_trim = randomized_region[0]
    seq_right_trim = input_length - randomized_region[1]
    position_bias_skip = RegularizedBiasLayer(
        position_regularization,
        adjacency_regularization,
        adjacency_regularization_so,
        adjacency_left_trim=seq_left_trim,
        adjacency_right_trim=seq_right_trim,
        name="position_bias_skip",
    )
    position_bias_incl = RegularizedBiasLayer(
        position_regularization,
        adjacency_regularization,
        adjacency_regularization_so,
        adjacency_left_trim=seq_left_trim,
        adjacency_right_trim=seq_right_trim,
        name="position_bias_incl",
    )

    dropout_skip_seq = Dropout(dropout_rate, name="dropout_skip_seq")
    dropout_incl_seq = Dropout(dropout_rate, name="dropout_incl_seq")

    # Structure layers
    c_skip_struct = Conv1D(
        num_structure_filters,
        structure_filter_width,
        padding="same",
        name="c_skip_struct",
    )
    c_incl_struct = Conv1D(
        num_structure_filters,
        structure_filter_width,
        padding="same",
        name="c_incl_struct",
    )

    position_bias_skip_struct = RegularizedBiasLayer(
        position_regularization_structure,
        adjacency_regularization_structure,
        adjacency_regularization_so_structure,
        name="position_bias_skip_struct",
    )
    position_bias_incl_struct = RegularizedBiasLayer(
        position_regularization_structure,
        adjacency_regularization_structure,
        adjacency_regularization_so_structure,
        name="position_bias_incl_struct",
    )

    dropout_skip_struct = Dropout(dropout_rate, name="dropout_skip_struct")
    dropout_incl_struct = Dropout(dropout_rate, name="dropout_incl_struct")

    # Energy layers
    energy_seq = SumDiff(name="energy_seq", freeze=not tune_energy)
    energy_seq_struct = SumDiff(name="energy_seq_struct", freeze=not tune_energy)

    # Generalized function layer
    gen_func = ResidualTuner(name="gen_func", hidden_units=4)

    # Final activation
    output_activation = Activation("sigmoid", name="output_activation")

    # Additional selectors
    output_selector = Selector(name="output_selector")

    ########################
    ## Define model logic ##
    ########################

    # Inputs
    seq_input = Input(shape=(input_length, 4), name="seq_input")
    struct_input = Input(shape=(input_length, 3), name="struct_input")
    wobble_input = Input(shape=(input_length, 1), name="wobble_input")

    # Sequence processing
    out_simple_skip = qc_skip(seq_input)
    out_simple_incl = qc_incl(seq_input)

    dropout_bias_skip = dropout_skip_seq(position_bias_skip(out_simple_skip))
    dropout_bias_incl = dropout_incl_seq(position_bias_incl(out_simple_incl))

    # Structure processing
    structure_out_skip = dropout_skip_struct(
        (
            position_bias_skip_struct(
                c_skip_struct(Concatenate()([seq_input, struct_input, wobble_input]))
            )
        )
    )[:, 2:-3, :]
    structure_out_incl = dropout_incl_struct(
        (
            position_bias_incl_struct(
                c_incl_struct(Concatenate()([seq_input, struct_input, wobble_input]))
            )
        )
    )[:, 2:-3, :]

    # Concatenate sequence (selector between sort vs no sort) and structure
    seq_struct_concat_skip = Concatenate()([dropout_bias_skip, structure_out_skip])
    seq_struct_concat_incl = Concatenate()([dropout_bias_incl, structure_out_incl])

    # Energy layers
    energy_seq_out = energy_seq(
        [
            regularized_act(
                dropout_bias_incl,
                activity_regularization,
                activation=energy_activation,
            ),
            regularized_act(
                dropout_bias_skip,
                activity_regularization,
                activation=energy_activation,
            ),
        ]
    )
    energy_seq_struct_out = energy_seq_struct(
        [
            regularized_act(
                seq_struct_concat_incl,
                activity_regularization,
                activation=energy_activation,
            ),
            regularized_act(
                seq_struct_concat_skip,
                activity_regularization,
                activation=energy_activation,
            ),
        ]
    )

    # Generalized function
    gen_func_out = gen_func(energy_seq_struct_out)

    # Model output
    out = output_activation(
        output_selector([energy_seq_out, energy_seq_struct_out, gen_func_out])
    )

    # create model
    model = Model(inputs=[seq_input, struct_input, wobble_input], outputs=out)
    model.compile(optimizer="adam", loss=binary_KL, metrics=[binary_KL])

    return model
