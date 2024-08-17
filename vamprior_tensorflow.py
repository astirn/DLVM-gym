import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tpd
from tensorflow_probability import layers as tpl


class Encoder(tf.keras.Model):
    def __init__(self, dim_x: int, dim_z: int):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(dim_x),
            tf.keras.layers.Dense(500),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(500),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(2000),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(tpl.IndependentNormal.params_size(dim_z)),
            tpl.IndependentNormal(dim_z),
        ])

    def call(self, x: tf.Tensor) -> tpd.Distribution:
        return self.net(x)

class Decoder(tf.keras.Model):
    def __init__(self, dim_x: int, dim_z: int):
        super().__init__()
        self.mu = tf.keras.Sequential([
            tf.keras.layers.InputLayer(dim_z),
            tf.keras.layers.Dense(2000),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(500),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(2000),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(dim_x),
        ])
        self.sigma = tfp.util.TransformedVariable(1.0, bijector=tfp.bijectors.Softplus())

    def call(self, x: tf.Tensor) -> tpd.Distribution:
        return tpd.Independent(tpd.Normal(self.mu(x), self.sigma), 1)

class VampPrior(tf.keras.Model):

    def __init__(
            self,
            encoder: tf.keras.Model,
            decoder: tf.keras.Model,
            u: tf.Tensor,
            kl_warmup_steps: int,
    ):
        super().__init__()

        self.qz_x = encoder
        self.px_z = decoder

        self.u = tf.Variable(initial_value=u)
        self.k = tf.constant(self.u.shape[0], dtype=tf.float32)

        self.kl_warmup_step = 0
        self.kl_warmup_steps_max = kl_warmup_steps
        self.elbo_tracker = tf.keras.metrics.Mean(name='elbo')

    def log_pz_c(self, z: tf.Tensor) -> tf.Tensor:
        return self.qz_x(self.u).log_prob(z[:, None])

    def dkl_z(self, qz_x, log_pz_c):
        dkl_z = -qz_x.entropy() - tf.math.reduce_logsumexp(log_pz_c - tf.math.log(self.k), axis=-1)
        return tf.reduce_sum(dkl_z)

    def evidence_lower_bound(self, x: tf.Tensor) -> tf.Tensor:
        qz_x = self.qz_x(x)
        z = qz_x.sample()

        log_pz_c = self.log_pz_c(z)

        kl_factor = min(1., self.kl_warmup_step / self.kl_warmup_steps_max)
        elbo = (
            tf.reduce_sum(self.px_z(z).log_prob(x)) -
            tf.reduce_sum(self.dkl_z(qz_x, log_pz_c)) * kl_factor
        ) / tf.cast(tf.shape(x)[0], tf.float32)

        return elbo

    def train_step(self, batch: dict):
        with tf.GradientTape() as tape:
            elbo = self.evidence_lower_bound(batch['x'])
            loss = -elbo

        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        for var in self.trainable_variables:
            tf.assert_equal(tf.math.logical_or(tf.math.is_inf(var), tf.math.is_nan(var)), False, message=var.name)

        self.kl_warmup_step = min(self.kl_warmup_step + 1, self.kl_warmup_steps_max)
        self.elbo_tracker.update_state(elbo)

        return self.get_metrics_result()

    def test_step(self, data: dict):
        elbo = self.evidence_lower_bound(data['x'])
        self.elbo_tracker.update_state(elbo)
        return self.get_metrics_result()

def collate(batch):
    batch['x'] = tf.cast(batch.pop('image'), tf.float32)
    batch['x'] = tf.keras.layers.Flatten()(batch['x']) / tf.reduce_max(batch['x'])
    batch['x'] = (batch['x']  - 0.5) * 2
    batch['y'] = batch.pop('label')
    return batch

if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--kl_warmup_steps', type=int, default=5_000)
    parser.add_argument('--latent_dim', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_clusters', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    assert args.batch_size > args.max_clusters, 'pseudo-inputs initialization makes this assumption'

    # random seed
    if args.seed is not None:
        tf.config.experimental.enable_op_determinism()
        tf.keras.utils.set_random_seed(args.seed)

    # load dataset
    ds_train = tfds.load(name=args.dataset, split=tfds.Split.TRAIN, data_dir='data')
    ds_test = tfds.load(name=args.dataset, split=tfds.Split.TEST, data_dir='data')

    # create dataloaders
    dl_train = ds_train.shuffle(buffer_size=1_000).batch(args.batch_size).map(collate)
    dl_test = ds_test.shuffle(buffer_size=1_000).batch(args.batch_size).map(collate)

    # build model
    data_dim = int(np.prod(next(iter(ds_train))['image'].shape))
    model = VampPrior(
        encoder=Encoder(dim_x=data_dim, dim_z=args.latent_dim),
        decoder=Decoder(dim_x=data_dim, dim_z=args.latent_dim),
        u=next(iter(dl_train))['x'][:args.max_clusters],
        kl_warmup_steps=args.kl_warmup_steps,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), run_eagerly=args.debug)

    # train model
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], enable=True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    hist = model.fit(x=dl_train, validation_data=dl_test, epochs=args.max_epochs, verbose=True)
