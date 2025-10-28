import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.losses import Huber
import tensorflow.keras.layers as kl

# 残差ブロック(Bottleneckアーキテクチャ)
class Res_Block(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = kl.Conv2D(256, kernel_size=(3,3), strides=1, padding='same')
        self.bn = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)
        self.conv2 = kl.Conv2D(256, kernel_size=(3,3), strides=1, padding='same')
        
        self.shortcut = self._scblock()
        self.add = kl.Add()
        self.av2 = kl.Activation(tf.nn.relu)

    # Shortcut Connection
    def _scblock(self):
        return lambda x: x

    def call(self, x):
        out1 = self.av1(self.bn(self.conv1(x)))
        out2 = self.conv2(out1)
        shortcut = self.shortcut(x)
        out3 = self.add([out2, shortcut])
        out4 = self.av2(out3)
        return out4


def makeModel():
    model = tf.keras.models.Sequential([
    Conv2D(256,kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(14, 6, 4)),
    kl.BatchNormalization(),
    kl.Activation(tf.nn.relu),
    *[Res_Block() for _ in range(6)],
    Conv2D(1,kernel_size=(1,1),strides=(1,1),padding='same'),
    kl.BatchNormalization(),
    kl.Activation(tf.nn.relu),
    Flatten(),
    Dense(256,activation='relu'),
    Dense(1,activation='linear'),
    ])
    model.compile(loss=Huber(), optimizer="adam")
    model.summary()
    return model