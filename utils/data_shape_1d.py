import math

class DataShape1d(object):
    def __init__(self, channels, length):
        self._channels = channels
        self._length = length
    
    @property
    def channels(self):
        return self._channels

    @property
    def length(self):
        return self._length

    @property
    def size(self):
        return self._channels * self._length

    @property
    def shape(self):
        return (self._channels, self._length)

    def conv(self, filters, kernel_size, stride=1, padding=None):
        padding = padding or [0, 0]

        self._channels = filters

        numerator = self._length - kernel_size + sum(padding)
        denominator = stride

        if numerator % denominator != 0:
            print("Warning: incorrect param combination, conv output must be integer")

        self._length = int(numerator / denominator + 1)
    
    def pool(self, kernel_size, stride=1, padding=None):
        padding = padding or [0, 0]

        numerator = self._length - kernel_size + sum(padding)
        denominator = stride

        if numerator % denominator != 0:
            print("Warning: incorrect param combination, pooling output must be integer")

        self._length = int(numerator / denominator + 1)

    def fc(self, output_size):
        self._channels = 1
        self._length = output_size

    def dense_block(self, depth, growth_rate):
        self._channels += depth * growth_rate

    def transition_layer(self, output_size, padding=None):
        padding = sum(padding or [0, 0])

        self._length += padding

        self._channels = output_size
        
        if self._length % 2:
            print("Transition layer warning: input length supposed to be even")
        self._length //= 2

    def suggest_padding(self, kernel_size, stride, output_length=None):
        if output_length:
            pad = (output_length - 1) * stride - self._length + kernel_size
        else:
            remainder = (self._length - kernel_size) % stride
            if remainder == 0:
                pad = 0
            else:
                pad = stride - remainder

        if pad:
            pad_left = pad // 2
            pad_right = pad - pad_left

            return (pad_left, pad_right)
        else:
            return None