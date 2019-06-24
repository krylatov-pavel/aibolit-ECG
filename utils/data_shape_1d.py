class DataShape1d(object):
    def __init__(self, channels, length):
        self._channels = channels
        self._length = length
    
    @property
    def channels(self):
        return self._channels

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

    def suggest_padding(self, kernel_size, stride):
        remainder = (self._length - kernel_size) % stride

        if remainder == 0:
            return None
        else:
            pad = stride - remainder
            pad_left = pad // 2
            pad_right = pad - pad_left
            return (pad_left, pad_right)