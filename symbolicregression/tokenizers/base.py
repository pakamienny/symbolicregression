from abc import ABC, abstractmethod

class Tokenizer(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """

    def __init__(self):
        pass

    @abstractmethod
    def encode(self, value):
        pass

    @abstractmethod
    def decode(self, value):
        pass