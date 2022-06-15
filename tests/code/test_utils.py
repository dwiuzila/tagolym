import numpy as np
from tagolym import utils

class TestNumpyEncoder:
    @classmethod
    def setup_class(cls):
        """Called before every class initialization."""
        pass

    @classmethod
    def teardown_class(cls):
        """Called after every class initialization."""
        pass

    def setup_method(self):
        """Called before every method."""
        self.numpy_encoder = utils.NumpyEncoder()

    def teardown_method(self):
        """Called after every method."""
        del self.numpy_encoder

    def test_default(self):
        enc = utils.NumpyEncoder()
        assert type(enc.default(np.int64(4))) == int
        assert type(enc.default(np.bool_(True))) == bool
