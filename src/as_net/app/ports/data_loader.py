from abc import ABC, abstractmethod
from typing import Any


class IDataLoader(ABC):
    """Interface for a data loader."""

    @abstractmethod
    def load_train_data(self) -> Any:
        """Loads the training data."""
        raise NotImplementedError

    @abstractmethod
    def load_val_data(self) -> Any:
        """Loads the validation data."""
        raise NotImplementedError

    @abstractmethod
    def load_test_data(self) -> Any:
        """Loads the test data."""
        raise NotImplementedError
