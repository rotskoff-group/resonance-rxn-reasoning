from abc import ABC, abstractmethod


class Parser(ABC):
    """Abstract base class for all parsers.

    Args:
        data_path: Path to the data directory containing the data files
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

    @abstractmethod
    def load_reactions(self, *args, **kwargs) -> list:
        """Load reactions from data source."""
        pass

    @abstractmethod
    def parse_reaction(self, *args, **kwargs) -> str:
        """Parse a reaction into the appropriate format."""
        pass
