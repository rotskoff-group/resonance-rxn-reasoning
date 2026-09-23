from abc import ABC, abstractmethod


class Prompt(ABC):
    """Abstract base class for all prompt templates."""

    @abstractmethod
    def get_system_prompt(self, *args, **kwargs) -> str:
        """Return the system prompt for this task."""
        pass

    @abstractmethod
    def get_task_template(self, *args, **kwargs) -> str:
        """Return the task template for this task."""
        pass
