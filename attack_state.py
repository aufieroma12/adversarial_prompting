"""Class for tracking the attack state."""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Input:
    """Class for storing the additional details for an attacked input."""

    prompt_template: str = "{}"
    static_instruction: str = ""

    def get_full_text(self, variable_text: str) -> str:
        """Format all text components into a single string."""
        return self.prompt_template.format(variable_text + self.static_instruction)


@dataclass
class AttackState:
    """Class for tracking the attack state."""

    variable_text: str
    inputs: List[Input]
    score: float
