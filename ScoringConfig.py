from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

from SatelliteImage import ImageRepresentation


class ScientificCategory(IntEnum):
    """True or predicted scientific category of a captured image.

    Integer values match the required order of classifier probabilities:
    UNUSABLE, LOW, MEDIUM, and HIGH.
    """

    UNUSABLE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Configurable scientific values and representation multipliers.

    Category values describe the value of a correctly identified image at full
    resolution. Representation multipliers describe how much of that value is
    retained when a thumbnail, compressed copy, or full-resolution copy reaches
    the ground.

    The default values are:
        UNUSABLE=0, LOW=1, MEDIUM=2, HIGH=3
        THUMBNAIL=0.25, COMPRESSED=0.75, FULL=1.0
    """

    unusable_value: float = 0.0
    low_value: float = 1.0
    medium_value: float = 2.0
    high_value: float = 3.0

    thumbnail_multiplier: float = 0.25
    compressed_multiplier: float = 0.75
    full_multiplier: float = 1.0

    def __post_init__(self) -> None:
        """Reject negative values and invalid retained-value multipliers."""
        if any(value < 0 for value in self.category_values):
            raise ValueError("scientific category values cannot be negative")
        if any(
            not 0.0 <= multiplier <= 1.0
            for multiplier in self.representation_multipliers
        ):
            raise ValueError("representation multipliers must be between 0 and 1")

    @property
    def category_values(self) -> tuple[float, float, float, float]:
        """Return values ordered as unusable, low, medium, and high."""
        return (
            self.unusable_value,
            self.low_value,
            self.medium_value,
            self.high_value,
        )

    @property
    def representation_multipliers(self) -> tuple[float, float, float]:
        """Return multipliers ordered as thumbnail, compressed, and full."""
        return (
            self.thumbnail_multiplier,
            self.compressed_multiplier,
            self.full_multiplier,
        )

    def category_value(self, category: ScientificCategory | int) -> float:
        """Return the configured full-resolution value of one category."""
        try:
            return self.category_values[int(category)]
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("category must be an integer from 0 through 3") from error

    def representation_multiplier(
        self,
        representation: ImageRepresentation | int,
    ) -> float:
        """Return the retained-value multiplier for one representation."""
        try:
            return self.representation_multipliers[int(representation)]
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                "representation must be THUMBNAIL, COMPRESSED, or FULL"
            ) from error

    def expected_value(self, probabilities: Sequence[float]) -> float:
        """Calculate predicted value from classifier probabilities.

        Probabilities must follow ScientificCategory order. Normal policies may
        use this value because it depends only on classifier output, not truth.
        """
        if len(probabilities) != len(self.category_values):
            raise ValueError("probabilities must contain exactly four categories")
        if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
            raise ValueError("probabilities must be between 0 and 1")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to 1")

        return sum(
            probability * value
            for probability, value in zip(probabilities, self.category_values)
        )

    def delivered_value(
        self,
        true_category: ScientificCategory | int,
        representation: ImageRepresentation | int,
    ) -> float:
        """Calculate actual value delivered to the ground.

        This uses the hidden true category, so it belongs in simulation outcome
        processing, metrics, and oracle evaluation—not ordinary policy logic.
        """
        return (
            self.category_value(true_category)
            * self.representation_multiplier(representation)
        )
