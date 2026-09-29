from dataclasses import dataclass
from enum import IntEnum


class ImageRepresentation(IntEnum):
    """Available forms in which an image can be stored or transmitted."""

    THUMBNAIL = 0
    COMPRESSED = 1
    FULL = 2


class ImageStatus(IntEnum):
    """Current lifecycle state of a simulated image."""

    STORED = 0
    FULLY_DELIVERED = 1
    DELETED = 2


@dataclass(slots=True)
class SatelliteImage:
    """Metadata for one simulated image; no pixels or image files are stored.

    The four predicted probabilities must follow the scientific-category order
    UNUSABLE, LOW, MEDIUM, HIGH. Sizes are measured in megabytes and times are
    measured in days since mission start.

    True category and value are retained for evaluation and oracle policies.
    Ordinary selection policies should use only predictions, uncertainty, image
    sizes, capture time, representation, and mission state.
    """

    image_id: int
    capture_time: float
    true_category: int
    true_value: float
    predicted_probabilities: tuple[float, ...]
    predicted_value: float
    uncertainty: float
    original_size: float
    compressed_size: float
    thumbnail_size: float
    representation: ImageRepresentation = ImageRepresentation.FULL
    status: ImageStatus = ImageStatus.STORED
    downlink_time: float | None = None

    def __post_init__(self) -> None:
        """Validate image metadata and representation-size relationships."""
        if self.image_id < 0:
            raise ValueError("image_id cannot be negative")
        if self.capture_time < 0:
            raise ValueError("capture_time cannot be negative")
        if not 0 <= self.true_category <= 3:
            raise ValueError("true_category must be between 0 and 3")
        if self.true_value < 0:
            raise ValueError("true_value cannot be negative")
        if self.predicted_value < 0:
            raise ValueError("predicted_value cannot be negative")
        if self.uncertainty < 0:
            raise ValueError("uncertainty cannot be negative")

        if any(size < 0 for size in self.representation_sizes):
            raise ValueError("image sizes cannot be negative")
        if not (
            self.thumbnail_size <= self.compressed_size <= self.original_size
        ):
            raise ValueError(
                "image sizes must satisfy thumbnail <= compressed <= original"
            )

        if len(self.predicted_probabilities) != 4:
            raise ValueError(
                "predicted_probabilities must contain exactly four categories"
            )
        if any(
            probability < 0 or probability > 1
            for probability in self.predicted_probabilities
        ):
            raise ValueError("probabilities must be between 0 and 1")
        if abs(sum(self.predicted_probabilities) - 1.0) > 1e-6:
            raise ValueError("predicted probabilities must sum to 1")

        if self.downlink_time is not None:
            if self.downlink_time < self.capture_time:
                raise ValueError("downlink_time cannot precede capture_time")

    @property
    def representation_sizes(self) -> tuple[float, float, float]:
        """Return thumbnail, compressed, and full sizes in ascending detail."""
        return (
            self.thumbnail_size,
            self.compressed_size,
            self.original_size,
        )

    @property
    def stored_size_mb(self) -> float:
        """Return the storage occupied by the image's current representation."""
        if self.representation == ImageRepresentation.THUMBNAIL:
            return self.thumbnail_size
        if self.representation == ImageRepresentation.COMPRESSED:
            return self.compressed_size
        return self.original_size


# Partial or resumable transmissions are intentionally unsupported for now.
# A selected representation must fit completely within the remaining contact.
