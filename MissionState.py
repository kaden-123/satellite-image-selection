from dataclasses import dataclass, field

from SatelliteImage import SatelliteImage
from SimulationConfig import SimulationConfig


@dataclass(slots=True)
class MissionState:
    """Mutable satellite state for one simulation run.

    Unlike ``SimulationConfig``, these values change as mission events occur.
    Times are measured in days and storage/downlink amounts in megabytes.

    Attributes:
        config: Fixed settings for this simulation run.
        current_time: Current mission time in days since mission start.
        onboard_images: Images currently occupying satellite storage.
        in_contact: Whether a ground-station contact is currently active.
        remaining_downlink_mb: Unused capacity in the current contact.
        action_history: Human-readable actions retained for debugging.
    """

    config: SimulationConfig
    current_time: float = 0.0
    #consider implementing a dictionary instead of list as many ops will be O(n) 
    onboard_images: list[SatelliteImage] = field(default_factory=list)
    remaining_downlink_mb: float = 0.0
    # in case for investigating action decisions
    action_history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the state's initial time and contact capacity."""
        if self.current_time < 0:
            raise ValueError("current_time cannot be negative")
        if self.remaining_downlink_mb < 0:
            raise ValueError("remaining_downlink_mb cannot be negative")

    @property
    def used_storage_mb(self) -> float:
        """Return storage occupied by all images currently onboard."""
        return sum(image.stored_size_mb for image in self.onboard_images)

    @property
    def available_storage_mb(self) -> float:
        """Return unused onboard storage, never less than zero."""
        return max(0.0, self.config.storage_capacity_mb - self.used_storage_mb)

    @property
    def storage_is_over_capacity(self) -> bool:
        """Return whether onboard images exceed configured storage capacity."""
        return self.used_storage_mb > self.config.storage_capacity_mb
