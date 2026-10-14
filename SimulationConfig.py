from dataclasses import dataclass

@dataclass(slots=True)
class SimulationConfig:
    """Fixed settings shared by every component in one simulation run.

    Times are measured in days unless a field states otherwise. Image sizes and
    downlink usage are measured in megabytes, while storage capacity is entered
    in gigabytes for convenience.

    Attributes:
        mission_duration: Length of the simulated mission in days.
        storage_capacity: Total onboard storage capacity in gigabytes.
        image_arrival_rate: Average number of images captured per day.
        const_take_image: Whether captures occur at a constant interval.
        contact_frequency: Number of ground-station contacts per day.
        contact_duration: Duration of each contact in minutes.
        bandwidth: Downlink speed during contact in megabits per second.
        rand_seed: Seed used to reproduce generated mission scenarios.
    """

    mission_duration: int = 365
    storage_capacity: float = 10.0
    image_arrival_rate: float = 10.0
    const_take_image: bool = True
    contact_frequency: int = 6
    contact_duration: float = 10.0
    bandwidth: float = 10.0
    rand_seed: int = 43
    
    def __post_init__(self) -> None:
        """Reject configuration values that cannot describe a valid run."""
        if self.mission_duration <= 0:
            raise ValueError("mission_duration must be greater than 0")
        if self.storage_capacity <= 0:
            raise ValueError("storage_capacity must be greater than 0")
        if self.image_arrival_rate < 0:
            raise ValueError("image_arrival_rate cannot be negative")
        if self.contact_frequency < 0:
            raise ValueError("contact_frequency cannot be negative")
        if self.contact_duration < 0:
            raise ValueError("contact_duration cannot be negative")
        if self.bandwidth < 0:
            raise ValueError("bandwidth cannot be negative")

    @property
    def storage_capacity_mb(self) -> float:
        """Return onboard storage capacity in megabytes."""
        return self.storage_capacity * 1024.0
