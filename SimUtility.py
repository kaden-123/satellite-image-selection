import numpy as np

from MetricsRecorder import MetricsRecorder
from Policy import Action
from SatelliteImage import ImageRepresentation, SatelliteImage
from ScoringConfig import ScoringConfig
from SimulationConfig import SimulationConfig

def generate_capture_times(config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate constant-rate or random image-capture times in mission days."""
    if config.image_arrival_rate == 0:
        return np.empty(0)

    if config.const_take_image:
        interval = 1.0 / config.image_arrival_rate
        return np.arange(interval, config.mission_duration + 1e-12, interval)

    times = []
    current_time = 0.0
    while True:
        current_time += rng.exponential(1.0 / config.image_arrival_rate)
        if current_time > config.mission_duration:
            return np.asarray(times)
        times.append(current_time)


def generate_contact_times(config: SimulationConfig) -> np.ndarray:
    """Generate regularly spaced ground-contact times in mission days."""
    if config.contact_frequency == 0:
        return np.empty(0)

    interval = 1.0 / config.contact_frequency
    return np.arange(interval, config.mission_duration + 1e-12, interval)


def create_image(
    image_id: int,
    capture_time: float,
    scoring: ScoringConfig,
    rng: np.random.Generator,
) -> SatelliteImage:
    """Create synthetic image metadata and an imperfect classifier prediction."""
    true_category = int(rng.choice(4, p=(0.10, 0.45, 0.30, 0.15)))
    true_value = scoring.category_value(true_category)

    # Dirichlet output is a probability vector that usually favors the truth.
    concentration = np.ones(4)
    concentration[true_category] = 7.0
    predicted_probabilities = tuple(
        float(p) for p in rng.dirichlet(concentration)
    )

    predicted_value = scoring.expected_value(predicted_probabilities)

    entropy = -sum(
        p * np.log(p) for p in predicted_probabilities if p > 0.0
    )
    uncertainty = float(entropy / np.log(4))

    original_size = float(rng.uniform(150.0, 300.0))
    compressed_size = original_size * 0.35
    thumbnail_size = original_size * 0.03

    return SatelliteImage(
        image_id=image_id,
        capture_time=capture_time,
        true_category=true_category,
        true_value=true_value,
        predicted_probabilities=predicted_probabilities,
        predicted_value=predicted_value,
        uncertainty=uncertainty,
        original_size=original_size,
        compressed_size=compressed_size,
        thumbnail_size=thumbnail_size,
    )
    

def set_image_representation(image, action):
    image.representation = {
        Action.THUMBNAIL: ImageRepresentation.THUMBNAIL,
        Action.COMPRESSED: ImageRepresentation.COMPRESSED,
        Action.FULL: ImageRepresentation.FULL,
    }[action]