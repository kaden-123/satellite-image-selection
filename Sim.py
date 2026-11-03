import numpy as np
import heapq

from MetricsRecorder import MetricsRecorder
from MissionState import MissionState
from Policy import FIFOPolicy, Action
from SatelliteImage import ImageRepresentation, ImageStatus, SatelliteImage
from ScoringConfig import ScoringConfig, ScientificCategory
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
    "Convert an image to its corresponding image representation from an action"
    image.representation = {
        Action.THUMBNAIL: ImageRepresentation.THUMBNAIL,
        Action.COMPRESSED: ImageRepresentation.COMPRESSED,
        Action.FULL: ImageRepresentation.FULL,
    }[action]

def run_simulation(config: SimulationConfig, policy, scoring: ScoringConfig, recorder : MetricsRecorder):
    """Run one policy and return its summary and final-image DataFrames."""
    rng = np.random.default_rng(config.rand_seed)
    state = MissionState(config)

    capture_times = generate_capture_times(config, rng)
    contact_times = generate_contact_times(config)

    events = [(time, 0, "capture") for time in capture_times]
    events += [(time, 1, "contact_start") for time in contact_times]
    contact_duration_days = config.contact_duration / 1440

    events.sort(key=lambda event: (event[0], event[1]))
    heapq.heapify(events)

    synthesize_images = True
    image_id = 0
    
    while events:
        event = heapq.heappop(events)
        match event[1]:
            case 0:
                #capture
                state.current_time = event[0]
                recorder.record_capture()
                if synthesize_images is True:
                    image = create_image(image_id, event[0], scoring, rng)
                else:
                    #find image from data base and ML classify it
                    raise NotImplementedError
                
                image_id += 1

                if image.original_size > state.available_storage_mb:
                    storage_plan = policy.on_storage_pressure(image, state)

                    action = storage_plan.incoming_action
                    if action == Action.DELETE:
                        image.status = ImageStatus.DELETED
                        recorder.record_final_image(image)
                        continue
                    else:
                        set_image_representation(image, action)

                    for target, action in storage_plan.onboard_actions:
                        if action == Action.DELETE:
                            target.status = ImageStatus.DELETED
                            state.onboard_images.remove(target)
                            recorder.record_final_image(target)
                        else:
                            set_image_representation(target, action)
                else:
                    action = policy.on_image_capture(image, state)
                    set_image_representation(image, action)
                    
                state.onboard_images.append(image)
                assert not state.storage_is_over_capacity
        
            case 1:
                #contact start
                state.current_time = event[0]
                recorder.record_contact()

                contact_end_time = contact_duration_days + event[0] 
                
                image, action, transmission_end_time = policy.on_contact_available(contact_end_time, state, config)
                # possibly add assert to account that action be FULL despite stored be COMPRESSED
                if action != Action.SKIP:
                    set_image_representation(image, action)
                    image.status = ImageStatus.TRANSMITTING
                    heapq.heappush(events, (transmission_end_time, 2, "transmission_end", 
                                            image, contact_end_time))

            case 2:
                #transmission end
                state.current_time = event[0]
                image = event[3]
                contact_end_time = event[4]

                image.status = ImageStatus.FULLY_DELIVERED
                image.downlink_time = event[0]
                state.onboard_images.remove(image)
                sent_value = scoring.delivered_value(image.true_category, image.representation)
                recorder.record_final_image(image, image.stored_size_mb, sent_value)

                image, action, transmission_end_time = policy.on_contact_available(contact_end_time, state, config)

                if action != Action.SKIP:
                    set_image_representation(image, action)
                    image.status = ImageStatus.TRANSMITTING
                    heapq.heappush(events, (transmission_end_time, 2, "transmission_end", 
                                            image, contact_end_time))

    
    for image in state.onboard_images:
        recorder.record_final_image(image)
    recorder.finish_run()