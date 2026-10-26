import xgboost as xgb
import numpy as np
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

from MissionState import MissionState
from SatelliteImage import ImageRepresentation, ImageStatus, SatelliteImage
from SimulationConfig import SimulationConfig
from ScoringConfig import ScoringConfig


SECONDS_PER_DAY = 86400


class Action(IntEnum):
    """Policy actions interpreted according to the current event.

    During capture/storage planning, THUMBNAIL, COMPRESSED, and FULL describe
    how the incoming image should be stored. For an onboard target, DELETE,
    THUMBNAIL, and COMPRESSED mean delete or irreversibly reduce it. During a
    contact, the representation actions describe what should be transmitted.
    """

    SKIP = 0
    THUMBNAIL = 1
    COMPRESSED = 2
    FULL = 3
    DELETE = 4


@dataclass(slots=True)
class StoragePlan:
    """Complete response to storage pressure caused by an incoming image.

    ``incoming_action`` determines whether the incoming image is deleted or
    stored as a thumbnail, compressed copy, or full-resolution copy.
    ``onboard_actions`` contains existing images to delete or reduce first.
    """

    incoming_action: Action
    onboard_actions: list[tuple[SatelliteImage, Action]]


class Policy(ABC):
    """Interface implemented by every onboard selection policy.

    Policies inspect predictions and mission state and return decisions. They
    must not directly add, remove, or modify images in ``MissionState``; the
    simulation validates and applies every returned decision.
    """

    name: ClassVar[str] = "base"

    @abstractmethod
    def on_image_capture(self, image: SatelliteImage, state: MissionState) -> Action:
        """Choose the preferred representation for a newly captured image."""
        raise NotImplementedError

    @abstractmethod
    def on_storage_pressure(self, image: SatelliteImage, state: MissionState) -> StoragePlan:
        """Plan how to make room when the incoming full image does not fit."""
        raise NotImplementedError

    @abstractmethod
    def on_contact_available(self, 
                             contact_end_time: float, 
                             state: MissionState, 
                             config : SimulationConfig) -> tuple[SatelliteImage | None, Action, float | None]:
        """Choose an image and representation that can finish in this contact."""
        raise NotImplementedError


class XGBPolicy(Policy):
    "Uses trained XGB ranker to select images to transmit while using FIFO policy on storage pressure"

    name: ClassVar[str] = "xgb"

    def __init__(self):
        self.ranker = xgb.XGBRanker()
        self.ranker.load_model("./data/XGBRanker.json")

    def on_image_capture(self, image: SatelliteImage, state: MissionState) -> Action:
        """Prefer to store every newly captured image at full resolution."""
        return FIFOPolicy().on_image_capture(image, state)

    def on_storage_pressure(self, image: SatelliteImage, state: MissionState) -> StoragePlan:
        """Plan deletion of the oldest images until the incoming image can fit."""
        return FIFOPolicy().on_storage_pressure(image, state)

    def on_contact_available(self,
                             contact_end_time: float,
                             state: MissionState, 
                             config :SimulationConfig) -> tuple[SatelliteImage | None, Action, float | None]:

        remaining_seconds = (
                contact_end_time - state.current_time
            ) * SECONDS_PER_DAY

        remaining_days = remaining_seconds / SECONDS_PER_DAY
        
        candidates = []
        features = []
        for image in state.onboard_images:
            if image.status != ImageStatus.STORED:
                continue
            for rep in ImageRepresentation:
                if rep > image.representation:
                    continue
                size_mb = image.representation_sizes[rep]
                transmission_seconds = size_mb * 8 / config.bandwidth
                if transmission_seconds <= remaining_seconds:
                    candidates.append((image, rep, transmission_seconds))
                    features.append((remaining_seconds / SECONDS_PER_DAY,
                                     image.predicted_value, size_mb))

        if not candidates:
            return None, Action.SKIP, None
        
        scores = self.ranker.predict(np.asarray(features, dtype=float))
        image, rep, transmission_seconds = candidates[int(np.argmax(scores))]
        end_time = state.current_time + transmission_seconds / SECONDS_PER_DAY
        
        return image, Action(int(rep) + 1), end_time


class GreedyPolicy(Policy):
    "Implements a greedy algorithm to optimize transmissions, storage pressure and capture"
    
    name: ClassVar[str] = "greedy"

    def __init__(self, scoring: ScoringConfig):
        self.scoring = scoring

    def on_image_capture(self, image: SatelliteImage, state: MissionState) -> Action:
        "Store full image as its the 'best' decision available"
        return Action.FULL

    def on_storage_pressure(self, image: SatelliteImage, state: MissionState) -> StoragePlan:
        """Free space by choosing the cheapest next reduction per MB."""
        candidates = [image] + [
            stored for stored in state.onboard_images
            if stored.status == ImageStatus.STORED
        ]
        reps = [ImageRepresentation.FULL] + [
            stored.representation for stored in candidates[1:]
        ]
        initial_reps = reps.copy()
        projected_used = state.used_storage_mb + image.original_size
        multipliers = self.scoring.representation_multipliers

        heap = []

        def add_next_reduction(i):
            rep = reps[i]
            if rep is None:
                return
            candidate = candidates[i]
            for lower in range(int(rep) - 1, -2, -1):
                next_rep = ImageRepresentation(lower) if lower >= 0 else None
                new_size = candidate.representation_sizes[next_rep] if next_rep is not None else 0.0
                freed = candidate.representation_sizes[rep] - new_size
                if freed > 0:
                    old_value = candidate.predicted_value * multipliers[rep]
                    new_value = candidate.predicted_value * multipliers[next_rep] if next_rep is not None else 0.0
                    heapq.heappush(heap, ((old_value - new_value) / freed, i, next_rep, freed))
                    return

        for i in range(len(candidates)):
            add_next_reduction(i)

        while projected_used > state.config.storage_capacity_mb:
            if not heap:
                return StoragePlan(Action.DELETE, [])

            _, chosen, next_rep, freed = heapq.heappop(heap)
            reps[chosen] = next_rep
            projected_used -= freed

            if chosen == 0 and next_rep is None:
                return StoragePlan(Action.DELETE, [])
            add_next_reduction(chosen)

        onboard_actions = [
            (candidate, Action.DELETE if rep is None else Action(rep + 1))
            for candidate, rep, initial in zip(candidates[1:], reps[1:], initial_reps[1:])
            if rep != initial
        ]
        incoming_action = Action.DELETE if reps[0] is None else Action(reps[0] + 1)
        return StoragePlan(incoming_action, onboard_actions)


    def on_contact_available(self,
                             contact_end_time: float,
                             state: MissionState,
                             config: SimulationConfig) -> tuple[SatelliteImage | None, Action, float | None]:
        """Send the feasible image/representation with the most predicted value per MB."""
    
        remaining_seconds = (contact_end_time - state.current_time) * SECONDS_PER_DAY

        best = None
        best_key = None

        for image in state.onboard_images:
            if image.status != ImageStatus.STORED:
                continue

            for rep in ImageRepresentation:
                if rep > image.representation:
                    continue

                size_mb = image.representation_sizes[rep]
                transmission_seconds = size_mb * 8 / config.bandwidth
                if transmission_seconds > remaining_seconds:
                    continue

                expected_value = image.predicted_value * self.scoring.representation_multipliers[rep]
                key = (expected_value / size_mb, expected_value)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (image, rep, transmission_seconds)

        if best is None:
            return None, Action.SKIP, None

        image, rep, transmission_seconds = best
        transmission_end_time = state.current_time + transmission_seconds / SECONDS_PER_DAY
        return image, Action(int(rep) + 1), transmission_end_time


class FIFOPolicy(Policy):
    """Keep new images at full resolution and prioritize the oldest captures."""

    name: ClassVar[str] = "fifo"

    def on_image_capture(self, image: SatelliteImage, state: MissionState) -> Action:
        """Prefer to store every newly captured image at full resolution."""
        return Action.FULL

    def on_storage_pressure(self, image: SatelliteImage, state: MissionState) -> StoragePlan:
        """Plan deletion of the oldest images until the incoming image can fit."""
        if image.original_size > state.config.storage_capacity_mb:
            return StoragePlan(Action.DELETE, [])

        projected_available_mb = state.available_storage_mb
        onboard_actions: list[tuple[SatelliteImage, Action]] = []

        for stored_image in sorted(
            (stored_image for stored_image in state.onboard_images
                if stored_image.status == ImageStatus.STORED),
            key=lambda stored: (stored.capture_time, stored.image_id),
        ):
            
            if projected_available_mb >= image.original_size:
                break

            onboard_actions.append((stored_image, Action.DELETE))
            projected_available_mb += stored_image.stored_size_mb

        if projected_available_mb < image.original_size:
            return StoragePlan(Action.DELETE, [])

        return StoragePlan(Action.FULL, onboard_actions)
        
    def on_contact_available(self, 
                             contact_end_time: float, 
                             state: MissionState, 
                             config : SimulationConfig) -> tuple[SatelliteImage | None, Action, float | None]:
        """Select the oldest image and highest-quality version that can finish."""
        remaining_seconds = (
            contact_end_time - state.current_time
        ) * SECONDS_PER_DAY

        for image in sorted(
            (image for image in state.onboard_images 
                if image.status == ImageStatus.STORED), 
            key=lambda stored: (stored.capture_time, stored.image_id)
        ):
            
            transmission_seconds = image.original_size * 8 / state.config.bandwidth

            if transmission_seconds <= remaining_seconds:
                transmission_end_time = (state.current_time + transmission_seconds / SECONDS_PER_DAY)
                return (image, Action.FULL, transmission_end_time)

        return None, Action.SKIP, None

