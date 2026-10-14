from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

from MissionState import MissionState
from SatelliteImage import ImageRepresentation, ImageStatus, SatelliteImage
from SimulationConfig import SimulationConfig


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

        if remaining_seconds <= 0 or state.config.bandwidth <= 0:
            return None, Action.SKIP, None

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