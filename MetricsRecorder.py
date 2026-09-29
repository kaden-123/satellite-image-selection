from time import perf_counter

import numpy as np

from SatelliteImage import ImageStatus, SatelliteImage

FINAL_IMAGE_RECORD_DTYPE = np.dtype([
    ("image_id", np.int64),
    ("capture_time", np.float64),
    ("true_category", np.int16),
    ("true_value", np.float32),
    ("predicted_value", np.float32),
    ("uncertainty", np.float32),
    ("final_status", np.int8),
    ("final_representation", np.int8),
    ("downlink_time", np.float64),
    ("mb_transmitted", np.float32),
    ("delivered_value", np.float32),
])


class MetricsRecorder:
    """Track run totals and one compact final record per simulated image."""

    def __init__(
        self,
        policy_name: str,
        random_seed: int,
        initial_capacity: int = 4096,
    ) -> None:
        """Initialize counters, compact record storage, and runtime timing."""
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be greater than 0")

        self.policy_name = policy_name
        self.random_seed = random_seed

        self._final_image_records = np.empty(
            initial_capacity,
            dtype=FINAL_IMAGE_RECORD_DTYPE,
        )
        self._final_image_count = 0

        # Wall-clock runtime measures Python execution time, not mission time.
        self._run_start_time = perf_counter()
        self.runtime_seconds = 0.0

        self.total_images_captured = 0
        self.storage_overflows = 0
        self.total_available_downlink_mb = 0.0
        self.total_transmitted_mb = 0.0

    def record_capture(self) -> None:
        """Increment the number of images captured during this run."""
        self.total_images_captured += 1

    def record_storage_overflow(self) -> None:
        """Increment the number of storage-overflow events."""
        self.storage_overflows += 1

    def record_contact(self, available_mb: float, transmitted_mb: float) -> None:
        """Record available and used capacity for one ground contact."""
        if available_mb < 0 or transmitted_mb < 0:
            raise ValueError("downlink amounts cannot be negative")
        if transmitted_mb > available_mb + 1e-9:
            raise ValueError("transmitted_mb cannot exceed available_mb")

        self.total_available_downlink_mb += available_mb
        self.total_transmitted_mb += transmitted_mb

    def record_final_image(
        self,
        image: SatelliteImage,
        mb_transmitted: float = 0.0,
        delivered_value: float = 0.0,
    ) -> None:
        """Save an image after delivery, deletion, or mission termination."""
        if mb_transmitted < 0 or delivered_value < 0:
            raise ValueError("final image values cannot be negative")
        if delivered_value > image.true_value + 1e-9:
            raise ValueError("delivered_value cannot exceed true_value")

        self._ensure_record_capacity()
        downlink_time = np.nan if image.downlink_time is None else image.downlink_time

        self._final_image_records[self._final_image_count] = (
            image.image_id,
            image.capture_time,
            image.true_category,
            image.true_value,
            image.predicted_value,
            image.uncertainty,
            int(image.status),
            int(image.representation),
            downlink_time,
            mb_transmitted,
            delivered_value,
        )
        self._final_image_count += 1

    def finish_run(self) -> None:
        """Stop the wall-clock timer after the simulation run completes."""
        self.runtime_seconds = perf_counter() - self._run_start_time

    @property
    def final_image_records(self) -> np.ndarray:
        """Return a view containing only finalized image records."""
        return self._final_image_records[:self._final_image_count]

    def run_summary(
        self,
        oracle_delivered_value: float | None = None,
    ) -> dict[str, int | float | str]:
        """Summarize this run, optionally against a matched oracle run."""
        if oracle_delivered_value is not None and oracle_delivered_value < 0:
            raise ValueError("oracle_delivered_value cannot be negative")

        records = self.final_image_records
        delivered = records["delivered_value"] > 0
        deleted = records["final_status"] == int(ImageStatus.DELETED)
        delays = records["downlink_time"] - records["capture_time"]
        valid_delays = delays[~np.isnan(delays)]

        total_available_value = float(records["true_value"].sum())
        total_delivered_value = float(records["delivered_value"].sum())
        total_value_loss = total_available_value - total_delivered_value

        normalized_delivered_value = (
            total_delivered_value / total_available_value
            if total_available_value > 0
            else 0.0
        )
        bandwidth_utilization = (
            self.total_transmitted_mb / self.total_available_downlink_mb
            if self.total_available_downlink_mb > 0
            else 0.0
        )
        value_per_mb = (
            total_delivered_value / self.total_transmitted_mb
            if self.total_transmitted_mb > 0
            else 0.0
        )

        if oracle_delivered_value is None:
            oracle_value = np.nan
            oracle_regret = np.nan
            normalized_oracle_regret = np.nan
        else:
            oracle_value = float(oracle_delivered_value)
            oracle_regret = oracle_value - total_delivered_value
            normalized_oracle_regret = (
                oracle_regret / oracle_value if oracle_value > 0 else 0.0
            )

        return {
            "policy_name": self.policy_name,
            "random_seed": self.random_seed,
            "total_images_captured": self.total_images_captured,
            "total_images_finalized": self._final_image_count,
            "total_images_delivered": int(delivered.sum()),
            "total_images_deleted": int(deleted.sum()),
            "total_available_true_value": total_available_value,
            "total_delivered_value": total_delivered_value,
            "total_value_loss": total_value_loss,
            "normalized_delivered_value": normalized_delivered_value,
            "oracle_delivered_value": oracle_value,
            "oracle_regret": oracle_regret,
            "normalized_oracle_regret": normalized_oracle_regret,
            "total_transmitted_mb": self.total_transmitted_mb,
            "storage_overflows": self.storage_overflows,
            "average_downlink_delay_days": (
                float(valid_delays.mean()) if valid_delays.size else np.nan
            ),
            "bandwidth_utilization": bandwidth_utilization,
            "value_per_mb": value_per_mb,
            "runtime_seconds": self.runtime_seconds,
        }

    def to_dataframes(self, oracle_delivered_value: float | None = None):
        """Convert the run summary and final image records to DataFrames."""
        import pandas as pd

        summary = pd.DataFrame([
            self.run_summary(oracle_delivered_value=oracle_delivered_value)
        ])
        image_records = pd.DataFrame.from_records(self.final_image_records)
        return summary, image_records

    def _ensure_record_capacity(self) -> None:
        """Double the record array when every preallocated row is occupied."""
        if self._final_image_count < len(self._final_image_records):
            return

        expanded_records = np.empty(
            len(self._final_image_records) * 2,
            dtype=FINAL_IMAGE_RECORD_DTYPE,
        )
        expanded_records[:self._final_image_count] = self._final_image_records
        self._final_image_records = expanded_records
