"""
Comprehensive simulation recorder for mesa-llm simulations.

This module provides tools to record all simulation events for post-analysis,
including agent observations, plans, actions, messages, and state changes.
"""

import json
import logging
import pickle
import uuid
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SimulationEvent:
    """
    Dataclass representing a single recorded event in the simulation with complete context and metadata.

    Attributes:
        - **event_id** (*str*) - Unique identifier for this event
        - **timestamp** (*datetime*) - UTC timestamp when event occurred
        - **step** (*int*) - Simulation step number
        - **agent_id** (*int | None*) - Agent associated with event (None for model events)
        - **event_type** (*str*) - Type of event (observation, plan, action, message, state_change, etc.)
        - **content** (*dict*) - Event-specific data and information
        - **metadata** (*dict*) - Additional contextual metadata
    """

    event_id: str
    timestamp: datetime
    step: int
    agent_id: int | None
    event_type: str
    content: dict[str, Any]
    metadata: dict[str, Any]


class SimulationRecorder:
    """
    Centralized recorder for capturing all simulation events for post-analysis.
    It captures agent observations, plans, actions, messages, state changes, etc.
    as well as model-level events and transitions.

    Attributes:
        - **model** - Reference to the Mesa model being recorded
        - **events** - List of all recorded SimulationEvent objects
        - **simulation_id** - Unique identifier for this recording session
        - **start_time** - Recording start timestamp
        - **simulation_metadata** - Recording metadata and statistics
    """

    def __init__(
        self,
        model,
        output_dir: str = "recordings",
        record_state_changes: bool = True,
        auto_save_interval: int | None = None,
        storage_mode: str = "memory",
        max_events_in_memory: int | None = None,
    ):
        """
        Initialize the simulation recorder.

        Parameters:
            - **model** (*Model*) - Mesa model instance to record
            - **output_dir** (*str*) - Directory for saving recordings (default: "recordings")
            - **record_state_changes** (*bool*) - Whether to track agent state changes (default: True)
            - **auto_save_interval** (*int | None*) - Automatic save frequency in events (default: None)
            - **storage_mode** (*str*) - "memory" to keep all events in memory, or "jsonl" to stream events to disk while retaining only an optional in-memory window
            - **max_events_in_memory** (*int | None*) - Maximum number of recent events to retain in memory when using streaming mode. If None, no in-memory event window is kept in streaming mode
        """

        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if storage_mode not in {"memory", "jsonl"}:
            raise ValueError("storage_mode must be either 'memory' or 'jsonl'")
        if max_events_in_memory is not None and max_events_in_memory < 0:
            raise ValueError("max_events_in_memory must be >= 0")

        # Recording configuration
        self.record_state_changes = record_state_changes
        self.auto_save_interval = auto_save_interval
        self.storage_mode = storage_mode
        self.max_events_in_memory = max_events_in_memory

        # Internal state
        self.events: list[SimulationEvent] = []
        self.simulation_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now(UTC)
        self.total_events_recorded = 0

        # Agent state tracking for change detection
        self.previous_agent_states: dict[int, dict[str, Any]] = {}
        self.agent_summaries: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "total_events": 0,
                "event_types": set(),
                "active_steps": set(),
                "first_event": None,
                "last_event": None,
            }
        )
        self.unique_agent_ids: set[int] = set()
        self.recorded_event_types: set[str] = set()

        # Auto-save counter
        self.events_since_save = 0

        # Optional streaming path for unbounded runs
        self.events_stream_path: Path | None = None
        if self.storage_mode == "jsonl":
            self.events_stream_path = (
                self.output_dir / f"simulation_{self.simulation_id}_events.jsonl"
            )
            self.events_stream_path.touch()

        # Initialize simulation metadata
        self.simulation_metadata = {
            "simulation_id": self.simulation_id,
            "start_time": self.start_time.isoformat(),
            "model_class": self.model.__class__.__name__,
            "storage_mode": self.storage_mode,
        }

    @property
    def has_recorded_events(self) -> bool:
        """Whether the recorder has captured any events."""
        return self.total_events_recorded > 0

    def _serialize_event(self, event: SimulationEvent) -> dict[str, Any]:
        """Convert an event to a JSON-safe dictionary."""
        serialized = asdict(event)
        serialized["timestamp"] = event.timestamp.isoformat()
        return serialized

    def _deserialize_event(self, data: dict[str, Any]) -> SimulationEvent:
        """Convert serialized event data back into a SimulationEvent."""
        return SimulationEvent(
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            step=data["step"],
            agent_id=data["agent_id"],
            event_type=data["event_type"],
            content=data["content"],
            metadata=data["metadata"],
        )

    def _iter_all_events(self):
        """Iterate over all recorded events, including streamed events."""
        if self.storage_mode == "jsonl":
            if self.events_stream_path is None:
                return
            with open(self.events_stream_path) as f:
                for line in f:
                    if line.strip():
                        yield self._deserialize_event(json.loads(line))
            return

        yield from self.events

    def _update_agent_summary(self, event: SimulationEvent):
        """Maintain agent-level summary data without depending on full event history."""
        if event.agent_id is None:
            return

        self.unique_agent_ids.add(event.agent_id)
        summary = self.agent_summaries[event.agent_id]
        summary["total_events"] += 1
        summary["event_types"].add(event.event_type)
        summary["active_steps"].add(event.step)
        timestamp = event.timestamp.isoformat()
        if summary["first_event"] is None:
            summary["first_event"] = timestamp
        summary["last_event"] = timestamp

    def record_event(
        self,
        event_type: str,
        content: dict[str, Any] | str | None = None,
        agent_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        recipient_ids: list[int] | None = None,
    ):
        """Record a simulation event.

        Args:
            event_type: Type of event to record (observation, plan, action, message, state_change, etc.)
            content: Event content as dict or string
            agent_id: ID of the agent associated with this event
            metadata: Additional metadata for the event
            recipient_ids: List of recipient IDs for message events
        """

        # Handle different content formats based on event type
        if event_type == "message":
            if isinstance(content, str | dict | list):
                formatted_content = {
                    "message": content,
                    "recipient_ids": recipient_ids or [],
                }
            else:
                formatted_content = {
                    "message": content,
                    "recipient_ids": recipient_ids or [],
                }
        else:
            if isinstance(content, dict):
                formatted_content = content
            else:
                formatted_content = {"data": content}

        # Create the event
        event_id = f"{self.simulation_id}_{self.total_events_recorded:06d}"

        event = SimulationEvent(
            event_id=event_id,
            timestamp=datetime.now(UTC),
            step=self.model.steps,
            agent_id=agent_id,
            event_type=event_type,
            content=formatted_content,
            metadata=metadata,
        )

        self.total_events_recorded += 1
        self.recorded_event_types.add(event.event_type)
        self._update_agent_summary(event)

        if self.storage_mode == "jsonl":
            if self.events_stream_path is None:
                raise RuntimeError("events_stream_path is not initialized")
            with open(self.events_stream_path, "a") as f:
                json.dump(self._serialize_event(event), f)
                f.write("\n")
            if self.max_events_in_memory:
                self.events.append(event)
                if len(self.events) > self.max_events_in_memory:
                    self.events.pop(0)
        else:
            self.events.append(event)

        self.events_since_save += 1

        # Auto-save if configured
        if (
            self.auto_save_interval
            and self.events_since_save >= self.auto_save_interval
        ):
            filename = (
                f"autosave_{self.simulation_id}_{self.total_events_recorded}.json"
            )
            self.save(filename)
            self.events_since_save = 0

    def record_model_event(self, event_type: str, content: dict[str, Any]):
        """Record a model-level event."""
        self.record_event(
            event_type=event_type,
            content=content,
            agent_id=None,
            metadata={"source": "model"},
        )

    def get_agent_events(self, agent_id: int) -> list[SimulationEvent]:
        """Get all events for a specific agent."""
        return [
            event for event in self._iter_all_events() if event.agent_id == agent_id
        ]

    def get_events_by_type(self, event_type: str) -> list[SimulationEvent]:
        """Get all events of a specific type."""
        return [
            event for event in self._iter_all_events() if event.event_type == event_type
        ]

    def get_events_by_step(self, step: int) -> list[SimulationEvent]:
        """Get all events from a specific simulation step."""
        return [event for event in self._iter_all_events() if event.step == step]

    def export_agent_memory(self, agent_id: int) -> dict[str, Any]:
        """Export agent memory state for external analysis."""
        agent_events = self.get_agent_events(agent_id)

        return {
            "agent_id": agent_id,
            "events": [asdict(event) for event in agent_events],
            "summary": {
                "total_events": len(agent_events),
                "event_types": list({event.event_type for event in agent_events}),
                "active_steps": list({event.step for event in agent_events}),
                "first_event": (
                    agent_events[0].timestamp.isoformat() if agent_events else None
                ),
                "last_event": (
                    agent_events[-1].timestamp.isoformat() if agent_events else None
                ),
            },
        }

    def save(self, filename: str | None = None, format: str = "json"):
        """Save complete simulation recording.

        Args:
            filename: Optional filename. If None, auto-generates based on format.
            format: Save format, either "json" or "pickle".
        """
        if format not in ["json", "pickle"]:
            raise ValueError("Format must be 'json' or 'pickle'")

        if filename is None:
            extension = "json" if format == "json" else "pkl"
            filename = f"simulation_{self.simulation_id}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.{extension}"

        filepath = self.output_dir / filename

        # Update metadata with final state
        self.simulation_metadata.update(
            {
                "end_time": datetime.now(UTC).isoformat(),
                "total_steps": self.model.steps,
                "total_events": self.total_events_recorded,
                "total_agents": len(self.model.agents),
                "duration_minutes": (
                    datetime.now(UTC) - self.start_time
                ).total_seconds()
                / 60,
                # Determine completion status gracefully when `max_steps` is absent
                "completion_status": (
                    "unknown"
                    if getattr(self.model, "max_steps", None) is None
                    else (
                        "interrupted"
                        if self.model.steps < self.model.max_steps
                        else "completed"
                    )
                ),
            }
        )

        # Record final model state
        self.record_model_event(
            event_type="simulation_end",
            content={
                "status": (
                    "unknown"
                    if getattr(self.model, "max_steps", None) is None
                    else (
                        "interrupted"
                        if self.model.steps < self.model.max_steps
                        else "completed"
                    )
                ),
                "final_step": self.model.steps,
                "total_events": self.total_events_recorded,
            },
        )

        events = list(self._iter_all_events())
        self.simulation_metadata["total_events"] = len(events)

        # Prepare export data
        export_data = {
            "metadata": self.simulation_metadata,
            "events": [self._serialize_event(event) for event in events],
            "agent_summaries": {
                str(agent_id): {
                    "total_events": summary["total_events"],
                    "event_types": sorted(summary["event_types"]),
                    "active_steps": sorted(summary["active_steps"]),
                    "first_event": summary["first_event"],
                    "last_event": summary["last_event"],
                }
                for agent_id, summary in self.agent_summaries.items()
            },
        }

        if format == "json":
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(export_data, f)

        logger.info("Simulation recording saved to: %s", filepath)
        return filepath

    def get_stats(self) -> dict[str, Any]:
        """Get recording statistics."""
        return {
            "total_events": self.total_events_recorded,
            "unique_agents": len(self.unique_agent_ids),
            "event_types": sorted(self.recorded_event_types),
            "simulation_steps": self.model.steps,
            "recording_duration_minutes": (
                datetime.now(UTC) - self.start_time
            ).total_seconds()
            / 60,
            "events_per_agent": {
                agent_id: summary["total_events"]
                for agent_id, summary in self.agent_summaries.items()
            },
        }
