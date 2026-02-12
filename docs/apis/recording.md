# Recording Module

The recording system in Mesa-LLM provides comprehensive tools for capturing, analyzing, and visualizing simulation events. It enables researchers and developers to record all agent behavior, communications, decisions, and state changes for post-simulation analysis, debugging, and research insights.

## Usage Examples

### Basic Recording Setup

```python
from mesa_llm.recording.record_model import record_model
from mesa import Model

@record_model(output_dir="my_recordings", auto_save_interval=50)
class MyModel(Model):
   def __init__(self, **kwargs):
      super().__init__(**kwargs)
      # Model initialization
      # Recorder automatically attached after __init__

   def step(self):
      super().step()
      # Step events automatically recorded

# Save recording manually
model = MyModel()
model.run_simulation()
model.save_recording("final_simulation.json")
```

### Manual Recorder Integration

```python
from mesa_llm.recording.simulation_recorder import SimulationRecorder
from mesa_llm.llm_agent import LLMAgent

class MyAgent(LLMAgent):
   def __init__(self, model, **kwargs):
      super().__init__(model, **kwargs)

   def step(self):
      # Automatic recording of observations, plans, actions
      obs = self.generate_obs()
      plan = self.reasoning.plan(obs=obs)
      self.apply_plan(plan)

class MyModel(Model):
   def __init__(self, **kwargs):
      super().__init__(**kwargs)
      self.recorder = SimulationRecorder(
            model=self,
            output_dir="recordings",
            auto_save_interval=100
      )

      # Attach to agents
      for agent in self.agents:
            if hasattr(agent, 'recorder'):
               agent.recorder = self.recorder
```

### Custom Event Recording

```python
def step(self):
   # Record custom domain-specific events
   if hasattr(self, 'recorder'):
      self.recorder.record_event(
            event_type="negotiation_result",
            content={
               "participants": [self.unique_id, other_agent.unique_id],
               "outcome": "agreement_reached",
               "terms": self.negotiation_terms
            },
            agent_id=self.unique_id,
            metadata={"negotiation_round": self.round_number}
      )
```

### Analysis and Visualization

```python
from mesa_llm.recording.agent_analysis import AgentViewer, quick_agent_view

# Interactive exploration
viewer = AgentViewer("recordings/simulation_abc123_20240101_120000.json")
viewer.interactive_mode()

# Quick specific views
quick_agent_view("recording.json", agent_id=5, view_type="timeline")
quick_agent_view("recording.json", agent_id=5, view_type="conversations")
quick_agent_view("recording.json", agent_id=5, view_type="decisions")
quick_agent_view("recording.json", view_type="info")  # Simulation overview
```

### Event Type Categories

**Agent Events:**

- **observation** - Environmental perception and state awareness
- **plan** - Reasoning output and decision-making processes
- **action** - Tool execution and environment interaction
- **message** - Agent-to-agent communication
- **state_change** - Internal agent state modifications

**Model Events:**

- **simulation_start** - Recording initialization
- **simulation_end** - Recording completion with status
- **step_start** - Beginning of simulation step
- **step_end** - Completion of simulation step

**Custom Events:**

- Domain-specific events can be recorded with custom event_type strings
- Useful for tracking domain logic, negotiations, transactions, etc.

### Export and Integration

```python
# Export specific agent data
recorder = model.recorder
agent_data = recorder.export_agent_memory(agent_id=1)

# Get recording statistics
stats = recorder.get_stats()
print(f"Total events: {stats['total_events']}")
print(f"Active agents: {stats['unique_agents']}")

# Filter events for analysis
observations = recorder.get_events_by_type("observation")
step_events = recorder.get_events_by_step(10)
```

## Core Components

```{eval-rst}
.. automodule:: mesa_llm.recording.simulation_recorder
   :members:
   :undoc-members:
   :show-inheritance:
```

## Model recording integration

```{eval-rst}
.. automodule:: mesa_llm.recording.record_model
   :members:
   :undoc-members:
   :show-inheritance:
```

## Analysis utilities

```{eval-rst}
.. automodule:: mesa_llm.recording.agent_analysis
   :members:
   :undoc-members:
   :show-inheritance:
```
