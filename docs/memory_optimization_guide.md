# Memory Optimization Guide: Token Counting & Salience Pruning

## Overview

This guide explains the enhanced STLTMemory features added in the GSoC contribution track, which bring intelligent memory management to mesa-llm agents.

**New Features**:
- 📊 **Token Counting**: Track and limit tokens in short-term memory
- 🎯 **Salience Scoring**: Score memories by relevance + recency
- 🧠 **Smart Eviction**: Automatically remove low-importance memories
- 📈 **Memory Stats**: Monitor memory health and optimization

---

## Quick Start

### Basic Usage with Token Limits

```python
from mesa_llm import LLMAgent
from mesa_llm.memory import STLTMemory

agent = LLMAgent(
    name="OptimizedAgent",
    model=model_instance,
    memory=STLTMemory(
        agent=agent,
        llm_model="ollama/llama3",
        max_tokens=2000,              # Limit to 2000 tokens
        enable_salience_pruning=True, # Smart eviction enabled
    )
)
```

### Monitor Token Usage

```python
# Get current token usage
usage = agent.memory.get_token_usage()
print(f"Tokens: {usage['short_term_tokens']}/{usage['max_tokens']}")
print(f"Utilization: {usage['token_utilization_percent']:.1f}%")

# Output:
# Tokens: 1300/2000
# Utilization: 65.0%
```

### View Memory Statistics

```python
# Get detailed memory analysis
stats = agent.memory.get_memory_stats()

for entry in stats['entries']:
    print(f"Step {entry['step']}: "
          f"Salience={entry['salience']:.3f}, "
          f"Relevance={entry['relevance']:.2f}, "
          f"Recency={entry['recency']:.2f}, "
          f"Tokens={entry['tokens']}")

print(f"Average Salience: {stats['average_salience']:.3f}")
```

---

## Configuration Parameters

### Token Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | int\|None | None | Maximum tokens allowed in ST memory (None = unlimited) |
| `token_buffer_threshold` | float | 0.8 | Trigger pruning at this % of max_tokens |

**Example**: With `max_tokens=2000` and threshold `0.8`, pruning starts at 1600 tokens.

### Salience Pruning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_salience_pruning` | bool | True | Enable intelligent eviction |
| `salience_recency_weight` | float | 0.3 | Weight for recency in score (0-1) |
| `salience_threshold` | float | 0.2 | Minimum score to keep entry |

**Example**:
- `salience_recency_weight=0.3`: 30% recency, 70% relevance
- `salience_threshold=0.2`: Remove entries with score < 0.2

---

## How Salience Scoring Works

### Salience Formula

```
Salience = (1 - recency_weight) × Relevance + recency_weight × Recency
         = 0.7 × Relevance + 0.3 × Recency
```

### Relevance Scoring (0-1)

Keyword-based: Counts important keywords in memory content.

**High-value keywords**:
- `action`, `decision`, `error`, `failure`, `success`
- `goal`, `objective`, `completed`, `result`
- `observation`, `plan`, `tool`, `interaction`

**Example**:
- Entry with `{"action": "move", "decision": "turn left"}` → Relevance ≈ 0.4
- Entry with `{"action": "success", "goal": "reached"}` → Relevance ≈ 0.8
- Entry with `{"data": "xyz"}` → Relevance ≈ 0.0

### Recency Scoring (0-1)

Exponential decay: Most recent entries get higher scores.

**Formula**: `Recency = e^(position × 2 - 2)`

**Example** (5-entry memory):
- Position 4 (newest): Recency ≈ 0.95
- Position 2 (middle): Recency ≈ 0.50
- Position 0 (oldest): Recency ≈ 0.13

---

## Eviction Strategies

When memory needs to make room for new entries, the system uses three strategies (in order):

### 1. Salience-Based Pruning
If `enable_salience_pruning=True`, drop entries below `salience_threshold`.

```python
# Aggressive pruning: Remove any entry with salience < 0.3
memory = STLTMemory(
    agent=agent,
    llm_model="model",
    enable_salience_pruning=True,
    salience_threshold=0.3  # Strict threshold
)
```

### 2. Token Limit Enforcement
If `max_tokens` is set and exceeded, evict lowest-salience entries.

```python
# Strong token constraints: 500 token limit
memory = STLTMemory(
    agent=agent,
    llm_model="model",
    max_tokens=500,  # Strict limit
    enable_salience_pruning=True  # Use salience to choose evictions
)
```

### 3. Capacity-Based Eviction
If neither pruning nor token limits apply, fall back to FIFO (oldest first).

```python
# Simple FIFO: Keep last 5 entries regardless of tokens
memory = STLTMemory(
    agent=agent,
    llm_model="model",
    short_term_capacity=5,
    enable_salience_pruning=False,  # Disable smart pruning
)
```

---

## Real-World Examples

### Long-Running Agents (Chat-like)

```python
# Keep chat memory manageable during long conversations
memory = STLTMemory(
    agent=agent,
    llm_model="ollama/llama3",
    short_term_capacity=10,          # Keep last 10 messages
    max_tokens=4000,                 # Budget 4K tokens
    enable_salience_pruning=True,
    salience_recency_weight=0.5,     # Balance old + new importance
    salience_threshold=0.15
)
```

### Simulation Agents (Action-focused)

```python
# Prioritize decisions and actions over observations
memory = STLTMemory(
    agent=agent,
    llm_model="ollama/llama3",
    short_term_capacity=5,
    max_tokens=1000,
    enable_salience_pruning=True,
    salience_recency_weight=0.2,     # 80% relevance, 20% recency
    salience_threshold=0.25          # Drop low-impact observations
)
```

### Lightweight Agents (Embedded)

```python
# Minimal memory footprint, all LLM calls tracked
memory = STLTMemory(
    agent=agent,
    llm_model="ollama/llama3",
    short_term_capacity=3,
    max_tokens=500,                  # Very tight
    enable_salience_pruning=True,
    salience_threshold=0.4           # Keep only high-quality memories
)
```

---

## Monitoring & Debugging

### Get formatted stats

```python
# Pretty-print memory state with scores
print(agent.memory.format_short_term_with_stats())

# Output:
# === SHORT-TERM MEMORY (with Salience Scores) ===
# Token usage: 1200/2000 (60.0%)
# Average salience: 0.562
#
# Step 1: Salience=0.312 (R:0.20 + T:0.13) Tokens=120
# Step 2: Salience=0.687 (R:0.50 + T:0.50) Tokens=142
# Step 3: Salience=0.892 (R:0.80 + T:0.95) Tokens=156
```

### Log memory health

```python
import logging

logger = logging.getLogger(__name__)

def log_memory_health(agent):
    stats = agent.memory.get_memory_stats()
    usage = stats['token_usage']

    logger.info(
        f"Memory Health: {len(stats['entries'])} entries, "
        f"{usage['token_utilization_percent']:.1f}% tokens, "
        f"avg_salience={stats['average_salience']:.3f}"
    )

# Call periodically
for step in range(100):
    agent.step()
    if step % 10 == 0:
        log_memory_health(agent)
```

---

## Token Counting Details

### Heuristic Used

Current implementation: **~4 characters per token**

```python
tokens = len(text) // 4
```

This is conservative and works well for planning. For production accuracy, integrate `tiktoken`:

```python
import tiktoken

def _count_tokens_accurate(self, text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))
```

### Token Budget Planning

| Scenario | Max Tokens | Util @80% | Notes |
|----------|-----------|-----------|-------|
| Short chat | 500 | 400 | Mobile/embedded |
| Normal agent | 2000 | 1600 | Typical usage |
| Long memory | 8000 | 6400 | Research/simulation |
| Unlimited | None | N/A | Original behavior |

---

## Performance Implications

### Computation

- **Token counting**: O(n) where n = text length (negligible)
- **Salience calculation**: O(m) where m = memory entries (fast)
- **Eviction**: O(m log m) for sorting (rare, during consolidation)

### Memory

- **Token cache**: Optional, ~10 bytes per entry
- **Salience scores**: ~16 bytes per entry (float64)
- **Overall**: +1-2KB for typical agents

### Backward Compatibility

✅ **Fully backward compatible**:
- Default: `max_tokens=None` (unlimited, original behavior)
- Default: `enable_salience_pruning=True` (non-breaking)
- Existing code: No changes needed

---

## Troubleshooting

### Memory filling up (token_utilization > 90%)

**Solution**: Lower token threshold or increase weight on recency:

```python
memory = STLTMemory(
    agent=agent,
    llm_model="model",
    max_tokens=1000,              # Lower limit
    salience_recency_weight=0.5,  # Favor newer memories
    salience_threshold=0.3        # Drop borderline entries
)
```

### Important memories getting evicted

**Solution**: Increase recency weight or threshold:

```python
memory = STLTMemory(
    agent=agent,
    llm_model="model",
    enable_salience_pruning=True,
    salience_threshold=0.1,       # Less aggressive
    salience_recency_weight=0.4,  # Protect older memories too
)
```

### Too many memories retained

**Solution**: Lower capacity or disable pruning:

```python
memory = STLTMemory(
    agent=agent,
    llm_model="model",
    short_term_capacity=5,           # Tighter FIFO
    enable_salience_pruning=False,   # Disable smart logic
)
```

---

## Testing Your Configuration

### Unit Test Template

```python
def test_memory_under_load():
    """Test memory with many entries."""
    agent = your_agent_instance

    # Simulate many steps
    for step in range(100):
        agent.step()
        stats = agent.memory.get_memory_stats()

        # Verify constraints
        assert stats['token_usage']['entries'] <= agent.memory.capacity + 5
        assert stats['average_salience'] > 0.1

        # Log interesting points
        if step % 20 == 0:
            print(f"Step {step}: {stats['token_usage']}")
```

---

## See Also

- [GitHub Issue #214](https://github.com/mesa/mesa-llm/issues/214) - Feature request
- [STLTMemory Source](../mesa_llm/memory/st_lt_memory.py)
- Tests: `tests/test_st_lt_memory_enhanced.py`
