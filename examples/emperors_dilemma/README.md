# The Emperor's Dilemma LLM Model

## Summary

This model shows how unpopular norms dominate a society even when the vast majority privately reject them. Agents are equipped with individual private beliefs and conviction levels. The model demonstrates the illusion of mass agreement: agents afraid of appearing disloyal not only comply with a norm they hate, but also aggressively enforce it on their neighbors. This creates a trap of False Enforcement, where the loudest defenders of a norm are often its secret opponents.

This model is implemented using Mesa-LLM, replacing the original mathematical threshold model with LLM-powered reasoning. Each agent uses natural language to reason about social pressure from neighbors and decides whether to publicly comply with the norm and whether to enforce it on others.

## Technical Details

The Emperor's Dilemma Model simulates the spread of an unpopular norm using a single type of agent, Citizens.

Each Citizen agent is characterized by:

- `private_belief`: Their true opinion of the norm (+1 supports, -1 rejects)
- `conviction`: How strongly they hold their private belief (0.0 to 1.0)
- `k`: The personal cost of enforcing the norm on neighbors
- `compliance`: Their current public stance (1 = publicly complying, -1 = resisting)
- `enforcement`: Whether they are actively pressuring neighbors (1 = enforcing, 0 = quiet)

---

### Compliance Rule (Original Mathematical Model)

In the original model, a Citizen publicly complies if:

```
social_pressure > conviction
```

Where:

```
social_pressure = (-private_belief / num_neighbors) * sum_of_neighbor_enforcement
```

---

### False Enforcement

A Citizen engages in False Enforcement when they:
- Privately reject the norm (`private_belief = -1`)
- But publicly enforce it on neighbors (`enforcement = 1`)

---

### LLM-Powered Agents

Citizens are implemented as LLM-powered agents, replacing the mathematical threshold with language-driven reasoning. Each agent:

- Receives a system prompt which defines their stand: their private belief, conviction strength, and background
- Observes their neighborhood: how many neighbors are complying and enforcing
- Reasons in natural language about whether to publicly comply and whether to enforce
- Uses tools to record their decisions back into the simulation



---

### The Model Tracks Three Key Metrics

- Compliance: Fraction of agents publicly obeying the norm
- Enforcement: Fraction of agents actively pressuring neighbors
- False Enforcement: Fraction of disbelievers who are enforcing the norm

## How to Run

Clone the mesa-llm repository and install it:

```bash
pip install -e .
```

You will need an API key from an LLM provider. This model makes a large number of calls per step — we recommend a provider with generous rate limits.

1. Install python-dotenv if not already installed:
```bash
pip install python-dotenv
```

2. In the root folder of the project, create a `.env` file.

3. Add your API key — for example for OpenAI:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```
Or for Groq:
```
GROQ_API_KEY=your-groq-api-key-here
```

4. Update the `LLM_MODEL` constant in `model.py` to your chosen model, in the format `{provider}/{model_name}`. For example:
```python
LLM_MODEL = "openai/gpt-4o-mini"
```

5. Run from this directory:
```bash
solara run app.py
```

**Note:** The default grid is 10x10 (100 agents). Each step requires approximately 200 LLM calls. For faster testing, reduce the grid size in `model.py` to 5x5.

## Files

* `model.py`: Core model code and data collection.
* `agents.py`: LLM-powered citizen agent class.
* `app.py`: Sets up the interactive Solara visualization.
* `tools.py`: Tools for agents to record compliance and enforcement decisions.

## Further Reading

This model is adapted from:

[Centola, D., Willer, R., & Macy, M. (2005). The Emperor's Dilemma: A Computational Model of Self-Enforcing Norms. American Journal of Sociology.](https://www.journals.uchicago.edu/doi/10.1086/427321)

You can also find Mesa's original rule-based version of the model here:
https://github.com/mesa/mesa-examples/tree/main/examples/emperor_dilemma