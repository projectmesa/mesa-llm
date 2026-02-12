# Sugarscape Constant Growback Model with Traders

## Summary

This model is based on Epstein & Axtell's classic "Sugarscape" simulation from Growing Artificial Societies (1996), specifically the G1MT (Growback 1, Metabolism, Trade) variation. Trader agents wander a grid populated with two unevenly distributed resources: Sugar and Spice. Agents are endowed with individual metabolic rates for each resource and a vision range; there are also Resource agents, which represent the landscape and regenerate food over time.

The model generates emergent economic dynamics through decentralized interactions. Traders must constantly harvest resources to satisfy their metabolic needs; if they run out of either sugar or spice, they starve. Crucially, agents can trade with neighbors. Decisions are governed by the Marginal Rate of Substitution (MRS); agents rich in sugar but poor in spice will trade sugar to acquire spice, and vice versa. Over time, this decentralized trading allows for the emergence of a price equilibrium and wealth distribution patterns.

This model is implemented using Mesa-LLM, unlike the original deterministic versions. All Trader agents use Large Language Models to "think" about their survival. They observe their internal inventory and MRS, then autonomously decide to use tools to move to high-value resource tiles or propose trades to neighbors to ensure their continued existence.

## Technical Details

### Agents

- `Trader (LLMAgent):` The primary actor equipped with STLTMemory and ReActReasoning.

        Internal State: Dynamically updates a context string with current inventory (Sugar, Spice) and hunger warnings to guide the LLM.

        Metabolism: Consumes a fixed amount of resources per step. Zero inventory results in agent removal (death).

        MRS Calculation: Computes the Marginal Rate of Substitution (MRS) using the Cobb-Douglas formula to value Sugar vs. Spice relative to biological needs.

- `Resource (CellAgent):` A passive environmental agent that acts as a container for resources. It regenerates its current_amount by 1 unit per step up to a max_capacity.

### Tools

- `move_to_best_resource:`

        Function: Scans the local grid within the agent's vision radius.

        Action: Identifies the cell with the highest current_amount of resources, moves the agent there, and automatically harvests the full amount into the agent's inventory.

-  `propose_trade:`

        Function: Targets a specific neighbor by unique_id.

        Logic: Executes a trade only if the partner's MRS is higher than the proposer's (indicating the partner values Sugar more highly). This ensures trades are mathematically rational and mutually beneficial.


### Movement Rule (Rule M)

A Trader agent moves to a new location and harvests resources if the following logic, executed by the *move_to_best_resource tool*, is satisfied:

Scan: The agent inspects all cells within its *vision range* (von Neumann neighborhood).

Identify: It identifies the cell containing a *Resource* agent with the highest *current_amount* of sugar/spice.

Harvest: The agent moves to that cell, sets the resource's amount to 0, and adds the harvested amount to its own inventory.

### Trade Rule (Rule T)

Agents determine whether to trade based on their Marginal Rate of Substitution (MRS). A trade is proposed via the propose_trade tool and accepted if it is mutually beneficial:

```
Trade occurs if: Partner_MRS > Agent_MRS
```

Where the MRS is calculated using the agent's inventory and metabolism:

```
MRS = (spice_inventory / spice_metabolism) / (sugar_inventory / sugar_metabolism)
```

In this implementation:

-    `Agent (Proposer):` Gives Sugar, Receives Spice.

-    `Partner (Receiver):` Receives Sugar, Gives Spice.

-    This flow ensures resources move from agents who value them less to agents who value them more.

### Resource Behavior

Resource Agents represent the landscape. They are passive agents that regenerate wealth over time:

-    `Growback:` At every step, a Resource agent increases its current_amount by growback (default: 1).

-    `Capacity:` This growth is capped at the agent's max_capacity.

### LLM-Powered Agents

Both Traders and the simulation logic are driven by LLM-powered agents, meaning:

-    Their actions (e.g., `move_to_best_resource`, `propose_trade`) are determined by a ReAct reasoning module.

-    This module takes as input:

        The agent’s internal state (current inventory, metabolic warnings, and calculated MRS).

        Local observations of the grid.

-        A set of available tools defined in `tools.py`.



## How to Run

If you have cloned the repo into your local machine, ensure you run the following command from the root of the library: ``pip install -e . ``. Then, you will need an api key of an LLM-provider of your choice. (This model in particular makes a large amount of calls per minute and we therefore recommend getting a paid version of an api-key that can offer high rate-limits). Once you have obtained the api-key follow the below steps to set it up for this model.
1) Ensure the dotenv package is installed. If not, run ``pip install python-dotenv``.
2) In the root folder of the project, create a file named .env.
3) If you are using openAI's api key, add the following command in the .env file: ``OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx``. If you have the paid version of Gemini, use this line instead: ``GEMINI_API_KEY=your-gemini-api-key-here``(the free ones tend to not work with this model).
4) Change the  ``api_key`` specification in app.py according to the provider you have chosen.
5) Similarly change the ``llm_model`` attribute as well in app.py to the name of a model you have access to. Ensure it is in the form of {provider}/{model_name}. For e.g. ``openai/gpt-4o-mini``.

Once you have set up the api-key in your system, run the following command from this directory:

```
    $ solara run app.py
```


## Files

* ``model.py``: Core model code.
* ``agents.py``: Agent classes.
* ``app.py``: Sets up the interactive visualization.
* ``tools.py``: Tools for the llm-agents to use.


## Further Reading

[Growing Artificial Societies](https://mitpress.mit.edu/9780262550253/growing-artificial-societies/)
[Complexity Explorer Sugarscape with Traders Tutorial](https://www.complexityexplorer.org/courses/172-agent-based-models-with-python-an-introduction-to-mesa#gsc.tab=0)