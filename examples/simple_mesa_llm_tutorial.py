"""
MESA BASIC TUTORIAL (FOR BEGINNERS)

Mesa is a Python library used to simulate many small "agents"
that act inside a "model" (world).

Think of it like a game:
- Agents = players
- Model = game controller
- Step = one turn of the game
"""

# Make sure Mesa is installed: pip install mesa
import mesa


class SimpleAgent(mesa.Agent):
    def __init__(self, model):
        # Connect the agent to the model (world)
        super().__init__(model)

    def step(self):
        # step() means what the agent does in one time step
        print("User: What are you doing?")
        print(f"Agent {self.unique_id}: I am doing my work.")


class SimpleModel(mesa.Model):
    def __init__(self):
        # Initialize the model
        super().__init__()

        # Create 6 agents (IDs: 0 to 5)
        for _ in range(6):
            SimpleAgent(self)

    def step(self):
        # Run all agents once in random order
        self.agents.shuffle_do("step")


# -------------------------
# RUN THE MODEL
# -------------------------
if __name__ == "__main__":
    model = SimpleModel()  # create the world
    model.step()           # run one step
