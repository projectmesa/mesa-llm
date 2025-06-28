import os

import pinecone
import weaviate
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import Pinecone, Qdrant, Weaviate
from mesa.agent import Agent
from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.model import Model
from mesa.space import (
    ContinuousSpace,
    MultiGrid,
    SingleGrid,
)
from qdrant_client import QdrantClient

from mesa_llm import Plan
from mesa_llm.memory import Memory
from mesa_llm.module_llm import ModuleLLM
from mesa_llm.reasoning.reasoning import (
    Observation,
    Reasoning,
)
from mesa_llm.recording.simulation_recorder import SimulationRecorder
from mesa_llm.tools.tool_manager import ToolManager


class LLMAgent(Agent):
    """
    LLMAgent manages an LLM backend and optionally connects to a memory module.

    Parameters:
        model (Model): The mesa model the agent in linked to.
        api_key (str): The API key for the LLM provider.
        llm_model (str): The model to use for the LLM in the format 'provider/model'. Defaults to 'gemini/gemini-2.0-flash'.
        system_prompt (str | None): Optional system prompt to be used in LLM completions.
        reasoning (str): Optional reasoning method to be used in LLM completions.

    Attributes:
        llm (ModuleLLM): The internal LLM interface used by the agent.
        memory (Memory | None): The memory module attached to this agent, if any.

    """

    def __init__(
        self,
        model: Model,
        api_key: str,
        reasoning: type[Reasoning],
        llm_model: str = "gemini/gemini-2.0-flash",
        system_prompt: str | None = None,
        vision: float | None = None,
        internal_state: list[str] | str | None = None,
        recorder: SimulationRecorder | None = None,
    ):
        super().__init__(model=model)

        self.model = model

        self.llm = ModuleLLM(
            api_key=api_key, llm_model=llm_model, system_prompt=system_prompt
        )

        self.memory = Memory(
            agent=self,
            short_term_capacity=5,
            consolidation_capacity=2,
            api_key=api_key,
            llm_model=llm_model,
        )

        self.tool_manager = ToolManager()
        self.recorder = recorder

        self.vision = vision
        self.reasoning = reasoning(agent=self)
        self.system_prompt = system_prompt
        self.is_speaking = False
        self._current_plan = None  # Store current plan for formatting

        # display coordination
        self._step_display_data = {}

        if isinstance(internal_state, str):
            internal_state = [internal_state]
        elif internal_state is None:
            internal_state = []

        self.internal_state = internal_state

    def __str__(self):
        return f"LLMAgent {self.unique_id}"

    def apply_plan(self, plan: Plan) -> list[dict]:
        """
        Execute the plan in the simulation.
        """
        # Store current plan for display
        self._current_plan = plan

        # Execute tool calls
        tool_call_resp = self.tool_manager.call_tools(
            agent=self, llm_response=plan.llm_plan
        )

        # Add to memory
        self.memory.add_to_memory(
            type="action",
            content={
                k: v
                for tool_call in tool_call_resp
                for k, v in tool_call.items()
                if k not in ["tool_call_id", "role"]
            },
        )

        if self.recorder is not None:
            self.recorder.record_event(
                event_type="action",
                content={"tool_call_response": tool_call_resp},
                agent_id=self.unique_id,
            )

        return tool_call_resp

    def add_doc(self, vector_db, embedding_model, doc_path, llm):
        # Load env variables
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")
        backend = vector_db.strip().lower()  # pinecone / weaviate / qdrant

        # Prompt user for document path

        # Load document
        if doc_path.endswith(".pdf"):
            loader = PyPDFLoader(doc_path)
        elif doc_path.endswith(".txt"):
            loader = TextLoader(doc_path)
        else:
            raise ValueError("Only .txt and .pdf files supported.")

        docs = loader.load()

        # Set up vector DBs
        if backend == "pinecone":
            # Init Pinecone
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV"),
            )
            index_name = "langchain-rag"
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(index_name, dimension=1536)
            vectorstore = Pinecone.from_documents(
                docs, embedding_model, index_name=index_name
            )

        elif backend == "weaviate":
            client = weaviate.Client(
                url=os.getenv("WEAVIATE_URL"),
                auth_client_secret=weaviate.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
            )

            index_name = "Document"
            if not client.schema.contains({"class": index_name}):
                client.schema.create_class({"class": index_name, "vectorizer": "none"})

            vectorstore = Weaviate.from_documents(
                documents=docs,
                embedding=embedding_model,
                client=client,
                index_name=index_name,
            )

        elif backend == "qdrant":
            client = QdrantClient(
                url="https://your-qdrant-cloud-instance.com",  # Replace with your Qdrant Cloud URL
                api_key=os.getenv("QDRANT_API_KEY"),
            )
            vectorstore = Qdrant.from_documents(
                documents=docs,
                embedding=embedding_model,
                qdrant_client=client,
                collection_name="rag-docs",
            )

        else:
            raise ValueError("Unsupported backend.")

        # LLM setup
        llm = ChatOpenAI(openai_api_key=openai_key, temperature=0)

        # Create RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True
        )
        return rag_chain

    def generate_obs(self) -> Observation:
        """
        Returns an instance of the Observation dataclass enlisting everything the agent can see in the model in that step.

        If the agents vision is set to anything above 0, the agent will get the details of all agents falling in that radius.
        If the agents vision is set to -1, then the agent will get the details of all the agents present in the simulation at that step.
        If it is set to 0 or None, then no information is returned to the agent.

        """
        step = self.model.steps

        self_state = {
            "agent_unique_id": self.unique_id,
            "system_prompt": self.system_prompt,
            "location": self.pos if self.pos is not None else self.cell.coordinate,
            "internal_state": self.internal_state,
        }
        if self.vision is not None and self.vision > 0:
            if isinstance(self.model.grid, SingleGrid | MultiGrid):
                neighbors = self.model.grid.get_neighbors(
                    tuple(self.pos), moore=True, include_center=False, radius=1
                )
            elif isinstance(
                self.model.grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid
            ):
                neighbors = []
                for neighbor in self.cell.connections.values():
                    neighbors.extend(neighbor.agents)

            elif isinstance(self.model.space, ContinuousSpace):
                neighbors, _ = self.get_neighbors_in_radius(radius=self.vision)

        elif self.vision == -1:
            all_agents = list(self.model.agents)
            neighbors = [agent for agent in all_agents if agent is not self]

        else:
            neighbors = []

        local_state = {}
        for i in neighbors:
            local_state[i.__class__.__name__ + " " + str(i.unique_id)] = {
                "position": i.pos if i.pos is not None else i.cell.coordinate,
                "internal_state": i.internal_state,
            }

        # Add to memory (memory handles its own display separately)
        self.memory.add_to_memory(
            type="observation",
            content={
                "self_state": self_state,
                "local_state": local_state,
            },
        )

        # --------------------------------------------------
        # Recording hook
        # --------------------------------------------------
        if self.recorder is not None:
            self.recorder.record_event(
                event_type="observation",
                content={"self_state": self_state, "local_state": local_state},
                agent_id=self.unique_id,
            )

            # Track state changes for the agent (location & internal state)
            self.recorder.track_agent_state(
                agent_id=self.unique_id,
                current_state={
                    "location": tuple(self.pos) if self.pos is not None else None,
                    "internal_state": self.internal_state,
                },
            )

        return Observation(step=step, self_state=self_state, local_state=local_state)

    def send_message(self, message: str, recipients: list[Agent]) -> str:
        """
        Send a message to the recipients.
        """
        for recipient in [*recipients, self]:
            recipient.memory.add_to_memory(
                type="message",
                content={
                    "message": message,
                    "sender": self,
                    "recipients": recipients,
                },
            )

        if self.recorder:
            self.recorder.record_event(
                event_type="message",
                content=message,
                agent_id=self.unique_id,
                recipient_ids=[recipient.unique_id for recipient in recipients],
            )

        return f"{self} → {recipients} : {message}"

    def pre_step(self):
        """
        This is some code that is executed before the step method of the child agent is called.
        """
        self.memory.process_step(pre_step=True)

    def post_step(self):
        """
        This is some code that is executed after the step method of the child agent is called.
        It functions because of the __init_subclass__ method that creates a wrapper around the step method of the child agent.
        """
        self.memory.process_step()

    def __init_subclass__(cls, **kwargs):
        """
        Wrapper - allows to automatically integrate code to be executed after the step method of the child agent (created by the user) is called.
        """
        super().__init_subclass__(**kwargs)
        # only wrap if subclass actually defines its own step
        user_step = cls.__dict__.get("step")
        if not user_step:
            return

        def wrapped(self, *args, **kwargs):
            """
            This is the wrapper that is used to integrate the pre_step and post_step methods into the step method of the child agent.
            """
            LLMAgent.pre_step(self, *args, **kwargs)
            result = user_step(self, *args, **kwargs)
            LLMAgent.post_step(self, *args, **kwargs)
            return result

        cls.step = wrapped
