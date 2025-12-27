from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_manager import ToolManager

seller_tool_manager = ToolManager()
buyer_tool_manager = ToolManager()


class SellerAgent(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )

        self.tool_manager = seller_tool_manager
        self.sales = 0

    def step(self):
        observation = self.generate_obs()
        prompt = "Don't move around. If there are any buyers in your cell or in the neighboring cells, pitch them your product using the speak_to tool. Talk to them until they agree or definitely refuse to buy your product."
        plan = self.reasoning.plan(
            prompt=prompt, obs=observation, selected_tools=["speak_to"]
        )
        self.apply_plan(plan)

    async def astep(self):
        observation = self.generate_obs()
        prompt = "Don't move around. If there are any buyers in your cell or in the neighboring cells, pitch them your product using the speak_to tool. Talk to them until they agree or definitely refuse to buy your product."
        plan = await self.reasoning.aplan(
            prompt=prompt, obs=observation, selected_tools=["speak_to"]
        )
        self.apply_plan(plan)


class BuyerAgent(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        budget,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = buyer_tool_manager
        self.budget = budget
        self.products = []

    def step(self):
        observation = self.generate_obs()
        prompt = f"Move around by using the teleport_to_location tool if you are not talking to a seller, grid dimensions are {self.model.grid.width} x {self.model.grid.height}. Seller agents around you might try to pitch their product by sending you messages, get as much information as possible. When you have enough information, decide what to buy the product."
        print(self.tool_manager.tools)
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["teleport_to_location", "speak_to", "buy_product"],
        )
        self.apply_plan(plan)

    async def astep(self):
        observation = self.generate_obs()
        prompt = f"Move around by using the teleport_to_location tool if you are not talking to a seller, grid dimensions are {self.model.grid.width} x {self.model.grid.height}. Seller agents around you might try to pitch their product by sending you messages, get as much information as possible. When you have enough information, decide what to buy the product."
        print(self.tool_manager.tools)
        plan = await self.reasoning.aplan(
            prompt=prompt,
            obs=observation,
            selected_tools=["teleport_to_location", "speak_to", "buy_product"],
        )
        self.apply_plan(plan)
