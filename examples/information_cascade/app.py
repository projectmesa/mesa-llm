import traceback

import solara
from dotenv import load_dotenv
from model import MarketPanicModel

load_dotenv()

sim_model = solara.reactive(None)
error_msg = solara.reactive(None)
step_count = solara.reactive(0)


@solara.component
def Page():
    if error_msg.value:
        solara.Error(f"CRASH DETECTED:\n\n{error_msg.value}")
        return

    if sim_model.value is None:
        try:
            sim_model.value = MarketPanicModel(num_agents=4)
        except Exception:
            error_msg.value = traceback.format_exc()
            return

    solara.Markdown("# 📈 Information Cascade (Financial Rumor Mill)")
    solara.Markdown(f"### Current Simulation Step: {step_count.value}")

    def on_step():
        try:
            sim_model.value.step()
            step_count.value += 1
        except Exception:
            error_msg.value = traceback.format_exc()

    solara.Button("Run 1 Step", on_click=on_step, color="primary")


page = Page
