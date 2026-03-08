# Negotiation Model

**Disclaimer**: This is a toy model for illustrative purposes only.

## Summary

Two agents (buyer and seller) negotiate over products.

**Seller A**: Skilled negotiator. Sells shoes ($40) and track suits ($50).

**Seller B**: Less persuasive. Sells shoes ($35) and track suits ($47).

**Buyers**: Budget of $50 or $100.

## How to Run

### Step 1 - Install

From the root of the repository run: pip install -e .

### Step 2 - Create a .env file in the project root

OpenAI: OPENAI_API_KEY=sk-xxx

Gemini (free at https://aistudio.google.com/apikey): GEMINI_API_KEY=your-key

Anthropic: ANTHROPIC_API_KEY=your-key

### Step 3 - Update llm_model in app.py

OpenAI: openai/gpt-4o-mini
Gemini: gemini/gemini-1.5-flash
Anthropic: anthropic/claude-3-haiku-20240307

### Step 4 - Run

solara run examples/negotiation/app.py

Open browser at http://localhost:8765

WSL users: Navigate to http://localhost:8765 manually.

## Files

- model.py: Core model code.
- agents.py: Agent classes.
- app.py: Visualization.
- tools.py: Tools for LLM agents.
