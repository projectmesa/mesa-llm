# Negotiation Model

**Disclaimer**: This is a toy model designed for illustrative purposes and is not based on a real research paper.

## Summary

This model is a simple negotiation model where two types of agents (buyer and seller) negotiate over a product. The objective is to test out which seller is able to capture most sales and therefore determine what influences customers the most: The Sales pitch, or the value for money.

**Seller A** : Has all the necessary skills for negotiation. It's comfortable with persuation, and is pretty committed to the job. It's pitching for Brand A and is aiming to sell running shoes and track suits. Although the seller has a more superior skill-set, Brand A products are slightly more over-prized for the same quality : $40 for Shoes and $50 for the Track Suit

**Seller B** : Is a bit more apathetic and is not particularly good at persuation or negotiation. It sells the same products as seller A, but is prized slightly cheaper. : $35 for Shoes and $47 for Track Suit

**Buyers** : there are mainly 2 types of buyers, buyers with a budget of $50 and buyers who have a budget of $100.

## Agent Decision Logic

Both buyers and sellers are implemented as LLM-powered agents. Their actions (such as moving, speaking, or buying) are determined by a reasoning module that receives the agent’s internal state, local observations, and a set of available tools.
Sellers do not move; they use the `speak_to` tool to pitch products to buyers in their cell or neighboring cells, attempting to persuade buyers until a sale is made or the buyer refuses.
Buyers can move using the `teleport_to_location` tool if not engaged with a seller, gather information from sellers using `speak_to`, and make purchases using the `buy_product` tool. Their decision is influenced by their budget and the information received from sellers.

## Agent Attributes

**Sellers**: Each seller has a set of internal attributes (e.g., persuasive, hardworking, lazy, unmotivated) and a sales counter.
**Buyers**: Each buyer has a budget ($50 or $100) and a list of purchased products.

## Negotiation Protocol

The negotiation is conducted through tool-based interactions, where sellers initiate conversations and buyers respond, gather information, and decide on purchases.
The reasoning module plans actions based on prompts and observations, simulating realistic negotiation dynamics.

## Data Collection

The model tracks the number of sales for each seller using a data collector, allowing for analysis of which seller is more successful.

## How to Run

If you have cloned the repo into your local machine, ensure you run the following command from the root of the library: ``pip install -e . ``. Then, you will need an api key of an LLM-provider of your choice. Once you have obtained the api-key follow the below steps to set it up for this model.
1) Ensure the dotenv package is installed. If not, run ``pip install python-dotenv``.
2) In the root folder of the project, create a file named .env.
3) If you are using openAI's api key, add the following command in the .env file: ``OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx``. If you have a Gemini based api-key, use this line instead: ``GEMINI_API_KEY=your-gemini-api-key-here``.
4) Change the  ``api_key`` specification in app.py according to the provider you have chosen.
5) Similarly change the ``llm_model`` attribute as well in app.py to the name of a model you have access to. Ensure it is in the form of {provider}/{model_name}. For e.g. ``openai/gpt-4o-mini``.

Once you have set up the api-key in your system, run the following command from this directory:

```
    $ solara run app.py
```

## Files

* [`model.py`](./model.py): Core model code.
* [`agents.py`](./agents.py): Agent classes.
* [`app.py`](./app.py): Sets up the interactive visualization.
* [`tools.py`](./tools.py): Tools for the llm-agents to use.