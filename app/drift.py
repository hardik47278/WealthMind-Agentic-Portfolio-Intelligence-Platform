"""debug_team.py -- calls team.py's internals step by step, without the
outer try/except swallowing the real traceback."""
import json, traceback
from app.team import _dispatch_role, route_with_fallback
from app.compilance_agent import build_ambiguity_classifier
from app.router import build_router_agent, build_strong_router_agent

book = json.load(open("data/client_book.json"))
market = json.load(open("data/market_data.json"))
client = next(c for c in book["clients"] if c["id"] == "cli_1014")
import os
from dotenv import load_dotenv
load_dotenv()

base_url = "https://api.openai.com/v1"
api_key = os.getenv("OPENAI_API_KEY")


prompt = "How much did technology exposure deviate from the mandate?"

router = build_router_agent(base_url, api_key, "gpt-4.1-mini")
strong = build_strong_router_agent(base_url, api_key, "gpt-4.1-mini")

try:
    roles, model_used, confidence = route_with_fallback(prompt, router, strong)
    print("ROLES:", roles, model_used, confidence)
    for role in roles:
        piece = _dispatch_role(role, client, market, prompt, base_url, api_key, "gpt-4.1-mini", "gpt-4.1-mini")
        print("PIECE:", piece)
except Exception:
    traceback.print_exc()