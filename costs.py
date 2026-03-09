# costs.py  –  Token usage and cost tracking

# GPT-4o-mini pricing (as of early 2025, USD per 1K tokens)
PRICING = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},   # per 1K tokens
    "text-embedding-3-small": {"input": 0.000020, "output": 0.0},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return cost in USD for a given model call."""
    p = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (input_tokens / 1000) * p["input"] + (output_tokens / 1000) * p["output"]


def format_cost(usd: float) -> str:
    """Format a USD amount nicely."""
    if usd < 0.001:
        return f"${usd:.6f}"
    return f"${usd:.4f}"