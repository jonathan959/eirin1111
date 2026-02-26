from typing import Any, Dict, Tuple
from symbol_classifier import classify_symbol


def _get_bot_client(bot: Dict[str, Any], kc: Any = None, alpaca_paper: Any = None, alpaca_live: Any = None) -> Tuple[Any, bool]:
    """
    Return (client, is_kraken) based on bot configuration.
    Uses classify_symbol to determine if stock or crypto.
    """
    symbol = str(bot.get("symbol", ""))

    if bot.get("market_type") == "stocks":
        client = alpaca_live if alpaca_live else alpaca_paper
        return client, False

    market_type = classify_symbol(symbol)
    if market_type == "stock":
        client = alpaca_live if alpaca_live else alpaca_paper
        return client, False

    return kc, True
