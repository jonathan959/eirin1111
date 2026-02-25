
def _get_bot_client(bot: Dict[str, Any]) -> Tuple[Any, bool]:
    """
    Return (client, is_kraken) based on bot configuration.
    Uses classify_symbol to determine if stock or crypto.
    """
    symbol = str(bot.get("symbol", ""))
    
    # 1. Check explicit market type
    if bot.get("market_type") == "stocks":
        client = alpaca_live if alpaca_live else alpaca_paper
        return client, False
        
    # 2. Check symbol classification
    market_type = classify_symbol(symbol)
    if market_type == "stock":
        client = alpaca_live if alpaca_live else alpaca_paper
        return client, False
        
    # 3. Default to Kraken
    return kc, True
