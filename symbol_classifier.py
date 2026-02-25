"""
Symbol Classification Module

Provides utilities to classify trading symbols as either 'stock' or 'crypto'
and route them to the appropriate trading provider (Alpaca for stocks, Kraken for crypto).

Rules:
- Symbols with '/' are crypto pairs (e.g., "BTC/USD", "ETH/USDT")
- Short symbols (< 6 chars) without '/' are stocks (e.g., "AAPL", "INTC", "MSFT")
- Long symbols (>= 6 chars) without '/' are crypto (e.g., "BTCUSD" which normalizes to "BTC/USD")
"""

from typing import Literal


def classify_symbol(symbol: str) -> Literal["stock", "crypto"]:
    """
    Classify a symbol as 'stock' or 'crypto'.
    
    Args:
        symbol: The trading symbol to classify (e.g., "INTC", "BTC/USD", "AAPL")
    
    Returns:
        "stock" if the symbol is a stock ticker
        "crypto" if the symbol is a cryptocurrency pair
    
    Examples:
        >>> classify_symbol("INTC")
        'stock'
        >>> classify_symbol("BTC/USD")
        'crypto'
        >>> classify_symbol("AAPL")
        'stock'
        >>> classify_symbol("ETH/USD")
        'crypto'
        >>> classify_symbol("BTCUSD")  # Long string, treated as crypto
        'crypto'
    """
    if not symbol:
        return "crypto"  # Default to crypto for empty strings
    
    s = symbol.strip().upper()
    
    # Rule 1: Contains '/' → definitely crypto pair
    if "/" in s:
        return "crypto"
    
    # Rule 2: Short symbol (< 6 chars) without '/' → stock
    # Examples: AAPL, MSFT, INTC, TSLA, AMD, NVDA, META
    if len(s) < 6:
        return "stock"
    
    # Rule 3: Long symbol (>= 6 chars) without '/' → crypto
    # Examples: BTCUSD, ETHUSD, XBTUSD (these get normalized to BTC/USD, ETH/USD, etc.)
    return "crypto"


def is_stock_symbol(symbol: str) -> bool:
    """
    Check if a symbol is a stock ticker.
    
    Args:
        symbol: The trading symbol to check
    
    Returns:
        True if the symbol is classified as a stock, False otherwise
    
    Examples:
        >>> is_stock_symbol("INTC")
        True
        >>> is_stock_symbol("BTC/USD")
        False
    """
    return classify_symbol(symbol) == "stock"


def is_crypto_symbol(symbol: str) -> bool:
    """
    Check if a symbol is a cryptocurrency pair.
    
    Args:
        symbol: The trading symbol to check
    
    Returns:
        True if the symbol is classified as crypto, False otherwise
    
    Examples:
        >>> is_crypto_symbol("BTC/USD")
        True
        >>> is_crypto_symbol("INTC")
        False
    """
    return classify_symbol(symbol) == "crypto"


# Validation function for explicit market type enforcement
def validate_symbol_type(symbol: str, expected_type: Literal["stock", "crypto"], caller: str = "function") -> None:
    """
    Validate that a symbol matches the expected type, raise ValueError if not.
    
    This is used as a guardrail to prevent routing errors (e.g., calling Kraken with stock symbols).
    
    Args:
        symbol: The trading symbol to validate
        expected_type: Either "stock" or "crypto"
        caller: Name of the calling function (for error messages)
    
    Raises:
        ValueError: If the symbol doesn't match the expected type
    
    Examples:
        >>> validate_symbol_type("INTC", "stock", "some_function")  # OK, no error
        >>> validate_symbol_type("INTC", "crypto", "kraken_fetch")  # Raises ValueError
    """
    actual_type = classify_symbol(symbol)
    
    if actual_type != expected_type:
        provider_map = {
            "stock": "AlpacaClient",
            "crypto": "KrakenClient"
        }
        correct_provider = provider_map.get(actual_type, "appropriate client")
        wrong_provider = provider_map.get(expected_type, "wrong client")
        
        raise ValueError(
            f"Developer Error in {caller}(): "
            f"Symbol '{symbol}' is a {actual_type} ticker but was passed to a {expected_type}-only function. "
            f"Please use {correct_provider} methods instead of {wrong_provider} methods."
        )
