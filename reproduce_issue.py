
import sys
import os

# Add local directory to sys.path
sys.path.append(os.getcwd())

try:
    from multi_timeframe import MultiTimeframeAnalyzer, Signal
    print(f"Signal type: {type(Signal.STRONG_BULLISH)}")
    print(f"Signal value: {Signal.STRONG_BULLISH.value}")
    
    analyzer = MultiTimeframeAnalyzer()
    
    # Create mock candles
    candles = [[0, 100, 105, 95, 101, 1000] for _ in range(100)]
    
    # Test analyze
    signal, details = analyzer.analyze(candles, candles, candles, candles)
    print(f"Analyze result signal type: {type(signal)}")
    print(f"Analyze result signal value: {signal}")
    
    if isinstance(signal, str):
        print("CRITICAL: Signal is a string! This will cause AttributeError when accessing .value")
    else:
        print(f"Signal.value works: {signal.value}")

except Exception as e:
    print(f"Error during MultiTimeframe check: {e}")

try:
    from intelligence_layer import IntelligenceLayer, RegimeType
    print(f"RegimeType type: {type(RegimeType.BULL)}")
    
    layer = IntelligenceLayer()
    # We can't easily run layer.evaluate without a Full Context, but checking the Enum is good start.
    
except Exception as e:
    print(f"Error during IntelligenceLayer check: {e}")
