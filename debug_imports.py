import sys
import os

sys.path.append(os.getcwd())

print("Importing intelligence_layer...")
try:
    import intelligence_layer
    print(f"intelligence_layer imported: {intelligence_layer}")
    from intelligence_layer import IntelligenceContext
    print(f"IntelligenceContext: {IntelligenceContext}")
except Exception as e:
    print(f"Error importing intelligence_layer: {e}")

print("Importing worker_api...")
try:
    import worker_api
    print(f"worker_api imported. IntelligenceContext in dir: {'IntelligenceContext' in dir(worker_api)}")
except Exception as e:
    print(f"Error importing worker_api: {e}")
