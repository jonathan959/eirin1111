
lines = []
with open('worker_api.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 0-indexed in list, but 1-indexed in editor
# Line 628 in editor is index 627
# Line 1043 in editor is index 1042. We want to keep 1043.
start_idx = 627 
end_idx = 1042

# Verify start
print(f"Start line content: {lines[start_idx]}")
if "def _scan_symbol_legacy" not in lines[start_idx]:
    print("ERROR: Start line mismatch! Aborting.")
    exit(1)

# Verify end (the line AFTER the block we delete)
print(f"End line content (kept): {lines[end_idx]}")
if "def _reco_symbols" not in lines[end_idx]:
    print("ERROR: End line mismatch! Aborting.")
    exit(1)

# Keep lines before start and from end onwards
new_lines = lines[:start_idx] + lines[end_idx:]

with open('worker_api.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Successfully removed legacy code block.")
