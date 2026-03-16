import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MASTER_BRAIN_PATH = PROJECT_ROOT / 'data' / 'brains' / 'runtime_brain.json'
DOWNLOADED_BRAIN_PATH = Path(r'c:/Users/amani/Downloads/camguard_brain_v7_2000_signatures.json')

def merge_brains():
    # 1. Load the current project brain
    with MASTER_BRAIN_PATH.open('r', encoding='utf-8') as f:
        master_data = json.load(f)
    
    # 2. Load the downloaded brain (2000 signatures)
    with DOWNLOADED_BRAIN_PATH.open('r', encoding='utf-8') as f:
        new_data = json.load(f)
    
    # Get current KB
    current_kb = master_data.get('knowledgeBase', [])
    new_kb = new_data.get('knowledgeBase', [])
    
    # Deduplication strategy: Use a unique key based on label and magValue
    # Since these are fingerprints, a combination of label and exact magValue is fairly unique.
    seen_keys = set()
    for item in current_kb:
        # Create a unique key
        key = f"{item.get('label')}|{item.get('magValue')}"
        seen_keys.add(key)
    
    added_count = 0
    unique_new_items = []
    
    for item in new_kb:
        key = f"{item.get('label')}|{item.get('magValue')}"
        if key not in seen_keys:
            unique_new_items.append(item)
            seen_keys.add(key)
            added_count += 1
            
    # Combine
    master_data['knowledgeBase'] = current_kb + unique_new_items
    
    # Update metadata
    master_data['version'] = 8 # Bump to v8 to trigger sync
    master_data['totalDetections'] = len(master_data['knowledgeBase'])
    
    # Save back to project
    MASTER_BRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MASTER_BRAIN_PATH.open('w', encoding='utf-8') as f:
        json.dump(master_data, f, indent=2)
        
    print(f"Successfully merged {added_count} unique signatures into Master Brain v8.")
    print(f"Total Database Size: {len(master_data['knowledgeBase'])} signatures.")

if __name__ == "__main__":
    merge_brains()
