import json

# Define the input and output paths
path = r"E:\Graph_writer\data\refined_content\refined_20251108_201946.jsonl"
output_path = r"full3.md" # Name of the output file

data = []

# --- Step 1: Read and Load Data ---
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        # Skip empty lines
        if not line.strip():
            continue
        
        try:
            obj = json.loads(line)
            # Only add objects that have both 'id' (for sorting) and 'content' (the text)
            if "content" in obj and "id" in obj:
                data.append(obj)
        except json.JSONDecodeError as e:
            print(f"Skipping malformed JSON line: {line.strip()}. Error: {e}")
            continue

# --- Step 2: Sort Data by 'id' ---
# Assuming 'id' is a number (integer) for proper numerical sorting.
# If 'id' is a string that represents a number (e.g., "1", "2", "10"), 
# we convert it to an integer for sorting.
try:
    data.sort(key=lambda x: int(x["id"]))
except ValueError:
    # If the 'id' cannot be converted to an integer, fall back to string sorting.
    print("Warning: 'id' values are not all valid integers. Falling back to string sort.")
    data.sort(key=lambda x: x["id"])


# --- Step 3: Extract Content in Sorted Order ---
all_text = [obj["content"] for obj in data]
final_article = "\n".join(all_text)

# --- Step 4: Save to full.md ---
with open(output_path, "w", encoding="utf-8") as f_out:
    f_out.write(final_article)

# --- Step 5: Print Confirmation ---
print(f"Successfully processed {len(data)} items.")
print(f"Final article saved to **{output_path}** after sorting by 'id'.")