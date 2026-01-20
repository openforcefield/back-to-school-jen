# type: ignore

import requests

# Fetch the JSON file from the URL
url = "https://raw.githubusercontent.com/openforcefield/ash-sage-rc2/32345dddeb6cb249367059fd99607ac2950a5c86/03_fit-valence/02_curate-data/output/optimizations-single-v3.json"
response = requests.get(url)
response.raise_for_status()  # Raise an error if the request fails

data = response.json()  # Load the JSON content into a Python dictionary

# Extract entries from the dictionary
entries = data["entries"]["https://api.qcarchive.molssi.org:443/"]

# Extract type and record_id from each entry
types = set(entry["type"] for entry in entries)
record_ids = [entry["record_id"] for entry in entries]

# Ensure there is only one unique type
if len(types) != 1:
    raise ValueError(f"Expected exactly one unique type, but found: {types}")

unique_type = types.pop()  # Get the single unique type

# Save the type and record IDs to a text file
output_file = "record_ids.txt"
with open(output_file, "w") as f:
    # Write the type as a commented header
    f.write(f"# Type: {unique_type}\n")
    # Write each record_id on a new line
    f.writelines(f"{record_id}\n" for record_id in record_ids)

print(f"Saved {len(record_ids)} record IDs to {output_file}")
