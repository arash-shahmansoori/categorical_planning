import json
import os


def modify_data(data):
    for d in data:
        for k, v in d.items():
            if k == "Blueprint":
                d[k] = str(v)
            elif k == "Detailed":
                d[k] = str(v)
            else:
                d[k] = d[k]
    return data


# Specify the source and target directories
source_directory = "data_bkup"
target_directory = "data_bkup_update"

# Create the target directory if it doesn't exist
os.makedirs(target_directory, exist_ok=True)

# Iterate through files in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith(".json"):
        # Read the content of the JSON file
        source_filepath = os.path.join(source_directory, filename)
        with open(source_filepath, "r") as source_file:
            data = json.load(source_file)

        data_updated = modify_data(data)

        # Write the modified data to a new JSON file in the target directory
        target_filepath = os.path.join(target_directory, filename)
        with open(target_filepath, "w") as target_file:
            json.dump(data_updated, target_file, indent=4)

        print(f"Modified and saved: {target_filepath}")

print("All files have been modified and saved to the target directory.")
