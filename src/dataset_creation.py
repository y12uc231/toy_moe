import datasets

## Create a "../data" before executing this file. 

def extract_string(byte_string):
    start_index = byte_string.find("'") + 1  # Find the index of the first single quote and add 1 to start from the next character
    end_index = byte_string.find("'", start_index)  # Find the index of the second single quote

    if start_index != -1 and end_index != -1:
        extracted_string = byte_string[start_index:end_index].strip()
        extracted_string = extracted_string.replace('\\n', '')
        return extracted_string
    else:
        print("Single quotes not found.")

def process_data(division):
    # Combine Samples 
    data_split = datasets.load_dataset("math_dataset", division)
    splits = ["train", "test"]
    for split in splits:
        samples = []
        for sample in range(data_split[split].num_rows):
            samples.append("Q: " + extract_string(data_split[split][sample]["question"]) + " A: " + extract_string(data_split[split][sample]["answer"]))
        all_data = "\n".join(samples)
        print("Sample data : ", all_data[:50])
        with open(f"../data/{split}_{division}.txt", "w") as f:
            f.write(all_data)        

if __name__ == "__main__":
    division = "algebra__linear_1d"
    process_data(division)