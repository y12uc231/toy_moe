import datasets

def process_data(division):
    # Combine Samples 
    data_split = datasets.load_dataset("math_dataset", division)
    splits = ["train", "test"]
    for split in splits:
        samples = []
        for sample in range(data_split[split].num_rows):
            samples.append("Q: " + data_split[split][sample]["question"] + " A: " + data_split[split][sample]["answer"].decode('utf-8').strip())
        all_data = "\n".join(samples)
        print("Sample data : ", all_data[:30])
        with open(f"../data/{split}_{division}.txt", "w") as f:
            f.write(all_data)        

if __name__ == "__main__":
    division = "algebra__linear_1d"
    process_data(division)