import torch 
# Define Dataloader
class LMDataset(Dataset):
    def __init__(self, file_path, store_file_path,max_length = SEQUENCE_LENGTH):
        self.lines = []
        num_lines = 100000
        max_length_dataset = 0
        if os.path.exists(store_file_path):
            self.lines = torch.load(store_file_path)
        else:                
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file, desc = "Dataset Prep"):
                    line = "<sos>" + line.strip() + "<eos> "
                    line,length_seq = encode_text(tokenizer, line)
                    if length_seq > max_length_dataset:
                        max_length_dataset = length_seq
                    self.lines.append(line)
                    # num_lines -= 1

            print("Max length in the dataset : ", max_length_dataset)
            torch.save(self.lines, store_file_path)       
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        return line


    
def collate_train_fn(batch):
    batch = torch.stack(batch)
    input_seq = batch[:, :-1]
    target_seq = batch[:, 1:]
    return input_seq, target_seq
