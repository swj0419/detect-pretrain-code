from datasets import load_dataset
from datasets import Dataset
from ipdb import set_trace as bp
# from options import Options
from src.eval import *


# data: different length:
def process_each_dict_length_data(data, length=32):
    new_data = []
    for ex in data:
        ex_copy = ex.copy()
        if len(ex_copy["input"].split()) < length:
            continue
        else:
            ex_copy["input"] = " ".join(ex_copy["input"].split()[:length])
        new_data.append(ex_copy)
    return new_data

def change_type(data):
    new_data = {"input": [], "label": []}
    for ex in data:
        ex["label"] = int(ex["label"])
        new_data["input"].append(ex["input"])
        new_data["label"].append(ex["label"])
    return new_data

if __name__ == "__main__":
    dataset = load_jsonl("/fsx-onellm/swj0419/attack/detect-pretrain-code/data/wikimia.jsonl")
    # bp()
    data = convert_huggingface_data_to_list_dic(dataset)
    for length in [128, 256]:
        new_data = process_each_dict_length_data(data, length=length)
        print(f"length {length} data size: {len(new_data)}")
        dump_jsonl(new_data, f"data/WikiMIA_length{length}.jsonl")
        huggingface_dataset = Dataset.from_dict(change_type(new_data))
        huggingface_dataset.push_to_hub("swj0419/WikiMIA", f"WikiMIA_length{length}")


