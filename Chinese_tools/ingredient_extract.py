from torch.utils.data import DataLoader
import os
import json
import sys
sys.path.append("..")
from ASP_utils import *

def extract_subjects(data_set_path: str, save_path: str, batch_size: int = 128):
    
    if data_set_path.split("|")[-1] == "skip.txt":
        return 

    saved_step = -1
    saved_id = -1
    finished_list = []
    subject_list = []
    adj_list = []

    if os.path.exists(save_path+".progress"):
        with open(save_path+".progress", "r") as f:
            saved_step = int(f.readline())
            saved_id = int(f.readline())
            while True:
                line = f.readline()
                if not line:
                    break
                finished_list.append(line.strip())
            f.close()
            
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                history = json.load(f)
                adj_list = history["adj"]
                subject_list = history["subject"]
                f.close()
        else:
            subject_list = []
            adj_list = []

    else:
        with open(save_path+".progress", "w") as f:
            f.write("0\n0")
            f.write("\n"+data_set_path.split("/")[-1])
            finished_list.append(data_set_path.split("/")[-1])
            f.close()
            
        subject_list = []
        adj_list = []

    #print(subject_list)

    if saved_step == 0 and saved_id == 0 and finished_list != []:
        if data_set_path.split("/")[-1] in finished_list:
            print(
                f"{data_set_path.split('/')[-1]} has already been processed.")
            return
        else:
            finished_list.append(data_set_path.split("/")[-1])
    elif finished_list != []:
        if data_set_path.split("/")[-1] != finished_list[-1]:
            return
        else:
            print("continuing from last progress...")

    print("=="*50)
    print(f"Extracting subjects from {data_set_path}...")

    premise_list, hypothesis_list, label = read_scale_data(data_set_path)
    
    data_set = APIuseDataset(
        {"premise": premise_list, "hypothesis": hypothesis_list}, label)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)

    for step, batch in enumerate(data_loader):

        if saved_step != -1 and step < saved_step:
            continue

        for id, (premise, hypothesis) in enumerate(zip(batch["premise"], batch["hypothesis"])):

            if saved_id != -1 and id < saved_id:
                continue
            
            compair_split_test = premise.split("比")
            negation_split_test = premise.split("不是")
            normal_split_test = premise.split("是")
            
            if len(negation_split_test) > 1:
                premise_subject = negation_split_test[0]
                premise_rest = negation_split_test[1]
            elif len(normal_split_test) > 1:
                premise_subject = normal_split_test[0]
                premise_rest = normal_split_test[1]
                
            else:
                premise_subject = compair_split_test[0]
                premise_rest = compair_split_test[1]
                
            reletive_split_test = premise_rest.split("相对比较")
            comparative_split_test = premise_rest.split("更")
            superlative_split_test = premise_rest.split("最")
            booster_split_test = premise_rest.split("非常")
            
            if len(reletive_split_test) > 1:
                premise_adj = reletive_split_test[1].split("的")[0]
            elif len(comparative_split_test) > 1:
                premise_adj = comparative_split_test[1].split("的")[0]
            elif len(superlative_split_test) > 1:
                premise_adj = superlative_split_test[1].split("的")[0]
            elif len(booster_split_test) > 1:
                premise_adj = booster_split_test[1].split("的")[0]
            else:
                premise_adj = premise_rest.split("的")[0]
            
            # subject extract
            if premise_subject not in subject_list:
                subject_list.append(premise_subject)
            
            if premise_adj not in adj_list:
                adj_list.append(premise_adj)
            
            compair_split_test = hypothesis.split("比")
            negation_split_test = hypothesis.split("不是")
            normal_split_test = hypothesis.split("是")
            
            if len(negation_split_test) > 1:
                hypothesis_subject = negation_split_test[0]
                hypothesis_rest = negation_split_test[1]
            elif len(normal_split_test) > 1:
                hypothesis_subject = normal_split_test[0]
                hypothesis_rest = normal_split_test[1]
            else:
                hypothesis_subject = compair_split_test[0]
                hypothesis_rest = compair_split_test[1]
                
            reletive_split_test = hypothesis_rest.split("相对比较")
            comparative_split_test = hypothesis_rest.split("更")
            superlative_split_test = hypothesis_rest.split("最")
            booster_split_test = hypothesis_rest.split("非常")
            
            if len(reletive_split_test) > 1:
                hypothesis_adj = reletive_split_test[1].split("的")[0]
            elif len(comparative_split_test) > 1:
                hypothesis_adj = comparative_split_test[1].split("的")[0]
            elif len(superlative_split_test) > 1:
                hypothesis_adj = superlative_split_test[1].split("的")[0]
            elif len(booster_split_test) > 1:
                hypothesis_adj = booster_split_test[1].split("的")[0]
            else:
                hypothesis_adj = hypothesis_rest.split("的")[0]
            

            if hypothesis_subject not in subject_list:
                subject_list.append(hypothesis_subject)
                
            if hypothesis_adj not in adj_list:
                adj_list.append(hypothesis_adj)

            with open(save_path+".progress", "w") as f:
                f.write(str(step)+"\n"+str(id))
                for item in finished_list:
                    f.write("\n"+item)
                f.close()

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"subject" : subject_list, "adj" : adj_list}, ensure_ascii=False, indent=4))

            print_and_clear(
                f"{data_set_path.split('/')[-1]} | step: {step+1}/{round(len(data_set)/batch_size)} progress: {(id+1)/len(batch['premise'])*100:5.2f}% | {len(subject_list)} subjects and {len(adj_list)} adjs extracted."
            )

    if os.path.exists(save_path+".progress"):
        with open(save_path+".progress", "w") as f:
            f.write("0\n0")
            for item in finished_list:
                f.write("\n"+item)
            f.close()

    print("\nSubjects saved to : "+save_path)


if __name__ == "__main__":

    data_sets_folder = "../data/EntailmentInference_Chinese/"
    data_set_path = os.listdir(data_sets_folder)
    save_path = "./subject_adj_list.json"

    for data_set in data_set_path:
        extract_subjects(data_sets_folder+data_set, save_path)
    # extract_subjects("../data/EntailmentInference_Chinese/argument_zh.txt", "./subject_list.json")
    
    os.remove("./subject_adj_list.json.progress")
