from torch.utils.data import DataLoader
import os
import json
import sys
sys.path.append("..")
from ASP_utils import *

def auto_label(data_set_path: str, batch_size: int = 128, rule : str = ""):
    
    if data_set_path.split("|")[-1] == "skip.txt":
        return 

    save_path = "./auto_label"

    saved_step = -1
    saved_id = -1
    finished_list = []

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

    else:
        with open(save_path+".progress", "w") as f:
            f.write("0\n0")
            f.write("\n"+data_set_path.split("/")[-1])
            finished_list.append(data_set_path.split("/")[-1])
            f.close()

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
    print(f"auto labeling {data_set_path}...")

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
            
            
            if rule == "booster":
                if str(premise).find("非常") != -1 and str(hypothesis).find("非常") == -1:
                    label[step*batch_size+id] = label_convert("entailment")
                else:
                    label[step*batch_size+id] = label_convert("not_entailment")
                    
            elif rule == "comparative":
                second_subject = str(premise).split("比")[-1].split("更")[0]
                
                judgement_index = str(hypothesis).find(second_subject) + len(second_subject)
                
                if judgement_index < len(str(hypothesis)):
                    label[step*batch_size+id] = label_convert("not_entailment")
                
                if str(hypothesis)[judgement_index] == "不":
                    label[step*batch_size+id] = label_convert("entailment")
                else:
                    label[step*batch_size+id] = label_convert("not_entailment")
                    
            elif rule == "diminisher":
                if str(premise).find("相对比较") == -1 and str(hypothesis).find("相对比较") != -1:
                    label[step*batch_size+id] = label_convert("entailment")
                else:
                    label[step*batch_size+id] = label_convert("not_entailment")
                
            elif rule == "superlative":
                if str(premise).find("世界上") != -1 and str(hypothesis).find("世界上") == -1:
                    label[step*batch_size+id] = label_convert("entailment")
                else:
                    label[step*batch_size+id] = label_convert("not_entailment")
            
            
            with open(save_path+".progress", "w") as f:
                f.write(str(step)+"\n"+str(id))
                for item in finished_list:
                    f.write("\n"+item)
                f.close()

            write_scale_data(premise_list, hypothesis_list, label, data_set_path)
           
            print_and_clear(
                f"{data_set_path.split('/')[-1]} | step: {step+1}/{round(len(data_set)/batch_size)} progress: {(id+1)/len(batch['premise'])*100:5.2f}%"
            )

    if os.path.exists(save_path+".progress"):
        with open(save_path+".progress", "w") as f:
            f.write("0\n0")
            for item in finished_list:
                f.write("\n"+item)
            f.close()
        
    print("\n")


if __name__ == "__main__":

    data_sets_folder = "../data/EntailmentInference_Chinese/"
    
    save_path = "./subject_adj_list.json"

    
    auto_label(data_sets_folder+"comparative_zh.txt", rule = "comparative")
    auto_label(data_sets_folder+"diminisher_zh.txt", rule = "diminisher")
    auto_label(data_sets_folder+"superlative_zh.txt", rule = "superlative")
    auto_label(data_sets_folder+"booster_zh.txt", rule = "booster")
    
    os.remove("./auto_label.progress")
