from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, default_data_collator
import torch
import numpy as np
import os
import sys
import time
sys.path.append('../')
from ASP_utils import *
from torch.utils.data import DataLoader
from natsort import natsorted
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="Data evaluation for the ASP.")
	parser.add_argument(
		"--test_type",
		type=str,
		default='test',
		required = True,
		help = "candidate: [test, swap_n, swap_a, swap_s, wider, api]"
	)
	args = parser.parse_args()

	return args

if __name__ == '__main__':
	os.chdir('../')

	task = 'mnli'
	args = parse_args()
	print(args)

	test_type = args.test_type
	swap_flag = False
	inv_flag = False
	sample_type = 'test'
	if test_type == 'wider':
		sample_type = test_type
	elif test_type == 'swap_n':
		inv_flag = True
	elif test_type == 'swap_s':
		swap_flag = True
	elif test_type == 'swap_a':
		inv_flag = True
		swap_flag = True

 
	num_labels = 3
	list_path = "./model/API_model/"
	model_list = os.listdir(list_path)
	
	if test_type == 'api':

		mapping_dict = {
			"entailment" : 0,
			"not_entailment" : 1,
			"discard" : -2
		}
		
		for model_name in model_list:

			if len(model_name.split('|')) > 1 and model_name.split('|')[1] == "skip":
				continue

			print("\n\n")
			print("%"*100)
			print(f'now turn to the {model_name}:')
			
			nli_classes = ['entailment','not_entailment']
			dev_path = "./data/EntailmentInference-Chinese/"
			task_list = os.listdir(dev_path)

			save_path = f'./result/EntailmentInference-Chinese/'
			# 判断save path是否存在，不存在则创建这个目录
			createDir(save_path)

			for task in task_list:

				dev_premise, dev_hypo, dev_label = read_scale_data(f'{dev_path}/{task}',p_id=0,h_id=1, l_id=2)

				print(f"test dataset: {task}, from {dev_path}")
				print(f"test model : {model_name}")
				print(f"save to: {save_path}")
				
				dataset = APIuseDataset({"premise" : dev_premise, "hypothesis": dev_hypo}, dev_label)
				dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

				pred_scale = []
				for step, batch in enumerate(dataloader):
					
					print("\n", "=="*50)
					print(f"in {step+1}/{len(dev_label)//128 + 1} steps, batch size: {len(batch['premise'])}")

					
					for i in range(len(batch['premise'])):
						prompt = f'Premise: {batch["premise"][i]}\nHypothesis: {batch["hypothesis"][i]}\n则Premise和Hypothesis的关系是？\n务必只回答: "entailment" 或 "not_entailment"'
						pred = call_api(model_name, prompt, show_respons=False)
						while pred == -1:
							time.sleep(5)
							pred = call_api(model_name, prompt, show_respons=False)
						
						if not pred.startswith("ent"):
							pred = "not_entailment"
						if pred.startswith("ent"):
							pred = "entailment"

						#print(pred)

						pred_scale.append(mapping_dict[pred])

						print_and_clear(f"progress: {i+1}/{len(batch['premise'])}")

				result_accuracy = accuracy_score(pred_scale, dev_label)

				with open(f"{save_path}/{model_name}.txt", "a") as f:
					f.write(f"{task[:-4]}: {str(result_accuracy)}\n")



			

		
	else:
		for model_name in model_list:


			print("\n\n")
			print("%"*100)
			print(f'now turn to the {model_name}:')
			model_path = f'{list_path}/{model_name}'
			config = AutoConfig.from_pretrained(model_path, num_labels=num_labels, finetuning_task=task)
			tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=True)

			model_mnli = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
			torch.cuda.set_device(0)
			device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
			model_mnli.to(device)
			model_type = model_name.split("/")[0]
			leaveout_list = model_type.split('-')[-3:-1]
			
			for leaveout_type in leaveout_list:
				if "ei" in leaveout_type and test_type != 'test':
					continue

				try:
					data_path = f'/data/ASP_fine-tuning/{sample_type}/{leaveout_type}'
					dir_list = os.listdir(data_path)
				except:
					continue

				bs_size = 128
				scale_class = ['entailment','not_entailment']
				for test_dir in dir_list:

					if sample_type == 'test':
						save_path = f"/result/ASP_model/{model_name}/{test_dir}"
						if swap_flag and inv_flag:
							save_path += '_sa'
						else:
							if swap_flag:
								save_path += '_ss'
							if inv_flag:
								save_path += '_sn'	 
					else:
						save_path = f"result/ASP_model/{model_name}/{sample_type}/{test_dir}"
					createDir(save_path)

					test_list = os.listdir(f'{data_path}/{test_dir}')
					test_list = natsorted(test_list)
					for test_type in test_list:

						print("="*100)
						
						dev_path = f'{data_path}/{test_dir}/{test_type}'
						nli_classes = ['entailment','not_entailment']
						dev_premise,dev_hypo, dev_label = read_scale_data(dev_path)
						
						if swap_flag:
							dev_premise_swap = []
							for p in dev_premise:
								sub_sent = p.split(".")[:-1]
								sub_sent.reverse()
								sent = ". ".join(sub_sent)+"."
								sent = sent.strip()
								dev_premise_swap.append(sent)
							dev_premise = dev_premise_swap
						if inv_flag:
							dev_premise_inv = []
							for p in dev_premise:
								token_list = p.split(" ")
								pos_id = None
								neg_id = None
								for idx, token in enumerate(token_list):
									if re.findall(r"\d+",token):
										if pos_id == None:
											pos_id = idx
										elif neg_id == None:
											neg_id = idx
								tmp = token_list[pos_id]
								token_list[pos_id] = token_list[neg_id]
								token_list[neg_id] = tmp
								dev_premise_inv.append(" ".join(token_list))
							dev_premise = dev_premise_inv

						print(f"test dataset: {test_type}, from {dev_path}")
						print(f"test model from: {model_path}")
						print(f"save to: {save_path}")
						dev_encoded = tokenizer(dev_premise, dev_hypo, truncation=True, padding='max_length', max_length=128)
						dev_dataset = ScaleDataset(dev_encoded,dev_label)
						data_loader = DataLoader(dev_dataset,batch_size=bs_size,shuffle=False)	

						acc = 0
						pred = []

						for i, batch in enumerate(data_loader):
							with torch.no_grad():
								input_ids = batch['input_ids'].to(device)
								attention_mask = batch['attention_mask'].to(device)
								labels = batch['labels'].to(device)
								if 'token_type_ids' in batch:
									token_type_ids = batch['token_type_ids'].to(device)
									outputs = model_mnli(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
								else:
									outputs = model_mnli(input_ids, attention_mask=attention_mask)
								logits = outputs['logits']
								output = logits.detach().cpu()
								
								poss = torch.softmax(output,dim=1).tolist()
								pred_batch = np.argmax(poss,axis=1)

								for idx in range(len(pred_batch)):
									pred_index = pred_batch[idx]
									pred_index = 1 if pred_index in [1,2] else 0
									pred_class = scale_class[pred_index]

									pred.append(pred_class)
						result = list(zip(dev_premise,dev_hypo,dev_label,pred))

						with open(f'{save_path}/{test_type}','w') as f:
							for pair in result:
								if pair[2] == -2:
									text = [pair[0],pair[1],'discard',pair[3]]
								else:
									text = [pair[0],pair[1],scale_class[pair[2]],pair[3]]
								text = "\t".join(text)
								f.write(text+"\n")
