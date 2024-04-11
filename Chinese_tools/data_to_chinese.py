import os
import sys
import time
from torch.utils.data import DataLoader
sys.path.append("..")
from ASP_utils import *

def translate_data(path_tuple : tuple):

	batch_size = 128

	data_set_path = path_tuple[0]

	if data_set_path.split("|")[-1] == "skip":
		return

	save_path = path_tuple[1]

	saved_step = -1
	saved_id = -1

	if os.path.exists(save_path+".progress"):
		with open(save_path+".progress", "r") as f:
			saved_step = int(f.readline())
			saved_id = int(f.readline())

			f.close()
		

	print("=="*50)
	print(f"Translating {data_set_path}...")


	premise_list, hypothesis_list, label = read_scale_data(data_set_path)        \
											if not os.path.exists(save_path+".progress") \
											else read_scale_data(save_path)

	data_set = APIuseDataset({"premise" : premise_list, "hypothesis" : hypothesis_list}, label)
	data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)


	for step, batch in enumerate(data_loader):

		if saved_step != -1 and step < saved_step:
			continue

		for id, (premise, hypothesis) in enumerate(zip(batch["premise"], batch["hypothesis"])):

			if saved_id != -1 and id < saved_id:
				continue

			translate_prompt = f"请将以下三引号内的英文句子翻译为中文：\n'''{premise}'''\n任何时候不要有译文以外的任何输出。"
			#print(translate_prompt)
			translate_premise = call_api("qwen-max",translate_prompt, show_respons=False)
			while translate_premise == -1:
				time.sleep(5)
				translate_premise = call_api("qwen-max",translate_prompt, show_respons=False)

			translate_prompt = f"请将以下三引号内的英文句子翻译为中文：\n'''{hypothesis}'''\n任何时候不要有译文以外的任何输出。"
			translate_hypothesis = call_api("qwen-max",translate_prompt, show_respons=False)
			while translate_hypothesis == -1:
				time.sleep(5)
				translate_hypothesis = call_api("qwen-max",translate_prompt, show_respons=False)

			premise_list[id+step*batch_size] = translate_premise
			hypothesis_list[id+step*batch_size] = translate_hypothesis

			print_and_clear(
				f"{data_set_path.split('/')[-1]}| \
step {step+1}, progress: {id+1}/{batch_size}| \
result: {translate_premise}|{translate_hypothesis}"
			)

			

			with open(save_path+".progress", "w") as f:
				f.write(str(step)+"\n"+str(id))
				f.close()

			write_scale_data(premise_list, hypothesis_list, label, save_path)

		print("\nsave to : "+save_path)

	if os.path.exists(save_path+".progress"):
		os.remove(save_path+".progress")


data_folder = "./data/EntailmentInference/"
save_path = "./data/EntailmentInference-Chinese/"

if __name__ == "__main__":

	os.chdir("..")

	createDir(save_path)

	data_set_path_list = ["booster.csv", "ordering.csv"]
	save_path_list = []

	for i, data_set_path in enumerate(data_set_path_list):
		data_set_path_list[i] = data_folder + data_set_path
		save_path_list.append(save_path + data_set_path)

	for data_set_path, save_path in zip(data_set_path_list, save_path_list):
		translate_data((data_set_path, save_path))