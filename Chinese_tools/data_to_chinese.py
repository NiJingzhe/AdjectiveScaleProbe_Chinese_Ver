import os
import sys
import time
sys.path.append("..")
from ASP_utils import *

def translate_data(path_tuple : tuple):

	data_set_path = path_tuple[0]

	if data_set_path.split("|")[-1] == "skip":
		return

	save_path = path_tuple[1]

	print("=="*50)
	print(f"Translating {data_set_path}...")

	premise_list, hypothesis_list, label = read_scale_data(data_set_path)

	for id, (premise, hypothesis) in enumerate(zip(premise_list, hypothesis_list)):

		translate_prompt = f"请将以下英文句子翻译为中文：\n{premise}\n不要有译文以外的任何输出。"
		#print(translate_prompt)
		translate_premise = call_api("qwen-max",translate_prompt, show_respons=False)
		while translate_premise == -1:
			time.sleep(5)
			translate_premise = call_api("qwen-max",translate_prompt, show_respons=False)

		translate_prompt = f"请将以下英文句子翻译为中文：\n{hypothesis}\n不要有译文以外的任何输出。"
		translate_hypothesis = call_api("qwen-max",translate_prompt, show_respons=False)
		while translate_hypothesis == -1:
			time.sleep(5)
			translate_hypothesis = call_api("qwen-max",translate_prompt, show_respons=False)

		premise_list[id] = translate_premise
		hypothesis_list[id] = translate_hypothesis

		print_and_clear(f"{data_set_path.split('/')[-1]}| progress: {id+1}/{len(premise_list)} \n")

	write_scale_data(premise_list, hypothesis_list, label, save_path)
	print("save to "+save_path)


data_folder = "./data/EntailmentInference/"
save_path = "./data/EntailmentInference-Chinese/"

if __name__ == "__main__":

	os.chdir("..")

	createDir(save_path)

	data_set_path_list = os.listdir(data_folder)
	save_path_list = []

	for i, data_set_path in enumerate(data_set_path_list):
		data_set_path_list[i] = data_folder + data_set_path
		save_path_list.append(save_path + data_set_path)

	import multiprocessing
	pool = multiprocessing.Pool(processes=10)
	pool.map(translate_data, zip(data_set_path_list, save_path_list))