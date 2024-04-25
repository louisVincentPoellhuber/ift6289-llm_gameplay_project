import json
import pandas as pd
import numpy as np
import os
from glob import glob

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
PROMPT_MODES =  ["final_baseline", "command_r", "add_restrict_info", "low_temperature", "high_temperature", "sub_cot", "sub_preamble"]

result_table = pd.DataFrame(columns=["N", "ST", "EE", "RLE", "CE", "AGR", "SLR"], index = PROMPT_MODES).fillna(0)
relative_result_table = result_table.copy()

slr = 0
swr = 0
for prompt_mode in PROMPT_MODES:
    folder_path = os.path.join(dir_path, "chat_history")
    folder_path = os.path.join(folder_path, prompt_mode)
    folder_rgx = os.path.join(folder_path, "*.json")

    result_table.at[prompt_mode, "N"] = len(glob(folder_rgx))
    relative_result_table.at[prompt_mode, "N"] = len(glob(folder_rgx))
    for conversation in glob(folder_rgx):
        with open(conversation, "r") as fp:
            conversation_json = json.load(fp)
        
        conversation_ending = conversation_json["metrics"]["end_condition"]
        nb_turns = conversation_json["metrics"]["nb_turns"]

        result_table.at[prompt_mode, "AGR"] += nb_turns
        if (conversation_ending == "SPYWINS") | (conversation_ending == "SPYLOSES"):
            result_table.at[prompt_mode, "ST"] += 1
            result_table.at[prompt_mode, "SLR"] += int(conversation_ending == "SPYLOSES")
            relative_result_table.at[prompt_mode, "ST"] = result_table.at[prompt_mode, "ST"] / result_table.at[prompt_mode, "N"]

        else:            
            result_table.at[prompt_mode, conversation_ending] += 1
            relative_result_table.at[prompt_mode, conversation_ending] = result_table.at[prompt_mode, conversation_ending] / result_table.at[prompt_mode, "N"]

    if result_table.at[prompt_mode, "ST"] == 0:
        result_table.at[prompt_mode, "SLR"] = 0
        relative_result_table.at[prompt_mode, "SLR"] = 0
    else:
        result_table.at[prompt_mode, "SLR"] = result_table.at[prompt_mode, "SLR"] / result_table.at[prompt_mode, "ST"]
        relative_result_table.at[prompt_mode, "SLR"] = result_table.at[prompt_mode, "SLR"] 
    
    result_table.at[prompt_mode, "AGR"] = result_table.at[prompt_mode, "AGR"] / result_table.at[prompt_mode, "N"] / 6
    relative_result_table.at[prompt_mode, "AGR"] = result_table.at[prompt_mode, "AGR"]

out_path = os.path.join(dir_path, "performance")
result_table.to_csv(os.path.join(out_path, "result_table.csv"))
relative_result_table.to_csv(os.path.join(out_path, "relative_result_table.csv"))
        