import sys
import os
from dotenv import load_dotenv
load_dotenv()
CHATARENA_PATH = os.getenv("CHATARENA_PATH")
sys.path.append(CHATARENA_PATH)

import json
import yaml
from time import strftime


from chatarena.agent import Player
from chatarena.backends import CohereAIChat, OpenAIChat, Claude
from chatarena.environments.taboo import Taboo


file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
chat_history_path = os.path.join(dir_path, "chat_history")
PROMPT_CONFIG_FILE = os.path.join(dir_path, "taboo_final_experiments.yaml")

with open(PROMPT_CONFIG_FILE) as file:
    prompts = yaml.safe_load(file)

role_description = "You are a player in a word guessing game called ask-guess. "

paya = Player(name="Paya",
                role_desc=role_description,
                backend=CohereAIChat())

toto = Player(name="Toto",
                role_desc=role_description,
                backend=CohereAIChat())

from chatarena.arena import Arena

with open(r"src\datasets\taboo\web.json", "r") as fp:
    taboo = json.load(fp)


NB_EXPERIMENTS = 10
MAX_STEPS = 20
PROMPT_MODES = ["final_baseline"]
# ================= EXPERIMENTS ================

for prompt_mode in PROMPT_MODES:
    print(f"======================= Doing {prompt_mode} =======================")
    
    backend = CohereAIChat(
        temperature=prompts[prompt_mode]["temperature"],
        max_tokens=prompts[prompt_mode]["max_tokens"],
        model=prompts[prompt_mode]["model_name"],
        preamble=prompts[prompt_mode]["preamble"],
    )

    for i in range(NB_EXPERIMENTS):
        print(f"Experiment {i}.")

        # ====== Baseline =======
        env = Taboo(
            player_names = ["Paya", "Toto"], 
            taboo = taboo, 
            prompt_config_file=PROMPT_CONFIG_FILE,
            prompt_config_mode=prompt_mode,
            )

        arena = Arena([paya, toto], env)
        arena.launch_cli(interactive=False, max_steps=MAX_STEPS)

        experiment_path = os.path.join(chat_history_path, prompt_mode)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)  # Create the directory if it doesn't exist

        # Saving history
        chat_path = os.path.join(
            experiment_path, f"taboo_{strftime('%Y_%m_%d_%H_%M_%S')}.json"
        )
        arena.save_chat(chat_path)

