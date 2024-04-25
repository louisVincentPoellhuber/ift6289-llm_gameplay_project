import sys
import os
from dotenv import load_dotenv

load_dotenv()
CHATARENA_PATH = os.getenv("CHATARENA_PATH")
sys.path.append(CHATARENA_PATH)

import json
import yaml
from time import strftime
import random

from chatarena.arena import Arena
from chatarena.agent import Player
from chatarena.backends import CohereAIChat
from chatarena.environments.spyfall import SpyFall

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
chat_history_path = os.path.join(dir_path, "chat_history")
PROMPT_CONFIG_FILE = os.path.join(dir_path, "spyfall_final_experiments.yaml")

# Name of the players
number_of_players = 6

# Opening configurations
with open(PROMPT_CONFIG_FILE) as file:
    prompts = yaml.safe_load(file)


with open(r"src\datasets\askguess.json", "r") as fp:
    topic_codes = json.load(fp)



NB_EXPERIMENTS = 5
MAX_STEPS = 48
PROMPT_MODES = [ "sub_preamble"]

for prompt_mode in PROMPT_MODES:
    print(f"======================= Doing {prompt_mode} =======================")
    # Defining players
    # First description of the game -> taken from GameEval
    role_description = prompts["role_description"].format(number_of_players=number_of_players)
    # Chatarena output format
    format_specification = ""

    # If we remove the preamble, it becomes the role description, and we remove it from the backend. 
    if prompt_mode == "sub_preamble":
        role_description = prompts[prompt_mode]["role_description"]

        backend = CohereAIChat(
        temperature=prompts[prompt_mode]["temperature"],
        max_tokens=prompts[prompt_mode]["max_tokens"],
        model=prompts[prompt_mode]["model_name"],
        )
    else:
        backend = CohereAIChat(
        temperature=prompts[prompt_mode]["temperature"],
        max_tokens=prompts[prompt_mode]["max_tokens"],
        model=prompts[prompt_mode]["model_name"],
        preamble=prompts[prompt_mode]["preamble"],
        )

    for i in range(NB_EXPERIMENTS):
        players = ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"]
        random.shuffle(players)
        players_list = [
            Player(
                name=players[i],
                role_desc="Your name is "
                + players[i]
                + ","
                + role_description,
                backend=backend,
            )
            for i in range(number_of_players)
        ]

        # Running experiment
        env = SpyFall(
            player_names=players,
            prompt_config_file=PROMPT_CONFIG_FILE,
            prompt_config_mode=prompt_mode,
            restrict_info=prompts[prompt_mode]["restrict_info"],
            topic_codes = topic_codes
        )
        arena = Arena(players_list, env)
        arena.launch_cli(interactive=False, max_steps=MAX_STEPS)

        experiment_path = os.path.join(chat_history_path, prompt_mode)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)  # Create the directory if it doesn't exist

        # Saving history
        chat_path = os.path.join(
            experiment_path, f"spyfall_{strftime('%Y_%m_%d_%H_%M_%S')}.json"
        )
        arena.save_chat(chat_path)
