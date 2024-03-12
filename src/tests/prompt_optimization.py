import sys
import os
from dotenv import load_dotenv
load_dotenv()
CHATARENA_PATH = os.getenv("CHATARENA_PATH")
sys.path.append(CHATARENA_PATH)

from chatarena.agent import Player
from chatarena.backends import CohereAIChat, OpenAIChat, Human
from chatarena.environments.conversation import Conversation
from chatarena.arena import Arena

import cohere
import openai

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from rich.color import ANSI_COLOR_NAMES
from rich.console import Console
from rich.text import Text

ASCII_ART = r"""
_________  .__               __      _____
\_   ___ \ |  |__  _____   _/  |_   /  _  \  _______   ____    ____  _____
/    \  \/ |  |  \ \__  \  \   __\ /  /_\  \ \_  __ \W/ __ \  /    \ \__  \
\     \____|   Y  \ / __ \_ |  |  /    |    \ |  | \/\  ___/ |   |  \ / __ \_
 \______  /|___|  /(____  / |__|  \____|__  / |__|    \___  >|___|  /(____  /
        \/      \/      \/                \/              \/      \/      \/
but modified by us >:)
"""
visible_colors = [
    color
    for color in ANSI_COLOR_NAMES.keys()
    if color not in ["black", "white", "red", "green"] and "grey" not in color
]

console = Console()
# Print ascii art
console.print(ASCII_ART, style="bold dark_orange3")

# Initialize model
model_cmd = prompt(
                [("class:message", "Choose your model (c/g/q) > ")],
                style=Style.from_dict({"message": "orange"}),
                completer=WordCompleter(
                    [
                        "cohere",
                        "c",
                        "gpt",
                        "g",
                        "exit",
                        "quit",
                        "q"
                    ]
                ),
                )
if model_cmd == "exit" or model_cmd == "quit" or model_cmd == "q":
    exit_cli = True
elif model_cmd == "cohere" or model_cmd == "c":
    agent_name = "Cohere"
    client  = cohere.Client(api_key=os.environ.get("COHEREAI_API_KEY"))
    backend = CohereAIChat()
elif model_cmd == "gpt" or model_cmd == "g":
    agent_name = "GPT-3.5"
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    backend = OpenAIChat()

if not exit_cli:
    agent = Player(name=agent_name,
                role_desc="LLM agent to test.",
                backend=backend)

    tester_name = prompt([("class:message", "What is your name?")], style=Style.from_dict({"message": "orange"}))

    tester = Player(name=tester_name, role_desc="Human tester.", backend=Human())

    # Conversation

    env = Conversation(player_names=[tester_name, agent_name])
    arena = Arena([tester, agent], env)
    arena.launch_cli(interactive=True)