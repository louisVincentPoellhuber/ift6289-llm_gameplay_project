# ift6289-llm_gameplay_project
This is the repository for the IFT6289 - Deep NLP class, specifically for the LLM and Gameplay project realized by Louis-Vincent Poellhuber, Bole Yi and Rafaela Pinter. 

***For Evaluators:*** This README is a light tutorial on how to use ChatArena. You'll find a map of all the files we've added / edited at the bottom of this document. 


# 1. Requirements

## Dependencies
First, install all the requirements in the `requirements.txt` file: `pip install -r requirements.txt`. It is highly recommended to use a virtual environment. 

## API Keys
You need API keys for each LLM you want to use. In our project, we've only used Cohere and OpenAI. For every API key, you need to set up an environment variable with the right name. The corresponding environment variable name will be put in parantheses (`KEY`) next to each LLM. 
- **OpenAI** (`OPENAI_API_KEY`): Go to the [openAI API website](https://platform.openai.com/api-keys) and register a new API key with all permissions. **New accounts** get 5$ of credits as free trial, but expire after **3 months**. If your account is older than three months old, you need to create a new account. 
- **Cohere** (`COHEREAI_API_KEY`): Go to the [Cohere API website](https://docs.cohere.com/reference/about) and create an account. From your account settings, you can find the API key section and register a new one. Do note that your account will have a limited rate or a maximum number of calls if you use the free trial.

## Environment Variables
It is **highly** recommended to use a `.env` file, as you'll need to setup a few environment variables. Here are the variables you need for the code to run:

- `OPENAI_API_KEY`: Your OpenAI API key retrieved in the last section. 
- `COHEREAI_API_KEY`: Your Cohere API key retrieved in the last section. 
- `CHATARENA_PATH`: The **absolute** path to the root of this directory. It should look something like this: `c/users/USER/.../this_directory`. 

It is crucial to add the `CHATARENA_PATH` environment variable, as it's what allows us to use our custom version of ChatArena. Otherwise, we'd have to edit the package itself, in the Python directory. This is also why you shouldn't install ChatArena using pip, as it's our custom version that we use. 

# 2. Directory
You'll find in this directory a couple of important folders. 

- **admin**: This folder hosts all three of our report submissions.
- **references**: This folder hosts all our related work, the different articles we took inspiration from. 
- **chatarena**: This folder is a fork from the [ChatArena github repository](https://github.com/Farama-Foundation/chatarena). This is the package we modified and it hosts our custom implementation of the framework. The next section will be dedicated to explain how it works.
- **src**: This folder hosts all the code and datasets necessary to run the arena. It is mostly separated in four subfolders, one for each game and the datasets. A further section will be dedicated to explain how to run our code. 

# 3. ChatArena

[ChatArena](https://github.com/Farama-Foundation/chatarena) is a multi-agent game environment framework that was used in the [werewolf article](references\werewolf.pdf) we quote in our final report. It allows for different LLMs to interact with each other in the context of conversational games. To better understand how this framework works, we'll explain it using three of its main components. 

## Environments
An **Environment** is a class that holds all the game logic for a specific game. It keeps track of the game state, as well as its rules, and how the different steps happen. To create an environment, you need to inherit from the abstract class from the same name and define a couple of generic functions. Besides that, all the game logic depends on your own implementation. This leaves lots of room for experimentation and design. 

To communicate the game rules to the agents, ChatArena uses a **Moderator**, which acts as a *fake* agent. The **Moderator** sends messages to the agents, through the **Message Pool**, giving them the context of the game. It's through it that we can give the agents the game rules, their roles, the explanation for different steps and phases through the game, etc. It acts as an interface between the developers and the agents. 

## Players
An **Player** is an LLM agent that plays the games. When creating an experiment, we need to initalize each **Player** with its *role* and its *backend*. For the role, we generally provide each agent with the global game rules rather than their specific roles. This is mostly because we want to be able to randomly pick each role during a game, rather than have it pretedermined beforehand. This showcases a weakness of ChatArena, as we then need the **Moderator** to give each player their role, rather than the *role* field in **Player** initialization. 

A **backend** is simply an interface to whatever LLM API we want to use. Again, this is an abstract class that can be inherited for each LLM. Much of these **backends** were quite outdated by the time we forked the repository, especially the **Cohere** one. Because each API is different, a lot of adaptation is necessary to make it work. There exists a **Human backend** that can be used to have a real person play with the LLMs as an agent. However, we haven't tested this. 

## Arena
The **Arena** is the main framework, it's what connects **Players** to the proper **Environment**. It iterates through the different players, checks for errors and ensures that the **Environment**'s game rules are respected. It makes use of a **CLI** UI to display the discussion. You can watch the discussion happen from your terminal. The conversation will appear quite slow because we've put a 3-second timer between each response to ensure we don't go over the call rate limit. Additionally, these conversations are saved on your computer. Next section will explain how. Finally, it is possible to use the **Arena** *interactively*. This allows us to save game states, save conversations up to that point, cancel a conversation, etc. 

# 4. Running the Code

In this section we'll cover how to run our code and how to recreate our experiments. The folder you'll want to look at is `src`. Inside it, there are three subfolders, `askguess`, `spyfall` and `taboo`, one for each game. Each subfolder has the following:

## Testing File
Each game should have a python file for testing, called `game_something_test.py`. This is where we test implementations and run quick experiments. You can use this file to see an example of the game using the pretermined configuration. The **midway** and **final** tags represent the tests we've done for the corresponding reports. 

## Game Configuration
Game configurations are hosted on YAML files. These files hold hardcoded prompts, befitting whichever prompt strategy we adopt. This is how we can quickly run experiments, by simply iterating through different configurations. Setting up the YAML file can be complicated, as it's also necessary to go edit the **Environment** file to make sure the game logic can retrieve the correct prompts at the correct time. It became necessary for us to define many game states and lots of **IF** statements to trigger the right prompts at the right time. 

## Experiment Files
These files are a generalization of the testing file. Each game should have a file called `game_something_experiment.py`. The **prompt** experiments are those ran for the midway report as well as for part of the final report, while the **final** experiments are those ran for the final report.. 

This file is quite simple, we provide a list of different prompt configurations to test, as well as the number of tests to run. Then, the experiments will run on your terminal. The different conversations will be saved in subfolders of the `chat_history` folder, each correpsonding to the title of the prompt configuration used. 

## Chat History
This subfolder holds all the chat histories of different experiments. It contains many subfolders corresponding to different prompt strategies. Each of those holds the conversations for the corresponding strategy. Each conversation is saved in a JSON file dated to the exact time it was saved. The JSON has four main components:

- **disposition**: This is largely game-specific, but it holds game information. For example, it might hold the secret word, the number of players, their roles, etc. 
- **players**: This contains a list of each **Player**, their backend, their role and their role description. 
- **metrics**: This holds the ending conditions of the game. It keeps how many turns the game lasted, as well as the ending state, as described in the reports. 

This way to save the chat history was largely implemented by us, expanding on ChatArena's base function. 

## Performance
To evaluate the performance of each experiment, we use the `performance_crawler.py` files, located in the subfolder for each game. This file simply goes through the different chat histories it is provided with and calculates how many of each ending state it finds, for each prompt strategy. It then saves those metrics in the `performance` subfolder of each game, as absolute or relative values. The CSV file can then easily be translated into visualisations or into LaTeX. 

## Chat
Finally, if you just want to talk to LLMs, you can use the `prompt_optimization.py` file. It launches an interactive **Arena** instance where you can simply ask questions to an agent. This is useful to quickly test agent capabilities and the impact of different prompts, hence the name. In between each answer, the interactive session will ask you, the user, to choose one of many options. 

First, it'll ask you for a model:
- **c**: Cohere
- **g**: GPT, OpenAI
- **q**: Quit

It will then ask for your name for display purposes. It will then provide you with five commands. You can choose the **h** command for a full explanation of each of them, but most of the time you'll just want to use the **n** command to move on to the next step. 

No experiments have been run using this file, but it's useful to assert LLM capabilities. 

# 5. File Map
Here is a map of the different files we created or edited in this repository. Please read the header of each of these files, as they'll explain what they are and what our contributions were. Look out for the tag used, as it describes what we did with the file:
- ***NEW***: This tags a new file that we've created. 
- ***EDIT***: This tags a file that we've edited, but not created. The header should describe what our edits are. You can also use ctrl+f to find the other ***EDIT*** comments throughout the code, as they mark what we've changed. 
- ***DEBUG***: This tags a file that we've lightly edited for debug purposes. 

**Environments**: These are the **Environments** we've added or modified. 

- [Ask-Guess](chatarena\environments\askguess.py)
- [Taboo](chatarena\environments\taboo.py)
- [Spyfall](chatarena\environments\spyfall.py)

**Backends**: These are the **backends** we've added or modified. 

- [Cohere](chatarena\backends\cohere.py)
- [OpenAI](chatarena\backends\openai.py)

**Framework**: These are the parts of the ChatArena framework we've modified. 

- [Agent](chatarena\agent.py)
- [Arena](chatarena\arena.py)
  
**Scripts**: Rather than link each script, you can just go in the [`src` folder](src), as everything in there is new. 