# ift6289-llm_gameplay_project
This is the repository for the IFT6289 - Deep NLP class, specifically for the LLM and Gameplay project realized by Louis-Vincent Poellhuber, Bole Yi and Rafaela Pinter. 

***For Evaluators:*** This README is a light tutorial on how to use ChatArena. You'll find a map of all the files we've added / edited at the bottom of this document. 


# 1. Requirements

## Dependencies
First, install all the requirements in the `requirements.txt` file: `pip install -r requirements. txt`. It is highly recommended to use a virtual environment. 

## API Keys
You need API keys for each LLM you want to use. In our project, we've only used Cohere and OpenAI. For every API key, you need to set up an environment variable with the right name so we avoid leaking the key with GitHub. The corresponding environment variable name will be put in parantheses (`KEY`) next to each LLM. 
- **OpenAI** (`OPENAI_API_KEY`): Go to the [openAI API website](https://platform.openai.com/api-keys) and register a new API key with all permissions. **New accounts** get 5$ of credits as free trial, but expire after **3 months**. If your account is older than three months old, you need to create a new account. Your UdeM email should work! 
- **Cohere** (`COHEREAI_API_KEY`): Go to the [Cohere API website](https://docs.cohere.com/reference/about) and create an account. From your account settings, you can find the API key section and register a new one. 
Do note that your account will have a limited rate or a maximum number of calls if you use the free trial.

## Environment Variables
It is **highly** recommended to use a `.env` file, as you'll need to setup a few environment variables. Here are the variables you need for the code to run:

- `OPENAI_API_KEY`: Your OpenAI API key retrieved in the last section. 
- `COHEREAI_API_KEY`: Your Cohere API key retrieved in the last section. 
- `CHATARENA_PATH`: The **absolute** path to the root of this directory. It should look something like this: `c/users/USER/.../this_directory`. 

It is crucial to add the path environment variable, as it's what allows us to use our custom version of ChatArena. Otherwise, we'd have to edit the package itself, in the Python directory. This is also why you shouldn't install ChatArena using pip, as it's our custom version that we use. 

# 2. Directory
You'll find in this directory a couple of important folders. 

- **admin**: This folder hosts all three of our report submissions.
- **datasets**: This folder holds the different datasets we use. The *ask-guess* and *spyfall* datasets are saved in a JSON file, while the *taboo* datasets are saved in multiple JSON files for different topics. 
- **references**: This folder hosts all our related work, the different articles we took inspiration from. 
- **chatarena**: This folder is a fork from the [ChatArena github repository](https://github.com/Farama-Foundation/chatarena). This is the package we modified and it hosts our custom implementation of the framework. The next section will be dedicated to explain how it works.
- **src**: This folder hosts all the code to run the arena. It is mostly separated in three subfolders, one for each game. A further section will be dedicated to explain how to run our code. 

# 3. ChatArena

[ChatArena](https://github.com/Farama-Foundation/chatarena) is a multi-agent game environment framework that was used in the [werewolf article](references\werewolf.pdf) we quote in our final report. 

# File Map
[a relative link](src\askguess\askguess_prompt_experiments.py)