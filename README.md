# ift6289-llm_gameplay_project
This is the repository for the IFT6289 - Deep NLP class, specifically for the LLM and Gameplay project realized by Louis-Vincent Poellhuber, Bole Yi and Raphaela Pinter. 


## 1. Requirements

### Dependencies
First, install all the requirements in the `requirements.txt` file: `pip install -r requirements. txt`. If you add any packages or see any packages that aren't in the file yet, don't hesitate to add them! 

### API Keys
For every API key, you need to set up an environment variable with the right name so we avoid leaking the key with GitHub. The corresponding environment variable name will be put in parantheses (`KEY`) next to each LLM. 
- **OpenAI** (`OPENAI_API_KEY`): Go to the (openAI API website)[https://platform.openai.com/api-keys] and register a new API key with all permissions. **New accounts** get 5$ of credits as free trial, but expire after **3 months**. If your account is older than three months old, you need to create a new account. Your UdeM email should work! 
- **Cohere (`COHEREAI_API_KEY`)**: No clue!
- **Gemini (Bard) (`_BARD_API_KEY`)**: No clue! Though it seems Bard isn't supported by default, meaning we might have to mess with the Chatarena package to get it to work. 
- **Claude (`ANTHROPIC_API_KEY`)**: No clue!
- **Any other**: No clue! Is it needed?

## 2. ChatArena Notes

First of all, there is a tutorial in the `src` folder that you can follow to get familiarized with the package. 