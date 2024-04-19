# Results
---


## Baseline

- They are able to provide descriptions about the word
- They do not interact with each other
- On the voting phase, they tend to modify the JSON format, breaking the code. They usually change the name of the keys in the JSON, so that we are unable to recover the information
- This shows the need of reminding about the correct JSON format each time they have to answer.

## Remember JSON

- They still follow the rules and do not interact with each other during the describing phase.
- They are capable of using the second template, but they were less prone to finish a complete round due to JSON format errors.
- They tend to incriminate the last player that spoke in the description phase as the spy
- They tend to follow the other's opinions in the voting phase


## Sentence and Bracket form

- Started talking to each other instead of playing the game
- Spoke as the host, ignoring instruction not to do so
- All other players followed the first guess as the spy, seeming that they did not analyse the texts
- Most of the games ended up with the round limit because they did not follow the rules of the game
- Started talking to each other and mentioning their names instead of voting, so vote counts cannot represent true intentions
- Little difference between results from sentence and bracket form
- Both of them show little potential of application in the game


# Future work
---

- Try to implement a dictionary format.
- Ask them explicitly to read all the history and then find the explanation that is the less likely to belong to the majority, then find the spy