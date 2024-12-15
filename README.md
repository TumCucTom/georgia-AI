# Georgia-AI
A machine learning AI that looks at a set of wordle results for a given day and tries to infer that days word given how people guessed.

## Overview
We give our bot an input representing several wordle results (the format shown below). We return a probability ditributed over the five letter words, giving the most likely words for that day.

```angular2html
Day 999 5/6
⬜⬜⬜🟨🟨
🟨🟨⬜⬜⬜
⬜🟩🟩🟩🟩
⬜🟩🟩🟩🟩
🟩🟩🟩🟩🟩
```
