![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
# Georgia-AI ![logo](images/logo.jpg)
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
