# Model Comparison Using TOPSIS

This project evaluates various language models based on multiple performance metrics using the TOPSIS method.

## Overview
The script generates text from multiple models, analyzes their features, and ranks them using the TOPSIS decision-making technique.

## Features Evaluated
1. **Word Count** - Number of words in the generated text.
2. **Readability Score** - Flesch reading ease score.
3. **Lexical Diversity** - Ratio of unique words to total words.
4. **Grammar Errors** - Number of detected grammar issues.
5. **Perplexity** - Model's ability to predict the next word (lower is better).

## Model Rankings
![image](https://github.com/user-attachments/assets/bef24394-34e3-4536-81b5-09ac55e20879)


## Visualizations
![image](https://github.com/user-attachments/assets/1619a0f0-b447-4638-aca4-a5f4a8e9dbc3)
![image](https://github.com/user-attachments/assets/076e9fd6-4131-4cb5-804a-013c00c47378)


## Dependencies
- numpy
- matplotlib
- seaborn
- sklearn
- transformers
- textstat
- language_tool_python
- torch

## Usage
Run the script in a Python environment with the required libraries installed. The output will include model rankings and visualized comparisons.

## License
MIT License

