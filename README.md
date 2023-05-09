# 3D_Modelling_AI
This repository contains the code for a system that generates 3D scenarios based on text input. The system uses natural language processing (NLP) techniques to extract relevant information from the input text, and 3D modeling software to create the corresponding 3D scenario.

## Requirements
The system requires the following dependencies:

Python 3.6+
nltk library
3D modeling software (e.g., Blender, Maya, 3ds Max)

## Installation
To install the required dependencies, run the following commands:
```
pip install nltk
pip install torch
```
You will also need to download additional resources for the nltk library. Run the following command to open the nltk downloader, and then select and download the resources:

```
python -m nltk.downloader
```
Finally, install the 3D modeling software of your choice and ensure that it is properly configured.

## Usage
To use the system, run the main.py script with the input text as an argument. For example:

```
python main.py "The quick brown fox jumps over the lazy dog. The dog barks loudly."
```
The script will pre-process the text data, extract relevant information using NLP techniques, and generate a 3D scenario using the configured 3
