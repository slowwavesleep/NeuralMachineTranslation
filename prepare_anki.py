from src.utils.data import load_anki

source, target = load_anki("data/rus.txt")

with open("data/anki_eng.txt", 'w') as file:
    for sentence in source:
        file.write(sentence + '\n')

with open("data/anki_rus.txt", 'w') as file:
    for sentence in target:
        file.write(sentence + '\n')