import sacrebleu
from src.evaluation.metrics import evaluate_corpus_bleu

r = "я спросил у тома зачем он хочет изучать французский"
t = "я спросил у тома зачем он хочет изучать французский"


references = [[r, r]]
hypothesis = [t, t]

blue = sacrebleu.corpus_bleu(hypothesis, references)
print(blue.score)

print(evaluate_corpus_bleu('results/main/translations.txt'))
