import sacrebleu
from src.utils.data import load_ted, load_anki, clean_anki, basic_load, basic_gzip_load
from src.evaluation.metrics import evaluate_corpus_bleu

r = "я спросил у тома зачем он хочет изучать французский"
t = "я спросил у тома зачем он хочет изучать французский"


references = [[r, r]]
hypothesis = [t, t]

blue = sacrebleu.corpus_bleu(hypothesis, references)
print(blue.score)

print(evaluate_corpus_bleu('results/translations.txt'))
