import sacrebleu
from typing import Callable, List
from tqdm import tqdm
from src.nn.translation import Translator
# BLEU, ROUGE, METEOR


def evaluate_corpus_bleu(source_sentences: List[str],
                         target_sentences: List[str],
                         translator: Translator) -> float:
    hypotheses = []
    for sentence in tqdm(source_sentences, total=len(source_sentences)):
        hypotheses.append(translator.translate(sentence))

    bleu = sacrebleu.corpus_bleu(hypotheses, [target_sentences])  # <- references expect a list of lists as input

    return bleu.score
