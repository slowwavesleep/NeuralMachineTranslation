import sacrebleu
# BLEU, ROUGE, METEOR


def read_translations(file_path):
    sources, targets, hypotheses = [], [], []

    with open(file_path) as file:
        for index, line in enumerate(file):
            position = index % 4
            if position == 0:
                sources.append(line)
            elif position == 1:
                targets.append(line)
            elif position == 2:
                hypotheses.append(line)
    return sources, targets, hypotheses


def evaluate_corpus_bleu(file_path):

    sources, references, hypotheses = read_translations(file_path)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    return bleu.score













