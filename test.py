from src.evaluation.metrics import evaluate_corpus_bleu
import argparse

parser = argparse.ArgumentParser(description='Evaluate BLEU score for given translations.')

parser.add_argument(dest='translations_path', type=str, help='Path to translations.')
args = parser.parse_args()


# 'results/main/translations.txt'

print(evaluate_corpus_bleu(args.translations_path))
