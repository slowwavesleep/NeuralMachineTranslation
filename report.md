### Baseline
```
model = BasicEncoderDecoder(vocab_size=VOCAB_SIZE,
                            emb_dim=256,
                            model_dim=256,
                            model_layers=2,
                            model_dropout=0.3,
                            padding_index=PAD_INDEX)
```

Начал с подготовки пайплайна для всего процесса обучения. Использовал датасет
субтитров выступлений TED. Оказалось, что там плохой перевод и плохое выравнивание.
Использовать такой датасет нецелесообразно.