class TokenizerWrapper:
    def __init__(self, vocab, splitter):
        self.splitter = splitter
        self.vocab = vocab

        self.id_pad = len(vocab)
        self.id_cls = len(vocab) + 1
        self.id_sep = len(vocab) + 2
        self.id_mask = len(vocab) + 3

        self.size_vocab = len(vocab) + 4
        self.vocab_size = self.size_vocab

        self.token_pad = '[P@D]'
        self.token_cls = '[CL$]'
        self.token_sep = '[$EP]'
        self.token_mask = '[M@$K]'

        self.index_id_token_special = {
            self.id_pad: self.token_pad,
            self.id_cls: self.token_cls,
            self.id_sep: self.token_sep,
            self.id_mask: self.token_mask
        }

    # end

    def encode(self, line):
        return self.vocab([doc.text.lower() for doc in self.splitter(line)])
    # end

    def decode(self, tokens):
        words = []
        for token in tokens:
            token = int(token)

            if token in self.index_id_token_special:
                word_target = self.index_id_token_special[token]
            else:
                try:
                    word_target = self.vocab.lookup_token(token)
                except:
                    word_target = '[ERROR_LOOKUP_{}]'.format(token)
                # end
            # end

            words.append(word_target)
        # end

        return ' '.join(words)
    # end
# end
