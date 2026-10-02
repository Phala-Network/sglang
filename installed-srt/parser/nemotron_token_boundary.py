"""Locate a real reasoning control token without matching quoted BPE text."""


class NemotronTokenBoundary:
    def __init__(self, tokenizer, decode_kwargs):
        self.tokenizer = tokenizer
        self.decode_kwargs = decode_kwargs
        self.start_id = tokenizer.convert_tokens_to_ids("<think>")
        self.end_id = tokenizer.convert_tokens_to_ids("</think>")
        if (
            not all(isinstance(i, int) for i in (self.start_id, self.end_id))
            or self.start_id == self.end_id
        ):
            raise ValueError("Nemotron reasoning requires distinct control token IDs")
        self.first_id = None
        self.leading_start_count = 0
        self._at_token_prefix = True
        self.seen_ids = 0
        self.text_length = 0
        self.end_offset = None
        self.prefix_ids = []

    def update(self, text, output_ids, incremental):
        self.text_length += len(text)
        if self.end_offset is not None:
            return
        if incremental:
            new_ids = output_ids
        else:
            if len(output_ids) < self.seen_ids:
                raise ValueError("Nemotron cumulative output token IDs moved backwards")
            new_ids = output_ids[self.seen_ids :]
        self.seen_ids += len(new_ids)
        if self.first_id is None and new_ids:
            self.first_id = new_ids[0]
        if self._at_token_prefix:
            for token_id in new_ids:
                if token_id != self.start_id:
                    self._at_token_prefix = False
                    break
                self.leading_start_count += 1
        # Decode the prefix exactly once, when the real closer first appears.
        # Do not repeatedly decode growing output or retain final-answer tokens.
        if self.end_id in new_ids:
            self.prefix_ids.extend(new_ids[: new_ids.index(self.end_id) + 1])
            prefix = self.tokenizer.decode(self.prefix_ids, **self.decode_kwargs)
            if not prefix.endswith("</think>"):
                raise ValueError("Nemotron reasoning control token was not preserved")
            self.end_offset = len(prefix) - len("</think>")
            self.prefix_ids.clear()
        else:
            self.prefix_ids.extend(new_ids)

    def end_index(self, buffer):
        if self.end_offset is None:
            return -1
        index = self.end_offset - (self.text_length - len(buffer))
        if index < 0:
            raise ValueError("Nemotron reasoning boundary preceded buffered text")
        if index + len("</think>") > len(buffer):
            return -1
        if buffer[index : index + len("</think>")] != "</think>":
            raise ValueError("Nemotron token boundary differs from detokenized text")
        return index
