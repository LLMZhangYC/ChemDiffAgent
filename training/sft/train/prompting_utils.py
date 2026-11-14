from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")


import torch
class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 max_prompt_len=8000, max_gen_length=377, ignore_id=-100,
                 max_seq_length=None, mask_tool_responses=True):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.max_gen_length = max_gen_length
        self.max_prompt_len = max_prompt_len
        self.max_seq_length = max_seq_length
        self.mask_tool_responses = mask_tool_responses
        self.kept_indices = []
        self.stats = {}
        if max_seq_length is not None and (not isinstance(max_seq_length, int) or max_seq_length < 3):
            raise ValueError("max_seq_length must be an integer >= 3")
        if max_prompt_len < 1 or max_gen_length < 1:
            raise ValueError("Prompt/response limits must be positive")

    def _bounded_prompt(self, prompts, responses):
        """Bound P + 2R before padding/duplication, preserving the existing API.

        The original SFT pipeline uses a shared prompt width for the whole
        dataset. Budget against that padded width, not each unpadded prompt.
        Long prompts retain the original drop policy. Responses keep a prefix.
        """
        if len(prompts) != len(responses) or not prompts:
            raise ValueError("Expected nonempty, equally sized prompt/response lists")
        tok = self.text_tokenizer
        if tok.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id")
        prompt_ids = tok(prompts, padding=False, truncation=False)["input_ids"]
        candidates = [i for i, ids in enumerate(prompt_ids) if 0 < len(ids) <= self.max_prompt_len]
        if not candidates:
            raise ValueError("No prompts remain after max_prompt_len filtering")
        prompt_width = max(len(prompt_ids[i]) for i in candidates)
        budget = min(self.max_gen_length, (self.max_seq_length - prompt_width) // 2)
        if budget < 1:
            raise ValueError("Padded prompt leaves no response budget under max_seq_length")

        begin_id = end_id = None
        if self.mask_tool_responses:
            tags = ("<tool_response>", "</tool_response>")
            ids = [tok.encode(tag, add_special_tokens=False) for tag in tags]
            if any(len(value) != 1 or value[0] == tok.unk_token_id for value in ids) or ids[0] == ids[1]:
                raise ValueError("Tokenizer must have distinct single-token tool_response tags")
            begin_id, end_id = ids[0][0], ids[1][0]

        kept, prefixes = [], []
        clipped = trimmed_tools = dropped_response = 0
        for index in candidates:
            # Tokenize without padding; never construct a full padded response
            # tensor before applying the sequence budget.
            original = tok(responses[index], padding=False, truncation=False)["input_ids"]
            prefix = original[:budget]
            clipped += len(original) > budget
            if self.mask_tool_responses:
                open_at = None
                for pos, token in enumerate(prefix):
                    if token == begin_id:
                        if open_at is not None:
                            raise ValueError(f"Nested tool_response tags in sample {index}")
                        open_at = pos
                    elif token == end_id:
                        if open_at is None:
                            raise ValueError(f"Unmatched tool_response end in sample {index}")
                        open_at = None
                if open_at is not None:
                    if len(original) <= budget:
                        raise ValueError(f"Unmatched tool_response begin in sample {index}")
                    prefix = prefix[:open_at]
                    trimmed_tools += 1

            trainable, inside_tool = False, False
            for token in prefix:
                if self.mask_tool_responses and token == begin_id:
                    inside_tool = True
                elif self.mask_tool_responses and token == end_id:
                    inside_tool = False
                elif not inside_tool and token != tok.pad_token_id:
                    trainable = True
            if not trainable:
                dropped_response += 1
                continue
            kept.append(index)
            prefixes.append(prefix)
        if not kept:
            raise ValueError("No samples with trainable response tokens remain after truncation")

        response_width = max(map(len, prefixes))
        sequences = [[tok.pad_token_id] * (prompt_width - len(prompt_ids[i])) + prompt_ids[i]
                     + response + [tok.pad_token_id] * (response_width - len(response))
                     for i, response in zip(kept, prefixes)]
        input_ids = torch.tensor(sequences, dtype=torch.long)
        actual = prompt_width + 2 * response_width
        if actual > self.max_seq_length:
            raise RuntimeError("Expanded sequence exceeded configured bound")
        self.kept_indices = kept
        self.stats = {"source_samples": len(prompts), "kept_samples": len(kept),
                      "dropped_prompts": len(prompts) - len(candidates),
                      "dropped_responses": dropped_response, "truncated_responses": clipped,
                      "trimmed_tool_spans": trimmed_tools, "prompt_width": prompt_width,
                      "response_width": response_width, "expanded_length": actual,
                      "max_seq_length": self.max_seq_length}
        return input_ids, input_ids.clone(), prompt_width, len(prompts) - len(kept)


    # language modeling
    def lm_prompt(self, text_ids_pairs):
        prompts_list, responses_list = text_ids_pairs
        pad_id = self.text_tokenizer.pad_token_id

        if responses_list.shape[1] < self.max_gen_length:
            max_seq_len = prompts_list.shape[1] + responses_list.shape[1]
        else:
            max_seq_len = prompts_list.shape[1] + self.max_gen_length

        sequence_ids = []
        attention_masks = []
        label_ids = []

        for prompt_ids, resp_ids in zip(prompts_list, responses_list):
            prompt_ids = prompt_ids.tolist()
            resp_ids   = resp_ids.tolist()

            temp_ids = prompt_ids + resp_ids
            temp_masks = [1] * len(temp_ids)
            temp_labels = temp_ids.copy()

            if len(temp_ids) < max_seq_len:
                pad_len = max_seq_len - len(temp_ids)
                temp_ids.extend([pad_id] * pad_len)
                temp_labels.extend([pad_id] * pad_len)
                temp_masks.extend([0] * pad_len)
            else:
                temp_ids = temp_ids[:max_seq_len]
                temp_labels = temp_labels[:max_seq_len]
                temp_masks = temp_masks[:max_seq_len]

            sequence_ids.append(torch.tensor(temp_ids).unsqueeze(0))
            attention_masks.append(torch.tensor(temp_masks).unsqueeze(0))
            label_ids.append(torch.tensor(temp_labels).unsqueeze(0))

        input_ids = torch.cat(sequence_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        label_ids = torch.cat(label_ids, dim=0)
        

        return input_ids, label_ids, prompts_list.shape[1]

        
    

    def mask_prompt(self):
        pass

    def __call__(self, input):
        prompts, responses = input
        if self.max_seq_length is not None:
            return self._bounded_prompt(prompts, responses)

        enc = self.text_tokenizer(
            prompts,
            padding=False,
            truncation=False,
            return_length=True
        )
        lengths = enc["length"]
        keep_indices = [i for i, L in enumerate(lengths) if L <= self.max_prompt_len]
        self.kept_indices = keep_indices
        drop_num = len(prompts) - len(keep_indices)
        
        prompts  = [prompts[i]  for i in keep_indices]
        responses = [responses[i] for i in keep_indices]

        prompt_ids = self.text_tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            padding_side = "left"
        )['input_ids']
        response_ids = self.text_tokenizer(
            responses,
            padding=True,
            return_tensors="pt",
            padding_side = "right"
        )['input_ids']
        input_ids_lm, labels_lm, start_pos = self.lm_prompt((prompt_ids, response_ids))
        return input_ids_lm, labels_lm, start_pos, drop_num


if __name__ == '__main__':
    pass
