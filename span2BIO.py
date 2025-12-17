def spans_to_bio(
    text: str,
    spans: list,  
    tokenizer,
    label2id: dict
):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True
    )

    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    labels = ["O"] * len(input_ids)

    for span_start, span_end, span_label in spans:
        span_tokens = []

        for i, (tok_start, tok_end) in enumerate(offsets):

            # 1. Skip special tokens
            if tok_start == tok_end == 0:
                continue

            # 2. Token overlaps span (robust version)
            if tok_end > span_start and tok_start < span_end:
                span_tokens.append(i)

        # 3. Assign BIO labels
        if span_tokens:
            labels[span_tokens[0]] = f"B-{span_label}"
            for idx in span_tokens[1:]:
                labels[idx] = f"I-{span_label}"

    label_ids = [label2id[label] for label in labels]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "label_ids": label_ids,
        "offsets": offsets,
    }
