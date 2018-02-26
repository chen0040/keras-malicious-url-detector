

def extract_text_model(urls):
    char2idx = dict()
    max_url_seq_length = 0
    for url in urls:
        max_url_seq_length = max(max_url_seq_length, len(url))
        for c in url:
            if c not in char2idx:
                char2idx[c] = len(char2idx)
    num_input_tokens = len(char2idx)
    idx2char = dict([(idx, c) for c, idx in char2idx.items()])

    config = dict()
    config['num_input_tokens'] = num_input_tokens
    config['char2idx'] = char2idx
    config['idx2char'] = idx2char
    config['max_url_seq_length'] = max_url_seq_length

    return config
