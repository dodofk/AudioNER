import re

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


def remove_special_characters(text):
    text = re.sub(chars_to_ignore_regex, '', text).lower()
    return text


def extract_all_chars(batch):
    all_text = " ".join(batch)
    vocab = list(set(all_text))
    return vocab
