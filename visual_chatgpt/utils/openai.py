import tiktoken


def num_tokens(model_name: str, text: str) -> int:
    """This is the official way to calculate the number of tokens, according to OpenAI.
    See:
        - https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        - https://github.com/openai/tiktoken
        - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    tokenizer = tiktoken.encoding_for_model(model_name)
    return len(tokenizer.encode(text))
