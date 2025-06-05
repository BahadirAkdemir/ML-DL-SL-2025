"""
GPT2 Byte Pair Encoding (BPE) Tokenizer with tiktoken library
"""

import tiktoken
import json
from collections import Counter, deque
from functools import lru_cache

class GPT2Tokenizer:
    """
    GPT-2-compatible tokenizer using OpenAI's tiktoken library.
    - Converts text to token IDs and back, using the 'gpt2' encoding.
    """
    def __init__(self):
        self.encoding = tiktoken.get_encoding("gpt2")
        
        #print("Special tokens and their IDs for GPT2:")
        #for token, token_id in self.encoding._special_tokens.items():
        #    print(f"{repr(token)}: {token_id}")

    def encode(self, text, allowed_special=None):
        return self.encoding.encode(text, allowed_special=allowed_special)

    def decode(self, tokens):
        return self.encoding.decode(tokens)

if __name__ == "__main__":
    # Example usage for GPT-4o tokenizer (tiktoken)
    print("--- GPT-2 Tokenizer (tiktoken) Example ---")
    gpt2_tokenizer = GPT2Tokenizer()
    text = "Hello LLM! <|endoftext|>"
    tokens = gpt2_tokenizer.encode(text, allowed_special={"<|endoftext|>","<|endofprompt|>"})
    print("Tokens:", tokens)
    print("Decoded:", gpt2_tokenizer.decode(tokens))