import random
import string

import torch


def _generate_word(start_letter, model, tokenizer, max_length, temperature=1.0):
    start_token_idx = tokenizer.encode("<s>").ids[0]
    start_letter_idx = tokenizer.encode(start_letter).ids[0]
    input_seq = torch.tensor([[start_token_idx, start_letter_idx]], dtype=torch.long)

    generated_word = [start_letter_idx]
    hidden = model.init_hidden(input_seq)
    for _ in range(max_length - 1):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
        output = output.squeeze(0)
        output = output[-1, :].view(-1).div(temperature).exp()
        next_token = torch.multinomial(output, 1).item()
        if next_token == tokenizer.token_to_id("<pad>"):
            break
        generated_word.append(next_token)
        input_seq = torch.tensor([generated_word], dtype=torch.long)
    return tokenizer.decode(generated_word)


def sample_n(n: int, model, tokenizer, max_length=20, temperature=1.0) -> list[str]:
    output_words = []
    for _ in range(n):
        random_start_letter = random.choice(string.ascii_lowercase)
        new_word = _generate_word(
            random_start_letter,
            model,
            tokenizer,
            max_length=max_length,
            temperature=temperature,
        )
        output_words.append(new_word)
    return output_words
