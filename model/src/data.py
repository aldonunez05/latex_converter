import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict

try:
    # Standard package import
    from src.constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, IMAGE_CHANNELS, MAX_SEQUENCE_LENGTH
except ImportError:
    # Fallback for direct execution: temporarily adjust path
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from constants import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, IMAGE_CHANNELS, MAX_SEQUENCE_LENGTH

class Vocabulary:
    """
    mock for now
    """
    def __init__(self, token_list: List[str]):
        self.itos = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + token_list
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, latex_string: str) -> List[int]:
        tokens = latex_string.split()
        indices = [self.stoi.get(token, UNK_TOKEN) for token in tokens]
        return indices
    
    def decode(self, indices: List[int]) -> str:
        tokens = [self.itos[i] for i in indices if i not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]]
        return ' '.join(tokens)

class HandwritingDataset(Dataset):
    """
    mock mimicking CHROME/Mathwriting structure
    """
    def __init__(self, num_samples, vocab: Vocabulary):
        self.num_samples = num_samples
        self.vocab = vocab
        self.data: List[Dict] = []

        mock_latex_tokens = ['0', '1', 'x', 'y', 'a', 'b', '\\frac', '\\int', '\\sum', '{', '}', '^', '_', '\\sqrt', '+', '=']

        for i in range(num_samples):
            seq_len = random.randint(5, 40)

            mock_latex_tokens_sample = random.choices(mock_latex_tokens, k=seq_len)
            mock_latex_string = ' '.join(mock_latex_tokens_sample)

            self.data.append({
                'image': torch.rand(IMAGE_CHANNELS, 64, 256),
                'latex': mock_latex_string
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']

        indicies = self.vocab.encode(item['latex'])
        indicies = [SOS_TOKEN] + indicies + [EOS_TOKEN]

        target_sequence = torch.tensor(indicies, dtype=torch.long)

        return image, target_sequence

def collate_fn(batch):
    images, target_sequences = zip(*batch)

    padded_sequences = []
    for seq in target_sequences:
        if len(seq) > MAX_SEQUENCE_LENGTH:
              padded_seq = seq[:MAX_SEQUENCE_LENGTH]
        else:
            padding_needed = MAX_SEQUENCE_LENGTH - len(seq)
            padded_seq = F.pad(seq, (0, padding_needed), 'constant', PAD_TOKEN)
            padded_sequences.append(padded_seq)

    images = torch.stack(images, 0)
    padded_sequences = torch.stack(padded_sequences, 0)

    return images, padded_sequences

