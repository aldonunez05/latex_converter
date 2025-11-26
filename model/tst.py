# hi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Dict

# --- Hyperparameters (Placeholders) ---
HIDDEN_SIZE = 256
# VOCAB_SIZE is now defined by the mock vocabulary
MAX_SEQUENCE_LENGTH = 150
IMAGE_CHANNELS = 1 # Grayscale handwritten image
# Common indices (must match the Vocabulary class)
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
UNK_TOKEN = 3

# --- Mock Vocabulary and Dataset Utilities ---
class Vocabulary:
    """
    Mock Vocabulary for tokenizing LaTeX strings.
    In a real scenario, this would be built from the entire dataset.
    """
    def __init__(self, token_list: List[str]):
        self.itos = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + token_list
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, latex_string: str) -> List[int]:
        """Convert a LaTeX string into a list of token indices."""
        # Simple split, assumes tokens are separated by spaces (e.g., '\frac { a } { b }')
        tokens = latex_string.split()
        indices = [self.stoi.get(token, UNK_TOKEN) for token in tokens]
        return indices

    def decode(self, indices: List[int]) -> str:
        """Convert a list of token indices back to a LaTeX string."""
        tokens = [self.itos[i] for i in indices if i not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]]
        return ' '.join(tokens)

class HandwritingDataset(Dataset):
    """
    Mock Dataset mimicking CROHME/MathWriting structure.
    Returns image and tokenized LaTeX sequence.
    """
    def __init__(self, num_samples, vocab: Vocabulary):
        self.num_samples = num_samples
        self.vocab = vocab
        self.data: List[Dict] = []
        
        # Mock data generation
        mock_latex_tokens = ['\\frac', '{', 'a', '}', '+', '1', '\\sqrt', 'x']
        
        for i in range(num_samples):
            # Generate a random length sequence
            seq_len = random.randint(5, 40)
            
            # Create a mock LaTeX string (e.g., '{ a } + \\frac { 1 } { x }')
            mock_latex_tokens_sample = random.choices(mock_latex_tokens, k=seq_len)
            mock_latex_string = ' '.join(mock_latex_tokens_sample)
            
            self.data.append({
                'image': torch.rand(IMAGE_CHANNELS, 64, 256), # Mock image tensor (C, H, W)
                'latex': mock_latex_string
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        
        # Encode LaTeX string
        indices = self.vocab.encode(item['latex'])
        # Add SOS and EOS tokens
        indices = [SOS_TOKEN] + indices + [EOS_TOKEN]
        
        # Note: Padding is handled in the collate function
        target_sequence = torch.tensor(indices, dtype=torch.long)
        
        return image, target_sequence

def collate_fn(batch):
    """
    Collate function to pad the target sequences to the same length
    and stack them into a single tensor.
    """
    images, target_sequences = zip(*batch)
    
    # 1. Pad sequences
    max_len = MAX_SEQUENCE_LENGTH # Fixed maximum length
    
    padded_sequences = []
    for seq in target_sequences:
        # Pad up to max_len
        if len(seq) > max_len:
            padded_seq = seq[:max_len]
        else:
            padding_needed = max_len - len(seq)
            padded_seq = F.pad(seq, (0, padding_needed), 'constant', PAD_TOKEN)
        padded_sequences.append(padded_seq)

    # 2. Stack and return
    images = torch.stack(images, 0)
    padded_sequences = torch.stack(padded_sequences, 0)
    
    return images, padded_sequences


# ----------------------------------------------------------------------
# 1. ENCODER: Extracts visual features from the handwritten image.
# ----------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, hidden_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        B, C, H, W = x.size()
        
        # Reshape to (Batch, Sequence_Length, HIDDEN_SIZE)
        feature_sequence = x.view(B, C, H * W).permute(0, 2, 1)
        return feature_sequence

# ----------------------------------------------------------------------
# 2. ATTENTION MECHANISM: Helps the decoder focus on relevant parts of the image.
# ----------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.squeeze(0)
        
        seq_len = encoder_outputs.size(1)
        tiled_hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        combined = torch.cat((tiled_hidden, encoder_outputs), dim=2)
        
        energy = torch.tanh(self.Wa(combined))
        
        scores = torch.sum(energy * self.v, dim=2)
        
        attn_weights = F.softmax(scores, dim=1).unsqueeze(1) # (Batch, 1, Sequence_Length)

        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)

        return context, attn_weights

# ----------------------------------------------------------------------
# 3. DECODER: Generates the LaTeX sequence token by token.
# ----------------------------------------------------------------------
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        embedded = self.embedding(input_token).unsqueeze(1)
        context, attn_weights = self.attention(hidden[0], encoder_outputs) 
        
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)

        output, hidden = self.lstm(rnn_input, hidden)

        # Predict next token using output and context
        output = self.out(torch.cat((output.squeeze(1), context), dim=1))
        
        return output, hidden, attn_weights

# ----------------------------------------------------------------------
# 4. FULL MODEL: Combines Encoder and Decoder
# ----------------------------------------------------------------------
class HTRModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(HTRModel, self).__init__()
        self.encoder = ImageEncoder(hidden_size)
        self.decoder = AttentionDecoder(vocab_size, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, input_image, target_sequence=None, teacher_forcing_ratio=0.5):
        encoder_outputs = self.encoder(input_image)

        batch_size = input_image.size(0)
        device = input_image.device
        
        # Initialize decoder state
        decoder_hidden = (
            torch.zeros(1, batch_size, self.hidden_size).to(device),
            torch.zeros(1, batch_size, self.hidden_size).to(device)
        )
        
        # First input token is <SOS>
        decoder_input = torch.tensor([SOS_TOKEN] * batch_size, device=device) 
        
        all_predictions = []

        # Time-step loop
        for t in range(MAX_SEQUENCE_LENGTH):
            output, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            all_predictions.append(output)

            # Get the predicted token index
            top_value, top_index = output.topk(1, dim=1)
            
            # Determine if we use teacher forcing
            use_teacher_forcing = target_sequence is not None and torch.rand(1).item() < teacher_forcing_ratio

            if use_teacher_forcing and t < target_sequence.size(1):
                # Use ground truth
                decoder_input = target_sequence[:, t]
            else:
                # Use model's own prediction
                decoder_input = top_index.squeeze(1).detach() 
        
        # Stack the predictions: (Batch, MAX_SEQUENCE_LENGTH, VOCAB_SIZE)
        return torch.stack(all_predictions, dim=0).permute(1, 0, 2)

# ----------------------------------------------------------------------
# 5. TRAINING FUNCTIONS
# ----------------------------------------------------------------------

def train_epoch(model, dataloader, criterion, optimizer, device, teacher_forcing_ratio):
    """
    Runs one full epoch of training.
    """
    model.train()
    total_loss = 0
    
    # [Image of Encoder-Decoder with Attention]
    
    for i, (images, target_sequences) in enumerate(dataloader):
        images = images.to(device)
        target_sequences = target_sequences.to(device)

        optimizer.zero_grad()

        # Forward pass
        # The model generates a sequence of predictions up to MAX_SEQUENCE_LENGTH
        predictions = model(images, target_sequences, teacher_forcing_ratio=teacher_forcing_ratio)

        # Calculate Loss
        # We need to flatten the predictions (Batch * Seq_Len, Vocab_Size) 
        # and the targets (Batch * Seq_Len) for CrossEntropyLoss.
        
        # Note: We compute loss on the entire MAX_SEQUENCE_LENGTH sequence,
        # but the criterion ignores the PAD_TOKEN (index 2).
        P_flat = predictions.reshape(-1, model.decoder.out.out_features)
        T_flat = target_sequences.reshape(-1)

        loss = criterion(P_flat, T_flat)
        loss.backward()
        
        # Basic gradient clipping helps stabilize RNN training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
        
        optimizer.step()
        total_loss += loss.item()
        
        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def train(model, train_loader, val_loader, vocab, num_epochs, device, initial_lr=0.001):
    """
    Main training loop orchestration.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN) 
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    
    # Teacher forcing ratio decay schedule
    teacher_forcing_schedule = {
        0: 1.0,  # Start with 100% teacher forcing
        5: 0.75,
        10: 0.5,
        15: 0.25,
        20: 0.1
    }
    
    print("\nStarting Training...")
    
    for epoch in range(num_epochs):
        # Update Teacher Forcing Ratio
        tfr = teacher_forcing_schedule.get(epoch, 0.0)
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} (TFR: {tfr:.2f}) ---")
        
        # Training Phase
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, tfr)
        
        # Validation Phase (Inference should always use 0.0 TFR)
        val_loss = evaluate(model, val_loader, criterion, device, vocab)
        
        print(f"Epoch {epoch+1} complete. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Simple stop condition could be added here (e.g., Early Stopping)

def evaluate(model, dataloader, criterion, device, vocab):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for i, (images, target_sequences) in enumerate(dataloader):
            images = images.to(device)
            target_sequences = target_sequences.to(device)
            
            # Inference mode: teacher_forcing_ratio=0.0
            predictions = model(images, target_sequences, teacher_forcing_ratio=0.0)
            
            P_flat = predictions.reshape(-1, model.decoder.out.out_features)
            T_flat = target_sequences.reshape(-1)
            
            loss = criterion(P_flat, T_flat)
            total_loss += loss.item()
            
            if i == 0:
                # Example inference output for the first batch
                predicted_indices = predictions.argmax(dim=2)
                
                print("\n  [Validation Sample Output]")
                print(f"  Ground Truth: {vocab.decode(target_sequences[0].tolist())}")
                print(f"  Prediction:   {vocab.decode(predicted_indices[0].tolist())}\n")
            
    return total_loss / len(dataloader)


# ----------------------------------------------------------------------
# 6. MAIN EXECUTION
# ----------------------------------------------------------------------
def main():
    print("Setting up HTR Training Pipeline...")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # 2. Setup Vocabulary and Hyperparameters
    # The actual vocabulary would be much larger (100-300 tokens)
    latex_tokens = ['0', '1', '2', '3', 'x', 'y', 'a', 'b', '\\frac', '\\int', '\\sum', '{', '}', '^', '_', '\\sqrt']
    vocab = Vocabulary(latex_tokens)
    VOCAB_SIZE = vocab.vocab_size
    
    # 3. Initialize Model
    model = HTRModel(VOCAB_SIZE, HIDDEN_SIZE).to(device)
    
    # 4. Setup DataLoaders (using mock data for demonstration)
    BATCH_SIZE = 16
    NUM_TRAIN_SAMPLES = 500
    NUM_VAL_SAMPLES = 100
    NUM_EPOCHS = 3
    
    train_dataset = HandwritingDataset(NUM_TRAIN_SAMPLES, vocab)
    val_dataset = HandwritingDataset(NUM_VAL_SAMPLES, vocab)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    print(f"Vocab Size: {VOCAB_SIZE}. Training on {NUM_TRAIN_SAMPLES} samples.")
    
    # 5. Run Training
    train(model, train_loader, val_loader, vocab, NUM_EPOCHS, device)
    
    print("\nTraining complete! Mock model is now 'trained'.")

if __name__ == "__main__":
    main()
