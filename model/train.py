import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys 
import os

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data import Vocabulary, HandwritingDataset, collate_fn
from src.img2seq import HTRModel
from src.constants import HIDDEN_SIZE, PAD_TOKEN, MAX_SEQUENCE_LENGTH

def train_epoch(model, dataloader, criterion, optimizer, device, teacher_forcing_ratio):
    model.train()
    total_loss = 0

    for i, (images, target_sequences) in enumerate(dataloader):
        images = images.to(device)
        target_sequences = target_sequences.to(device)

        optimizer.zero_grad()

        predicitons = model(images, target_sequences, teacher_forcing_ratio=teacher_forcing_ratio)

        P_flat = predictions.reshape(-1, model.decoder.out.out_features)
        T_flat = target_sequences.reshape(-1)

        loss = criterion(P_flat, T_flat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()

        if (i+1) % 5 == 0:
            print(f" Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, vocab):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, (images, target_sequences) in enumerate(dataloader):
            images = images.to(device)
            target_sequences = target_sequences.to(device)

            predicitons = model(images, target_sequences, teacher_forcing_ratio=0.0)

            P_flat = predictions.reshape(-1, model.decoder.out.out_features)
            T_flat = target_sequences.reshape(-1)

            loss = criterion(P_flat, T_flat)
            total_loss += loss.item()

            if i == 0:
                predicted_indices = predictions.argmax(dim=2)

                print("\n [Validation Sample Output]")
                print(f" Ground Truth: {vocab.decode(target_sequences[0].tolist())}")
                print(f" Prediction: {vocab.decode(predicted_indices[0].tolist())}\n")
    return total_loss / len(dataloader)

def train(model, train_loader, val_loader, vocab, num_epochs, device, initial_lr=0.001):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    teacher_forcing_schedule = {0: 1.0, 1:0.75, 2:0.5, 3: 0.25}

    print("\nStarting Training")

    for epoch in range(num_epochs):
        tfr = teacher_forcing_schedule.get(epoch, 0.0)

        print(f"\n---Epoch {epoch+1}/{num_epochs} (TFR: {tfr:.2f}) ---")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, tfr)

        val_loss = evaluate(model, val_loader, criterion, device, vocab)

        print(f"Epoch {epoch+1} complete. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    print("setting up htr training pipeline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on: {device}")

    latex_tokens = ['0', '1', '2', '3', 'x', 'y', 'a', 'b', '\\frac', '\\int', '\\sum', '{', '}', '^', '_', '\\sqrt', '+', '=', '(', ')', '\\sin', '\\cos']
    vocab = Vocabulary(latex_tokens)
    VOCAB_SIZE = vocab.vocab_size

    model = HTRModel(VOCAB_SIZE, HIDDEN_SIZE).to(device)

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
            collate_fn = collate_fn
    )
    val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
    )

    print(f"Vocab Size: {VOCAB_SIZE}. Max Seq Length: {MAX_SEQUENCE_LENGTH}. Training on {NUM_TRAIN_SAMPLES} samples")

    train(model, train_loader, val_loader, vocab, NUM_EPOCHS, device)

    print("\nTraining complete.")
