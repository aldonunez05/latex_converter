import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.constants import HIDDEN_SIZE, IMAGE_CHANNELS, SOS_TOKEN, MAX_SEQUENCE_LENGTH
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from constants import HIDDEN_SIZE, IMAGE_CHANNELS, SOS_TOKEN, MAX_SEQUENCE_LENGTH


class ImageEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(ImageEncoder, self).__init__()
        # 3 convolutional blocks to extract features (C,H,W) -> (hidden_size, H/4, W/4)
        self.conv1 = nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size =3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d(64, hidden_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        B, C, H, W = x.size()


        #reshape to (batch, seq len, hidden size) where seq len = H*W (flattened spatial dims)
        feature_sequence = x.view(B,C, H*W).permute(0,2,1)
        return feature_sequence

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

        attn_weights = F.softmax(scores, dim=1).unsqueeze(1)

        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)

        return context, attn_weights

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

        output = self.out(torch.cat((output.squeeze(1), context), dim=1))

        return output, hidden, attn_weights


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

        decoder_hidden = (
            torch.zeros(1, batch_size, self.hidden_size).to(device),
            torch.zeros(1, batch_size, self.hidden_size).to(device)
        )

        decoder_input = torch.tensor([SOS_TOKEN] * batch_size, device=device)

        all_predicitions = []

        for t in range(MAX_SEQUENCE_LENGTH):
            output, decoder_hidden, attn_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
            )
            
            all_predicitions.append(output)

            top_value, top_index = output.topk(1, dim=1)

            use_teacher_forcing = target_sequence is not None and torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing and t < target_sequence.size(1):
                decoder_input = target_sequence[:, t]
            else:
                decoder_input = top_index.squeeze(1).detach()


        return torch.stack(all_predictions, dim=0).permute(1, 0, 2)
