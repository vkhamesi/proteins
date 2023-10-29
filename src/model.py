import torch
import torch.nn as nn
from transformers import AutoModel

class ProteinPredictor(nn.Module):

    def __init__(self):
        super(ProteinPredictor, self).__init__()

        # Load the pre-trained models
        self.protein_model = AutoModel.from_pretrained("yarongef/DistilProtBert")
        self.go_model = AutoModel.from_pretrained("nlpie/distil-biobert")

        # Define linear layers for mapping embeddings to hidden dimensions
        self.protein_embedding_dim = self.protein_model.config.hidden_size
        self.go_embedding_dim = self.go_model.config.hidden_size

        # Reshape
        self.protein_reshape = nn.Linear(self.protein_embedding_dim, 256)
        self.go_reshape = nn.Linear(self.go_embedding_dim, 256)

        # Final MLP prediction
        self.mlp = nn.Sequential(
            nn.Linear(3 * 256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, protein_input_ids, protein_attention_mask, go_input_ids, go_attention_mask):

        # Generate protein embeddings
        protein_outputs = self.protein_model(input_ids=protein_input_ids, attention_mask=protein_attention_mask)
        protein_embeddings = protein_outputs.last_hidden_state.mean(dim=1)  # Average pooling over sequence length

        # Generate function embeddings
        go_outputs = self.go_model(input_ids=go_input_ids, attention_mask=go_attention_mask)
        go_embeddings = go_outputs.last_hidden_state.mean(dim=1)  # Average pooling over sequence length

        # Reshape
        protein_embeddings_reshaped = torch.tanh(self.protein_reshape(protein_embeddings))
        go_embeddings_reshaped = torch.tanh(self.go_reshape(go_embeddings))

        # Concatenate embeddings
        combined_embeddings = torch.cat((protein_embeddings_reshaped,
                                         go_embeddings_reshaped,
                                         torch.absolute(protein_embeddings_reshaped - go_embeddings_reshaped)), dim=1)

        output = self.mlp(combined_embeddings)

        return output