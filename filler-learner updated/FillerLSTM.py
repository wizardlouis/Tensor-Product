import torch
import torch.nn as nn

class FillerLSTM(nn.Module):
    def __init__(self,
                 num_fillers,filler_embedding_dim,
                 Symbol_embedding,
                 hidden_dim,num_layers=1,bidirectional=True,softmax_fillers=True,Symbol_learning=False):
        super(FillerLSTM,self).__init__()
        self.num_fillers=num_fillers
        self.filler_embedding_dim=filler_embedding_dim
        if not isinstance(Symbol_embedding,nn.Embedding):
            Symbol_embedding=nn.Embedding(Symbol_embedding.shape[0],Symbol_embedding.shape[1],_weight=Symbol_embedding)
        self.Symbol_embedding=Symbol_embedding
        self.Symbol_learning=Symbol_learning

        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.bidirectional=bidirectional

        #decide if we should learn the embedding of symbols
        self.Symbol_embedding.requires_grad_(Symbol_learning)
        self.lstm=nn.LSTM(self.Symbol_embedding.embedding_dim,self.hidden_dim,bidirectional=self.bidirectional,num_layers=self.num_layers)

        #different weighting function input dim of bidirectional
        if bidirectional:
            print("The fillerLSTM is bidirectional")
            self.filler_weight_predictions = nn.Linear(hidden_dim * 2, num_fillers)
        else:
            self.filler_weight_predictions = nn.Linear(hidden_dim, num_fillers)

        self.softmax_fillers=softmax_fillers
        if softmax_fillers:
            print("Use softmax for role predictions")
            # The output of filler_weight_predictions is shape
            # (sequence_length, batch_size, num_roles)
            # We want to softmax across the roles so set dim=2
            self.softmax = nn.Softmax(dim=2)

        self.filler_embedding = nn.Embedding(num_fillers, filler_embedding_dim)
        self.filler_indices= torch.tensor([x for x in range(num_fillers)])

        self.snap_one_hot_predictions=False

    def forward(self,Seq_tensor):
        """
        :param Seq_tensor: This input tensor should be of shape (batch_size, sequence_length)
        :param Seq_lengths: A list of the length of each sequence in the batch. This is used
            for padding the sequences.
        :return: A tensor of size (sequence_length, batch_size, filler_embedding_dim) with the filler
            embeddings for the input filler_tensor.
        """
        batch_size = len(Seq_tensor)
        hidden = self.init_hidden(batch_size)

        Symbols_embedded = self.Symbol_embedding(Seq_tensor)
        # The shape of fillers_embedded should be
        # (batch_size, sequence_length, filler_embedding_dim)
        # Pytorch LSTM expects data in the shape (sequence_length, batch_size, feature_dim)
        Symbols_embedded = torch.transpose(Symbols_embedded, 0, 1)


        lstm_out, hidden = self.lstm(Symbols_embedded, hidden)
        filler_predictions = self.filler_weight_predictions(lstm_out)

        if self.softmax_fillers:
            filler_predictions = self.softmax(filler_predictions)
        # filler_predictions is size (sequence_length, batch_size, num_fillers)

            # Normalize the embeddings. This is important so that role attention is not overruled by
            # embeddings with different orders of magnitude.
        filler_embeddings = self.filler_embedding(self.filler_indices)
            # filler_embeddings is size (num_fillers, filler_embedding_dim)
        filler_embeddings = filler_embeddings / torch.norm(filler_embeddings, dim=1).unsqueeze(1)


        # During evaluation, we want to snap the role predictions to a one-hot vector
        if self.snap_one_hot_predictions:
            one_hot_predictions = self.one_hot_embedding(torch.argmax(filler_predictions, 2),
                                                        self.num_fillers)
            fillers = torch.matmul(one_hot_predictions, filler_embeddings)
        else:
            fillers = torch.matmul(filler_predictions, filler_embeddings)
        # fillers is size (sequence_length, batch_size, filler_embedding_dim)

        return fillers, filler_predictions

    def init_hidden(self, batch_size):
        layer_multiplier = 1
        if self.bidirectional:
            layer_multiplier = 2

        # The axes semantics are (num_layers, batch_size, hidden_dim)
        # We need a tuple for the hidden state and the cell state of the LSTM.
        return (torch.zeros(self.num_layers * layer_multiplier, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * layer_multiplier, batch_size, self.hidden_dim,))

    def one_hot_embedding(self, labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        y = torch.eye(num_classes)
        return y[labels]

if __name__=='__main__':
    Assigner=torch.load('TPE//')