from __future__ import division

import torch
import torch.nn as nn

from binder_operations import CircularConvolution, EltWise, SumFlattenedOuterProduct
from FillerLSTM import FillerLSTM

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# A tensor product encoder layer
# Takes a list of fillers and a list of roles and returns an encoding
class FillerTPE(nn.Module):
    def __init__(
            self,
            Symbol_embedding,
            role_embedding,
            free_dim,
            final_layer_width=None,
            n_roles=3,
            n_fillers=2,filler_dim=3,
            Symbol_learning=False,
            binder="tpr",
            hidden_dim=20,
            bidirectional=True,
            num_layers=1,
            softmax_fillers=True,
            one_hot_regularization_weight=1.0,
            l2_norm_regularization_weight=1.0,
            unique_role_regularization_weight=1.0,
    ):

        super(FillerTPE, self).__init__()
        if not isinstance(Symbol_embedding,nn.Embedding):
            Symbol_embedding=nn.Embedding(Symbol_embedding.shape[0],Symbol_embedding.shape[1],_weight=Symbol_embedding)
        self.Symbol_embedding=Symbol_embedding
        self.Symbol_embedding.requires_grad_(Symbol_learning)

        if not isinstance(role_embedding,nn.Embedding):
            role_embedding=nn.Embedding(role_embedding.shape[0],role_embedding.shape[1],_weight=role_embedding)
        self.role_embedding=role_embedding
        self.role_embedding.requires_grad_(False)
        self.role_dim=self.role_embedding.weight.shape[1]

        self.free_dim=free_dim
        self.n_roles = n_roles  # number of roles
        self.n_fillers = n_fillers  # number of fillers
        self.filler_dim=filler_dim  # Set the dimension for the filler embeddings

        self.one_hot_regularization_weight = one_hot_regularization_weight
        self.l2_norm_regularization_weight = l2_norm_regularization_weight
        self.unique_role_regularization_weight = unique_role_regularization_weight
        self.regularize = False

        self.filler_assigner = FillerLSTM(
            self.n_fillers,self.filler_dim,
            self.Symbol_embedding,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            softmax_fillers=softmax_fillers,
            Symbol_learning=Symbol_learning
        )

        # Create a SumFlattenedOuterProduct layer that will
        # take the sum flattened outer product of the filler
        # and role embeddings (or a different type of role-filler
        # binding function, such as circular convolution)
        if binder == "tpr":
            self.sum_layer = SumFlattenedOuterProduct()
        elif binder == "hrr":
            self.sum_layer = CircularConvolution(self.filler_dim)
        elif binder == "eltwise" or binder == "elt":
            self.sum_layer = EltWise()
        else:
            print("Invalid binder")

        # This final part if for including a final linear layer that compresses
        # the sum flattened outer product into the dimensionality you desire
        # But if self.final_layer_width is None, then no such layer is used
        self.final_layer_width = final_layer_width
        if self.final_layer_width is None:
            self.has_last = 0
        else:
            self.has_last = 1
            if binder == "tpr":
                self.last_layer = nn.Linear(self.filler_dim * self.role_dim+self.free_dim, self.final_layer_width)
            else:
                self.last_layer = nn.Linear(self.filler_dim+self.free_dim, self.final_layer_width)

    # Function for a forward pass through this layer. Takes a list of fillers and
    # a list of roles and returns an single vector encoding it.
    def forward(self, Seq_list):
        # fillers_embedded is size (sequence_length, batch_size, filler_embedding_dim)
        # filler_predictions is size (sequence_length, batch_size, num_fillers)
        fillers_embedded, filler_predictions = self.filler_assigner(Seq_list)
        fillers_embedded = fillers_embedded.transpose(0, 1)
        # 在任务中我们用到的role就是1，2，3，所以嵌入role是三维I矩阵
        role_list=torch.zeros_like(Seq_list)
        for i in range(Seq_list.shape[1]):
            role_list[:,i]=i
        roles_embedded=self.role_embedding(role_list)

        # Create the sum of the flattened tensor products of the
        # filler and role embeddings
        output = self.sum_layer(fillers_embedded, roles_embedded)
        state=torch.randn(output.shape[0],output.shape[1],self.free_dim)
        out_state=torch.cat((output,state),dim=2)

        # If there is a final linear layer to change the output's dimensionality, apply it
        if self.has_last:
            out_state = self.last_layer(out_state)

        return out_state, filler_predictions

    def use_regularization(self, use_regularization):
        self.regularize = use_regularization

    def set_regularization_temp(self, temp):
        self.regularization_temp = temp

    def get_regularization_loss(self, filler_predictions):
        if not self.regularize:
            return 0, 0, 0

        one_hot_temperature = self.regularization_temp
        batch_size = filler_predictions.shape[1]

        softmax_fillers = self.filler_assigner.softmax_fillers

        if softmax_fillers:
            # For RoleLearningTensorProductEncoder, we encourage one hot vector weight predictions
            # by regularizing the role_predictions by `w * (1 - w)`
            one_hot_reg = torch.sum(filler_predictions * (1 - filler_predictions))
        else:
            one_hot_reg = torch.sum((filler_predictions ** 2) * (1 - filler_predictions) ** 2)
        one_hot_loss = one_hot_temperature * one_hot_reg / batch_size

        if softmax_fillers:
            l2_norm = -torch.sum(filler_predictions * filler_predictions)
        else:
            l2_norm = (torch.sum(filler_predictions ** 2) - 1) ** 2
        l2_norm_loss = one_hot_temperature * l2_norm / batch_size

        # We also want to encourage the network to assign each filler in a sequence to a
        # different role. To encourage this, we sum the vector predictions across a sequence
        # (call this vector w) and add `(w * (1 - w))^2` to the loss function.
        exclusive_filler_vector = torch.sum(filler_predictions, 0)
        unique_filler_loss = one_hot_temperature * torch.sum(
            (exclusive_filler_vector * (1 - exclusive_filler_vector)) ** 2) / batch_size
        return self.one_hot_regularization_weight * one_hot_loss,\
               self.l2_norm_regularization_weight * l2_norm_loss,\
               self.unique_role_regularization_weight * unique_filler_loss

    def train(self):
        self.filler_assigner.snap_one_hot_predictions = False

    def eval(self):
        self.filler_assigner.snap_one_hot_predictions = True
