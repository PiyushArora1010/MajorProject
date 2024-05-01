import torch
import torch.nn as nn
from entmax import sparsemax


_EPSILON = 1e-6

def _vector_norms(v:torch.Tensor)->torch.Tensor:

    squared_norms = torch.sum(v * v, dim=1, keepdim=True)
    return torch.sqrt(squared_norms + _EPSILON)

def _distance(x:torch.Tensor , y:torch.Tensor, type:str='cosine')->torch.Tensor:

        if type == 'cosine':
            x_norm = x / _vector_norms(x)
            y_norm = y / _vector_norms(y)
            d = 1 - torch.mm(x_norm,y_norm.transpose(0,1))
        elif type == 'l2':
            d = (
                x.unsqueeze(1).expand(x.shape[0], y.shape[0], -1) -
                y.unsqueeze(0).expand(x.shape[0], y.shape[0], -1)
        ).pow(2).sum(dim=2)
        elif type == 'dot':
            expanded_x = x.unsqueeze(1).expand(x.shape[0], y.shape[0], -1)
            expanded_y = y.unsqueeze(0).expand(x.shape[0], y.shape[0], -1)
            d = -(expanded_x * expanded_y).sum(dim=2)
        else:
            raise NameError('{} not recognized as valid distance. Acceptable values are:[\'cosine\',\'l2\',\'dot\']'.format(type))
        return d

class MLP(nn.Module):
    '''
    Multi-layer perceptron class
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, output_size)
    
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

class multiHeadAttention(nn.Module):
    def __init__(self, encoder_output_dim:int, heads:int=1, sparse:bool=True):

        super(multiHeadAttention, self).__init__()

        self.encoder_output_dim = encoder_output_dim
        self.heads = heads
        self.sparse = sparse

        self.Wq = nn.Linear(encoder_output_dim, encoder_output_dim)
        self.Wk = nn.Linear(encoder_output_dim, encoder_output_dim)
        self.Wv = nn.Linear(encoder_output_dim, encoder_output_dim)

        self.fc = nn.Linear(encoder_output_dim, encoder_output_dim)

    def forward(self, encoder_output:torch.Tensor, memory_set:torch.Tensor)->torch.Tensor:
        #encoder_output: (batch_size, 1, encoder_output_dim)
        #memory_set: (1, memory_set_size, encoder_output_dim)

        q = self.Wq(encoder_output)
        k = self.Wk(memory_set)
        v = self.Wv(memory_set)

        q = q.view(-1, self.heads, self.encoder_output_dim//self.heads)
        k = k.view(-1, self.heads, self.encoder_output_dim//self.heads)
        v = v.view(-1, self.heads, self.encoder_output_dim//self.heads)

        q = q.transpose(0,1)
        k = k.transpose(0,1)
        v = v.transpose(0,1)

        attention_weights = torch.matmul(q,k.transpose(-2,-1))
        attention_weights = attention_weights / (self.encoder_output_dim ** 0.5)

        if self.sparse:
            attention_weights = sparsemax(attention_weights,dim=-1)
        else:
            attention_weights = torch.softmax(attention_weights,dim=-1)
            
        output = torch.matmul(attention_weights,v)
        output = output.transpose(0,1).contiguous().view(-1,self.encoder_output_dim)

        output = self.fc(output)

        return output, attention_weights

class CosineSimLayer(nn.Module):

    def __init__(self, encoder_output_dim:int, output_dim:int):

        super(CosineSimLayer, self).__init__()

        self.distance_name = 'cosine'

        self.classifier = MLP(encoder_output_dim*2, encoder_output_dim*4, output_dim)
    

    def forward(self, encoder_output:torch.Tensor, memory_set:torch.Tensor, return_weights:bool=False)->torch.Tensor:

        dist = _distance(encoder_output,memory_set,self.distance_name)
        content_weights = sparsemax(-dist,dim=1)

        memory_vector = torch.matmul(content_weights,memory_set)

        final_input = torch.cat([encoder_output,memory_vector],1)
        output = self.classifier(final_input)

        if return_weights:
            return output, content_weights
        else: 
            return output

class AttentionSparseMax(nn.Module):
    def __init__(self, encoder_output_dim:int, output_dim:int, heads:int=1, sparse=True):

        super(AttentionSparseMax, self).__init__()

        self.classifier = MLP(encoder_output_dim*2, encoder_output_dim*4, output_dim)
        self.attention = multiHeadAttention(encoder_output_dim, heads, sparse)
        self.sparse = sparse


    def forward(self, encoder_output:torch.Tensor, memory_set:torch.Tensor, return_weights:bool=False)->torch.Tensor:

        encoder_output = encoder_output.unsqueeze(1)
        memory_set = memory_set.unsqueeze(0)

        output, attention_weights = self.attention(encoder_output, memory_set)

        final_input = torch.cat([encoder_output.squeeze(1),output],1)
        output = self.classifier(final_input)

        if return_weights:
            return output, attention_weights
        else:
            return output
