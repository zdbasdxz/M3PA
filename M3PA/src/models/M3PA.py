
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss












class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class GeoAttention(nn.Module):

    def __init__(self,
                 num_nearest,
                 shape_input_phenomenon,
                 label=None,
                 graph_label=None,
                 suffix_mean=None,
                 **kwargs):
        super().__init__()
        self.label = label
        self.num_nearest = num_nearest,  # 最近邻居个数
        self.shape_input_phenomenon = shape_input_phenomenon,  # 输入向量维度

        self.graph_label = graph_label,  # softmax层维度
        self.suffix_mean = suffix_mean

        self.kernel = nn.Parameter(torch.randn(self.num_nearest[0], self.num_nearest[0]))

        self.bias = nn.Parameter(torch.randn(self.num_nearest[0]))

    # 这是定义层功能的方法
    def forward(self, source_distance, context):

        source_distance.to(torch.float32)
        context.to(torch.float32)
        ######################## Attention data ########################
        self.distance = source_distance

        self.simi = Lambda(lambda x: torch.exp(-x * (10 ** 2 / 2)))(self.distance)

        # calculates the weights associated with each neighbor (m, seq)
        self.simi = self.simi.to(torch.float32)

        self.weight = torch.mm(self.simi, self.kernel) + self.bias

        self.weight = torch.softmax(self.weight, dim=1)

        repeattimes = self.shape_input_phenomenon[0]
        prob_repeat = torch.repeat_interleave(self.weight, repeattimes, dim=1).reshape(self.weight.shape[0],
                                                                                       self.weight.shape[1],
                                                                                       repeattimes)
        relevance = torch.multiply(prob_repeat, context)

        self.mean = torch.sum(relevance, axis=1)

        return self.mean

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.l1 = torch.nn.Linear(in_size, hidden_size, bias=True)
        self.ac = nn.Tanh()
        self.l2 = torch.nn.Linear(int(hidden_size), 1, bias=False)

    def forward(self, z):
        w = self.l1(z)
        w = self.ac(w)
        w = self.l2(w)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)
class linearRegression_multi_att(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression_multi_att, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, inputSize, bias=True)
        self.act1 = nn.ReLU()
        self.linear2 = torch.nn.Linear(inputSize, outputSize, bias=True)

        self.attention = Attention(in_size=inputSize)


    def forward(self, x1, x2):  #
        Features = torch.stack([x1, x2], dim=1)

        Features = self.attention(Features)

        out = self.linear1(Features)
        out = self.act1(out)
        out = self.linear2(out)

        return out




class M3PA(nn.Module):
    def __init__(self, geobert, bert_input_dim, street_input_dim, hidden_dim, num_nearest, shape_input_phenomenon, num_semantic_types):
        super().__init__()

        self.geobert =geobert
        self.attention = GeoAttention(num_nearest=num_nearest, shape_input_phenomenon=shape_input_phenomenon)
        self.projector1 = nn.Sequential(
            nn.Linear(bert_input_dim, bert_input_dim),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(bert_input_dim, hidden_dim)
        )
        self.projector2 = nn.Sequential(
            nn.Linear(street_input_dim, street_input_dim),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(street_input_dim, hidden_dim)
        )



        self.change_shape = nn.Linear(shape_input_phenomenon, street_input_dim)

        self.fusion_layer = linearRegression_multi_att(hidden_dim, hidden_dim)
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_semantic_types)
        )

        self.num_semantic_types = num_semantic_types
        self.shape_input_phenomenon = shape_input_phenomenon
        self.street_input_dim = street_input_dim

    def forward(self, input_ids, attention_mask, sent_position_ids, sent_coordinate_list, poi_len_list, labels, poi_street_embedding, near_feature, n_dist):

        bert_embedding = self.geobert(input_ids, attention_mask=attention_mask, sent_position_ids=sent_position_ids,
                                   sent_coordinate_list=sent_coordinate_list,
                                   poi_len_list=poi_len_list).hidden_states

        neiber_embedding = self.attention(n_dist, near_feature)
        if self.street_input_dim != self.shape_input_phenomenon:
            neiber_embedding  =self.change_shape(neiber_embedding)

        batch_vector = neiber_embedding + poi_street_embedding
        hidden_bert_embedding = self.projector1(bert_embedding)
        hidden_street_embedding = self.projector2(batch_vector)

        fusion_embedding = self.fusion_layer(hidden_bert_embedding, hidden_street_embedding)

        type_prediction_score = self.cls(fusion_embedding)

        loss_fct = CrossEntropyLoss()
        typing_loss = loss_fct(type_prediction_score.view(-1, self.num_semantic_types), labels.view(-1))

        return typing_loss, type_prediction_score