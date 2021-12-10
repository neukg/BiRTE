from transformers.modeling_bert import BertModel,BertPreTrainedModel
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

class Biaffine(nn.Module):
    '''
        Args:
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: ``[True, True]``
        bias[0, 1]: the bias of U_m
        bias[2]: the b_m
    '''

    def __init__(self, in1_features, in2_features, out_features, bias=(True, True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))  # 3-dim -> 2-dim
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)
        self.linear_1 = nn.Linear(in_features=2*self.in1_features+1,
                                out_features=self.out_features,
                                bias=False)
        self.linear_2 = nn.Linear(in_features=2*self.in1_features+1,
                                out_features=self.out_features,
                                bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        U = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        W1 = np.zeros((self.out_features, 1+2*self.in1_features), dtype=np.float32)
        W2 = np.zeros((self.out_features, 1+2*self.in1_features), dtype=np.float32)

        self.linear.weight.data.copy_(torch.from_numpy(U))
        self.linear_1.weight.data.copy_(torch.from_numpy(W1))
        self.linear_2.weight.data.copy_(torch.from_numpy(W2))

    def forward(self, input1, input2):
        input1=input1.unsqueeze(dim=1)
        input2=input2.unsqueeze(dim=1)
        input3=torch.cat([input1, input2],dim=-1)

        # batch_size, len1, dim1 = input1.size()
        # batch_size, len2, dim2 = input2.size()
        batch_size,_, dim1 = input1.size()
        batch_size,_, dim2 = input2.size()

        if self.bias[0]:
            ones = input1.data.new(batch_size, 1,1).zero_().fill_(1)
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, 1,1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1
        if self.bias[2]:
            ones = input3.data.new(batch_size, 1,1).zero_().fill_(1)
            input3 = torch.cat((input3, Variable(ones)), dim=2)

        affine = self.linear(input1)

        affine = affine.view(batch_size, self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, 1, 1, self.out_features)

        # affine_1 = self.linear_1(input3)
        # affine_1 = affine_1.view(batch_size, 1, 1, self.out_features)
        # biaffine = biaffine + affine_1

        return biaffine.squeeze(dim=1).squeeze(dim=1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'in1_features=' + str(self.in1_features) \
               + ', in2_features=' + str(self.in2_features) \
               + ', out_features=' + str(self.out_features) + ')'


class BiRTE(BertPreTrainedModel):
    def __init__(self, config):
        super(BiRTE, self).__init__(config)
        self.bert=BertModel(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.w1=nn.Linear(config.hidden_size,config.hidden_size)
        self.w2=nn.Linear(config.hidden_size,config.hidden_size)
        self.w3=nn.Linear(config.hidden_size,config.hidden_size)

        #s
        self.s_classier=nn.Linear(config.hidden_size,2)
        self.s_classier_from_o=nn.Linear(config.hidden_size,2)

        #o
        self.o_classier=nn.Linear(config.hidden_size,2)
        self.o_classier_from_s=nn.Linear(config.hidden_size,2)

        #p
        self.biaffine=Biaffine(config.hidden_size,config.hidden_size,config.num_p)

        self.sigmoid=nn.Sigmoid()
        self.init_weights()

    def forward(self, token_ids, mask_token_ids,s2_mask,o2_mask,s3_mask,o3_mask):
        '''
        :param token_ids:
        :param token_type_ids:
        :param mask_token_ids:
        :param s_loc:
        :return: s_pred: [batch,seq,2]
        op_pred: [batch,seq,p,2]
        '''

        #获取表示
        head,tail,rel,cls=self.get_embed(token_ids, mask_token_ids)
        #初步预测s o
        s1_pred=self.s_pred(head,cls=cls)
        o1_pred=self.o_pred(tail,cls=cls)

        #进一步预测 s,o
        o2_pred=self.o_pred_from_s(head,tail,s2_mask,cls)
        s2_pred=self.s_pred_from_o(head,tail,o2_mask,cls)

        #预测r
        p_pred=self.p_pred(rel,s3_mask,o3_mask)

        return s1_pred,o1_pred,s2_pred,o2_pred,p_pred

    def get_embed(self,token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed=bert_out[0]
        head=self.w1(embed)
        tail=self.w2(embed)
        rel=self.w3(embed)
        cls=bert_out[1]
        head=head+tail[:,0,:].unsqueeze(dim=1)
        tail=tail+head[:,0,:].unsqueeze(dim=1)

        head, tail,rel,cls=self.dropout(head),self.dropout(tail),self.dropout(rel),self.dropout(cls)
        return head, tail,rel,cls

    def extract_entity(self, input, mask):
        '''
        取首尾平均
        :param input:BLH
        :param mask:BL
        :return: BH
        '''
        _,_,dim=input.shape
        entity=input*mask.unsqueeze(dim=-1) #BLH
        entity=entity.sum(dim=1)/mask.sum(dim=-1,keepdim=True) #BH/B1
        return entity

    def s_pred(self,head,cls):
        s_logist=self.s_classier(head+cls.unsqueeze(dim=1)) #BL,2
        s_pred=self.sigmoid(s_logist)
        return s_pred

    def o_pred(self,tail,cls):
        o_logist=self.o_classier(tail+cls.unsqueeze(dim=1)) #BL,2
        o_pred=self.sigmoid(o_logist)
        return o_pred

    def o_pred_from_s(self,head,tail,s_mask,cls):
        s_entity=self.extract_entity(head,s_mask)
        s2o_embed=tail*s_entity.unsqueeze(dim=1) #BLH
        o_logist=self.o_classier_from_s(s2o_embed+cls.unsqueeze(dim=1)) #BL2
        o_pred=self.sigmoid(o_logist)
        return o_pred #BL2

    def s_pred_from_o(self,head,tail,o_mask,cls):
        o_entity=self.extract_entity(tail,o_mask)
        o2s_embed=head*o_entity.unsqueeze(dim=1) #BLH
        s_logist=self.s_classier_from_o(o2s_embed+cls.unsqueeze(dim=1)) #BL2
        s_pred=self.sigmoid(s_logist)
        return s_pred #BL2

    def p_pred(self, rel, s_mask, o_mask):
        s_entity=self.extract_entity(rel,s_mask) #BH
        o_entity=self.extract_entity(rel,o_mask) #BH
        logist=self.biaffine(s_entity,o_entity) #bc
        r_pred=self.sigmoid(logist)
        return r_pred #BR