from transformers import WEIGHTS_NAME,AdamW, get_linear_schedule_with_warmup
from bert4keras.tokenizers import Tokenizer
from model import BiRTE
from util import *
from tqdm import tqdm
import random
import os
import torch.nn as nn
import torch
from transformers.modeling_bert import BertConfig
import json

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def judge(ex):
    '''判断样本是否正确'''
    for s,p,o in ex["triple_list"]:
        if s=='' or o=='' or s not in ex["text"] or o not in ex["text"]:
            return False
    return True

class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, args, train_data, tokenizer, predicate2id, id2predicate):
        super(data_generator, self).__init__(train_data, args.batch_size)
        self.max_len=args.max_len
        self.tokenizer=tokenizer
        self.predicate2id=predicate2id
        self.id2predicate=id2predicate

    def __iter__(self, is_random=True): 
        batch_token_ids, batch_mask = [], []
        batch_s1_labels, batch_o1_labels,\
        batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels,\
        batch_s3_mask, batch_o3_mask, batch_r = [],[],[],[],[],[],[],[],[]

        for is_end, d in self.sample(is_random):
            if judge(d)==False:
                continue
            token_ids, _ ,mask = self.tokenizer.encode(
                d['text'], max_length=self.max_len
            )
            # 整理三元组 {s: [(o, p)]}
            spoes_s = {}
            spoes_o = {}
            for s, p, o in d['triple_list']:
                s = self.tokenizer.encode(s)[0][1:-1]
                p = self.predicate2id[p]
                o = self.tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s_loc = (s_idx, s_idx + len(s) - 1)
                    o_loc = (o_idx, o_idx + len(o) - 1)
                    if s_loc not in spoes_s:
                        spoes_s[s_loc] = []
                    spoes_s[s_loc].append((o_loc,p))
                    if o_loc not in spoes_o:
                        spoes_o[o_loc] = []
                    spoes_o[o_loc].append((s_loc,p))
            if spoes_s and spoes_o:
                # s1_labels o1_labels
                def get_entity1_labels(item,l):
                    res=np.zeros([l,2])
                    for start,end in item:
                        res[start][0]=1
                        res[end][1]=1
                    return res
                s1_labels = get_entity1_labels(spoes_s, len(token_ids))
                o1_labels = get_entity1_labels(spoes_o, len(token_ids))

                # s2_labels,o2_labels,s2_mask,o2_mask
                def get_entity2_labels_mask(item,l):
                    start, end = random.choice(list(item.keys()))
                    #构造labels
                    labels = np.zeros((l, 2))
                    if (start,end) in item:
                        for loc,_ in item[(start,end)]:
                            labels[loc[0], 0] = 1
                            labels[loc[1], 1] = 1
                    #构造mask
                    mask=np.zeros(l)
                    mask[start]=1
                    mask[end]=1
                    return labels,mask
                o2_labels,s2_mask=get_entity2_labels_mask(spoes_s,len(token_ids))
                s2_labels,o2_mask=get_entity2_labels_mask(spoes_o,len(token_ids))

                #s3_mask,o3_mask,r
                s_loc=random.choice(list(spoes_s.keys()))
                o_loc,_=random.choice(spoes_s[s_loc])
                r=np.zeros(len(self.id2predicate))
                if s_loc in spoes_s:
                    for loc,the_r in spoes_s[s_loc]:
                        if loc==o_loc:
                            r[the_r]=1
                s3_mask=np.zeros(len(token_ids))
                o3_mask=np.zeros(len(token_ids))
                s3_mask[s_loc[0]]=1
                s3_mask[s_loc[1]]=1
                o3_mask[o_loc[0]]=1
                o3_mask[o_loc[1]]=1

                # 构建batch
                batch_token_ids.append(token_ids)
                batch_mask.append(mask)

                batch_s1_labels.append(s1_labels)
                batch_o1_labels.append(o1_labels)

                batch_s2_mask.append(s2_mask)
                batch_o2_mask.append(o2_mask)
                batch_s2_labels.append(s2_labels)
                batch_o2_labels.append(o2_labels)

                batch_s3_mask.append(s3_mask)
                batch_o3_mask.append(o3_mask)
                batch_r.append(r)

                if len(batch_token_ids) == self.batch_size or is_end:   #输出batch
                    batch_token_ids,batch_mask,\
                    batch_s1_labels,batch_o1_labels,\
                    batch_s2_mask,batch_o2_mask,batch_s2_labels,batch_o2_labels,\
                    batch_s3_mask,batch_o3_mask=\
                        [sequence_padding(i).astype(np.int)
                         for i in [batch_token_ids,batch_mask,
                                   batch_s1_labels,batch_o1_labels,
                                   batch_s2_mask,batch_o2_mask,batch_s2_labels,batch_o2_labels,
                                   batch_s3_mask,batch_o3_mask]]

                    batch_r = np.array(batch_r).astype(np.int)

                    yield [
                        batch_token_ids, batch_mask,
                        batch_s1_labels, batch_o1_labels,
                        batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels,
                        batch_s3_mask, batch_o3_mask,batch_r
                    ]
                    batch_token_ids, batch_mask = [], []
                    batch_s1_labels, batch_o1_labels, \
                    batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels, \
                    batch_s3_mask, batch_o3_mask, batch_r = [], [], [], [], [], [], [], [], []


class CE():
    def __call__(self,args,targets, pred, from_logist=False):
        '''
        计算二分类交叉熵
        :param targets: [batch,seq,2]
        :param pred: [batch,seq,2]
        :param from_logist:是否没有经过softmax/sigmoid
        :return: loss.shape==targets.shape==pred.shape
        '''
        if not from_logist:
            '''返回到没有经过softmax/sigmoid得张量'''
            # 截取pred，防止趋近于0或1,保持在[min_num,1-min_num]
            pred = torch.where(pred < 1 - args.min_num, pred, torch.ones(pred.shape).to("cuda") * 1 - args.min_num).to("cuda")
            pred = torch.where(pred > args.min_num, pred, torch.ones(pred.shape).to("cuda") * args.min_num).to("cuda")
            pred = torch.log(pred / (1 - pred))
        relu = nn.ReLU()
        # 计算传统的交叉熵loss
        loss = relu(pred) - pred * targets + torch.log(1 + torch.exp(-1 * torch.abs(pred).to("cuda"))).to("cuda")
        return loss

def train(args):
    output_path = os.path.join(args.base_path, args.dataset, "output", args.file_id)
    train_path=os.path.join(args.base_path,args.dataset,"train.json")
    dev_path=os.path.join(args.base_path,args.dataset,"dev.json")
    test_path=os.path.join(args.base_path,args.dataset,"test.json")
    rel2id_path=os.path.join(args.base_path,args.dataset,"rel2id.json")
    test_pred_path=os.path.join(output_path,"test_pred.json")
    dev_pred_path=os.path.join(output_path,"dev_pred.json")
    log_path=os.path.join(output_path,"log.txt")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print_config(args)
    # 加载数据集
    train_data = json.load(open(train_path))
    valid_data = json.load(open(dev_path))
    test_data = json.load(open(test_path))
    id2predicate, predicate2id = json.load(open(rel2id_path))

    tokenizer = Tokenizer(args.bert_vocab_path)  # 注意修改
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p=len(id2predicate)
    torch.cuda.set_device(int(args.cuda_id))
    train_model = BiRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
    train_model.to("cuda")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataloader = data_generator(args, train_data, tokenizer, predicate2id, id2predicate)

    t_total = len(dataloader) * args.num_train_epochs

    """ 优化器准备 """
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_model.named_parameters() if "bert." in n],
            "weight_decay": args.weight_decay,
            "lr": args.bert_learning_rate,
        },
        {
            "params": [p for n, p in train_model.named_parameters() if "bert." not in n],
            "weight_decay": args.weight_decay,
            "lr": args.other_learning_rate,
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=args.min_num)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )
    best_f1 = -1.0  # 全局的best_f1
    step = 0
    binary_crossentropy=CE()
    no_change=0
    for epoch in range(args.num_train_epochs):
        train_model.train()
        epoch_loss = 0
        with tqdm(total=dataloader.__len__(), desc="train", ncols=80) as t:
            for i, batch in enumerate(dataloader):
                batch = [torch.tensor(d).to("cuda") for d in batch]
                batch_token_ids, batch_mask,\
                batch_s1_labels, batch_o1_labels,\
                batch_s2_mask, batch_o2_mask, batch_s2_labels, batch_o2_labels,\
                batch_s3_mask, batch_o3_mask, batch_r = batch

                s1_pred,o1_pred,s2_pred,o2_pred,p_pred = train_model(batch_token_ids, batch_mask,
                                                                     batch_s2_mask, batch_o2_mask,
                                                                     batch_s3_mask, batch_o3_mask)

                #计算损失
                def get_loss(target,pred,mask):
                    loss = binary_crossentropy(args, targets=target, pred=pred)  # BL2
                    loss = torch.mean(loss, dim=2).to("cuda")  # BL
                    loss = torch.sum(loss * mask).to("cuda") / torch.sum(mask).to("cuda")
                    return loss
                s1_loss=get_loss(target=batch_s1_labels,pred=s1_pred,mask=batch_mask)
                o1_loss=get_loss(target=batch_o1_labels,pred=o1_pred,mask=batch_mask)
                s2_loss=get_loss(target=batch_s2_labels,pred=s2_pred,mask=batch_mask)
                o2_loss=get_loss(target=batch_o2_labels,pred=o2_pred,mask=batch_mask)
                r_loss=binary_crossentropy(args,targets=batch_r,pred=p_pred)
                r_loss=r_loss.mean()

                loss=s1_loss+o1_loss+s2_loss+o2_loss+r_loss

                loss.backward()
                step += 1
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                train_model.zero_grad()
                t.set_postfix(loss="%.4lf"%(loss.cpu().item()))
                t.update(1)
        f1, precision, recall = evaluate(args,tokenizer,id2predicate,train_model,valid_data,dev_pred_path)

        if f1 > best_f1:
            # Save model checkpoint
            best_f1 = f1
            torch.save(train_model.state_dict(), os.path.join(output_path, WEIGHTS_NAME))  # 保存最优模型权重

        epoch_loss = epoch_loss / dataloader.__len__()
        with open(log_path, "a", encoding="utf-8") as f:
            print("epoch:%d\tloss:%f\tf1:%f\tprecision:%f\trecall:%f\tbest_f1:%f" % (
                int(epoch), epoch_loss, f1, precision, recall, best_f1), file=f)

    #对test集合进行预测
    #加载训练好的权重
    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    f1, precision, recall = evaluate(args,tokenizer,id2predicate,train_model, test_data, test_pred_path)
    with open(log_path, "a", encoding="utf-8") as f:
        print("test： f1:%f\tprecision:%f\trecall:%f" % (f1, precision, recall), file=f)

def extract_spoes(args,tokenizer,id2predicate,model,text,entity_start=0.5,entity_end=0.5,p_num=0.5):
    """抽取输入text所包含的三元组
    """
    #sigmoid=nn.Sigmoid()
    if isinstance(model,torch.nn.DataParallel):
        model=model.module
    model.to("cuda")
    tokens = tokenizer.tokenize(text, max_length=args.max_len)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, _ ,mask = tokenizer.encode(text, max_length=args.max_len)
    #获取BERT表示
    model.eval()
    with torch.no_grad():
        head,tail,rel,cls = model.get_embed(torch.tensor([token_ids]).to("cuda"), torch.tensor([mask]).to("cuda"))
        head = head.cpu().detach().numpy() #[1,L,H]
        tail = tail.cpu().detach().numpy()
        rel = rel.cpu().detach().numpy()
        cls = cls.cpu().detach().numpy()

    def get_entity(entity_pred):
        start = np.where(entity_pred[0, :, 0] > entity_start)[0]
        end = np.where(entity_pred[0, :, 1] > entity_end)[0]
        entity = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                entity.append((i, j))
        return entity

    #抽取s1 o1
    model.eval()
    with torch.no_grad():
        s1_preds = model.s_pred(torch.tensor(head).to("cuda"),torch.tensor(cls).to("cuda"))
        o1_preds = model.o_pred(torch.tensor(tail).to("cuda"),torch.tensor(cls).to("cuda"))

        s1_preds = s1_preds.cpu().detach().numpy() #[1,L,2]
        o1_preds = o1_preds.cpu().detach().numpy() #[1,L,2]

        s1_preds[:,0,:],s1_preds[:,-1,:]=0.0,0.0
        o1_preds[:,0,:],o1_preds[:,-1,:]=0.0,0.0

    s1=get_entity(s1_preds)
    o1=get_entity(o1_preds)

    #获得s_loc,o_loc
    pairs_0=[]
    for s in s1:
        for o in o1:
            pairs_0.append((s[0],s[1],o[0],o[1]))

    pairs_1=[]
    for s in s1:
        #s:(start,end)
        s2_mask=np.zeros(len(token_ids)).astype(np.int)
        s2_mask[s[0]] = 1
        s2_mask[s[1]] = 1

        model.eval()
        with torch.no_grad():
            o2_pred=model.o_pred_from_s(torch.tensor(head).to("cuda"),torch.tensor(tail).to("cuda"),
                                       torch.tensor([s2_mask]).to("cuda"),cls=torch.tensor(cls).to("cuda"))
            o2_pred = o2_pred.cpu().detach().numpy()  # [1,L,2]
            o2_pred[:, 0, :], o2_pred[:, -1, :] = 0.0, 0.0
        objects2 = get_entity(o2_pred)
        if objects2:
            for o in objects2:
                pairs_1.append((s[0],s[1],o[0],o[1]))

    pairs_2=[]
    for o in o1:
        #o:(start,end)
        o2_mask=np.zeros(len(token_ids)).astype(np.int)
        o2_mask[o[0]] = 1
        o2_mask[o[1]] = 1

        model.eval()
        with torch.no_grad():
            s2_pred=model.s_pred_from_o(torch.tensor(head).to("cuda"),torch.tensor(tail).to("cuda"),
                                       torch.tensor([o2_mask]).to("cuda"),cls=torch.tensor(cls).to("cuda"))
            s2_pred = s2_pred.cpu().detach().numpy()  # [1,L,2]
            s2_pred[:, 0, :], s2_pred[:, -1, :] = 0.0, 0.0
        subjects2 = get_entity(s2_pred)
        if subjects2:
            for s in subjects2:
                pairs_2.append((s[0],s[1],o[0],o[1]))

    pairs_1=set(pairs_1)
    pairs_2=set(pairs_2)

    pairs=list(pairs_1|pairs_2)


    if pairs: # m * 4
        s_mask=np.zeros([len(pairs),len(token_ids)]).astype(np.int)
        o_mask=np.zeros([len(pairs),len(token_ids)]).astype(np.int)

        for i,pair in enumerate(pairs):
            s1, s2, o1, o2=pair
            s_mask[i,s1]=1
            s_mask[i,s2]=1
            o_mask[i,o1]=1
            o_mask[i,o2]=1

        spoes = []
        rel=np.repeat(rel,len(pairs),0)

        # 传入subject，抽取object和predicate
        model.eval()
        with torch.no_grad():
            p_pred = model.p_pred(
                                  rel=torch.tensor(rel).to("cuda"),
                                  s_mask=torch.tensor(s_mask).to("cuda"),
                                  o_mask=torch.tensor(o_mask).to("cuda"),
                                  )
            p_pred = p_pred.cpu().detach().numpy() #BR

        index,p_index=np.where(p_pred>p_num)
        for i,p in zip(index,p_index):
            s1,s2,o1,o2=pairs[i]
            spoes.append(
                (
                 (mapping[s1][0],mapping[s2][-1]),
                  p,
                 (mapping[o1][0], mapping[o2][-1])
                )
            )

        return [(text[s[0]:s[1] + 1], id2predicate[str(p)], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []

def evaluate(args,tokenizer,id2predicate,model,evl_data,evl_path):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(evl_path, 'w', encoding='utf-8')
    pbar = tqdm()
    for d in evl_data:
        R = set(extract_spoes(args,tokenizer,id2predicate,model,d['text']))
        T = set([(i[0],i[1],i[2]) for i in d['triple_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'triple_list': list(T),
            'triple_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },ensure_ascii=False,indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall

def test(args):
    torch.cuda.set_device(int(args.cuda_id))
    test_path = os.path.join(args.base_path, args.dataset, "test.json")
    output_path=os.path.join(args.base_path,args.dataset,"output",args.file_id)
    test_pred_path = os.path.join(output_path, "test_pred.json")
    rel2id_path=os.path.join(args.base_path,args.dataset,"rel2id.json")
    test_data = json.load(open(test_path))
    id2predicate, predicate2id = json.load(open(rel2id_path))
    config = BertConfig.from_pretrained(args.bert_config_path)
    tokenizer = Tokenizer(args.bert_vocab_path)
    config.num_p=len(id2predicate)
    train_model = BiRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
    train_model.to("cuda")

    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))
    f1, precision, recall = evaluate(args,tokenizer,id2predicate,train_model, test_data, test_pred_path)
    print("f1:%f, precision:%f, recall:%f"%(f1, precision, recall))