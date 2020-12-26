import torch
import csv

def square_euclidean_metric(a, b):
    """ Measure the euclidean distance (optional)
    Args:
        a : torch.tensor, features of data query
        b : torch.tensor, mean features of data shots or embedding features

    Returns:
        A torch.tensor, the minus euclidean distance
        between a and b
    """

    n = a.shape[0]
    m = b.shape[0]

    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)

    logits = torch.pow(a - b, 2).sum(2)

    return logits

def cosine_similarity_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return logits

def loss_mode(args, model, data_shot, data_query, labels):
    mode=args.mymode
    nway = args.nway
    kshot = args.kshot
    query = args.query
    qshot = query // nway

    # [Terminology]
    # CE: cross entropy loss    
    # EU: euclidean distance
    # COS: cosine distance
    # L2N: L2 normalization of feature from (SimpleShot paper)
    # CL2N: Center(Mean) and L2 normalization of feature (from SimpleShot paper)
    
    if mode == 1:
        # version 1: CE + EU
        s_out = model(data_shot)
        q_out = model(data_query)
        s_out = s_out.reshape([nway, kshot, -1]) # kshot(1~5) for each nway
        s_out = torch.mean(s_out, dim=1) # normalize for kshot == output nway prototype

        dist = square_euclidean_metric(q_out, s_out)
        logits = torch.nn.functional.softmax(-dist, dim=1) 
        if type(labels)==type(None): return logits
        loss = torch.nn.functional.cross_entropy(-dist, labels)

    elif mode == 2:
        # version 2: CE + COS
        s_out = model(data_shot)
        q_out = model(data_query)
        s_out = s_out.reshape([nway, kshot, -1]) # kshot(1~5) for each nway
        s_out = torch.mean(s_out, dim=1) # normalize for kshot == output nway prototype

        cos_sim = cosine_similarity_metric(q_out, s_out) # cos_sim-1 = -2(worst) ~ 0(best)
        logits = torch.nn.functional.softmax(cos_sim-1, dim=1) 
        if type(labels)==type(None): return logits
        loss = torch.nn.functional.cross_entropy(cos_sim-1, labels)     

    elif mode == 3:  
        # version 3: CE + EU + L2N (Best result)
        s_out = model(data_shot)
        q_out = model(data_query)
        s_out = s_out / torch.norm(s_out, p=2, dim=0)
        q_out = q_out / torch.norm(q_out, p=2, dim=0)
        s_out = s_out.reshape([nway, kshot, -1]) # kshot(1~5) for each nway
        s_out = torch.mean(s_out, dim=1) # normalize for kshot == output nway prototype

        dist = square_euclidean_metric(q_out, s_out)
        logits = torch.nn.functional.softmax(-dist, dim=1) 
        if type(labels)==type(None): return logits
        loss = torch.nn.functional.cross_entropy(-dist, labels)   

    elif mode == 4:
        # version 4: CE + COS + L2N
        s_out = model(data_shot)
        q_out = model(data_query)
        s_out = s_out / torch.norm(s_out, p=2, dim=0)
        q_out = q_out / torch.norm(q_out, p=2, dim=0)
        s_out = s_out.reshape([nway, kshot, -1]) # kshot(1~5) for each nway
        s_out = torch.mean(s_out, dim=1) # normalize for kshot == output nway prototype

        cos_sim = cosine_similarity_metric(q_out, s_out) # cos_sim-1 = -2(worst) ~ 0(best)
        logits = torch.nn.functional.softmax(cos_sim-1, dim=1) 
        if type(labels)==type(None): return logits
        loss = torch.nn.functional.cross_entropy(cos_sim-1, labels)  
 
    elif mode == 5:                   
        # version 5: CE + EU + CL2N
        s_out = model(data_shot)
        q_out = model(data_query)
        s_out = (s_out-torch.mean(s_out, dim=0)) / torch.norm(s_out, p=2, dim=0)
        q_out = (q_out-torch.mean(q_out, dim=0)) / torch.norm(q_out, p=2, dim=0)
        s_out = s_out.reshape([nway, kshot, -1]) # kshot(1~5) for each nway
        s_out = torch.mean(s_out, dim=1) # normalize for kshot == output nway prototype

        dist = square_euclidean_metric(q_out, s_out)
        logits = torch.nn.functional.softmax(-dist, dim=1) 
        if type(labels)==type(None): return logits
        loss = torch.nn.functional.cross_entropy(-dist, labels)     
  
    elif mode == 6:
        # version 6: CE + COS + CL2N
        s_out = model(data_shot)
        q_out = model(data_query)
        mean_ = torch.mean(s_out, dim=0)
        s_out = (s_out-torch.mean(s_out, dim=0)) / torch.norm(s_out, p=2, dim=0)
        q_out = (q_out-torch.mean(q_out, dim=0)) / torch.norm(q_out, p=2, dim=0)
        s_out = s_out.reshape([nway, kshot, -1]) # kshot(1~5) for each nway
        s_out = torch.mean(s_out, dim=1) # normalize for kshot == output nway prototype

        cos_sim = cosine_similarity_metric(q_out, s_out) # cos_sim-1 = -2(worst) ~ 0(best)
        logits = torch.nn.functional.softmax(cos_sim-1, dim=1) 
        if type(labels)==type(None): return logits
        loss = torch.nn.functional.cross_entropy(cos_sim-1, labels)           
           
    elif mode == 7:
        # version 7 : CE + EU + L2N + Data augmentation only for proto 
        [_, ch, w, h]= data_shot.shape
        data_shot = data_shot.reshape([nway, kshot, ch, w, h])
        data_shot = torch.stack([
                            data_shot.clone(),
                            torch.rot90(data_shot.clone(), 1, [3,4]),
                            torch.rot90(data_shot.clone(), 2, [3,4]),
                            torch.rot90(data_shot.clone(), 3, [3,4])],dim=1)
        data_shot = data_shot.reshape([nway * kshot * 4, ch, w, h])

        s_out = model(data_shot)
        q_out = model(data_query)
        s_out = s_out / torch.norm(s_out, p=2, dim=0)
        q_out = q_out / torch.norm(q_out, p=2, dim=0)
        s_out = s_out.reshape([nway, kshot*4, -1]) # kshot(1~5) for each nway
        s_out = torch.mean(s_out, dim=1) # normalize for kshot == output nway prototype

        dist = square_euclidean_metric(q_out, s_out)
        logits = torch.nn.functional.softmax(-dist, dim=1) 
        if type(labels)==type(None): return logits
        loss = torch.nn.functional.cross_entropy(-dist, labels)       

    else: print('You Should Select MYMODE: \nExample) python main.py ~~~ --mymode 3'); assert(0)
    return logits, loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_acc(logits, label):
    """ In each query set, the index with the highest probability or lowest distance is determined
    Args:
        logits : torch.tensor, distance or probabilty
        label : ground truth

    Returns:
        float, mean of accuracy
    """

    # when logits is distance
    #pred = torch.argmin(logits, dim=1)

    # when logits is prob
    pred = torch.argmax(logits, dim=1)

    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Averager():
    """ During training, update the average of any values.
    Returns:
        float, the average value
    """

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class csv_write():

    def __init__(self, args):
        self.f = open('20192643_NamHyunjoon.csv', 'w', newline='')
        self.write_number = 1
        self.wr = csv.writer(self.f)
        self.wr.writerow(['id', 'prediction'])
        self.query_num = args.query

    def add(self, prediction):

        for i in range(self.query_num):
          self.wr.writerow([self.write_number, int(prediction[i].item())])
          self.write_number += 1

    def close(self):
        self.f.close()
