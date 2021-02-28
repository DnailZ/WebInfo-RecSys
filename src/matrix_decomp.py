import pandas as pd
import numpy as np
import torch
import torch.utils.data as D

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.distributed as distributed
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel as DP



def read_data(datapath, social_relation):
    """
    read_data:
        read train data from `datapath` and `social_relation`.
    Returns:
        (data, env, param)
        data = (users, movies, score, weight) 4 1DTensors with same length.
        env = (user_dict, movie_dict, tag_dict) convert user_id, movie_id, tag to internal repersentation.
        param = (user_n, movie_n) abstract user/movie count. (tag included)
    """
    ratings = []
    user_dict = {}
    movie_dict = {}
    tag_dict = {}
    score_sum = 0
    with open(datapath) as f:
        for line in f:
            entry = line.strip().split(',')
            if len(entry) == 0:
                continue
            [user, movie, score, date, *tags] = entry
            u, m, s = int(user), int(movie), int(score)/5
            score_sum += s
            if u not in user_dict:
                user_dict[u] = len(user_dict)
            u = user_dict[u]
            
            if m not in movie_dict:
                movie_dict[m] = len(movie_dict)
            m = movie_dict[m]
            
            for t in tags:
                if t in tag_dict:
                    tag_dict[t].append(m)
                else:
                    tag_dict[t] = [m]

            ratings.append((u, m, s, 1))
    for i, (tag, movies) in enumerate(tag_dict.items()):
        for m in movies:
            ratings.append((len(user_dict) + i, m, 0.95, 0.1))
    with open(social_relation) as f:
        for line in f:
            u, fellow_li = line.strip().split(":")
            if int(u) not in user_dict:
                continue
            u = user_dict[int(u)]
            fellow_li = map(lambda x: int(x), fellow_li.split(","))
            for v in fellow_li:
                if v not in user_dict:
                    continue
                v = user_dict[v]
                ratings.append((u, len(movie_dict) + v, 0.95, 0.05))
            ratings.append((u, len(movie_dict) + u, 0.95, 0.1))
            
    ratings = pd.DataFrame(ratings, columns=['user', 'movie', 'score', 'weight'])
    users = torch.LongTensor(ratings['user'])
    movies = torch.LongTensor(ratings['movie'])
    scores = torch.FloatTensor(ratings['score'])
    weight = torch.FloatTensor(ratings['weight'])
    
    return (users, movies, scores, weight), (user_dict, movie_dict, tag_dict), (len(user_dict) + len(tag_dict), len(movie_dict) + len(user_dict))

def read_test_data(datapath, user_dict, movie_dict):
    """
    read_test_data:
        read test data from `datapath` using `user_dict`, `movie_dict`.
        user_dict, movie_dict: convert user_id, movie_id, tag to internal repersentation.
    Returns:
        (users, movies) 2 1DTensors with same length.
    """
    ratings = []
    with open(datapath) as f:
        for line in f:
            entry = line.strip().split(',')
            if len(entry) == 0:
                continue
            [user, movie, date, *tags] = entry
            u, m = int(user), int(movie)
            u = user_dict[u]
            m = movie_dict[m]
            ratings.append((u, m))
    ratings = pd.DataFrame(ratings, columns=['user', 'movie'])
    
    users = torch.LongTensor(ratings['user'])
    movies = torch.LongTensor(ratings['movie'])
    return (users, movies)

class DualEmbedding(nn.Module):
    def __init__(self, user_n, movie_n, k):
        super(DualEmbedding, self).__init__()
        self.user_embed = nn.Embedding(user_n, k)
        self.user_bias = nn.Embedding(user_n, 1)
        self.movie_embed = nn.Embedding(movie_n, k)
        self.movie_bias = nn.Embedding(movie_n, 1)
        self.total_bias = 0.51465023748498
    
    def forward(self, user, movie):
        user_feat = self.user_embed(user)
        movie_feat = self.movie_embed(movie)
        dot_product = torch.sum(user_feat * movie_feat, dim=-1)
        
        user_bias = torch.reshape( self.user_bias(user), user.shape )
        movie_bias = torch.reshape( self.movie_bias(movie), movie.shape )
        
        result = dot_product + user_bias + movie_bias
        return (torch.sigmoid(dot_product), self.l1_loss())
    
    def l1_loss(self):
        params = torch.cat([x.view(-1) for x in self.user_embed.parameters()]
                       + [x.view(-1) for x in self.movie_embed.parameters()])
        return torch.norm(params, 1)

def train_model(user_n, movie_n, train_data, val_data, gpus=[], epochs=100, lr=0.3, k=17, batch_size=1000):
    """
    train_model:
        (user_n, movie_n): net parameter.
        train_data = (users, movies, scores, weight) 4 1DTensor of Train Data, in same length.
        val_data = (users, movies, scores) 3 1DTensor of Validation Data, in same length.
    Returns:
        model: PyTorch model. 
    """
    dataset = D.TensorDataset(*train_data)
    dataloader = D.DataLoader(dataset, batch_size)
    
    model = DualEmbedding(user_n, movie_n, k).cuda()
    model = DP(model, device_ids=gpus, output_device=gpus[0])

    optimizer = optim.SGD(model.parameters(), lr)
    
    def criterion(pred, score, weight):
        return torch.dot(weight, (pred - score)**2) / len(pred)
    mseloss = nn.MSELoss()
    (val_users, val_movies, val_scores)= val_data

    li = list(dataloader)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (user, movie, score, weight) in enumerate(li):
            user = user.cuda(non_blocking=True)
            movie = movie.cuda(non_blocking=True)
            score = score.cuda(non_blocking=True)
            weight = weight.cuda(non_blocking=True)

            optimizer.zero_grad()

            pred, l1_loss = model(user, movie)
            
            
            loss = criterion(pred, score, weight) + 1e-2 * l1_loss/len(li)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 0:
                print(f"batch: {i}")
        pred, _ = model(val_users, val_movies)
        val_loss = mseloss(torch.round(pred*5), val_scores*5)
        print(f"epoch: {epoch}, loss: {running_loss / len(li)}, val_loss:{val_loss}")
    return model

def varify(model, val_data):
    """
    varify:
        model: PyTorch model. 
        val_data = (users, movies, scores) 3 1DTensor of Validation Data, in same length.
    Returns:
        val_loss: difference between predicted and given.
    """
    criterion = nn.MSELoss()
    (val_users, val_movies, val_scores) = val_data
    pred, _ = model(val_users, val_movies)
    val_loss = torch.sqrt(criterion(torch.round(pred*5), val_scores*5))
    return val_loss

def generate_test(test_data_file, result_file, user_dict, movie_dict):
    """
    generate_test:
        test_data_file: test data file path. 
        result_file: result file path.
        user_dict, movie_dict: convert user_id, movie_id, tag to internal repersentation.
    """
    (users, movies) = read_test_data(test_data_file, user_dict, movie_dict)
    pred, _ = model(users, movies)
    print(torch.round(pred*5)[:10])
    pred = torch.round(pred*5)

    with open(result_file, 'w') as f:
        f.writelines([str(int(e)) + '\n' for e in pred])