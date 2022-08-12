import streamlit as sl

# -----------------------------------------------------------------------------------------------------------------
# LOADING OF MODELS; takes time
# -----------------------------------------------------------------------------------------------------------------

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from time import time
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split

CUDA = True
device = "cuda" if (torch.cuda.is_available() and CUDA) else "cpu"
print(torch.cuda.is_available())
print(device)
batch_size = 10
resized = 512
num_classes = 42
lr = 0.001
epochs = 25
top_n_results = 10
mods = transforms.Compose([transforms.Resize((resized,resized)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                          ])

model = models.resnet50(pretrained=True)
final_num_feats = model.fc.in_features
model = torch.nn.Sequential(*(list(model.children())[:-1]))

for param in model.parameters():
    param.require_grad = False

model.to(device)
model.eval()

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(final_num_feats, num_classes)
    def forward(self, x):
        return self.fc(x)

with open('cluster_book-84.pickle', 'rb') as f:
    cluster_book = pickle.load(f)
with open('kmeans-84.pickle', 'rb') as f:
    kmeans = pickle.load(f)
classifier = torch.load('classifier-100.pth')
classifier.eval()
with open('cat_book.pickle', 'rb') as f:
    category_book = pickle.load(f)
#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

header = sl.container()
searcher = sl.container()
trf_class = sl.container()
cluster = sl.container()

with header:
    sl.title('E-commerce Image Searcher')

with searcher:
    sl.text('Search here:')
    query_img_file = sl.file_uploader('Upload image to search', type=['png', 'jpg'])

    if query_img_file is not None:
        query_img = Image.open(query_img_file)
        fig = plt.figure()
        plt.imshow(plt.imread(query_img_file))
        plt.title('Query Image')
        plt.xticks([])
        plt.yticks([])
        sl.pyplot(fig)
        with trf_class:
            sl.text('Transfer learning classifier')
            start_time = time()
            query_tensor = mods(query_img)
            query_tensor = query_tensor.reshape(1, *query_tensor.shape).to(device)

            with torch.no_grad():
                query_feats = torch.flatten(model(query_tensor), 1)
                query_cat = classifier(query_feats)
                sm = nn.Softmax(dim=1)
                query_conf = sm(query_cat)

            # print(query_conf)
            pred_class = torch.argmax(query_conf).item()
            print(f'Predicted class is : {pred_class}')

            results = [(0, '') for i in range(top_n_results)]  # keep ranking of (similarity score, path)
            cat = category_book[pred_class]
            cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
            for img, path in cat:
                img = img.to(device)
                # query_feats = query_feats.squeeze()
                cos_sim_score = cos_sim(query_feats,img)
                
                # 'bubble up' found img has higher similarity score than prev lowest bound, then sort
                if cos_sim_score > results[-1][0]:
                    results[-1] = (cos_sim_score, path)
                    
                results = sorted(results, reverse=True)
                
            # Displaying results
            if top_n_results%2 == 0:
                rows = 2
                cols = top_n_results//2
            elif top_n_results%3 == 0:
                rows = 3
                cols = top_n_results//3
            else:
                rows = 1
                cols = top_n_results

            fig, chungus = plt.subplots(rows, cols)
            fig.set_size_inches(18.5, 10.5)
            axes = [ax for axes in chungus for ax in axes]
            for i in range(len(results)):
                axes[i].imshow(plt.imread(results[i][1]))
                axes[i].set_yticks([])
                axes[i].set_xticks([])
                axes[i].set_title(f'Rank {i+1}')
            sl.pyplot(fig)
            sl.text(f'Time taken to retrieve: {time()-start_time} seconds')

        with cluster:
            sl.text('\nKMeans Cluster')
            start_time = time()
            query_tensor = mods(query_img)
            query_tensor = query_tensor.reshape(1, *query_tensor.shape).to(device)

            with torch.no_grad():
                query_feats = model(query_tensor).detach().squeeze().to('cpu')
                
            cluster_indices = kmeans.predict(query_feats.reshape(1,-1))
            cluster_knn_model, cluster_path_list = cluster_book[cluster_indices[0]]

            dists, indices = cluster_knn_model.kneighbors(query_feats.reshape(1,-1), return_distance=True)

            if top_n_results%2 == 0:
                rows = 2
                cols = top_n_results//2
            elif top_n_results%3 == 0:
                rows = 3
                cols = top_n_results//3
            else:
                rows = 1
                cols = top_n_results

            fig, chungus = plt.subplots(rows, cols)
            fig.set_size_inches(18.5, 10.5)
            axes = [ax for axes in chungus for ax in axes]
            for i in range(len(indices[0])):
                axes[i].imshow(plt.imread(cluster_path_list[i]))
                axes[i].set_yticks([])
                axes[i].set_xticks([])
                axes[i].set_title(f'Rank {i+1}')
            sl.pyplot(fig)
            sl.text(f'Time taken to retrieve: {time()-start_time} seconds')
