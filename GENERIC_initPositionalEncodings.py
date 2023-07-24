'''
Kleinen Beispielgraph basteln und darauf das ganze ausführen


Zeitplanung: bis 13.30 Laplacian fertig       DONE 
13.30 - 14.30: RWE                             DONE
14.30 - 16.30: Struktur von Node- und Edgefeatures für Amazon Review verstehen   DONE
17.30 - 21.00: GCN Layer anpassen auf Struktur von Amazon Review (Positional Features erstmal null)       DOING
21.00 - 0.00: GCN Layer anpassen auf Struktur von Amazon Review hoffentlich fertig

morgen: 
2h: Struktur von Positional Features definieren  --> Einfach nur ein Array! Teil von Data! 
2h: Arbeitsmeeting
--- Essenspause ---
4h: Initialisierung von Positional Features für Amazon Review Dataset

mittwoch vormittag: 
Initialisierung fertig

'''


import torch
import torch.nn.functional as F
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx

import numpy as np
from scipy.sparse import csr_matrix



'''
Input: num_nodes und edge_matrix (hier mit Beispiel graphen codiert)
Output: matrix mit anzahl knoten einträgen, für jeden Knoten das positional encoding
'''
def calculatePositionalEncodings_Laplacian():
    #Beispiel graph
    num_nodes = 20
    edges = [
        (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3),
        (4, 5), (4, 11), (5, 4), (5, 6), (5, 10), (6, 5), (6, 7), (7, 6),
        (7, 8), (8, 7), (8, 9), (8, 10), (9, 8), (10, 5), (10, 8), (11, 4),
        (11, 12), (11, 19), (12, 11), (12, 13), (13, 12), (13, 14), (13, 18),
        (14, 13), (14, 15), (15, 14), (15, 16), (16, 15), (16, 17), (17, 16),
        (17, 18), (18, 13), (18, 17), (18, 19), (19, 11), (19, 18)
    ]

    # Create the adjacency matrix in CSR format -> das wird dann für die encodings benutzt!
    rows, cols = zip(*edges)
    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    ''' this code computes the in_degrees matrix from the edge list. it can later be adapted to compute the in-degrees matrix from the adjacency matrix (however, then, we should
    do some tests with small sample graphs to ensure everything is correct
    '''
    in_degrees_dict = {node: 0 for node in range(num_nodes)}
    # Calculate the in-degrees for each node
    for edge in edges:
        _, dst = edge
        in_degrees_dict[dst] += 1

    in_degrees = np.array([in_degrees_dict[i] for i in range(len(in_degrees_dict))], dtype=float)
    in_degrees = in_degrees.clip(1)  # Clip to ensure no division by zero
    in_degrees = np.power(in_degrees, -0.5)  # Take the element-wise inverse square root

    # Create the sparse diagonal matrix N
    N = sp.diags(in_degrees, dtype=float)

    L = sp.eye(num_nodes) - N * A * N

    #calc eigvals and eigVecs, equivalent to the original code
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])

    #pos_enc_dim = hyperparameter!
    pos_enc_dim = 1
    RESULT_POS_ENCODING = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    RESULT_EIG_VEC = RESULT_POS_ENCODING

    #Ergebnis: Array mit pos_enc_dim * Anzahl Knoten --> 
    # für jeden Knoten ein Eigenvector mit Anzahl Dims, d.h. result[0] = pos_enc für node 0 usw. 
    print(RESULT_EIG_VEC)


'''
Input: num_nodes und edge_matrix (hier mit Beispiel graphen codiert)
Output: matrix mit anzahl knoten einträgen, für jeden Knoten das positional encoding
'''
def calculatePositionalEncodings_RSWE():
    num_nodes = 20
    edges = [
        (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3),
        (4, 5), (4, 11), (5, 4), (5, 6), (5, 10), (6, 5), (6, 7), (7, 6),
        (7, 8), (8, 7), (8, 9), (8, 10), (9, 8), (10, 5), (10, 8), (11, 4),
        (11, 12), (11, 19), (12, 11), (12, 13), (13, 12), (13, 14), (13, 18),
        (14, 13), (14, 15), (15, 14), (15, 16), (16, 15), (16, 17), (17, 16),
        (17, 18), (18, 13), (18, 17), (18, 19), (19, 11), (19, 18)
    ]


    '''
    Berechung von A, RW und M analog zu original code
    '''
    rows, cols = zip(*edges)
    data = np.ones(len(rows))
    A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))

    in_degrees_dict = {node: 0 for node in range(num_nodes)}
    # Calculate the in-degrees for each node
    for edge in edges:
        _, dst = edge
        in_degrees_dict[dst] += 1

    in_degrees = np.array([in_degrees_dict[i] for i in range(len(in_degrees_dict))], dtype=float)
    in_degrees = in_degrees.clip(1)  # Clip to ensure no division by zero
    in_degrees = np.power(in_degrees, -1)  # Take the element-wise inverse square root

    Dinv = sp.diags(in_degrees, dtype=float)

    RW = A * Dinv  
    M = RW
    
    # das ist wieder ein Hyperparameter; sollte >1 sein weil eins immer 0 ist irgendwie!
    pos_enc_dim = 2

    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc-1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE,dim=-1)

    #ERGEBNIS
    RESULT_POS_ENCODING = PE 
    print(RESULT_POS_ENCODING) 

calculatePositionalEncodings_RSWE()

'''  KEINE AHNUNG WAS DAS SOLL
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, labels, snorm_n
    '''