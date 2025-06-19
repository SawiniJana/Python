import torch
file = open("edge_list.txt","w")
l = [['Kolkata Mumbai 2031'],
    ['Mumbai Pune 155'],
    ['Mumbai Goa 571'],
    ['Kolkata Delhi 1492'],
    ['Kolkata Bhubaneshwar 444'],
    ['Mumbai Delhi 1424'],
    ['Delhi Chandigarh 243'],
    ['Delhi Surat 1208'],
    ['Kolkata Hyderabad 1495'],
    ['Hyderabad Chennai 626'],
    ['Chennai Thiruvananthapuram 773'],
    ['Thiruvananthapuram Hyderabad 1299'],
    ['Kolkata Varanasi 679'],
    ['Delhi Varanasi 821'],
    ['Mumbai Bangalore 984'],
    ['Chennai Bangalore 347'],
    ['Hyderabad Bangalore 575'],
    ['Kolkata Guwahati 1031']]

for i in l:
    file.write(i[0] + '\n')
file.close()

#IMPORTANT
edge = []
import networkx as nx
G = nx.read_weighted_edgelist('edge_list.txt', create_using=nx.Graph)
with open("edge_list.txt","r") as f:
    for i in f.readlines():
        l = i.split(" ")[0], i.split(" ")[1]
        #print(l)
        edge.append(l)

population = {
        'Kolkata' : 4486679,
        'Delhi' : 11007835,
        'Mumbai' : 12442373,
        'Guwahati' : 957352,
        'Bangalore' : 8436675,
        'Pune' : 3124458,
        'Hyderabad' : 6809970,
        'Chennai' : 4681087,
        'Thiruvananthapuram' : 460468,
        'Bhubaneshwar' : 837737,
        'Varanasi' : 1198491,
        'Surat' : 4467797,
        'Goa' : 40017,
        'Chandigarh' : 961587
        }

G.add_edges_from(edge)
#print(G.nodes())
#setting population attributes to each node
for i in list(G.nodes()):
    G.nodes[i]['population'] = population[i]


import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
node_color = [G.degree(v) for v in G] #list of degrees of nodes
node_size = [0.0005* nx.get_node_attributes(G, 'population')[v] for v in G] 
edge_width = [0.0015 * G[u][v]['weight'] for u,v in G.edges()] #storing weight of edges
pos = nx.planar_layout(G)

nx.draw_networkx(G, pos,
                 node_size = node_size,
                 node_color= node_color,
                 width = edge_width,
                 with_labels=True,
                 cmap = plt.cm.Blues,
                 alpha = 0.7
                 )
plt.axis("off")
plt.show()

import numpy as np
def Adjacency(edge_ind):
        dict_new = {}
        new = torch.zeros(size=(len(population),len(population)))
        val = np.arange(len(population))
        c = 0
        for i in population:
            dict_new[i] = val[c]
            c+= 1
        
        for i in edge_ind:
            x_co, y_co = i[0], i[1]
            new[dict_new[x_co]][dict_new[y_co]] = 1
        
        return new

adjacency_matrix = Adjacency(edge)
print(adjacency_matrix)
plt.imshow(adjacency_matrix.tolist())
plt.show()


