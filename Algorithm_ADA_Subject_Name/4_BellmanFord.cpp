#include <iostream>
#include <climits>
using namespace std;

struct Edge {
    int src, dest, weight;
};

int main() {
    int V, E;
    cout << "Enter number of vertices and edges: ";
    cin >> V >> E;

    Edge edges[E];
    cout << "Enter each edge as (src dest weight):" << endl;
    for (int i = 0; i < E; i++)
        cin >> edges[i].src >> edges[i].dest >> edges[i].weight;

    int src;
    cout << "Enter source vertex: ";
    cin >> src;

    int dist[V];
    for (int i = 0; i < V; i++) dist[i] = INT_MAX;
    dist[src] = 0;

    for (int i = 1; i <= V - 1; i++) {
        for (int j = 0; j < E; j++) {
            int u = edges[j].src, v = edges[j].dest, w = edges[j].weight;
            if (dist[u] != INT_MAX && dist[u] + w < dist[v])
                dist[v] = dist[u] + w;
        }
    }

    // Check for negative weight cycles
    for (int j = 0; j < E; j++) {
        int u = edges[j].src, v = edges[j].dest, w = edges[j].weight;
        if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
            cout << "Graph contains a negative weight cycle!" << endl;
            return 0;
        }
    }

    cout << "\nShortest distances from vertex " << src << ":" << endl;
    for (int i = 0; i < V; i++) {
        if (dist[i] == INT_MAX)
            cout << src << " -> " << i << " : INF" << endl;
        else
            cout << src << " -> " << i << " : " << dist[i] << endl;
    }

    return 0;
}
