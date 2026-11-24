#include <iostream>
#include <algorithm>
using namespace std;

struct Edge {
    int src, dest, weight;
};

int parent[100], rankArr[100];

int find(int x) {
    if (parent[x] != x)
        parent[x] = find(parent[x]);
    return parent[x];
}

void unionSets(int x, int y) {
    int px = find(x), py = find(y);
    if (rankArr[px] < rankArr[py]) parent[px] = py;
    else if (rankArr[px] > rankArr[py]) parent[py] = px;
    else { parent[py] = px; rankArr[px]++; }
}

int main() {
    int V, E;
    cout << "Enter number of vertices and edges: ";
    cin >> V >> E;

    Edge edges[E];
    cout << "Enter each edge as (src dest weight):" << endl;
    for (int i = 0; i < E; i++)
        cin >> edges[i].src >> edges[i].dest >> edges[i].weight;

    sort(edges, edges + E, [](Edge a, Edge b) {
        return a.weight < b.weight;
    });

    for (int i = 0; i < V; i++) { parent[i] = i; rankArr[i] = 0; }

    cout << "\nMST Edges:" << endl;
    int totalWeight = 0;
    for (int i = 0; i < E; i++) {
        int u = find(edges[i].src);
        int v = find(edges[i].dest);
        if (u != v) {
            cout << edges[i].src << " -- " << edges[i].dest << " == " << edges[i].weight << endl;
            totalWeight += edges[i].weight;
            unionSets(u, v);
        }
    }
    cout << "Total MST Weight: " << totalWeight << endl;

    return 0;
}
