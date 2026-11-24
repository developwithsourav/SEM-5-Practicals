#include <iostream>
using namespace std;

class BTreeNode {
public:
    int *keys;
    int t;          // Minimum degree
    BTreeNode **children;
    int n;          // Current number of keys
    bool leaf;

    BTreeNode(int t, bool leaf) {
        this->t = t;
        this->leaf = leaf;
        keys = new int[2 * t - 1];
        children = new BTreeNode *[2 * t];
        n = 0;
    }

    void traverse() {
        for (int i = 0; i < n; i++) {
            if (!leaf) children[i]->traverse();
            cout << keys[i] << " ";
        }
        if (!leaf) children[n]->traverse();
    }

    BTreeNode *search(int k) {
        int i = 0;
        while (i < n && k > keys[i]) i++;
        if (i < n && keys[i] == k) return this;
        if (leaf) return NULL;
        return children[i]->search(k);
    }

    void insertNonFull(int k) {
        int i = n - 1;
        if (leaf) {
            while (i >= 0 && keys[i] > k) {
                keys[i + 1] = keys[i];
                i--;
            }
            keys[i + 1] = k;
            n++;
        } else {
            while (i >= 0 && keys[i] > k) i--;
            if (children[i + 1]->n == 2 * t - 1) {
                splitChild(i + 1, children[i + 1]);
                if (keys[i + 1] < k) i++;
            }
            children[i + 1]->insertNonFull(k);
        }
    }

    void splitChild(int i, BTreeNode *y) {
        BTreeNode *z = new BTreeNode(y->t, y->leaf);
        z->n = t - 1;
        for (int j = 0; j < t - 1; j++) z->keys[j] = y->keys[j + t];
        if (!y->leaf)
            for (int j = 0; j < t; j++) z->children[j] = y->children[j + t];
        y->n = t - 1;
        for (int j = n; j >= i + 1; j--) children[j + 1] = children[j];
        children[i + 1] = z;
        for (int j = n - 1; j >= i; j--) keys[j + 1] = keys[j];
        keys[i] = y->keys[t - 1];
        n++;
    }
};

class BTree {
    BTreeNode *root;
    int t;
public:
    BTree(int t) { root = NULL; this->t = t; }

    void insert(int k) {
        if (!root) {
            root = new BTreeNode(t, true);
            root->keys[0] = k;
            root->n = 1;
        } else {
            if (root->n == 2 * t - 1) {
                BTreeNode *s = new BTreeNode(t, false);
                s->children[0] = root;
                s->splitChild(0, root);
                int i = (s->keys[0] < k) ? 1 : 0;
                s->children[i]->insertNonFull(k);
                root = s;
            } else {
                root->insertNonFull(k);
            }
        }
    }

    void traverse() {
        if (root) { root->traverse(); cout << endl; }
    }

    void search(int k) {
        BTreeNode *res = root ? root->search(k) : NULL;
        if (res) cout << k << " found in B-Tree." << endl;
        else cout << k << " not found in B-Tree." << endl;
    }
};

int main() {
    int t;
    cout << "Enter minimum degree of B-Tree (t >= 2): ";
    cin >> t;

    BTree bt(t);
    int n, key;
    cout << "Enter number of keys to insert: ";
    cin >> n;
    cout << "Enter keys: ";
    for (int i = 0; i < n; i++) {
        cin >> key;
        bt.insert(key);
    }

    cout << "B-Tree traversal (inorder): ";
    bt.traverse();

    cout << "Enter a key to search: ";
    cin >> key;
    bt.search(key);

    return 0;
}
