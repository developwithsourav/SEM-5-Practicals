#include <iostream>
#include <string>
#include <map>
using namespace std;

struct TrieNode {
    map<char, TrieNode*> children;
    bool isEnd;
    TrieNode() : isEnd(false) {}
};

class SuffixTrie {
    TrieNode* root;

    void insertSuffix(string suffix) {
        TrieNode* cur = root;
        for (char c : suffix) {
            if (!cur->children.count(c))
                cur->children[c] = new TrieNode();
            cur = cur->children[c];
        }
        cur->isEnd = true;
    }

public:
    SuffixTrie(string text) {
        root = new TrieNode();
        // Brute force: insert all suffixes
        for (int i = 0; i < (int)text.size(); i++)
            insertSuffix(text.substr(i));
    }

    bool search(string pattern) {
        TrieNode* cur = root;
        for (char c : pattern) {
            if (!cur->children.count(c)) return false;
            cur = cur->children[c];
        }
        return true;
    }
};

int main() {
    string text;
    cout << "Enter text to build suffix trie: ";
    cin >> text;

    SuffixTrie st(text);

    int q;
    cout << "Enter number of patterns to search: ";
    cin >> q;
    while (q--) {
        string pattern;
        cout << "Enter pattern: ";
        cin >> pattern;
        if (st.search(pattern))
            cout << "\"" << pattern << "\" found in text." << endl;
        else
            cout << "\"" << pattern << "\" NOT found in text." << endl;
    }

    return 0;
}
