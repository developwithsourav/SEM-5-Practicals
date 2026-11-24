#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

int randomizedPartition(int arr[], int low, int high) {
    int randIndex = low + rand() % (high - low + 1);
    swap(arr[randIndex], arr[high]);
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

int randomizedSelect(int arr[], int low, int high, int i) {
    if (low == high) return arr[low];
    int pi = randomizedPartition(arr, low, high);
    int k = pi - low + 1;
    if (i == k) return arr[pi];
    else if (i < k) return randomizedSelect(arr, low, pi - 1, i);
    else return randomizedSelect(arr, pi + 1, high, i - k);
}

int main() {
    srand(time(0));
    int n;
    cout << "Enter number of elements: ";
    cin >> n;
    int arr[n];
    cout << "Enter elements: ";
    for (int i = 0; i < n; i++) cin >> arr[i];

    int i;
    cout << "Enter i (to find ith smallest): ";
    cin >> i;

    if (i < 1 || i > n) {
        cout << "Invalid i!" << endl;
    } else {
        int result = randomizedSelect(arr, 0, n - 1, i);
        cout << "The " << i << "th smallest element is: " << result << endl;
    }

    return 0;
}
