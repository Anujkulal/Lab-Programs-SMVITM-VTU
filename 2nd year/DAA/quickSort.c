//Quick sort
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>

int partition(int arr[], int low, int high){
    int pivot = arr[high];
    int i = low - 1;
    for(int j=low; j<high; j++){
        if(arr[j] <= pivot){
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i+1];
    arr[i+1] = arr[high];
    arr[high] = temp;
    return i+1;
}

void quickSort(int arr[], int low, int high){
    if(low < high){
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi-1);
        quickSort(arr, pi+1, high);
    }
}

int main() {
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);
    int arr[n];
    srand(time(NULL));
    printf("Enter the elements: ");
    for(int i=0; i<n; i++){
        // arr[i] = rand() % 10;
        scanf("%d", &arr[i]);
    }
    clock_t start = clock();
    quickSort(arr, 0, n-1);
    clock_t end = clock();
    double timeTaken = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("Time taken for sorting %f\n", timeTaken);
    for(int i=0; i<n; i++){
        printf("%d ", arr[i]);
    }
    printf("\n");
    return 0;
}