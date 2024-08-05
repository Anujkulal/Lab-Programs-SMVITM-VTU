//Selection sort
//
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>

void selection(int arr[], int n){
    for(int i=0; i<n-1; i++){
        int minIndex = i;
        for(int j=0; j<n; j++){
            if(arr[j] < arr[minIndex]){
                minIndex = j;
            }
        }
        if(minIndex !=i){
            int temp = arr[i];
            arr[i] = arr[minIndex];
            arr[minIndex] = temp;
        }
    }
}
int main() {
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);
    int arr[n];
    srand(time(NULL));
    for(int i =0;i<n; i++){
        arr[i] = rand() % 1000;
    }
    clock_t start = clock();
    selection(arr, n);
    clock_t end = clock();
    double timeTaken = ((double)(end - start))/CLOCKS_PER_SEC;
    printf("Time taken for sorting: %f\n", timeTaken);
    return 0;
}