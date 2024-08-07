//
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#define max 20
int x[max];
int s[max];
int d, flag=0;
void sumofsub(int m, int k, int r){
    x[k]=1;
    if(m+s[k] == d){
        flag=1;
        printf("{");
        for(int i=1; i<=k; i++){
            if(x[i]==1){
                printf("%d", s[i]);
            }
        }
        printf("}\n");
    }
    else if((m+s[k]+s[k+1]) <=d){
        sumofsub(m+s[k], k+1, r-s[k]);
    }
    // if((m+r-s[k]>=d) && (m+s[k+1] <=d)){ x[k]=0;
    else{ 
        x[k]=0;
        sumofsub(m, k+1, r-s[k]);
    }
}
void inputarr(int arr[], int n){
    for(int i=1; i<=n; i++){
        scanf("%d", &arr[i]);
    }
}
int main() {
    int i, n, sum=0;
    printf("Enterthe number of elements:");
    scanf("%d", &n);
    printf("Enter the elements: ");
    inputarr(s, n);
    printf("Enter the value of d: ");
    scanf("%d", &d);
    for(i=1; i<=n; i++){
        sum += s[i];
    }
    if(sum < d || s[1] > d){
        printf("\n the givvrn problem doesnot have a solution\n");
        return 0;
    }
    else{
        printf("The subsets are:\n");
        sumofsub(0, 1, sum);
    }
    if(flag==0){
        printf("no subssets possible\n");
    }
    return 0;
}