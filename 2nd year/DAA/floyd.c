//
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#define max 20
int min(int a, int b);
void floyd(int n, int d[max][max], int w[max][max]){
    int i, j, k;
    for(i=1; i<=n; i++){
        for(j=1; j<=n; j++){
            d[i][j] = w[i][j];
        }
    }
    for(k=1; k<=n; k++){
        for(i=1; i<=n; i++){
            for(j=1; j<=n; j++){
                d[i][j] = min(d[i][j], d[i][k]+d[k][j]);
            }
        }
    }
    printf("All pairs shortest path is:\n");
    for(i=1; i<=n; i++){
        printf("\t%d ", i);
    }
    printf("------------------------------\n");
    for(i=1; i<=n; i++){
        printf("%d\t", i);
        for(j=1; j<=n; j++){
            printf("%d\t", d[i][j]);
        }
        printf("\n");
    }
}
int min(int a, int b){
    return (a<b)? a:b;
}
int main() {
    int n, i, j, w[max][max], d[max][max];
    printf("Enter the no of nodes: ");
    scanf("%d", &n);
    printf("Enter the cost matrix:\n");
    for(i=1; i<=n; i++){
        for(j=1; j<=n; j++){
            scanf("%d", &w[i][j]);
        }
    }
    floyd(n, d, w);
    return 0;
}