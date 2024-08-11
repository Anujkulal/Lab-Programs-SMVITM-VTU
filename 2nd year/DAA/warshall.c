//
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#define max 20
void warshall(int n, int r[max][max]){
    int i,j,k;
    for(k=1; k<=n; k++){
        for(i=1; i<=n; i++){
            for(j=1; j<=n; j++){
                r[i][j] = r[i][j] || (r[i][k] && r[k][j]);
            }
        }
    }
    printf("Transitive closure matrix is:\n");
    for(i=1; i<=n; i++){
        printf("\t %d", i);
    }printf("\n");
    printf("---------------------------------------\n");
    for(i=1; i<=n; i++){
        printf("%d\t", i);
        for(j=1; j<=n; j++){
            printf("%d\t", r[i][j]);
        }
        printf("\n");
    }
}
int main() {
    int r[max][max], n;
    printf("Enter the no of nodes: ");
    scanf("%d", &n);
    printf("Enter the adjacency matrix:\n");
    for(int i=1; i<=n; i++){
        for(int j=1; j<=n; j++){
            scanf("%d", &r[i][j]);
        }
    }
    warshall(n, r);
    return 0;
}