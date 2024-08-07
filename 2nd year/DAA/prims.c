//
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<limits.h>
int main() {
    int i,j,k,sv=0, ev=0, source, minwt, totalcost=0, n;
    int w[50][50], visited[50];
    printf("Enter the number of nodes: ");
    scanf("%d", &n);
    printf("Enter the cost matrix:\n");
    for(i=1; i<=n; i++){
        for(j=1; j<=n; j++){
            scanf("%d", &w[i][j]);
        }
    }
    printf("Enter the source matrix: ");
    scanf("%d", &source);
    for(i=1; i<=n; i++){
        visited[i]=0;
    }
    visited[source] = 1;
    printf("Minimum cost edge selected for spanning tree are:\n");
    for(i=1; i<n; i++){
        minwt = INT_MAX;
        for(j=1; j<=n; j++){
            if(visited[j] == 1){
                for(k=1; k<=n; k++){
                    if(visited[k] !=1 && w[j][k]<minwt){
                        sv = j;
                        ev = k;
                        minwt = w[j][k];
                    }
                }
            }
        }
        totalcost += minwt;
        visited[ev] = 1;
        printf("%d --> %d cost: %d\n", sv, ev, minwt);
    }
    printf("Total cost of minimum spanning tree is %d\n", totalcost);
    return 0;
}