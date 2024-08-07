//Kruskal's algorithm
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#define INF 999
int parent[100], cost[100][100];

int findParent(int i){
    while(parent[i] != 0){
        i=parent[i];
    }
    return i;
}
int unionVertices(int i, int j){
    if(i != j){
        parent[j] = i;
        return 1;
    }
    return 0;
}

int main() {
    int min, mincost=0, u=0, v=0, a=0, b=0, n, ne=1;
    printf("Enter the number of nodes: ");
    scanf("%d", &n);
    printf("Enter the cost/weight matrix:\n");
    for(int i=1; i<=n; i++){
        parent[i] = 0;
        for(int j=1; j<=n; j++){
            scanf("%d", &cost[i][j]);
            if(cost[i][j] == 0) cost[i][j] = INF;
        }
    }
    printf("Edges of minimum spanning tree are:\n");
    while(ne < n){
        min = INF;
        for(int i=1; i<=n; i++){
            for(int j=1; j<=n; j++){
                if(cost[i][j]<min){
                    min = cost[i][j];
                    a = u = i;
                    b = v = j;
                }
            }
        }
        u = findParent(u);
        v = findParent(v);
        if(unionVertices(u, v)){
            printf("%d edge selected (%d --> %d), Cost: %d\n", ne++, a, b, min);
            mincost += min;
        }
        cost[a][b] = cost[b][a] = INF;
    }
    printf("Minimum cost is %d\n", mincost);
    return 0;
}