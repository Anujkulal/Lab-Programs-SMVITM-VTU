// //
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
void readgraph(int n, int a[10][10])
{
int i,j;
for(i=0; i<n; i++)
{
for(j=0; j<n; j++)
scanf("%d", &a[i][j]);
}
}
void findIndegree(int n, int a[10][10], int indegree[])
{
int i, j, sum;
for(j=0; j<n; j++)
{
sum = 0;
for(i=0; i<n; i++)
sum += a[i][j];
indegree[j] = sum;
}
}
void topology(int n, int a[10][10])
{
int i, k=0, u, v, top, t[10], indegree[10], s[10];
findIndegree(n, a, indegree);
top=-1;
for(i=0; i<n; i++){
if(indegree[i] == 0) s[++top] = i;
}
while(top!=-1){
u=s[top--];
t[k++] = u;
for(v=0; v<n; v++){
if(a[u][v] == 1){
indegree[v]--;
if(indegree[v] == 0){
s[++top] = v;
}
}
}
}
    printf("Topological sort sequence is: ");
    for(i=0; i<n; i++){
        printf("%d", t[i]);
    }
}
void main() {
    int n, a[10][10];
    printf("\n Enter the number of values: ");
    scanf("%d", &n);
    printf("\n Enter the adjacency matrix: \n");
    readgraph(n,a);
    topology(n,a);
    // return 0;
}
