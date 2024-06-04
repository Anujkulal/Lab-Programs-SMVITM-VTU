public class MatrixAddition {
    public static void main(String[] args) {
        if(args.length!=1){
            System.out.println("Usage: Java matrix addition <N>");
            return;
        }

        int N = Integer.parseInt(args[0]);
        int[][] matrixA = new int[N][N] ;
        int[][] matrixB = new int[N][N];
        int[][] result = new int[N][N];

        fillmatrix(matrixA,N);
        fillmatrix(matrixB,N);

        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                result[i][j] = matrixA[i][j] + matrixB[i][j];
            }
        }

        System.out.println("Matrix A:");
        printMatrix(matrixA,N);
        System.out.println("Matrix B:");
        printMatrix(matrixB,N);
        System.out.println("Sum of matrices A and B:");
        printMatrix(result,N);
    }
    private static void fillmatrix(int[][]Matrix,int N){
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                Matrix[i][j] = (int)(Math.random()*100);
            }
        }
    }
    private static void printMatrix(int[][] Matrix,int N){
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                System.out.print(Matrix[i][j]+" ");
            }
            System.out.println();
        }
        System.out.println();
    }
}
