import java.util.Scanner;
public class LeakyBucket {

     public static void leakyBucket(int n, int capacity, int rate, int[] size){
          System.out.println("CLOCK\t SIZE\t ACCEPT\t SENT\t REM");
          int accept = 0, sent =0, rem = 0;
          for(int i=0; i<n; i++){
               int packetSize = size[i];
               if(packetSize <= capacity){
                    accept = packetSize;
                    
                    if(accept <=rate){
                         sent = accept + rem;
                         rem=0;
                    }
                    else{
                         rem = accept - rate;
                         sent = rate;
                    }
                    
                    System.out.println((i+1)+"\t "+packetSize+"\t "+accept+"\t "+sent+"\t "+rem);
               }
               else{
                    String msg = "dropped";
                    sent = rem;
                    rem=0;
                    System.out.println((i+1)+"\t "+packetSize+" "+msg+"\t "+sent+"\t "+rem);
               }
          }
     }
  
   public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    System.out.print("Enter the number of packets: ");
    int n = sc.nextInt();
    System.out.print("Enter bucket capacity: ");
    int capacity = sc.nextInt();
    System.out.print("Enter output rate: ");
    int rate = sc.nextInt();
    System.out.println("Enter size of packets: ");
    int[] size = new int[n];
    for(int i=0; i<n; i++){
     size[i] = sc.nextInt(); 
    }

    leakyBucket(n, capacity, rate, size);
    sc.close();
   }
}

