import java.util.Scanner;
class Stack{
    int top;
    int s[] = new int[10];
    public Stack(){
        top=-1;
    }

    void push(int item){
        if(s.length-1 == top){
            System.out.println("Stack overflow");;
        }
        else{
            top = top+1;
            s[top]=item;
        }
    }

    void pop(){
        if(top==-1){
            System.out.println("Stack underflow");
        }
        else{
            int item = s[top];
            System.out.println("Item popped is: "+item);
            top=top-1;
        }
    }

    void display(){
        if(top==-1){
            System.out.println("Stack is empty");
        }
        else{
            System.out.println("Stack contents are:");
            for(int i = top;i>=0;i--){
                System.out.println(s[i]);
            }
        }
    }
}

public class Stackdemo {
    public static void main(String[] args) {
        int ch;
        Stack stk = new Stack();
        Scanner in = new Scanner(System.in);
        while(true){
            System.out.println("Stack operations Demo");
            System.out.println("1: push\n 2: pop\n 3: Display\n 4:Exit");
            ch=in.nextInt();
            switch(ch){
                case 1: System.out.println("Enter the item to be"+" pushed "+"into the stack:");
                int item = in.nextInt();
                stk.push(item);
                break;

                case 2: stk.pop();
                break;

                case 3: stk.display();
                break;

                case 4: System.exit(0);

                default:
                System.out.println("Invalid choice!");
            }
        }
    }
}
