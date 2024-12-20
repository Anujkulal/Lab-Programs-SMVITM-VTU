import java.util.*;
public class RSA_encr_decr {
    public static void main(String[] args) {
        String msg; // Declare a string variable to store the message

        int pt[] = new int[100]; // Array to hold plain text characters
        int ct[] = new int[100]; // Array to hold cipher text characters
        int z, n, d, e, p, q, mlen; // Variables for RSA calculation

        Scanner in = new Scanner(System.in); // Initialize Scanner for input     SF Pro Display Medium       'Cascadia Code',  Consolas,  'Courier New',  monospace
        do { 
            // Prompt the user to enter two prime numbers
            System.out.println("Enter the two large prime numbers for p and q");
            p = in.nextInt(); // Input first prime number
            q = in.nextInt(); // Input second prime number
        } while (prime(p) == 0 || prime(q) == 0); // Repeat until both numbers are prime

        n = p * q; // Calculate n as the product of p and q
        z = (p - 1) * (q - 1); // Calculate z = (p-1)*(q-1)

        // Display the calculated values of n and z
        System.out.println("Value of n " + n + "\nValue of z is :" + z);

        // Find encryption key e such that gcd(e, z) == 1
        for (e = 2; e < z; e++) {
            if (gcd(e, z) == 1)
                break;
        }
        // Print the encryption key e and the public key (e, n)
        System.out.println("Encryption key e is " + e);
        System.out.println("Public key is (e, n) : " + e + "," + n);

        // Find decryption key d such that (e * d) % z == 1
        for (d = 2; d < z; d++) {
            if ((e * d) % z == 1)
                break;
        }
        // Print the decryption key d and the private key (d, n)
        System.out.println("Decryption key d is : " + d);
        System.out.println("Private key is (d, n) => " + d + "," + n);

        in.nextLine(); // Consume the new line character from previous input

        // Prompt the user to enter the message for encryption
        System.out.println("Enter the message for encryption");
        msg = in.nextLine(); // Input the message to be encrypted
        // in.close();

        mlen = msg.length(); // Get the length of the message
        for (int i = 0; i < mlen; i++) {
            pt[i] = msg.charAt(i); // Convert each character of the message to its ASCII value and store in the pt array
        }

        for (int i = 0; i < mlen; i++) {
            System.out.println(pt[i]);
        }

        // Encrypt each character using the public key (e, n) and store in ct array
        System.out.println("Encryption: Cipher Text Obtained : ");
        for (int i = 0; i < mlen; i++) {
            ct[i] = mult(pt[i], e, n); // Encrypt the character using the RSA encryption formula
            System.out.print(ct[i] + "\t"); // Print the cipher text value
        }

        // Decrypt the cipher text back to plain text using the private key (d, n)
        System.out.println("\nDecryption: Plain Text Obtained: ");
        for (int i = 0; i < mlen; i++) {
            pt[i] = mult(ct[i], d, n); // Decrypt the character using the RSA decryption formula
            System.out.println(pt[i] + ":" + (char) pt[i]); // Print the ASCII value and corresponding character
        }
        
    }

    // Method to calculate the greatest common divisor (GCD) of two numbers x and y
    public static int gcd(int x, int y) {
        if (y == 0)
            return x; // If y is 0, GCD is x
        else
            return gcd(y, x % y); // Recursively calculate GCD using the Euclidean algorithm
    }

    // Method to check if a number is prime
    public static int prime(int num) {
        int i;
        for (i = 2; i <= num / 2; i++) {
            if (num % i == 0)
                return 0; // If num is divisible by any number between 2 and num/2, it's not prime
        }
        return 1; // If no divisors are found, num is prime
    }

    // Method to perform modular exponentiation: (base^exp) % n
    public static int mult(int base, int exp, int n) {
        int res = 1, j;
        for (j = 1; j <= exp; j++)
            res = ((res * base) % n); // Multiply base and take modulo n
        return res; // Return the result
    }
}