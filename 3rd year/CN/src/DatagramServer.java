import java.util.*;
import java.net.*;

public class DatagramServer {
    public static void main(String[] args) throws Exception {
        Scanner in = new Scanner(System.in);
        DatagramSocket serverSocket = new DatagramSocket(9000);
        byte[] sendData = new byte[1024];
        byte[] receiveData = new byte[1024];

        System.out.println("Server display");
        
        DatagramPacket receivePacket = new DatagramPacket(receiveData, receiveData.length);
        serverSocket.receive(receivePacket);
        System.out.println(new String(receivePacket.getData()));
        InetAddress IPAddress = receivePacket.getAddress();
        int port = receivePacket.getPort();

        while (true) {
            System.out.println("Type some msg to display at client end");
            String msg = in.nextLine();
            sendData = msg.getBytes();
            System.out.println("Message sent from server: "+new String(sendData));
            DatagramPacket sendpacket = new DatagramPacket(sendData, sendData.length, IPAddress, port);
            serverSocket.send(sendpacket);
        }
    }
}
