import java.util.*;
import java.net.*;

public class DatagramClient {
    public static void main(String[] args) throws Exception {
        String line = "connected with client";
        DatagramSocket clientSocket = new DatagramSocket();
        InetAddress IPAddress = InetAddress.getByName("localhost");
        byte[] sendData = new byte[1024];
        byte[] receiveData = new byte[1024];
        sendData = line.getBytes();

        DatagramPacket sendPacket = new DatagramPacket(sendData, sendData.length, IPAddress, 9000);
        clientSocket.send(sendPacket);
        System.out.println("Client display");
        while(true){
            DatagramPacket recievepacket = new DatagramPacket(receiveData, receiveData.length);
            clientSocket.receive(recievepacket);
            String messageRecieved = new String(recievepacket.getData(), recievepacket.getOffset(), recievepacket.getLength());
            System.out.println("message typed at server side is: "+messageRecieved);
        }
    }
}
