import java.util.*;
import java.net.*;
import java.io.*;

public class SocketServer {
    public static void main(String[] args) throws Exception {
        ServerSocket serverSocket = new ServerSocket(4000);
        System.out.println("***** server side *****");
        System.out.println("Server ready for connection");

        Socket connSocket = serverSocket.accept();
        System.out.println("Connection is successfull and ready for file transfer");

        InputStream iStream = connSocket.getInputStream();
        BufferedReader fileRead = new BufferedReader(new InputStreamReader(iStream));
        String fname = fileRead.readLine();
        File filename = new File(fname);

        OutputStream oStream = connSocket.getOutputStream();
        PrintWriter pwrite = new PrintWriter(oStream, true);
        if(filename.exists()){
            BufferedReader contentRead = new BufferedReader(new FileReader(fname));
            System.out.println("Writing the contents to the socket");
            String str;
            while((str = contentRead.readLine())!= null){
                pwrite.println(str);
            }
            // contentRead.close();
        }
        else{
            System.out.println("Requested file doesnot exists");
            String msg = "Requested file doesnot exists at server side";
            pwrite.println(msg);
        }
    }
}
