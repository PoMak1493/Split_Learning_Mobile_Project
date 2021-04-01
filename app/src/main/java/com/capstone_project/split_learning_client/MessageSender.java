package com.capstone_project.split_learning_client;

import android.content.Context;
import android.os.AsyncTask;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;

public class MessageSender extends AsyncTask<String, Void , String>
{
    private final String server_addr = "10.0.2.2";
    private final int server_port = 1493;
    private Socket socket;
    DataOutputStream dos;
    PrintWriter pw;
    public AsyncResponse delegate = null;

    @Override
    protected String doInBackground(String... voids) {
        System.out.print("Start Working.");
        String message = voids[0];
        String result = "";
        try {
            socket = new Socket(server_addr, server_port);
            pw = new PrintWriter(socket.getOutputStream(), true);
            pw.write(message);
            pw.flush();
            BufferedReader bufferedReader = new BufferedReader( new InputStreamReader( socket.getInputStream())); //BufferedReader aus Socket inputStreamReader
            String line = null;

            boolean wait = true;
            while((line = bufferedReader.readLine()) != null) {
                result = result + line;
            }
            pw.close();
            socket.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return result;
    }


    //The "onPostExecute()" function will receive the message from the server resposne.
    @Override
    protected void onPostExecute(String result) {
        System.out.println(result);
        delegate.processFinish(result);
    }

}
