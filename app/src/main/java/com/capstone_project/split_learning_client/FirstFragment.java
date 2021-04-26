package com.capstone_project.split_learning_client;

import android.app.ProgressDialog;
import android.bluetooth.BluetoothAdapter;
import android.graphics.Color;
import android.os.AsyncTask;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;

public class FirstFragment extends Fragment {

    @Override
    public View onCreateView(
            LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState
    ) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_first, container, false);
    }

    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        TextView textView = view.findViewById(R.id.textview_first);
        textView.setMovementMethod(new ScrollingMovementMethod());
        ProgressBar progressBar = (ProgressBar) getView().findViewById(R.id.progressbar);
        progressBar.setVisibility(view.INVISIBLE);

        view.findViewById(R.id.button_train).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // (Note) Originally -> This click is used to switch to fragment 2.
                //NavHostFragment.findNavController(FirstFragment.this)
                //        .navigate(R.id.action_FirstFragment_to_SecondFragment);
                progressBar.setVisibility(view.VISIBLE);
                TextView textview = view.findViewById(R.id.textview_first);
                textView.setText("The Split Learning training process is processing.\nGive me just one moment please.");
                Button btn_train = view.findViewById(R.id.button_train);
                btn_train.setClickable(false);
                btn_train.setBackgroundColor(Color.GRAY);
                SplitLearningTrainingTask split_task = new SplitLearningTrainingTask();
                split_task.execute();

            }

        });




    }

    public void sendMessageToSplitServer(MainActivity activity)
    {
        activity.callPythonCode();
        MessageSender ms = new MessageSender();
        ms.delegate = activity;
        System.out.println(getDeviceName());
        ms.execute("I am an " + getDeviceName()+".");

    }

    private class SplitLearningTrainingTask extends AsyncTask<MainActivity, ProgressBar, ProgressBar> {
        protected ProgressBar doInBackground(MainActivity... activities) {
            ProgressBar progressBar = (ProgressBar) getView().findViewById(R.id.progressbar);
            sendMessageToSplitServer((MainActivity) getParentFragment().getActivity());

            return progressBar;
        }


        protected void onPostExecute(ProgressBar progress) {
            progress.setVisibility(View.INVISIBLE);
            Button btn_train = getView().findViewById(R.id.button_train);
            btn_train.setBackgroundColor(0xFF6200EE);
            btn_train.setClickable(true);
        }
    }






    public String getDeviceName() {
        BluetoothAdapter myDevice = BluetoothAdapter.getDefaultAdapter();
        String deviceName = myDevice.getName();
        return deviceName;
    }





}