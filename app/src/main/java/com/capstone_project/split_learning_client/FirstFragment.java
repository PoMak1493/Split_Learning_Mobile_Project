package com.capstone_project.split_learning_client;

import android.bluetooth.BluetoothAdapter;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

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

        view.findViewById(R.id.button_train).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // (Note) Originally -> This click is used to switch to fragment 2.
                //NavHostFragment.findNavController(FirstFragment.this)
                //        .navigate(R.id.action_FirstFragment_to_SecondFragment);

                sendMessageToSplitServer((MainActivity) getParentFragment().getActivity(), view);
            }
        });
    }

    public void sendMessageToSplitServer(MainActivity activity, View v)
    {
        activity.callPythonCode();
        MessageSender ms = new MessageSender();
        ms.delegate = activity;
        System.out.println(getDeviceName());
        ms.execute("I am an " + getDeviceName()+".");

    }

    public String getDeviceName() {
        BluetoothAdapter myDevice = BluetoothAdapter.getDefaultAdapter();
        String deviceName = myDevice.getName();
        return deviceName;
    }










}