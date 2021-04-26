package com.capstone_project.split_learning_client;

import android.os.AsyncTask;
import android.os.Bundle;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.view.View;

import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;
import android.widget.Toast;


public class MainActivity extends AppCompatActivity implements AsyncResponse{

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        initPython();

        //FloatingActionButton fab = findViewById(R.id.fab);
        //fab.setOnClickListener(new View.OnClickListener() {
        //    @Override
        //    public void onClick(View view) {
        //        Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
        //                .setAction("Action", null).show();
        //    }
        //});
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public void processFinish(String output) {
        Toast.makeText(this, output,Toast.LENGTH_SHORT).show();
    }




    // The following part is related to the python implementation:
    void initPython(){
        if (! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
    }


    void callPythonCode(){
        Python py = Python.getInstance();
        //py.getModule("hello").get("greet").call("Android");
        py.getModule("split_mobile_train").callAttr("greet", "Android");

        String default_csv_file_path = "custom_dataset.csv";
        PyObject log_obj = py.getModule("split_mobile_train").callAttr("execute_train", default_csv_file_path);
        String training_log = log_obj.toString();
        TextView training_log_view = findViewById(R.id.textview_first);
        training_log_view.setText(training_log);
        //PyObject obj4 = py.getModule("hello").callAttr("get_java_bean");
        //JavaBean data = obj4.toJava(JavaBean.class);
        //data.print();
    }








}