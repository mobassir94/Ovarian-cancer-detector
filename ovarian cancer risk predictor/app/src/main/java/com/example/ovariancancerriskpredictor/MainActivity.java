package com.example.ovariancancerriskpredictor;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;


import android.view.View;
import android.view.View.OnClickListener;

import android.widget.Button;
import android.widget.Spinner;
import android.widget.Toast;


import com.google.firebase.firestore.FirebaseFirestore;

import weka.core.Attribute;

import weka.classifiers.Classifier;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;


import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.DatabaseReference;



import static weka.core.SerializationHelper.read;

public class MainActivity extends AppCompatActivity {

    private FirebaseFirestore mFirestore;

    private Spinner spinner1,spinner3,spinner4,spinner5,spinner6,spinner7,spinner8,spinner9,spinner10,spinner11,spinner12;
    private Button btnSubmit;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        mFirestore = FirebaseFirestore.getInstance();
        // Write a message to the database

        addListenerOnButton();

    }






// get the selected dropdown list value

    public void addListenerOnButton() {
        spinner1 = findViewById(R.id.spinner1);
        spinner3 = findViewById(R.id.spinner3);
        spinner4 = findViewById(R.id.spinner4);
        spinner5 = findViewById(R.id.spinner5);
        spinner6 = findViewById(R.id.spinner6);
        spinner7 = findViewById(R.id.spinner7);
        spinner8 = findViewById(R.id.spinner8);
        spinner9 = findViewById(R.id.spinner9);
        spinner10 = findViewById(R.id.spinner10);
        spinner11 = findViewById(R.id.spinner11);
        spinner12 = findViewById(R.id.spinner12);

        // Write a message to the database
        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference myRef = database.getReference("Patients");



        btnSubmit = findViewById(R.id.btnSubmit);
        btnSubmit.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View v) {

                String menarche = String.valueOf(spinner1.getSelectedItem());
                String oral = String.valueOf(spinner3.getSelectedItem());
                String diet = String.valueOf(spinner4.getSelectedItem());
                String breast = String.valueOf(spinner5.getSelectedItem());
                String cervical = String.valueOf(spinner6.getSelectedItem());
                String history = String.valueOf(spinner7.getSelectedItem());
                String education = String.valueOf(spinner8.getSelectedItem());
                String aohusband = String.valueOf(spinner9.getSelectedItem());
                String menopause = String.valueOf(spinner10.getSelectedItem());
                String foodfat = String.valueOf(spinner11.getSelectedItem());
                String abortion = String.valueOf(spinner12.getSelectedItem());

                int count = 0;

                if(menarche.equals("early"))count+=2;
                else if(menarche.equals("late"))count+=3;
                else count+=1;

                if(oral.equals("yes"))count+=3;
                else count+=1;

                if(diet.equals("yes"))count+=1;
                else count+=2;


                if(breast.equals("yes"))count+=3;
                else count+=1;

                if(cervical.equals("yes"))count+=3;
                else count+=1;


                if(history.equals("yes"))count+=2;
                else count+=1;


                if(education.equals("illitarate"))count+=3;
                else count+=1;

                if(aohusband.equals("46-60"))count+=4;
                else if(aohusband.equals("31-45"))count+=1;
                else if(aohusband.equals("above 60"))count+=3;
                else count+=2;


                if(menopause.equals("40-51"))count+=1;
                else if(menopause.equals("before 40"))count+=2;
                else count+=3;

                if(foodfat.equals("yes"))count+=3;
                else count+=1;

                if(abortion.equals("yes"))count+=3;
                else count+=1;




                try{

               // Bayes Net

                    Classifier cls = null;

                    Toast.makeText(MainActivity.this, "Using Machine Learning Algorithms For prediction ", Toast.LENGTH_LONG).show();
                    cls = (Classifier) read(getAssets().open("BayesNet.model"));

                    ArrayList<Attribute> attributes = new ArrayList<>();

                    attributes.add(new Attribute("Menarche start early?", Arrays.asList("normal","late","early"),0));
                    attributes.add(new Attribute("Oral Contraception?",Arrays.asList("yes","no"),1));
                    attributes.add(new Attribute("Diet Maintaining?",Arrays.asList("yes","no"),2));
                    attributes.add(new Attribute("Affected By Breast Cancer?", Arrays.asList("yes","no"),3));
                    attributes.add(new Attribute("Affected By cervical Cancer?",Arrays.asList("yes","no"),4));

                    attributes.add(new Attribute("Cancer History In family?",Arrays.asList("yes","no"),5));
                    attributes.add(new Attribute("Education?",Arrays.asList("primary level","illitarate","graduated"),6));
                    attributes.add(new Attribute("Age of Husband?",Arrays.asList("46-60","31-45","above 60","below 30"),7));
                    attributes.add(new Attribute("Menopause End age?",Arrays.asList("40-51","before 40","after 52"),8));
                    attributes.add(new Attribute("Food contains high fat?",Arrays.asList("yes","no"),9));
                    attributes.add(new Attribute("Abortion?",Arrays.asList("yes","no"),10));

                    attributes.add(new Attribute("Affected by ovarian Cancer?",Arrays.asList("yes","no"),11));


                    // new instance to classify.
                    Instance instance = new SparseInstance(11);
                    instance.setValue(attributes.get(0), menarche);

                    instance.setValue(attributes.get(1), oral);
                    instance.setValue(attributes.get(2), diet);
                    instance.setValue(attributes.get(3), breast);
                    instance.setValue(attributes.get(4), cervical);
                    instance.setValue(attributes.get(5), history);
                    instance.setValue(attributes.get(6), education);
                    instance.setValue(attributes.get(7), aohusband);
                    instance.setValue(attributes.get(8), menopause);
                    instance.setValue(attributes.get(9), foodfat);
                    instance.setValue(attributes.get(10), abortion);

                    // Create an empty set
                    Instances datasetConfiguration;
                    datasetConfiguration = new Instances("ovarian.symbolic", attributes, 0);

                    datasetConfiguration.setClassIndex(11);
                    instance.setDataset(datasetConfiguration);

                    double[] distribution;
                    distribution = cls.distributionForInstance(instance);


                    Bundle bundle = new Bundle();

                    bundle.putDoubleArray("BayesNet",distribution);


               cls = null;
               cls = (Classifier) read(getAssets().open("RandomForest.model"));

               // Create an empty set
               Instances datasetConfigurations;
               datasetConfigurations = new Instances("ovarians.symbolic", attributes, 0);

               datasetConfigurations.setClassIndex(11);
               instance.setDataset(datasetConfigurations);

               double[] distribution1;
               distribution1 = cls.distributionForInstance(instance);

               bundle.putDoubleArray("RandomForest",distribution1);



               cls = null;
               cls = (Classifier) read(getAssets().open("J48.model"));

               // Create an empty set
               Instances datasetConfigurationss;
               datasetConfigurationss = new Instances("ovarianss.symbolic", attributes, 0);

               datasetConfigurationss.setClassIndex(11);
               instance.setDataset(datasetConfigurationss);

               double[] distribution2;
               distribution2 = cls.distributionForInstance(instance);

               bundle.putDoubleArray("J48",distribution2);




               cls = null;
               cls = (Classifier) read(getAssets().open("NaiveBayes.model"));

               // Create an empty set
               Instances datasetConfigurationn;
               datasetConfigurationn = new Instances("ovarianc.symbolic", attributes, 0);

               datasetConfigurationn.setClassIndex(11);
               instance.setDataset(datasetConfigurationn);

               double[] distribution3;
               distribution3 = cls.distributionForInstance(instance);


               bundle.putDoubleArray("NaiveBayes",distribution3);



               cls = null;
               cls = (Classifier) read(getAssets().open("LMT.model"));

               // Create an empty set
               Instances datasetConfigurationnn;
               datasetConfigurationnn = new Instances("ovarianca.symbolic", attributes, 0);
               datasetConfigurationnn.setClassIndex(11);
               instance.setDataset(datasetConfigurationnn);

               double[] distribution4;
               distribution4 = cls.distributionForInstance(instance);

               bundle.putDoubleArray("LMT",distribution4);


               cls = null;
               cls = (Classifier) read(getAssets().open("SVM.model"));

               // Create an empty set
               Instances datasetConfigur;
               datasetConfigur = new Instances("ovariancan.symbolic", attributes, 0);
               datasetConfigur.setClassIndex(11);
               instance.setDataset(datasetConfigur);

               double[] distribution5;
               distribution5 = cls.distributionForInstance(instance);

               bundle.putDoubleArray("SVM",distribution5);
               bundle.putInt("score",count);

               Intent myPred = new Intent(MainActivity.this,predictions.class);

                    myPred.putExtras(bundle);

                    startActivity(myPred);



                    //sending Data to firebase Realtime Database





                    HashMap<String, Object> patientRecord = new HashMap<>();

                    patientRecord.put("Menarche Starts Early?",menarche);
                    patientRecord.put("Oral Contraception?",oral);
                    patientRecord.put("Diet Maintaining?",diet);
                    patientRecord.put("Affected By Breast Cancer?",breast);
                    patientRecord.put("Affected By cervical Cancer?",cervical);
                    patientRecord.put("Cancer History In family?",history);
                    patientRecord.put("Education?",education);
                    patientRecord.put("Age of Husband?",aohusband);
                    patientRecord.put("Menopause End age?",menopause);
                    patientRecord.put("Food contains high fat?",foodfat);
                    patientRecord.put("Abortion?",abortion);

                    String rf = distribution1[0]+"";
                    String j48 = distribution2[0]+"";
                    String nb = distribution3[0]+"";
                    String lmt = distribution4[0]+"";
                    String svm = distribution5[0]+"";
                    String bnet = distribution[0]+"";

                    patientRecord.put("Random Forest",rf);
                    patientRecord.put("Naive Bayes",nb);
                    patientRecord.put("Decision Tree(J48)",j48);
                    patientRecord.put("Bayesian Network",bnet);
                    patientRecord.put("Logistic Model Trees(LMT)",lmt);
                    patientRecord.put("Support Vector Machines(SVM)",svm);

                    //mFirestore.collection("Patients").add(patientRecord);

                    myRef.push().updateChildren(patientRecord);



                }catch(Exception e){
                    System.out.println(e);
                }

            }
        });
    }
}
