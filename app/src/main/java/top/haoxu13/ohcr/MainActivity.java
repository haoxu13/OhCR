package top.haoxu13.ohcr;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.*;
import java.util.List;

import android.content.Context;
import android.os.AsyncTask;
import android.support.annotation.NonNull;
import android.support.design.widget.BottomNavigationView;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.EditText;
import android.widget.ImageView;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.database.Cursor;
import android.widget.ViewFlipper;

import com.googlecode.tesseract.android.TessBaseAPI;

import com.theartofdev.edmodo.cropper.CropImageView;
import com.theartofdev.edmodo.cropper.CropImage;

import org.apache.commons.lang3.StringUtils;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("opencv_java3");
    }

    protected EditText _editText;
    protected ImageView _image;
    protected String _path;
    protected boolean _taken;
    protected static final String PHOTO_TAKEN = "photo_taken";
    protected int RESULT_LOAD_IMAGE;
    protected Uri _image_uri;
    protected static Bitmap _bitmap;
    protected static Bitmap _origin_img;
    protected static String recognizedText;

    private Context mContext;

    private BottomNavigationView.OnNavigationItemSelectedListener mOnNavigationItemSelectedListener
            = new BottomNavigationView.OnNavigationItemSelectedListener() {

        @Override
        public boolean onNavigationItemSelected(@NonNull MenuItem item) {

            ViewFlipper vf = (ViewFlipper)findViewById(R.id.vf);

            switch (item.getItemId()) {
                case R.id.navigation_home:
                    vf.setDisplayedChild(0);
                    return true;
                case R.id.navigation_dashboard:
                    vf.setDisplayedChild(1);
                    return true;
            }
            return false;
        }

    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        BottomNavigationView navigation = (BottomNavigationView) findViewById(R.id.navigation);
        navigation.setOnNavigationItemSelectedListener(mOnNavigationItemSelectedListener);

        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        _image = (ImageView) findViewById(R.id.image);

        File folder = new File(Environment.getExternalStorageDirectory() + "/OhCR");
        if (!folder.exists()) {
            Log.i("MakeMachine", "folder.mkdir()");
            folder.mkdir();
        }
        _path = folder + "/OhCR_photo.jpg";

        _editText = (EditText) findViewById(R.id.edittext);

        mContext = this;

        //Test_Origin();
        //Test_Photo1();
        //getBestResult();
    }

    protected void startCameraActivity() {
        Log.i("MakeMachine", "startCameraActivity()");
        File file = new File(_path);
        Uri outputFileUri = Uri.fromFile(file);

        Intent intent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, outputFileUri);

        startActivityForResult(intent, 0);
    }


    private class OcrTask extends AsyncTask<Bitmap, String, String> {
        Context context;

        public OcrTask(Context mContext) {
            this.context = mContext;
        }

        protected String doInBackground(Bitmap... bitmap) {
            int confidence[];

            String result;
            Log.i("Ocr", "OcrButtonClickHandler.onClick()");
            TessBaseAPI baseApi = new TessBaseAPI();
            Log.i("MakeMachine", "TessBaseAPI()");

            // For most variables, it is wise to set them before calling Init.
            baseApi.setVariable("chop_enable", "T");
            baseApi.setVariable("use_new_state_cost", "F");
            baseApi.setVariable("segment_segcost_rating", "F");
            baseApi.setVariable("enable_new_segsearch", "0");
            baseApi.setVariable("language_model_ngram_on", "0");
            baseApi.setVariable("textord_force_make_prop_words", "F");
            baseApi.setVariable("edges_max_children_per_outline", "40");
            baseApi.setVariable("tessedit_char_blacklist", "|_`/");

            baseApi.init(Environment.getExternalStorageDirectory() + "/", "chi_sim", baseApi.OEM_DEFAULT);

            baseApi.setPageSegMode(3);
            baseApi.setImage(bitmap[0]);

            result = baseApi.getUTF8Text();

            confidence = baseApi.wordConfidences();

            Log.d("Confidence", confidence.toString());

            baseApi.end();
            return  result;
        }

        protected void onPostExecute(String result) {
            _editText.setText(result);

            AlertDialog.Builder goLogin = new AlertDialog.Builder(context);
            goLogin.setMessage("OCR Done");
            goLogin.setCancelable(false);
            goLogin.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int id) {
                    dialog.cancel();
                }
            });
            AlertDialog alertLogin = goLogin.create();
            alertLogin.show();
        }
    }

    public void startOcrHandler() {

        new OcrTask(mContext).execute(_bitmap);

        AlertDialog alertDialog = new AlertDialog.Builder(MainActivity.this).create();
        alertDialog.setTitle("Notice");
        alertDialog.setMessage("Start OCR");
        alertDialog.setButton(AlertDialog.BUTTON_NEUTRAL, "OK",
                new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.dismiss();
                    }
                });
        alertDialog.show();
    }

    public void startBinarize() {
        Bitmap bitmap = _bitmap.copy(_bitmap.getConfig(), true);

        Mat imgMAT = new Mat();
        Utils.bitmapToMat(bitmap, imgMAT);
        Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_BGR2GRAY);

        BinarizeProccess my_imgproc = new BinarizeProccess();

        //imgMAT = my_imgproc.preProcess(imgMAT);
        imgMAT = my_imgproc.AdaptiveBinary(imgMAT);
        //imgMAT = my_imgproc.Percentile_Binarilze(imgMAT);

        Utils.matToBitmap(imgMAT, bitmap);

            /* Tess
            BinarizeProccess my_imgproc = new BinarizeProccess();
            bitmap = my_imgproc.tessThreshold(bitmap);
            */

        _image.setImageBitmap(bitmap);

        _bitmap = bitmap;
    }

    public void startTextDetection() {
        //long tStart = System.currentTimeMillis();

        Bitmap bitmap = _bitmap.copy(_bitmap.getConfig(), true);

        Mat imgMAT = new Mat();
        Utils.bitmapToMat(bitmap, imgMAT);
        Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_BGR2GRAY);

        TextDetection my_textdetec = new TextDetection();
        imgMAT = my_textdetec.swtFindRect_C(imgMAT);

        Utils.matToBitmap(imgMAT, bitmap);

        _image.setImageBitmap(bitmap);


        /*
        long tEnd = System.currentTimeMillis();
        long tDelta = tEnd - tStart;
        double elapsedSeconds = tDelta / 1000.0;
        AlertDialog alertDialog = new AlertDialog.Builder(MainActivity.this).create();
        alertDialog.setTitle("Reslut");
        alertDialog.setMessage("time:" + elapsedSeconds);
        alertDialog.setButton(AlertDialog.BUTTON_NEUTRAL, "OK",
                new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.dismiss();
                    }
                });
        alertDialog.show();
        */
    }

    public void startTableDetection() {

        String result = "";

        Log.i("Ocr", "OcrButtonClickHandler.onClick()");
        TessBaseAPI baseApi = new TessBaseAPI();
        Log.i("MakeMachine", "TessBaseAPI()");

        baseApi.setPageSegMode(7);

        baseApi.init(Environment.getExternalStorageDirectory() + "/", "chi_sim", baseApi.OEM_DEFAULT);
        //
        TableDetection td = new TableDetection();
        Bitmap bitmap = _bitmap.copy(_bitmap.getConfig(), true);


        Mat imgMAT = new Mat();
        Utils.bitmapToMat(bitmap, imgMAT);
        Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_BGR2GRAY);


        List<Mat> elements;
        elements = td.table_element_detect(imgMAT);
        for(int j = 0; j < elements.size(); j++)
        {
            Bitmap new_test_bitmap;
            new_test_bitmap = Bitmap.createBitmap(elements.get(j).width(), elements.get(j).height(), bitmap.getConfig());
            Utils.matToBitmap(elements.get(j), new_test_bitmap);
            baseApi.setImage(new_test_bitmap);

            result = baseApi.getUTF8Text() + result + "\n";
        }
        //
        baseApi.end();


        _editText.setText(result);

        // binarize
        //BinarizeProccess my_imgproc = new BinarizeProccess();
        //imgMAT = my_imgproc.AdaptiveBinary(imgMAT);
        //imgMAT = td.table_element_detect(imgMAT);

        //Utils.matToBitmap(imgMAT, bitmap);

        //_image.setImageBitmap(bitmap);
    }

    public void startTableClean() {
        TableDetection td = new TableDetection();
        Bitmap bitmap = _bitmap.copy(_bitmap.getConfig(), true);

        Mat imgMAT = new Mat();
        Utils.bitmapToMat(bitmap, imgMAT);
        Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_BGR2GRAY);

        // binarize
        //BinarizeProccess my_imgproc = new BinarizeProccess();
        //imgMAT = my_imgproc.AdaptiveBinary(imgMAT);

        imgMAT = td.cleanTable(imgMAT);

        Utils.matToBitmap(imgMAT, bitmap);

        _image.setImageBitmap(bitmap);

        _bitmap = bitmap;
    }

    public void startReset() {
        _bitmap = _origin_img;
        _image.setImageBitmap(_bitmap);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        Log.i("MakeMachine", "resultCode: " + resultCode);
        switch (resultCode) {
            case 0:
                Log.i("MakeMachine", "User cancelled");
                break;
            case -1:
                onPhotoTaken();
                break;
        }
        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };
            Cursor cursor = getContentResolver().query(selectedImage,filePathColumn, null, null, null);
            cursor.moveToFirst();
            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();
            _bitmap = BitmapFactory.decodeFile(picturePath);

            ImageView imageView = (ImageView) findViewById(R.id.image);
            imageView.setImageBitmap(_bitmap);
            _origin_img = _bitmap;
            _path = picturePath;
            _image_uri = selectedImage;
        }
        else if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {
            CropImage.ActivityResult result = CropImage.getActivityResult(data);
            if (resultCode == RESULT_OK) {
                Uri resultUri = result.getUri();
                ImageView imageView = (ImageView) findViewById(R.id.image);
                try {
                    Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), resultUri);

                    _bitmap = bitmap.copy(bitmap.getConfig(), true);
                    _image_uri = resultUri;
                    _origin_img = _bitmap;
                    imageView.setImageBitmap(_bitmap);

                }
                catch (IOException e) {
                }

            } else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE) {
                Exception error = result.getError();
            }
        }
    }

    protected void onPhotoTaken() {
        Log.i("MakeMachine", "onPhotoTaken");

        _taken = true;

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 4;

        Bitmap bitmap = BitmapFactory.decodeFile(_path, options);

        _image.setImageBitmap(bitmap);
        _bitmap = bitmap;
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle item selection
        switch (item.getItemId()) {
            case R.id.action_camera:
                Log.i("MakeMachine", "startCameraActivity");
                startCameraActivity();
                return true;
            case R.id.action_gallery:
                Log.i("MakeMachine", "Gallery");
                Intent i = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(i, RESULT_LOAD_IMAGE);
                return true;
            case R.id.action_binary:
                if(_bitmap != null)
                    startBinarize();
                return true;
            case R.id.action_crop:
                CropImage.activity(_image_uri)
                        .setGuidelines(CropImageView.Guidelines.ON)
                        .start(MainActivity.this);
                return true;
            case R.id.action_ocr:
                if(_bitmap != null)
                    startOcrHandler();
                return true;
            case R.id.action_table:
                if(_bitmap != null)
                    startTableDetection();
                return true;
            case R.id.action_text:
                if(_bitmap != null)
                    startTextDetection();
                return true;
            case R.id.action_Table_Clean:
                if(_bitmap != null)
                    startTableClean();
                return true;
            case R.id.action_reset:
                if(_bitmap != null)
                    startReset();
                return true;

            default:
                return super.onOptionsItemSelected(item);
        }
    }

    protected  void Test_Origin() {
        TableDetection td = new TableDetection();

        long tStart = System.currentTimeMillis();

        File sdcard = Environment.getExternalStorageDirectory();
        int counter = 1;
        int test_amount = 37;

        double similarity;
        double average = 0;

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;

        Log.i("Ocr", "OcrButtonClickHandler.onClick()");
        TessBaseAPI baseApi = new TessBaseAPI();
        Log.i("MakeMachine", "TessBaseAPI()");

        // For most variables, it is wise to set them before calling Init.
        /*
        baseApi.setVariable("chop_enable", "T");
        baseApi.setVariable("use_new_state_cost", "F");
        baseApi.setVariable("segment_segcost_rating", "F");
        baseApi.setVariable("enable_new_segsearch", "0");
        baseApi.setVariable("language_model_ngram_on", "0");
        baseApi.setVariable("textord_force_make_prop_words", "F");
        baseApi.setVariable("edges_max_children_per_outline", "40");
        baseApi.setVariable("tessedit_char_blacklist", "|_`/");
*/
        baseApi.setPageSegMode(7);

        baseApi.init(Environment.getExternalStorageDirectory() + "/", "chi_sim", baseApi.OEM_DEFAULT);


        for(; counter <= test_amount; counter++) {
            String result = "";
            Bitmap test_bitmap = BitmapFactory.decodeFile(sdcard.toString()+"/DCIM/TABLE/T"+Integer.toString(counter)+".png", options);

            // clean table
            Bitmap new_test_bitmap;

            List<Mat> elements;
            Mat imgMAT = new Mat();
            Utils.bitmapToMat(test_bitmap, imgMAT);
            Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_BGR2GRAY);
            //imgMAT = td.cleanTable(imgMAT);
            elements = td.table_element_detect(imgMAT);

            //Utils.matToBitmap(imgMAT, test_bitmap);

            for(int j = 0; j < elements.size(); j++)
            {
                new_test_bitmap = Bitmap.createBitmap(elements.get(j).width(), elements.get(j).height(), test_bitmap.getConfig());
                Utils.matToBitmap(elements.get(j), new_test_bitmap);
                baseApi.setImage(new_test_bitmap);

                result = baseApi.getUTF8Text() + result + "\n";
            }

            // Get Ground Truth

            File testFile = new File(sdcard + "/GroundTruth/", "T" + Integer.toString(counter) + ".txt");

            StringBuilder text = new StringBuilder();

            try {
                BufferedReader br = new BufferedReader(new FileReader(testFile));
                String line;

                while ((line = br.readLine()) != null) {
                    text.append(line);
                    text.append('\n');
                }
                br.close();
            } catch (IOException e) {
            }


            similarity = similarity(text.toString().replaceAll("\\s+", ""), result.replaceAll("\\s+", ""));
            average += similarity;

            Log.i("Test " + Integer.toString(counter), Double.toString(similarity));
        }

        average /= test_amount;
        long tEnd = System.currentTimeMillis();
        long tDelta = tEnd - tStart;

        Log.i("Test Average ", Double.toString(average));
        Log.i("Test Time(s) ", Double.toString(tDelta));


        baseApi.end();
    }

    protected  void Test_Photo1() {
        File sdcard = Environment.getExternalStorageDirectory();
        int counter = 1;
        int test_amount = 37;

        double similarity;
        double average = 0;

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;

        String result;
        Log.i("Ocr", "OcrButtonClickHandler.onClick()");
        TessBaseAPI baseApi = new TessBaseAPI();
        Log.i("MakeMachine", "TessBaseAPI()");

        // For most variables, it is wise to set them before calling Init.
        baseApi.setVariable("chop_enable", "T");
        baseApi.setVariable("use_new_state_cost", "F");
        baseApi.setVariable("segment_segcost_rating", "F");
        baseApi.setVariable("enable_new_segsearch", "0");
        baseApi.setVariable("language_model_ngram_on", "0");
        baseApi.setVariable("textord_force_make_prop_words", "F");
        baseApi.setVariable("edges_max_children_per_outline", "40");
        baseApi.setVariable("tessedit_char_blacklist", "|_`/");

        baseApi.init(Environment.getExternalStorageDirectory() + "/", "chi_sim", baseApi.OEM_DEFAULT);

        baseApi.setPageSegMode(3);

        for(; counter <= test_amount; counter++) {
            Bitmap test_bitmap = BitmapFactory.decodeFile(sdcard.toString()+"/DCIM/TEST_TABLE/T"+Integer.toString(counter)+".png", options);

            // Bi
            Mat imgMAT = new Mat();
            Utils.bitmapToMat(test_bitmap, imgMAT);
            Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_BGR2GRAY);
            BinarizeProccess my_imgproc = new BinarizeProccess();
            imgMAT = my_imgproc.AdaptiveBinary(imgMAT);
            Utils.matToBitmap(imgMAT, test_bitmap);

            baseApi.setImage(test_bitmap);

            result = baseApi.getUTF8Text();

            // Get Ground Truth

            File testFile = new File(sdcard + "/GroundTruth/", "T" + Integer.toString(counter) + ".txt");

            StringBuilder text = new StringBuilder();

            try {
                BufferedReader br = new BufferedReader(new FileReader(testFile));
                String line;

                while ((line = br.readLine()) != null) {
                    text.append(line);
                    text.append('\n');
                }
                br.close();
            } catch (IOException e) {
            }


            similarity = similarity(text.toString().replaceAll("\\s+", ""), result.replaceAll("\\s+", ""));
            average += similarity;

            Log.i("Test " + Integer.toString(counter), Double.toString(similarity));
        }

        average /= test_amount;
        Log.i("Test Average ", Double.toString(average));

        baseApi.end();
    }

    private double similarity(String s1, String s2) {
        String longer = s1, shorter = s2;
        if (s1.length() < s2.length()) { // longer should always have greater length
            longer = s2; shorter = s1;
        }
        int longerLength = longer.length();
        if (longerLength == 0) { return 1.0; /* both strings are zero length */ }

        return (longerLength - StringUtils.getLevenshteinDistance(longer, shorter)) / (double) longerLength;
    }

    private void getBestResult() {
        File sdcard = Environment.getExternalStorageDirectory();

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;

        String result;
        double average = 0;
        double similarity = 0;
        int counter = 1;
        int test_amount = 37;

        int blocksize = 21;
        int best_blocksize = 21;
        double best_avg = 0;

        Log.i("Ocr", "OcrButtonClickHandler.onClick()");
        TessBaseAPI baseApi = new TessBaseAPI();
        Log.i("MakeMachine", "TessBaseAPI()");

        // For most variables, it is wise to set them before calling Init.
        baseApi.setVariable("chop_enable", "T");
        baseApi.setVariable("use_new_state_cost", "F");
        baseApi.setVariable("segment_segcost_rating", "F");
        baseApi.setVariable("enable_new_segsearch", "0");
        baseApi.setVariable("language_model_ngram_on", "0");
        baseApi.setVariable("textord_force_make_prop_words", "F");
        baseApi.setVariable("edges_max_children_per_outline", "40");
        baseApi.setVariable("tessedit_char_blacklist", "|_`/");

        baseApi.init(Environment.getExternalStorageDirectory() + "/", "chi_sim", baseApi.OEM_DEFAULT);

        baseApi.setPageSegMode(3);

        for(; blocksize <= 111; blocksize += 2) {
            counter = 1;
            test_amount = 37;
            average = 0;
            similarity = 0;
            for (; counter <= test_amount; counter++) {
                Bitmap test_bitmap = BitmapFactory.decodeFile(sdcard.toString() + "/DCIM/TEST_TABLE/T" + Integer.toString(counter) + ".png", options);

                // Bi
                Mat imgMAT = new Mat();
                Utils.bitmapToMat(test_bitmap, imgMAT);
                Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_BGR2GRAY);
                BinarizeProccess my_imgproc = new BinarizeProccess();
                imgMAT = my_imgproc.AdaptiveBinary_2(imgMAT, blocksize);
                Utils.matToBitmap(imgMAT, test_bitmap);

                baseApi.setImage(test_bitmap);

                result = baseApi.getUTF8Text();

                // Get Ground Truth

                File testFile = new File(sdcard + "/GroundTruth/", "T" + Integer.toString(counter) + ".txt");

                StringBuilder text = new StringBuilder();

                try {
                    BufferedReader br = new BufferedReader(new FileReader(testFile));
                    String line;

                    while ((line = br.readLine()) != null) {
                        text.append(line);
                        text.append('\n');
                    }
                    br.close();
                } catch (IOException e) {
                }


                similarity = similarity(text.toString().replaceAll("\\s+", ""), result.replaceAll("\\s+", ""));
                average += similarity;

                Log.i("Test " + Integer.toString(counter), Double.toString(similarity));
            }

            average /= test_amount;
            if(average > best_avg) {
                best_avg = average;
                best_blocksize = blocksize;
            }

            Log.i("Test BlockSize " + Integer.toString(blocksize)+ " Average ", Double.toString(average));
        }

        Log.i("Best BlockSize " + Integer.toString(best_blocksize)+ " Best Average ", Double.toString(best_avg));

        baseApi.end();
    }
}



