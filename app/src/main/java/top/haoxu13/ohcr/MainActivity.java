package top.haoxu13.ohcr;

import java.io.File;
import java.io.IOException;
import java.lang.*;

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

import org.opencv.android.Utils;
import org.opencv.core.Mat;
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
    protected static String recognizedText;

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
                case R.id.navigation_notifications:
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
        protected String doInBackground(Bitmap... bitmap) {
            String result;
            Log.i("Ocr", "OcrButtonClickHandler.onClick()");
            TessBaseAPI baseApi = new TessBaseAPI();
            Log.i("MakeMachine", "TessBaseAPI()");
            baseApi.init(Environment.getExternalStorageDirectory() + "/", "chi_sim", baseApi.OEM_DEFAULT);
            baseApi.setImage(bitmap[0]);
            result = baseApi.getUTF8Text();
            baseApi.end();
            return  result;
        }

        protected void onPostExecute(String result) {
            _editText.setText(result);
        }
    }

    public void startOcrHandler() {
        long tStart = System.currentTimeMillis();

        new OcrTask().execute(_bitmap);

        long tEnd = System.currentTimeMillis();
        long tDelta = tEnd - tStart;
        double elapsedSeconds = tDelta / 1000.0;

        AlertDialog alertDialog = new AlertDialog.Builder(MainActivity.this).create();
        alertDialog.setTitle("Reslut");
        alertDialog.setMessage("time:" + elapsedSeconds + "s\n" + recognizedText);
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
        imgMAT = my_imgproc.AdaptiveBinary(imgMAT);
        //imgMAT = my_imgproc.Binarilze(imgMAT);

        Utils.matToBitmap(imgMAT, bitmap);

            /* Tess
            BinarizeProccess my_imgproc = new BinarizeProccess();
            bitmap = my_imgproc.tessThreshold(bitmap);
            */
        _image.setImageBitmap(bitmap);

        _bitmap = bitmap;
    }

    public void startTextDetection() {
        long tStart = System.currentTimeMillis();

        Bitmap bitmap = _bitmap.copy(_bitmap.getConfig(), true);

        Mat imgMAT = new Mat();
        Utils.bitmapToMat(bitmap, imgMAT);
        Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_BGR2GRAY);

        TextDetection my_textdetec = new TextDetection();
        imgMAT = my_textdetec.swtFindRect_C(imgMAT);

        Utils.matToBitmap(imgMAT, bitmap);

        _image.setImageBitmap(bitmap);

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
    }

    public void startTableDetection() {
        TableDetection tb = new TableDetection();
        Bitmap bitmap = _bitmap.copy(_bitmap.getConfig(), true);

        Mat imgMAT = new Mat();
        Utils.bitmapToMat(bitmap, imgMAT);
        Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_BGR2GRAY);

        imgMAT = tb.HoughlineDetect(imgMAT);

        Utils.matToBitmap(imgMAT, bitmap);

        _image.setImageBitmap(bitmap);
    }

    protected String getOcrResult(Bitmap bitmap) {
        TessBaseAPI baseApi = new TessBaseAPI();
        Log.i("MakeMachine", "TessBaseAPI()");
        baseApi.init(Environment.getExternalStorageDirectory() + "/", "chi_sim", baseApi.OEM_DEFAULT);
        baseApi.setImage(bitmap);
        String recognizedText = baseApi.getUTF8Text();
        baseApi.end();

        return recognizedText;
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
                    _bitmap = Bitmap.createScaledBitmap(_bitmap, _bitmap.getWidth()/2, _bitmap.getHeight()/2, true);
                    _image_uri = resultUri;

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
                startBinarize();
                return true;
            case R.id.action_crop:
                CropImage.activity(_image_uri)
                        .setGuidelines(CropImageView.Guidelines.ON)
                        .start(MainActivity.this);
                return true;
            case R.id.action_ocr:
                startOcrHandler();
                return true;
            case R.id.action_table:
                startTableDetection();
                return true;
            case R.id.action_text:
                startTextDetection();
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }
}



