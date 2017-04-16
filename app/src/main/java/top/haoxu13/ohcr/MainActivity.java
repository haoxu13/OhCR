package top.haoxu13.ohcr;

import java.io.File;
import java.io.IOException;
//import java.util.Arrays;
import java.lang.*;

import android.support.v7.app.AppCompatActivity;
//import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.database.Cursor;

import com.googlecode.tesseract.android.TessBaseAPI;

import com.theartofdev.edmodo.cropper.CropImageView;
import com.theartofdev.edmodo.cropper.CropImage;

import org.opencv.android.Utils;
//import org.opencv.core.Core;
import org.opencv.core.Mat;
//import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;



public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("opencv_java3");
    }

    protected Button _button;
    protected ImageView _image;
    protected TextView _field;
    protected String _path;
    protected boolean _taken;
    protected Button _ocr_button;
    protected Button _crop_button;
    protected Button _gallery_button;
    protected Button _bi_button;
    protected Button _detect_button;
    protected Button _table_button;
    protected static final String PHOTO_TAKEN = "photo_taken";
    protected int RESULT_LOAD_IMAGE;
    protected Uri _image_uri;
    protected Bitmap _bitmap;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        _image = (ImageView) findViewById(R.id.image);
        _field = (TextView) findViewById(R.id.field);

        _button = (Button) findViewById(R.id.button);
        _button.setOnClickListener(new PhotoButtonClickHandler());

        _ocr_button = (Button) findViewById(R.id.ocr_button);
        _ocr_button.setOnClickListener(new OcrButtonClickHandler());

        _crop_button = (Button) findViewById(R.id.crop_button);
        _crop_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                CropImage.activity(_image_uri)
                        .setGuidelines(CropImageView.Guidelines.ON)
                        .start(MainActivity.this);
            }
        });

        _gallery_button = (Button) findViewById(R.id.gallery_button);
        _gallery_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(i, RESULT_LOAD_IMAGE);
            }
        });

        _bi_button = (Button) findViewById(R.id.bi_button);
        _bi_button.setOnClickListener(new BiButtonClickHandler());

        _detect_button = (Button) findViewById(R.id.detec_button);
        _detect_button.setOnClickListener(new DetectButtonClickHandler());

        _table_button = (Button) findViewById(R.id.table_button);
        _table_button.setOnClickListener(new TableButtonClickHandler());

        File folder = new File(Environment.getExternalStorageDirectory() + "/OhCR");
        if (!folder.exists()) {
            Log.i("MakeMachine", "folder.mkdir()");
            folder.mkdir();
        }
        _path = folder + "/OhCR_photo.jpg";
    }

    public class PhotoButtonClickHandler implements View.OnClickListener {
        public void onClick(View view) {
            Log.i("MakeMachine", "PhotoButtonClickHandler.onClick()");
            startCameraActivity();
        }
    }

    protected void startCameraActivity() {
        Log.i("MakeMachine", "startCameraActivity()");
        File file = new File(_path);
        Uri outputFileUri = Uri.fromFile(file);

        Intent intent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, outputFileUri);

        startActivityForResult(intent, 0);
    }


    public class OcrButtonClickHandler implements View.OnClickListener {
        public void onClick(View view) {
            Log.i("MakeMachine", "OcrButtonClickHandler.onClick()");
            startOcrActivity();
        }
    }

    // Binarize
    public class BiButtonClickHandler implements View.OnClickListener {
        public void onClick(View view) {
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
    }

    // Text Detection
    public class DetectButtonClickHandler implements View.OnClickListener {
        public void onClick(View view) {

            Bitmap bitmap = _bitmap.copy(_bitmap.getConfig(), true);

            Mat imgMAT = new Mat();
            Utils.bitmapToMat(bitmap, imgMAT);
            Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_BGR2GRAY);

            TextDetection my_textdetec = new TextDetection();
            //imgMAT = my_textdetec.SWTtextDetection(imgMAT);
            //imgMAT = my_textdetec.MSER_Detection(imgMAT);
            imgMAT = my_textdetec.swtFindRect_C(imgMAT);
            //imgMAT = my_textdetec.getSWTImage_C(imgMAT);
            //imgMAT = my_textdetec.TestRead_C(imgMAT);

            Utils.matToBitmap(imgMAT, bitmap);

            //TextDetection my_textdetec = new TextDetection();
            //bitmap = my_textdetec.tessDetection(bitmap);

            _image.setImageBitmap(bitmap);

        }
    }


    // Table Detection
    public class TableButtonClickHandler implements View.OnClickListener {
        public void onClick(View view) {
            TableDetection tb = new TableDetection();
            Bitmap bitmap = _bitmap.copy(_bitmap.getConfig(), true);

            Mat imgMAT = new Mat();
            Utils.bitmapToMat(bitmap, imgMAT);
            Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_BGR2GRAY);

            imgMAT = tb.maskLine(imgMAT);

            Utils.matToBitmap(imgMAT, bitmap);

            _image.setImageBitmap(bitmap);
        }
    }


    // OCR
    protected void startOcrActivity() {
        TessBaseAPI baseApi = new TessBaseAPI();
        Log.i("MakeMachine", "TessBaseAPI()");
        baseApi.init(Environment.getExternalStorageDirectory() + "/", "chi_sim", baseApi.OEM_DEFAULT);
        baseApi.setImage(_bitmap);
        long tStart = System.currentTimeMillis();
        String recognizedText = baseApi.getUTF8Text();
        long tEnd = System.currentTimeMillis();
        long tDelta = tEnd - tStart;
        double elapsedSeconds = tDelta / 1000.0;
        baseApi.end();

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

        _field.setVisibility(View.GONE);
    }

}



