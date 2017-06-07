package top.haoxu13.ohcr;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Environment;
import android.util.Log;

import com.googlecode.leptonica.android.Pix;
import com.googlecode.leptonica.android.Pixa;
import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Created by Hao on 2017/3/27.
 */

public class BinarizeProccess {

    //      2D PERCENTILE FILTER WINDOW SEARCH
    //      image  - input image
    //      result - output image
    //      P      - percentile
    //      KS     - kernel size(window size KS * KS)
    public Mat percentileFilter(Mat src, int P, int KS) {
        byte buff[] = new byte[src.cols()*src.rows()];
        byte new_buff[] = new byte[src.cols()*src.rows()];
        src.get(0, 0, buff);

        int offset = (KS-1)/2;
        int n = (int)Math.ceil(P*KS*KS/100.0);
        if(n == 0)
            n = 1;

        for(int row = 1; row <= src.rows(); row++) {
            for(int col = 1; col <= src.cols(); col++){

                // boundary
                if(row <= offset || row > src.rows()-offset || col <= offset || col > src.cols()-offset) {
                    new_buff[(row - 1) * src.cols() + col - 1] = buff[(row - 1) * src.cols() + col - 1];
                    continue;
                }

                // pick up window elements
                byte window[] = new byte[KS*KS];
                //  window's row
                for(int j = 1; j <= KS; j++) {
                    // window's col
                    for(int k = 1; k <= KS; k++)
                    {
                        int wrib = row-offset+j-1;
                        int wcib = col-offset+k-1;

                        window[(j-1)*KS+k-1] = buff[(wrib-1)*src.cols()+wcib-1];
                    }
                }
                // sort
                Arrays.sort(window);
                // filter
                new_buff[(row-1)*src.cols()+col-1] = window[n-1];
            }
        }

        src.put(0, 0, new_buff);
        return src;
    }

    // Constant Time Percentile Filter
    // Simon Perreault and Patrick He ́bert
    public Mat fastPercentileFilter(Mat src, int P, int KS) {
        if(P > 100 || P < 0)
            return src;
        if(KS % 2 == 0)
            return src;

        byte buff[] = new byte[src.cols()*src.rows()];
        byte new_buff[];
        src.get(0, 0, buff);

        new_buff = buff;

        // Threshold
        int Amount;
        Amount = (int)Math.ceil(P*KS*KS/100);

        // radius of kernel
        int r;
        r = (KS-1)/2;

        // row
        int M = src.rows();
        // column
        int N = src.cols();
        // default value is 0
        int col_hist[][] = new int[N][256];
        int kernel_hist[] = new int[256];

        for(int i = 0; i < KS; i++)
            for(int j = 0; j < N; j++)
            {
                col_hist[j-1][i]= 0;
            }

        for(int i = 0; i < 256; i++)
            for(int j = 0; j < KS; j++) {
                kernel_hist[i] = 0;
            }

        // init column histogram
        for(int i = 1; i <= KS; i++)
            for(int j = 1; j <= N; j++)
            {
                int pix;
                pix = Byte2Int(buff[(i-1)*N+j-1]);

                col_hist[j-1][pix]++;
            }

        // init kernel histogram
        for(int i = 0; i < 256; i++)
            for(int j = 0; j < KS; j++) {
                kernel_hist[i] += col_hist[j][i];
            }

        // pos
        int pos = 0;
        int counter = 0;
        for(int i = 0; i < 256; i++) {
            counter += kernel_hist[i];
            if(counter > Amount)
            {
                pos = i;
                break;
            }
        }

        // O(1)
        int sum = 0;
        for(int i = 1+r; i <= M-r; i++)
            for(int j = 1+r; j <= N-r; j++)
            {
                new_buff[(i - 1) * N + j - 1] = (byte)kernel_hist[pos];

                if(i < M-r) {
                    if (j != N - r) {
                        for (int k = 0; k < KS; k++) {
                            kernel_hist[k] = kernel_hist[k] - col_hist[j-r-1][k] + col_hist[j+r+1- 1][k];
                        }
                    } else {
                        for (int k = 0; k < 256; k++)
                            for (int z = 0; z < KS; z++) {
                                kernel_hist[k] += col_hist[z][k];
                            }

                    }

                    counter = 0;
                    for(int ii = 0; ii < 256; ii++) {
                        counter += kernel_hist[ii];
                        if(counter > Amount)
                        {
                            pos = ii;
                            break;
                        }
                    }

                    col_hist[j - r - 1][Byte2Int(buff[(i-r-1) * N + j - r - 1])]--;
                    col_hist[j - r - 1][Byte2Int(buff[(i+r+1-1) * N + j - r - 1])]++;

                    if (j == N - r) {
                        for (int k = 1; k <= r + 1; k++) {
                            col_hist[j + k - 1 - 1][Byte2Int(buff[(i - r - 1) * N + j + k - 1 - 1])]--;
                            col_hist[j + k - 1 - 1][Byte2Int(buff[(i + r + 1 - 1) * N + j + k - 1 - 1])]++;
                        }
                    }
                }
            }

        Mat result = src.clone();
        result.put(0, 0, new_buff);
        return result;
    }

    // DOG
    public Mat differenceOfGaussian(Mat src) {
        Mat g1, g2,result;
        g1 = new Mat(src.size(), src.type());
        g2 = new Mat(src.size(), src.type());
        result =  new Mat(src.size(), src.type());
        Imgproc.GaussianBlur(src, g1, new Size(3,3), 0.6);
        Imgproc.GaussianBlur(src, g2, new Size(31,31), 5);
        Core.subtract(g1, g2, result);
        Imgproc.GaussianBlur(result, result, new Size(31,31), 5);

        return result;
    }

    // Dilation
    public Mat dilation(Mat src) {
        byte buff[] = new byte[src.cols()*src.rows()];
        byte new_buff[] = new byte[src.cols()*src.rows()];
        src.get(0, 0, buff);
        Log.d("Dilation", "Dilation");
        double sum = 0;  // sum of all the elements
        for (int i=0; i<src.cols()*src.rows(); i++) {
            sum += buff[i];
        }
        sum = sum / (src.cols()*src.rows());

        for(int row = 1; row <= src.rows(); row++) {
            for(int col = 1; col <= src.cols(); col++){
                if(buff[(row-1)*src.cols()+col-1] > 0.5 * sum)
                    new_buff[(row-1)*src.cols()+col-1] = (byte)255;
                else
                    new_buff[(row-1)*src.cols()+col-1] = 0;
            }
        }
        src.put(0, 0, new_buff);
        Mat element;
        element = Imgproc.getStructuringElement(0, new Size(10,10));
        Imgproc.dilate(src, src, element);
        return src;
    }

    public Mat topHat(Mat src) {
        Core.bitwise_not(src,src);
        Mat element;
        element = Imgproc.getStructuringElement(0, new Size(3,3));
        Imgproc.morphologyEx(src, src, Imgproc.MORPH_TOPHAT,element);

        return src;
    }

    public Mat brightnessAndContrastAuto(Mat src)
    {
        int histSize = 256;
        double alpha, beta;
        double minGray = 0, maxGray = 0;

        //to calculate grayscale histogram
        Mat gray;
        gray = src;
        minGray = Core.minMaxLoc(gray).minVal;
        maxGray = Core.minMaxLoc(gray).maxVal;

        // current range
        double inputRange = maxGray - minGray;

        alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
        beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

        // Apply brightness and contrast normalization
        // convertTo operates with saurate_cast
        src.convertTo(src, -1, alpha, beta);

        return  src;
    }

    public Mat AdaptiveBinary(Mat src)
    {
        Mat result = new Mat();
        /*
        Mat blur = new Mat();
        Imgproc.GaussianBlur(src, blur, new Size(3,3), 0);

        Core.addWeighted(src, 1.5, blur, -0.5, 0, src);
        */

        //Imgproc.medianBlur(src, src, 3);
        Imgproc.adaptiveThreshold(src, result, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 53, 15);
        //Imgproc.adaptiveThreshold(src, result, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 71, 21);


        Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
        Mat element2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5));

        Imgproc.morphologyEx(result, result, Imgproc.MORPH_CLOSE, element1);
        Imgproc.erode(result, result, element1);


        return  result;
    }

    public Mat AdaptiveBinary_2(Mat src, int block_size)
    {
        Mat result = new Mat();
        Imgproc.adaptiveThreshold(src, result, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, block_size, 15);

        Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
        Mat element2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5));

        Imgproc.morphologyEx(result, result, Imgproc.MORPH_CLOSE, element1);
        Imgproc.erode(result, result, element1);

        return  result;
    }

    // Preprocess
    public Mat preProcess(Mat src) {
        /*
        Imgproc.cvtColor(src, src, Imgproc.COLOR_RGB2HSV);
        List<Mat> mat_list = new ArrayList<Mat>();
        Core.split(src, mat_list);
        Imgproc.createCLAHE().apply(mat_list.get(2), mat_list.get(2));
        Core.merge(mat_list, src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_HSV2RGB);
        */

        //
        src.convertTo(src, -1, 1.4, 50);

        Mat blur = new Mat();
        Imgproc.GaussianBlur(src, blur, new Size(3,3), 0);

        Core.addWeighted(src, 1.5, blur, -0.5, 0, src);

        Imgproc.threshold(src, src, Imgproc.THRESH_OTSU, 255, Imgproc.THRESH_BINARY);

        return  src;
    }

    public Mat thresholdPercentile(Mat src, Mat Mask, int KS)
    {
        byte buff[] = new byte[src.cols()*src.rows()];
        byte new_buff[] = new byte[src.cols()*src.rows()];
        byte mask_buff[] = new byte[src.cols()*src.rows()];
        src.get(0, 0, buff);
        Mask.get(0, 0, mask_buff);
        Log.d("threshold", "threshold");
        int sum = 0;
        int counter = 0;
        int max = 0;
        int min = 255;
        for(int row = 1; row <= src.rows(); row++) {
            for(int col = 1; col <= src.cols(); col++){
                if(mask_buff[(row-1)*src.cols()+col-1] == (byte)255) {
                    new_buff[(row - 1) * src.cols() + col - 1] = buff[(row - 1) * src.cols() + col - 1];
                    int value = Byte2Int(buff[(row - 1) * src.cols() + col - 1]);
                    sum += value;
                    counter++;
                    if(value > max)
                        max = value;
                    if(value < min)
                        min = value;

                }
                else
                    new_buff[(row-1)*src.cols()+col-1] = (byte)255;
            }
        }

        int avg = sum / counter;

        for(int row = 1; row <= src.rows(); row++) {
            for(int col = 1; col <= src.cols(); col++){
                if(new_buff[(row - 1) * src.cols() + col - 1] != (byte)255) {
                    //int value = Byte2Int(new_buff[(row - 1) * src.cols() + col - 1]);
                    //int new_value = (int)Math.round((double)(value-min)/(max-min));
                    if(Byte2Int(new_buff[(row - 1) * src.cols() + col - 1]) > 216)
                        new_buff[(row - 1) * src.cols() + col - 1] = 0;
                    else
                        new_buff[(row - 1) * src.cols() + col - 1] = (byte)255;
                }
            }
        }

        src.put(0, 0, new_buff);
        return src;
    }


    // I = (I-Imin)/Imax * Range
    public Mat Normalize(Mat src) {

        int histSize = 255;
        double alpha, beta;
        double minGray = 0, maxGray = 0;

        Mat gray;
        gray = src;
        minGray = Core.minMaxLoc(gray).minVal;
        maxGray = Core.minMaxLoc(gray).maxVal;

        alpha = histSize / maxGray;
        beta = - alpha * minGray;

        src.convertTo(src, -1, alpha, beta);
        return src;
    }


    /**
     * Percentile Filter
     * @param src
     * @return
     */
    public Mat Percentile_Binarilze(Mat src) {
        Mat imgPW = new Mat();
        Mat imgMAT = src;
        //imgMAT = Normalize(imgMAT);
        //imgPW = percentileFilter(imgMAT, 70, 11);
        //imgMAT = imgPW;
        imgMAT = differenceOfGaussian(imgMAT);
        //imgMAT = dilation(imgMAT);
        //imgMAT = thresholdPercentile(imgPW, imgMAT, 3);
        return imgMAT;
    }

    private int Byte2Int(byte x) {
        return x & 0xff;
    }

    public Bitmap tessThreshold(Bitmap bitmap) {
        Pix pix;
        TessBaseAPI baseApi = new TessBaseAPI();
        baseApi.init(Environment.getExternalStorageDirectory() + "/", "chi_sim", baseApi.OEM_DEFAULT);
        baseApi.setImage(bitmap);
        pix = baseApi.getThresholdedImage();

        for(int row = 0; row < bitmap.getHeight(); row++)
            for(int col = 0; col < bitmap.getWidth(); col++)
            {
                bitmap.setPixel(col,row, pix.getPixel(col,row));
            }

        return bitmap;
    }
}
