package top.haoxu13.ohcr;

import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;



/**
 * Created by Hao on 2017/6/12.
 */

public class Deskewing {

    public Mat deskew(Mat src) {

        double angle = 0;
        angle = computeSkew(src);

        Mat reverse = src.clone();
        Core.bitwise_not(src, reverse);

        Mat idx = new Mat();
        Core.findNonZero(reverse, idx);

        MatOfPoint2f mop2f = new MatOfPoint2f();
        RotatedRect box;
        idx.copyTo(mop2f);
        box = Imgproc.minAreaRect(mop2f);

        Mat rot_mat = Imgproc.getRotationMatrix2D(box.center, angle, 1);

        Mat rotated = new Mat();

        Imgproc.warpAffine(src, rotated, rot_mat, src.size(), Imgproc.INTER_LANCZOS4);

        //    std::swap(box_size.width, box_size.height);

        Size box_size = box.size;
        if(box.angle < -45)
        {
            double temp = box_size.width;
            box_size.width = box_size.height;
            box_size.height = temp;
        }

        Mat cropped = new Mat();
        Imgproc.getRectSubPix(rotated, box_size, box.center, cropped);

        return src;
    }

    private double computeSkew_Hough(Mat src) {

    }

    private double computeSkew(Mat src) {
        Mat black_white = src.clone();

        Imgproc.threshold(src, black_white, 255, Imgproc.THRESH_OTSU, Imgproc.THRESH_BINARY);
        Core.bitwise_not(black_white, black_white);

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5));
        Imgproc.erode(black_white, black_white, element);


        Mat idx = new Mat();
        Core.findNonZero(black_white, idx);
        MatOfPoint2f mop2f = new MatOfPoint2f();
        RotatedRect box;
        idx.copyTo(mop2f);
        box = Imgproc.minAreaRect(mop2f);

        double angle = box.angle;
        if(angle < -45)
            angle += 90;

        return  angle;
    }
}
