package top.haoxu13.ohcr;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Vector;

import java.lang.Math;

/**
 * Created by Hao on 2017/4/1.
 */


/*
    Copyright 2012 Andrew Perrault and Saurav Kumar.
*/

public class TextDetection {

    // for debug
    private int counter = 0;

    private class SWTPoint2d  implements Comparator<SWTPoint2d>, Comparable<SWTPoint2d> {
        int x; // col
        int y; // row
        float SWT;

        @Override
        public int compareTo(SWTPoint2d d) {
            if(this.SWT > d.SWT)
                return 1;
            if(this.SWT < d.SWT)
                return -1;
            return 0;
        }

        @Override
        public int compare(SWTPoint2d d, SWTPoint2d d1) {
            if(d.SWT > d1.SWT)
                return 1;
            if(d.SWT < d1.SWT)
                return -1;
            return 0;
        }
    }

    private class SWTPoint2f {
        float x;
        float y;
    }

    private class Ray {
        SWTPoint2d p;
        SWTPoint2d q;
        Vector<SWTPoint2d> points;

    }

    private Mat edgeImage;
    private Mat gradientX;
    private Mat gradientY;
    private Mat gradient;
    private Vector<Ray> rays;
    private Mat SWTImage;

    // Mat edgeImage, Mat gradientX, Mat gradientY, Mat SWTImage, Vector<Ray> rays
    private void strokeWidthTransform() {
        // First pass
        float prec = 0.05f;
        float buff_swt[] = new float[edgeImage.rows()*edgeImage.cols()];
        Arrays.fill(buff_swt, Float.POSITIVE_INFINITY);
        for( int row = 0; row < edgeImage.rows(); row++ ){
            for ( int col = 0; col < edgeImage.cols(); col++ ){
                // edge
                float pixel = (float)(edgeImage.get(row,col)[0]);
                if (pixel > 0) {
                    Ray r = new Ray();
                    SWTPoint2d p = new SWTPoint2d();
                    p.x = col;
                    p.y = row;
                    r.p = p;
                    Vector<SWTPoint2d> points = new Vector<SWTPoint2d>();
                    points.add(p);

                    float curX = (float)col + 0.5f;
                    float curY = (float)row + 0.5f;
                    int curPixX = col;
                    int curPixY = row;
                    float G_x = (float)gradientX.get(row, col)[0];
                    float G_y = (float)gradientY.get(row, col)[0];

                    // normalize gradient
                    float mag = (float)Math.sqrt( (G_x * G_x) + (G_y * G_y) );
                    G_x = -G_x/mag;
                    G_y = -G_y/mag;


                    if(G_x == 0 && G_y == 0) {
                        counter++;
                        Log.d("Oach..", "Zero "+String.valueOf(counter));
                        continue;
                    }

                    if(Float.isNaN(G_x)) {
                        counter++;
                        Log.d("Oach..", "Gx "+String.valueOf(counter));
                        continue;
                    }

                    if(Float.isNaN(G_y)) {
                        counter++;
                        Log.d("Oach..", "Gy "+String.valueOf(counter));
                        continue;
                    }
                    while (true) {
                        curX += G_x*prec;
                        curY += G_y*prec;
                        if ((int)(Math.floor(curX)) != curPixX || (int)(Math.floor(curY)) != curPixY) {
                            curPixX = (int)(Math.floor(curX));
                            curPixY = (int)(Math.floor(curY));
                            // check if pixel is outside boundary of image
                            if (curPixX < 0 || (curPixX >= SWTImage.cols()) || curPixY < 0 || (curPixY >= SWTImage.rows())) {
                                break;
                            }
                            SWTPoint2d pnew = new SWTPoint2d();
                            pnew.x = curPixX;
                            pnew.y = curPixY;
                            points.add(pnew);

                            if ((float)edgeImage.get(curPixY, curPixX)[0] > 0) {
                                r.q = pnew;
                                // dot product
                                float G_xt = (float)gradientX.get(curPixY, curPixX)[0];
                                float G_yt = (float)gradientY.get(curPixY, curPixX)[0];
                                if(G_x == 0 && G_y == 0) {
                                    counter++;
                                    Log.d("Oach..", "Zero "+String.valueOf(counter));
                                    break;
                                }

                                if(Float.isNaN(G_x)) {
                                    counter++;
                                    Log.d("Oach..", "Gx "+String.valueOf(counter));
                                    break;
                                }

                                if(Float.isNaN(G_y)) {
                                    counter++;
                                    Log.d("Oach..", "Gy "+String.valueOf(counter));
                                    break;
                                }
                                mag = (float)Math.sqrt( (G_xt * G_xt) + (G_yt * G_yt) );
                                G_xt = -G_xt / mag;
                                G_yt = -G_yt / mag;

                                if (Math.acos(G_x * -G_xt + G_y * -G_yt) < Math.PI/2.0 ) {
                                    float length = (float)Math.sqrt( ((float)r.q.x - (float)r.p.x)*((float)r.q.x - (float)r.p.x) + ((float)r.q.y - (float)r.p.y)*((float)r.q.y - (float)r.p.y));
                                    for(int i = 0; i < points.size(); i++) {
                                        float pixValue = (float)SWTImage.get(points.elementAt(i).y, points.elementAt(i).x)[0];
                                        if(Float.isInfinite(pixValue))
                                            buff_swt[points.elementAt(i).y*edgeImage.cols()+points.elementAt(i).x] = (float)length;
                                        else
                                            buff_swt[points.elementAt(i).y*edgeImage.cols()+points.elementAt(i).x] = Math.min((float)length, buff_swt[points.elementAt(i).y*edgeImage.cols()+points.elementAt(i).x]);
                                        Log.d("buff_swt",  String.valueOf(buff_swt[points.elementAt(i).y*edgeImage.cols()+points.elementAt(i).x]));
                                    }
                                    r.points = points;
                                    rays.add(r);
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
        Log.d("SWT BUFFER", buff_swt.toString());
        SWTImage.put(0, 0, buff_swt);
    }

    // Step 2 in SWT
    private void SWTMedianFilter(Mat SWTImage, Vector<Ray> rays) {
        for (int rit = 0; rit < rays.size(); rit++) {
            for (int pit = 0; pit < rays.elementAt(rit).points.size(); pit++) {
                SWTPoint2d point = new SWTPoint2d();
                point = rays.get(rit).points.get(pit);
                rays.get(rit).points.get(pit).SWT = (float)SWTImage.get(point.y, point.x)[0];
            }
        }

        for (int rit = 0; rit < rays.size(); rit++)
            Collections.sort(rays.get(rit).points, new SWTPoint2d());

        for (int rit = 0; rit < rays.size(); rit++) {
            int index = (int)Math.floor(rays.get(rit).points.size()/2);
            float median = rays.get(rit).points.get(index).SWT;
            for (int pit = 0; pit < rays.elementAt(rit).points.size(); pit++) {
                SWTPoint2d point = new SWTPoint2d();
                point = rays.get(rit).points.get(pit);
                SWTImage.put(point.y, point.x, Math.min(point.SWT, median));
            }
        }
    }

    // Return Masked Image
    // Dark On Light Only
    public Mat SWTtextDetection(Mat src) {

        // Create Canny Image
        Mat result = new Mat();
        float threshold_low = 140;
        float threshold_high = 255;
        edgeImage = src;
        Imgproc.Canny(src, edgeImage, threshold_low, threshold_high, 3, false);
        Imgproc.GaussianBlur(edgeImage, edgeImage, new Size(5, 5), 0);
        Imgproc.threshold(edgeImage, edgeImage, 0, 255, Imgproc.THRESH_BINARY_INV);

        //return edgeImage;

        // Create gradient X, gradient Y
        Mat gaussianImage = new Mat(src.size(), CvType.CV_32FC1);
        src.convertTo(gaussianImage, CvType.CV_32FC1, 1f/255f);
        Imgproc.GaussianBlur(gaussianImage, gaussianImage, new Size(5, 5), 0);
        gradientX = new Mat(src.size(), CvType.CV_32FC1);
        gradientY = new Mat(src.size(), CvType.CV_32FC1);
        Imgproc.Scharr(gaussianImage, gradientX, -1, 1, 0);
        Imgproc.Scharr(gaussianImage, gradientY, -1, 0, 1);
        Imgproc.GaussianBlur(gradientX, gradientX, new Size(3, 3), 0);
        Imgproc.GaussianBlur(gradientY, gradientY, new Size(3, 3), 0);

        // Calculate SWT
        SWTImage = new Mat(src.size(), CvType.CV_32FC1);
        rays = new Vector<Ray>();
        strokeWidthTransform();
        SWTMedianFilter ( SWTImage, rays );

        Log.d("Ray", rays.toString());

        Core.convertScaleAbs(SWTImage, SWTImage);

        return SWTImage;

        // Calculate legally connect components from SWT and gradient image.
        // return type is a vector of vectors, where each outer vector is a component and
        // the inner vector contains the (y,x) of each pixel in that component.
        /*
        std::vector<std::vector<SWTPoint2d> > components = findLegallyConnectedComponents(SWTImage, rays);

        // Filter the components
        std::vector<std::vector<SWTPoint2d> > validComponents;
        std::vector<SWTPointPair2d > compBB;
        std::vector<Point2dFloat> compCenters;
        std::vector<float> compMedians;
        std::vector<SWTPoint2d> compDimensions;
        filterComponents(SWTImage, components, validComponents, compCenters, compMedians, compDimensions, compBB );

        Mat output3( input.size(), CV_8UC3 );
        renderComponentsWithBoxes (SWTImage, validComponents, compBB, output3);
        imwrite ( "components.png",output3);
        //

        // Make chains of components
        std::vector<Chain> chains;
        chains = makeChains(input, validComponents, compCenters, compMedians, compDimensions, compBB);

        Mat output4( input.size(), CV_8UC1 );
        renderChains ( SWTImage, validComponents, chains, output4 );
        //imwrite ( "text.png", output4);

        Mat output5( input.size(), CV_8UC3 );
        cvtColor (output4, output5, CV_GRAY2RGB);
        */
    }

    // An attempt to detect text area via Contours
    public Mat areaDetection(Mat src) {
        float threshold_low = 140;
        float threshold_high = 255;
        edgeImage = src;
        Imgproc.Canny(src, edgeImage, threshold_low, threshold_high, 3, false);
        Imgproc.GaussianBlur(edgeImage, edgeImage, new Size(5, 5), 0);

        Rect rect = new Rect();
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        List<MatOfPoint> lines = new ArrayList<MatOfPoint>();

        MatOfInt4 hierarchy = new MatOfInt4();
        Imgproc.findContours( edgeImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // contours bubble sort
        // be careful! y is row, x is col
        int lines_num = 0;
        for(int i = 0; i < contours.size(); i++) {
            rect = Imgproc.boundingRect(contours.get(i));
            int curX = (int)rect.br().x;
            int curY = (int)rect.tl().y;
            int distY = (int)rect.br().y;
            for(int col = curY; col <= distY; col++){
                if(lines_num == 0) {
                    Point p_tl = new Point();
                    Point p_br = new Point();

                }
            }
            Imgproc.rectangle(edgeImage, rect.tl(), rect.br(), new Scalar(0, 0, 255));
        }

        Imgproc.threshold(edgeImage, edgeImage, 0, 255, Imgproc.THRESH_BINARY_INV);
        Imgproc.cvtColor(edgeImage, edgeImage, Imgproc.COLOR_GRAY2BGR);

        for(int i = 0; i < contours.size(); i++) {
            rect = Imgproc.boundingRect(contours.get(i));

            Imgproc.rectangle(edgeImage, rect.tl(), rect.br(), new Scalar(0, 0, 255));
        }

        return  edgeImage;
    }
}