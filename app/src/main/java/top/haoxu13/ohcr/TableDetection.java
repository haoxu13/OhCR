package top.haoxu13.ohcr;


import org.apache.commons.lang3.tuple.ImmutablePair;
import org.opencv.core.Core;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.Objdetect;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.List.*;

import static top.haoxu13.ohcr.TextDetection.swtWordHeight_C;

/**
 * Created by Hao on 2017/4/5.
 */

public class TableDetection {

    public Mat detectTable_complex(Mat src) {
        Mat origin = src.clone();

        Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 5,5));

        Core.bitwise_not(src, src);

        Mat horizontal = src.clone();
        Mat vertical = src.clone();

        int scale = 4;

        int horizontalsize = horizontal.cols() / scale;

        Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(horizontalsize,1));

        Imgproc.erode(horizontal, horizontal, horizontalStructure);
        Imgproc.dilate(horizontal, horizontal, horizontalStructure);
        Imgproc.threshold(horizontal, horizontal, 100, 255, Imgproc.THRESH_BINARY);
        Imgproc.dilate(horizontal, horizontal, dilateElement);


        int factor = 4;
        if(horizontal.cols()/vertical.rows() > 5)
            factor = 2;
        if(horizontal.cols() < vertical.rows())
            factor = 8;

        int verticalsize = vertical.rows() / factor;
        //int verticalsize = swtWordHeight_C(src);

        Mat verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 1,verticalsize));

        Imgproc.erode(vertical, vertical, verticalStructure);
        Imgproc.dilate(vertical, vertical, verticalStructure);
        Imgproc.threshold(vertical, vertical, 100, 255, Imgproc.THRESH_BINARY);
        Imgproc.dilate(vertical, vertical, dilateElement);

        //
        List<MatOfPoint> horizon_lines = new ArrayList<MatOfPoint>();

        MatOfInt4 hierarchy = new MatOfInt4();
        Imgproc.findContours(horizontal, horizon_lines, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        int min_vertical = Integer.MAX_VALUE;

        for(int i = 0; i < horizon_lines.size()-1; i++) {
            Rect boundRect1;
            boundRect1 = Imgproc.boundingRect(horizon_lines.get(i));

            Rect boundRect2;
            boundRect2 = Imgproc.boundingRect(horizon_lines.get(i+1));

            Point Big_tl, Big_br, Small_tl, Small_br;
            if(boundRect1.br().y < boundRect2.br().y) {
                Small_br = boundRect1.br();
                Big_br = boundRect2.br();
                Small_tl = boundRect1.tl();
                Big_tl = boundRect2.tl();
            } else {
                Small_br = boundRect2.br();
                Big_br = boundRect1.br();
                Small_tl = boundRect2.tl();
                Big_tl = boundRect1.tl();
            }

            verticalsize = (int)(Big_tl.y - Small_br.y);
            verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 1,verticalsize));

            if(verticalsize < min_vertical)
                min_vertical = verticalsize;

            int width = (int)(Big_br.x - Small_tl.x);
            int height = (int)(Big_br.y - Small_tl.y);

            Rect roi_rect = new Rect((int)Small_tl.x,(int)Small_tl.y, width,height);

            Mat ROI = src.submat(roi_rect);
            Mat vertical_roi = vertical.submat(roi_rect);

            Mat small_vertical = new Mat();

            Imgproc.erode(ROI, small_vertical, verticalStructure);
            Imgproc.dilate(small_vertical, small_vertical, verticalStructure);
            Imgproc.dilate(small_vertical, small_vertical, dilateElement);

            Core.add(small_vertical, vertical_roi, vertical_roi);
        }

        Mat mask = new Mat();
        Core.add(horizontal, vertical, mask);


        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Rect> Table_area = new ArrayList<Rect>();
        for(int i = 0; i < contours.size(); i++) {
            Rect boundRect;
            boundRect = Imgproc.boundingRect(contours.get(i));
            Mat sub_area;
            sub_area = origin.submat(boundRect);

            List<MatOfPoint> elements = new ArrayList<MatOfPoint>();
            Imgproc.findContours(sub_area, elements, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            if(elements.size() > 6) {
                if(boundRect.width * boundRect.height < min_vertical * boundRect.width)
                    continue;
                Table_area.add(boundRect);
            }
        }

        Imgproc.cvtColor(origin, origin, Imgproc.COLOR_GRAY2BGR);

        for(int i = 0; i < Table_area.size(); i++) {
            Imgproc.rectangle(origin, Table_area.get(i).tl(), Table_area.get(i).br(), new Scalar(255, 0, 0), 4);
        }

        return origin;
    }


    public Mat detectTable_simple(Mat src) {
        Mat origin = src.clone();

        Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 5,5));

        Core.bitwise_not(src, src);

        Mat horizontal = src.clone();
        Mat vertical = src.clone();

        int scale = 4;

        int horizontalsize = horizontal.cols() / scale;

        Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(horizontalsize,1));

        Imgproc.erode(horizontal, horizontal, horizontalStructure);
        Imgproc.dilate(horizontal, horizontal, horizontalStructure);
        Imgproc.threshold(horizontal, horizontal, 100, 255, Imgproc.THRESH_BINARY);
        Imgproc.dilate(horizontal, horizontal, dilateElement);


        int factor = 4;
        if(horizontal.cols()/vertical.rows() > 5)
            factor = 2;
        if(horizontal.cols() < vertical.rows())
            factor = 8;

        int verticalsize = vertical.rows() / factor;

        Mat verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 1,verticalsize));

        Imgproc.erode(vertical, vertical, verticalStructure);
        Imgproc.dilate(vertical, vertical, verticalStructure);
        Imgproc.threshold(vertical, vertical, 100, 255, Imgproc.THRESH_BINARY);
        Imgproc.dilate(vertical, vertical, dilateElement);

        Mat mask = new Mat();
        Core.add(horizontal, vertical, mask);


        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        MatOfInt4 hierarchy = new MatOfInt4();

        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Rect> Table_area = new ArrayList<Rect>();
        for(int i = 0; i < contours.size(); i++) {
            Rect boundRect;
            boundRect = Imgproc.boundingRect(contours.get(i));
            Mat sub_area;
            sub_area = origin.submat(boundRect);

            List<MatOfPoint> elements = new ArrayList<MatOfPoint>();
            Imgproc.findContours(sub_area, elements, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            if(elements.size() > 6) {
                if(boundRect.width * boundRect.height < boundRect.width * 2)
                    continue;
                Table_area.add(boundRect);
            }
        }

        Imgproc.cvtColor(origin, origin, Imgproc.COLOR_GRAY2BGR);

        for(int i = 0; i < Table_area.size(); i++) {
            Imgproc.rectangle(origin, Table_area.get(i).tl(), Table_area.get(i).br(), new Scalar(255, 0, 0), 4);
        }

        return origin;
    }

    public Mat houghLineDetec(Mat src) {
        //Mat canny = new Mat();
        Mat result = src.clone();
        Mat vertical_lines = new Mat();
        Mat horizontal_lines = new Mat();

        /*
        float threshold_low = 100;
        float threshold_high = 200;
        Imgproc.Canny(src, canny, threshold_low, threshold_high, 3, true);
        */

        Core.bitwise_not(src, src);
        Imgproc.blur(src, src, new Size(11.0, 11.0));

        // C++:  void HoughLinesP(Mat image, Mat& lines, double rho, double theta, int threshold, double minLineLength = 0, double maxLineGap = 0)
        int height = src.height();
        int width = src.width();
        // Vertical lines
        //int vertical_threshold = swtWordHeight_C(src) * 3;

        int vertical_threshold = height/4;
        int horizontal_threshold = width/4;

        Imgproc.HoughLinesP(src, vertical_lines, 1, Math.PI, vertical_threshold, vertical_threshold, 5);
        // Horizontal lines
        Imgproc.HoughLinesP(src, horizontal_lines, 1, Math.PI/2, horizontal_threshold, horizontal_threshold, 5);

        Imgproc.cvtColor(result, result, Imgproc.COLOR_GRAY2BGR);

        for (int x = 0; x < vertical_lines.rows(); x++)
        {
            double[] vec = vertical_lines.get(x, 0);
            double x1 = vec[0],
                    y1 = vec[1],
                    x2 = vec[2],
                    y2 = vec[3];
            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);

            Imgproc.line(result, start, end, new Scalar(255,0,0), 3);

        }

        for (int x = 0; x < horizontal_lines.rows(); x++)
        {
            double[] vec = horizontal_lines.get(x, 0);
            double x1 = vec[0],
                    y1 = vec[1],
                    x2 = vec[2],
                    y2 = vec[3];
            Point start = new Point(x1, y1);
            Point end = new Point(x2, y2);

            Imgproc.line(result, start, end, new Scalar(0,0,255), 3);

        }

        return result;
    }


    // Refer to theodore's
    // http://answers.opencv.org/question/63847/how-to-extract-tables-from-an-image/
    public Mat cleanTable(Mat src) {
        Mat origin = src.clone();

        Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 5,5));

        Core.bitwise_not(src, src);

        Mat horizontal = src.clone();
        Mat vertical = src.clone();

        int scale = 4;
        int horizontalsize = horizontal.cols() / scale;
        Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(horizontalsize,1));

        Imgproc.erode(horizontal, horizontal, horizontalStructure);
        Imgproc.dilate(horizontal, horizontal, horizontalStructure);
        Imgproc.dilate(horizontal, horizontal, dilateElement);


        int factor = 4;
        if(horizontal.cols()/vertical.rows() > 5)
            factor = 2;
        if(horizontal.cols() < vertical.rows())
            factor = 8;

        int verticalsize = vertical.rows() / factor;

        Mat verticalStructure;
        verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 1,verticalsize));
        Imgproc.erode(vertical, vertical, verticalStructure);
        Imgproc.dilate(vertical, vertical, verticalStructure);
        Imgproc.dilate(vertical, vertical, dilateElement);

        List<MatOfPoint> horizon_lines = new ArrayList<MatOfPoint>();

        MatOfInt4 hierarchy = new MatOfInt4();
        Imgproc.findContours(horizontal, horizon_lines, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for(int i = 0; i < horizon_lines.size()-1; i++) {
            Rect boundRect1;
            boundRect1 = Imgproc.boundingRect(horizon_lines.get(i));

            Rect boundRect2;
            boundRect2 = Imgproc.boundingRect(horizon_lines.get(i+1));

            Point Big_tl, Big_br, Small_tl, Small_br;
            if(boundRect1.br().y < boundRect2.br().y) {
                Small_br = boundRect1.br();
                Big_br = boundRect2.br();
                Small_tl = boundRect1.tl();
                Big_tl = boundRect2.tl();
            } else {
                Small_br = boundRect2.br();
                Big_br = boundRect1.br();
                Small_tl = boundRect2.tl();
                Big_tl = boundRect1.tl();
            }

            verticalsize = (int)(Big_tl.y - Small_br.y);
            verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 1,verticalsize));

            int width = (int)(Big_br.x - Small_tl.x);
            int height = (int)(Big_br.y - Small_tl.y);

            Rect roi_rect = new Rect((int)Small_tl.x,(int)Small_tl.y, width,height);

            Mat ROI = src.submat(roi_rect);
            Mat vertical_roi = vertical.submat(roi_rect);

            Mat small_vertical = new Mat();

            Imgproc.erode(ROI, small_vertical, verticalStructure);
            Imgproc.dilate(small_vertical, small_vertical, verticalStructure);
            Imgproc.dilate(small_vertical, small_vertical, dilateElement);

            Core.add(small_vertical, vertical_roi, vertical_roi);
        }

        Mat mask = new Mat();
        Core.add(horizontal, vertical, mask);

        Imgproc.threshold(mask, mask, 100, 255, Imgproc.THRESH_BINARY);

        Core.add(mask, origin, origin);

        return origin;
    }

    public Mat detectTable_swt(Mat src) {
        Mat origin = src.clone();

        Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 5,5));

        Imgproc.cvtColor(src, origin, Imgproc.COLOR_GRAY2RGB);

        Core.bitwise_not(src, src);

        Mat horizontal = src.clone();
        Mat vertical = src.clone();

        int scale = 4;

        int horizontalsize = horizontal.cols() / scale;

        Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(horizontalsize,1));

        Imgproc.erode(horizontal, horizontal, horizontalStructure);
        Imgproc.dilate(horizontal, horizontal, horizontalStructure);
        Imgproc.threshold(horizontal, horizontal, 100, 255, Imgproc.THRESH_BINARY);
        Imgproc.dilate(horizontal, horizontal, dilateElement);


        int verticalsize = swtWordHeight_C(src);

        Mat verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 1,verticalsize));

        Imgproc.erode(vertical, vertical, verticalStructure);
        Imgproc.dilate(vertical, vertical, verticalStructure);
        Imgproc.threshold(vertical, vertical, 100, 255, Imgproc.THRESH_BINARY);
        Imgproc.dilate(vertical, vertical, dilateElement);

        Mat mask = new Mat();
        Core.add(horizontal, vertical, mask);

        return mask;
    }

    public List<Mat> table_element_detect(Mat src) {
        Mat origin = src.clone();

        Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 5,5));

        Core.bitwise_not(src, src);

        Mat horizontal = src.clone();
        Mat vertical = src.clone();

        int scale = 4;

        int horizontalsize = horizontal.cols() / scale;

        Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(horizontalsize,1));

        Imgproc.erode(horizontal, horizontal, horizontalStructure);
        Imgproc.dilate(horizontal, horizontal, horizontalStructure);
        Imgproc.threshold(horizontal, horizontal, 100, 255, Imgproc.THRESH_BINARY);
        Imgproc.dilate(horizontal, horizontal, dilateElement);


        int factor = 4;
        if(horizontal.cols()/vertical.rows() > 5)
            factor = 2;
        if(horizontal.cols() < vertical.rows())
            factor = 8;

        int verticalsize = vertical.rows() / factor;
        //int verticalsize = swtWordHeight_C(src);

        Mat verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 1,verticalsize));

        Imgproc.erode(vertical, vertical, verticalStructure);
        Imgproc.dilate(vertical, vertical, verticalStructure);
        Imgproc.threshold(vertical, vertical, 100, 255, Imgproc.THRESH_BINARY);
        Imgproc.dilate(vertical, vertical, dilateElement);

        //
        List<MatOfPoint> horizon_lines = new ArrayList<MatOfPoint>();

        MatOfInt4 hierarchy = new MatOfInt4();
        Imgproc.findContours(horizontal, horizon_lines, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        int min_vertical = Integer.MAX_VALUE;

        for(int i = 0; i < horizon_lines.size()-1; i++) {
            Rect boundRect1;
            boundRect1 = Imgproc.boundingRect(horizon_lines.get(i));

            Rect boundRect2;
            boundRect2 = Imgproc.boundingRect(horizon_lines.get(i+1));

            Point Big_tl, Big_br, Small_tl, Small_br;
            if(boundRect1.br().y < boundRect2.br().y) {
                Small_br = boundRect1.br();
                Big_br = boundRect2.br();
                Small_tl = boundRect1.tl();
                Big_tl = boundRect2.tl();
            } else {
                Small_br = boundRect2.br();
                Big_br = boundRect1.br();
                Small_tl = boundRect2.tl();
                Big_tl = boundRect1.tl();
            }

            verticalsize = (int)(Big_tl.y - Small_br.y);
            verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size( 1,verticalsize));

            if(verticalsize < min_vertical)
                min_vertical = verticalsize;

            int width = (int)(Big_br.x - Small_tl.x);
            int height = (int)(Big_br.y - Small_tl.y);

            Rect roi_rect = new Rect((int)Small_tl.x,(int)Small_tl.y, width,height);

            Mat ROI = src.submat(roi_rect);
            Mat vertical_roi = vertical.submat(roi_rect);

            Mat small_vertical = new Mat();

            Imgproc.erode(ROI, small_vertical, verticalStructure);
            Imgproc.dilate(small_vertical, small_vertical, verticalStructure);
            Imgproc.dilate(small_vertical, small_vertical, dilateElement);

            Core.add(small_vertical, vertical_roi, vertical_roi);
        }

        Mat mask = new Mat();
        Core.add(horizontal, vertical, mask);


        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Rect> Table_element = new ArrayList<Rect>();
        List<MatOfPoint> element_contours = new ArrayList<MatOfPoint>();

        List<Mat> result = new ArrayList<Mat>();
        Mat result_sub;
        //Imgproc.cvtColor(result, result, Imgproc.COLOR_GRAY2BGR);

        Mat mask_clone = new Mat();
        for(int i = 0; i < contours.size(); i++) {
            mask_clone = mask.clone();

            Rect boundRect;
            boundRect = Imgproc.boundingRect(contours.get(i));
            Mat sub_area = mask_clone.submat(boundRect);

            Core.bitwise_not(sub_area, sub_area);

            Imgproc.erode(sub_area, sub_area, dilateElement);
            Imgproc.erode(sub_area, sub_area, dilateElement);


            List<MatOfPoint> elements = new ArrayList<MatOfPoint>();
            Imgproc.findContours(sub_area, elements, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            for(int j = 0; j < elements.size(); j++)
            {
                Rect elementRect;
                elementRect = Imgproc.boundingRect(elements.get(j));
                if(elementRect.height*elementRect.width > boundRect.height*boundRect.width*0.8 )
                    continue;
                result_sub = origin.submat(boundRect);

                //Table_element.add(elementRect);
                //Imgproc.rectangle(result_sub, elementRect.tl(),  elementRect.br(), new Scalar(255, 0, 0), 3);
                Mat table_element = result_sub.submat(elementRect);
                result.add(table_element.clone());
            }
        }
        return result;
    }

}
