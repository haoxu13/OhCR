package top.haoxu13.ohcr;


import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.List.*;

import static top.haoxu13.ohcr.TextDetection.*;

/**
 * Created by Hao on 2017/4/5.
 */

public class TableDetection {

    private class TableLine {
        // 0 -- Horizontal, 1 -- Vertical
        public int type;
        public int length;
        public Point sp;
        public Point ep;
    }

    List<TableLine> row_line;
    List<TableLine> col_line;

    public Mat HoughlineDetect(Mat src) {
        float threshold_low = 50;
        float threshold_high = 200;
        Imgproc.Canny(src, src, threshold_low, threshold_high, 3, false);

        Mat lines = new Mat();

        int threshold = 70;
        int minLineSize = 30;
        int lineGap = 10;

        minLineSize = (int)Math.round(swtWordMedianHeight_C(src) * 1.5);

        Imgproc.HoughLinesP(src, lines, 1, Math.PI / 180, threshold,
                minLineSize, lineGap);

        for (int x = 0; x < lines.cols(); x++) {

            double[] vec = lines.get(0, x);
            double[] val = new double[4];

            val[0] = 0;
            val[1] = ((float) vec[1] - vec[3]) / (vec[0] - vec[2]) * -vec[0] + vec[1];
            val[2] = src.cols();
            val[3] = ((float) vec[1] - vec[3]) / (vec[0] - vec[2]) * (src.cols() - vec[2]) + vec[3];

            lines.put(0, x, val);

        }

        return  lines;
    }

    /**
     * Base on LONG RUN LINE, O(n^2) n = image size
     * @param src
     * @return
     */
    public void detectTable(Mat src) {
        boolean flag = false;
        int threshoLd_width = src.width()/2;
        int threshold_height = src.height()/2;
        int current_len = 0;

        threshold_height = (int)Math.round(swtWordMedianHeight_C(src) * 1.5);

        row_line = new ArrayList<TableLine>();
        col_line = new ArrayList<TableLine>();

        // Horizontal
        Point sp = new Point(), ep = new Point();
        for(int row = 0; row < src.rows(); row++) {
            TableLine rl = new TableLine();
            for (int col = 0; col < src.cols(); col++) {
                if (src.get(row, col)[0] < 255) {
                    if(!flag) {
                        sp.x = col;
                        sp.y = row;
                        rl.sp = sp;
                    }
                    current_len++;
                    if (current_len > threshoLd_width)
                        flag = true;
                } else {
                    current_len = 0;
                    if(flag) {
                        ep.x = col;
                        ep.y = row;
                        rl.ep = ep;
                        rl.type = 0;
                        row_line.add(rl);
                    }
                    flag = false;
                }
            }
        }

        current_len = 0;
        flag = false;

        // Vertical
        for (int col = 0; col < src.cols(); col++) {
            TableLine cl = new TableLine();
            for(int row = 0; row < src.rows(); row++) {
                if (src.get(row, col)[0] < 255) {
                    if(!flag) {
                        sp.x = col;
                        sp.y = row;
                        cl.sp = sp;
                    }
                    current_len++;
                    if (current_len > threshoLd_width)
                        flag = true;
                } else {
                    current_len = 0;
                    if(flag) {
                        ep.x = col;
                        ep.y = row;
                        cl.ep = ep;
                        cl.type = 1;
                        col_line.add(cl);
                    }
                    flag = false;
                }
            }
        }
    }

    private void mergeLine() {
        if(row_line.isEmpty() || row_line.size() == 0 || col_line.isEmpty() || col_line.size() == 0)
            return;

        double threshold = 10;

        List<TableLine> new_rl = new ArrayList<TableLine>();
        List<TableLine> new_cl = new ArrayList<TableLine>();

        new_rl.add(row_line.get(0));
        int rl_counter = 1;
        for(int i = 1; i < row_line.size(); i++) {
            Point p1 = row_line.get(i-1).sp;
            Point p2 = row_line.get(i-1).ep;
            Point p3 = row_line.get(i).sp;
            Point p4 = row_line.get(i).ep;
            double distance1 = Math.hypot(p1.x-p3.x, p1.y-p3.y);
            double distance2 = Math.hypot(p2.x-p4.x, p2.y-p4.y);
            if(distance1 < threshold && distance2 < threshold)
                continue;
            else {
                TableLine e2 = new TableLine();
                e2.sp = p3;
                e2.ep = p4;
                new_rl.add(e2);
            }
        }

        new_cl.add(col_line.get(0));
        int cl_counter = 1;
        for(int i = 1; i < col_line.size(); i++) {
            Point p1 = col_line.get(i-1).sp;
            Point p2 = col_line.get(i-1).ep;
            Point p3 = col_line.get(i).sp;
            Point p4 = col_line.get(i).ep;
            double distance1 = Math.hypot(p1.x-p3.x, p1.y-p3.y);
            double distance2 = Math.hypot(p2.x-p4.x, p2.y-p4.y);
            if(distance1 < threshold && distance2 < threshold)
                continue;
            else {
                TableLine e2 = new TableLine();
                e2.sp = p3;
                e2.ep = p4;
                new_cl.add(e2);
            }
        }


        row_line = new_rl;
        col_line = new_cl;
    }

    private void findCrossPoint (){

    }

    public Mat maskLine(Mat src) {
        detectTable(src);

        Imgproc.cvtColor(src, src, Imgproc.COLOR_GRAY2BGR);

        for(int i = 0; i < row_line.size(); i++) {
            Point p1 = row_line.get(i).sp;
            Point p2 = row_line.get(i).ep;
            Imgproc.line(src, p1, p2, new Scalar(255, 0, 0), 3);
        }

        for(int i = 0; i < col_line.size(); i++) {
            Point p1 = col_line.get(i).sp;
            Point p2 = col_line.get(i).ep;
            Imgproc.line(src, p1, p2, new Scalar(255, 0, 0), 3);
        }

        return src;
    }

}
