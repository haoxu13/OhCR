package top.haoxu13.ohcr;


import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Hao on 2017/4/5.
 */

public class TableDetection {

    List<List<Point>> row_line;
    List<List<Point>> col_line;


    /**
     * Base on LONG RUN LINE
     * @param src
     * @return
     */
    public void detectTable(Mat src) {
        boolean flag = false;
        int threshoLd_width = src.width()/2;
        int threshold_height = src.height()/2;
        int current_len = 0;
        row_line = new ArrayList<List<Point>>();
        col_line = new ArrayList<List<Point>>();

        // Vertical
        for(int row = 0; row < src.cols(); row++) {
            List<Point> rl = new ArrayList<Point>();
            for (int col = 0; col < src.cols(); col++) {
                if (src.get(row, col)[0] < 255) {
                    Point p = new Point();
                    p.x = col;
                    p.y = row;
                    rl.add(p);
                    current_len++;
                    if (current_len > threshoLd_width)
                        flag = true;
                } else {
                    current_len = 0;
                    if(flag) {
                        row_line.add(rl);
                        flag = false;
                    }
                }
            }
        }

        current_len = 0;

        // Horizontal
        for (int col = 0; col < src.cols(); col++) {
            List<Point> cl = new ArrayList<Point>();
            for(int row = 0; row < src.cols(); row++) {
                if (src.get(row, col)[0] < 255) {
                    Point p = new Point();
                    p.x = col;
                    p.y = row;
                    cl.add(p);
                    current_len++;
                    if (current_len > threshold_height)
                        flag = true;
                } else {
                    current_len = 0;
                    if(flag) {
                        col_line.add(cl);
                        flag = false;
                    }
                }
            }
        }
    }

    public Mat maskLine(Mat src) {
        detectTable(src);
        double pix[] = new  double[3];
        pix[0] = 255;
        pix[1] = 0;
        pix[2] = 0;
        Imgproc.cvtColor(src, src, Imgproc.COLOR_GRAY2BGR);

        int x, y;
        for(int i = 0; i < row_line.size(); i++) {
            for(int j = 0; j < row_line.get(i).size(); j++)
            {
                x = (int)row_line.get(i).get(j).x;
                y = (int)row_line.get(i).get(j).y;
                src.put(y, x, pix);
            }
        }

        for(int i = 0; i < col_line.size(); i++) {
            for(int j = 0; j < col_line.get(i).size(); j++)
            {
                x = (int)col_line.get(i).get(j).x;
                y = (int)col_line.get(i).get(j).y;
                src.put(y, x, pix);
            }
        }

        return src;
    }

}
