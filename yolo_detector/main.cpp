#include "opencv2/opencv.hpp"
extern  "C"{
#include "network.h"
}
static network *net;
static float **predictions;
static float *avg;

static float demo_thresh = 0;
static float demo_hier = .5;


image cvmat_to_image(const cv::Mat &src)
{
    unsigned char *data = src.data;
    const int h = src.rows;
    const int w = src.cols;
    const int c = src.channels();
    const int area = w*h;
    image out = make_image(w, h, c);

    for(int i = 0; i < h; ++i){
        for(int j = 0; j < w; ++j){
            for(int k = 0; k < c; ++k, ++data){
                out.data[k*area + i*w + j] = data[0] / 255.;
            }
        }
    }

    return out;
}

int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

//void remember_network(network *net)
//{
//    int i;
//    int count = 0;
//    for(i = 0; i < net->n; ++i){
//        layer l = net->layers[i];
//        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
//            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
//            count += l.outputs;
//        }
//    }
//}

//detection *avg_predictions(network *net, int *nboxes)
//{
//    int i, j;
//    int count = 0;
//    fill_cpu(demo_total, 0, avg, 1);
//    for(j = 0; j < demo_frame; ++j){
//        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
//    }
//    for(i = 0; i < net->n; ++i){
//        layer l = net->layers[i];
//        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
//            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
//            count += l.outputs;
//        }
//    }
//    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
//    return dets;
//}

int main(int argc, char *argv[])
{
    char *cfgfile = "/home/lochappy/workspace/ros/src/open_ptrack/yolo_detector/darknet_opt/cfg/yolov3.cfg";
    char *weightfile = "/home/lochappy/workspace/ros/src/open_ptrack/yolo_detector/darknet_opt/yolov3.weights";

    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    std::cout << net->w << "," << net->h << std::endl << std::flush;

    cv::VideoCapture cap(0);
    while(1){
        cv::Mat mFrame;
        cap >> mFrame;
        if (!mFrame.data) continue;
        cv::resize(mFrame,mFrame,cv::Size(net->w,net->h));

        image im = cvmat_to_image(mFrame);
        rgbgr_image(im);

        //cv::Mat mIM(mFrame.rows*mFrame.channels(),mFrame.cols,CV_32FC1,im.data);
        //cv::imshow("mIM",mIM);

        float nms = .4;

        layer l = net->layers[net->n-1];
        float *X = im.data;
        network_predict(net, X);

        int nboxes = 0;
        detection *dets = get_network_boxes(net, net->w, net->h, demo_thresh, demo_hier, 0, 1, &nboxes);
        std::cout << nboxes << std::endl << std::flush;
        if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);
        std::cout << nboxes << std::endl << std::flush;

        cv::imshow("mFrame",mFrame);
        cv::waitKey(100);

        free_image(im);

    }

    return 1;
}
