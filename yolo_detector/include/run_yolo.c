#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "run_yolo.h"

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "option_list.h"

#include <time.h>


box* init_boxes(network *net)
{
    layer l = net->layers[net->n-1];
    box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
    
    return boxes;
}

float** init_probs(network *net)
{
	int j;
    layer l = net->layers[net->n-1];
    float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
    
    for(j = 0; j < l.w*l.h*l.n; ++j) 
    {
    	probs[j] = (float*)calloc(l.classes + 1, sizeof(float *));
    }
    
    return probs;
}

//IplImage* image_to_ipl(image p)
//{
//	image copy = copy_image(p);
//	if(p.c == 3) rgbgr_image(copy);
//	int x,y,k;

//	IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
//	int step = disp->widthStep;
//	for(y = 0; y < p.h; ++y){
//	    for(x = 0; x < p.w; ++x){
//	        for(k= 0; k < p.c; ++k){
//	            disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
//	        }
//	    }
//	}
	
//	free_image(copy);
	
//	return disp;
//}

image **load_alphabet_(char* path)
{
    int i, j;
    const int nsize = 8;
    image **alphabets = calloc(nsize, sizeof(image));
    for(j = 0; j < nsize; ++j){
        alphabets[j] = calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i){
            char buff[256];
            sprintf(buff, "%s/labels/%d_%d.png", path, i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}

void draw_people_detections(image im, int num,  box *boxes)
{
	int i;
	for(i = 0; i < num; i++)
    {
		int width = im.h * .012;

        float red = 1;
        float green = 0;
        float blue = 1;
        float rgb[3];

        box b = boxes[i];

        int left  = (b.x-b.w/2.)*im.w;
        int right = (b.x+b.w/2.)*im.w;
        int top   = (b.y-b.h/2.)*im.h;
        int bot   = (b.y+b.h/2.)*im.h;

        if(left < 0) left = 0;
        if(right > im.w-1) right = im.w-1;
        if(top < 0) top = 0;
        if(bot > im.h-1) bot = im.h-1;

        draw_box_width(im, left, top, right, bot, width, red, green, blue);
	}
}

void extractPerson(int imW, int imH, int num, float thresh, box *boxes, float **probs, char **names, int classes, boxInfo *result)
{
	int i;
	int newNum = 0;
	
	//clock_t t;
    //t = clock();
	
    for(i = 0; i < num; i++)
    {
        int classI = max_index(probs[i], classes);
        float prob = probs[i][classI];
        if(prob > thresh)
        {	
	        if (strcmp(names[classI], "person") == 0)
			{
				newNum++;
				
				if(newNum == result->num)
				{
					break;
				}
				
				int j = newNum - 1;
				
				int left  = (boxes[i].x-boxes[i].w/2.)*imW;
				int right = (boxes[i].x+boxes[i].w/2.)*imW;
				int top   = (boxes[i].y-boxes[i].h/2.)*imH;
				int bot   = (boxes[i].y+boxes[i].h/2.)*imH;

				if(left < 0) left = 0;
				if(right > imW-1) right = imW-1;
				if(top < 0) top = 0;
				if(bot > imH-1) bot = imH-1;

				result->boxes[j].x = left;
				result->boxes[j].y = top;
				result->boxes[j].w = right-left;
				result->boxes[j].h = bot-top;
			}
        }
    }
    
    result->num = newNum;
    
    //t = clock() - t;
    //double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
    //printf("took %f seconds to extract person \n", time_taken);
    //clock_t t;
    //t = clock();
    
    
    //printf( "H in extract = %d\n", personBoxes->im->h);
    //printf( "Number of People In Extract = %d\n", newNum);
   // printf( "Size of boxinfo = %d\n", sizeof(boxInfo));
   
    
    
    //t = clock() - t;
    //double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
   // printf("took %f seconds to execute \n", time_taken);
    //free_ptrs((void **)probs, num);
}

void extractPersonv2(int imW, int imH, int num, float thresh, detection *det, char **names, int classes, boxInfo *result)
{
    int i;
    int newNum = 0;

    //clock_t t;
    //t = clock();

    for(i = 0; i < num; i++)
    {
        int classI = max_index(det[i].prob, classes);
        float prob = det[i].prob[classI];
        if(prob > thresh)
        {
            if (strcmp(names[classI], "person") == 0)
            {
                newNum++;

                if(newNum == result->num)
                {
                    break;
                }

                int j = newNum - 1;

                int left  = (det[i].bbox.x-det[i].bbox.w/2.)*imW;
                int right = (det[i].bbox.x+det[i].bbox.w/2.)*imW;
                int top   = (det[i].bbox.y-det[i].bbox.h/2.)*imH;
                int bot   = (det[i].bbox.y+det[i].bbox.h/2.)*imH;

                if(left < 0) left = 0;
                if(right > imW-1) right = imW-1;
                if(top < 0) top = 0;
                if(bot > imH-1) bot = imH-1;

                result->boxes[j].x = left;
                result->boxes[j].y = top;
                result->boxes[j].w = right-left;
                result->boxes[j].h = bot-top;
            }
        }
    }

    result->num = newNum;

    //t = clock() - t;
    //double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
    //printf("took %f seconds to extract person \n", time_taken);
    //clock_t t;
    //t = clock();


    //printf( "H in extract = %d\n", personBoxes->im->h);
    //printf( "Number of People In Extract = %d\n", newNum);
   // printf( "Size of boxinfo = %d\n", sizeof(boxInfo));



    //t = clock() - t;
    //double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
   // printf("took %f seconds to execute \n", time_taken);
    //free_ptrs((void **)probs, num);
}

void run_yolo_detection(image im, network *net, box *boxes, float **probs, float thresh, float hier_thresh, char **names, boxInfo *result)
{
    float nms=.4;
    layer l = net->layers[net->n-1];
    //clock_t t;
    //t = clock();
    
    image sized = resize_image(im, net->w, net->h);


	
    
    float *X = sized.data;
    
    
    network_predict(net, X);
//    get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);

//    if (l.softmax_tree && nms)
//    {
//        do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
//    }
//    else if (nms)
//    {
//        do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
//    }
    
    

    int imW = im.w;
    int imH = im.h;

    free_image(im);
    free_image(sized);
    
    //t = clock() - t;
    //double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
    //printf("took %f seconds to predict yolo \n", time_taken);
   // free_ptrs((void **)probs, l.w*l.h*l.n);
  // printf( "detect layer (layer %d) w = %d h = %d n = %d\n", net.n, l.w, l.h, l.n);
    extractPerson(imW, imH, l.w*l.h*l.n, thresh, boxes, probs, names,  l.classes, result);
}

void run_yolo_detection_v2(image im, network *net, float thresh, float hier_thresh, char **names, boxInfo *result)
{
    float nms=.4;
    layer l = net->layers[net->n-1];
    //clock_t t;
    //t = clock();

    image sized = resize_image(im, net->w, net->h);

    float *X = sized.data;

    network_predict(net, X);

    int nboxes = 0;
    detection *dets = get_network_boxes(net, net->w, net->h, thresh, hier_thresh, 0, 1, &nboxes);
    //printf("nboxes = %d \n", nboxes);
    if (l.softmax_tree && nms)
    {
        do_nms_obj(dets, nboxes, l.classes, nms);
    }
    else if (nms)
    {
        do_nms_sort(dets, nboxes, l.classes, nms);
    }

    int imW = im.w;
    int imH = im.h;

    free_image(im);
    free_image(sized);

    //t = clock() - t;
    //double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
    //printf("took %f seconds to predict yolo \n", time_taken);
   // free_ptrs((void **)probs, l.w*l.h*l.n);
  // printf( "detect layer (layer %d) w = %d h = %d n = %d\n", net.n, l.w, l.h, l.n);
    //extractPerson(imW, imH, l.w*l.h*l.n, thresh, boxes, probs, names,  l.classes, result);
    extractPersonv2(imW, imH, nboxes, thresh, dets, names,  l.classes, result);
    free_detections(dets, nboxes);
}
