#include <iostream>
#include <time.h>
#include <opencv\cxcore.h>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\gpu\gpu.hpp>
#include <omp.h>
#include "ctmf.h"

using namespace std;
using namespace cv;
using namespace gpu;

double **mag_l,**mag_r;
int **mage_l,**mage_r;
double mag_max_l,mag_max_r;
#define QX_DEF_CHAR_MAX 255
#define QX_DEF_DOUBLE_MAX				1.7E+308
float**m_second_derivative_left,**m_second_derivative_right,**m_second_derivative_shifted;
unsigned char***m_image_shifted,***m_image_temp;
clock_t start,finish;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////底层函数，没有争议的函数
inline unsigned char** qx_allocu(int r,int c)
{
	unsigned char *a,**p;
	a=(unsigned char*) malloc(sizeof(unsigned char)*(r*c));
	if(a==NULL) {printf("qx_allocu() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(unsigned char**) malloc(sizeof(unsigned char*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void qx_freeu_4(unsigned char ****p)
{
	if(p!=NULL)
	{
		free(p[0][0][0]);
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline void qx_freed_4(double ****p)
{
	
	if(p!=NULL)
	{
		free(p[0][0][0]);
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline void qx_freed_3(double ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline void qx_freei_3(int ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline double *** qx_allocd_3(int n,int r,int c,int padding=10)
{
	double *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(double*) malloc(sizeof(double)*(n*rc+padding));
	if(a==NULL) {printf("qx_allocd_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(double**) malloc(sizeof(double*)*n*r);
    pp=(double***) malloc(sizeof(double**)*n);
    for(i=0;i<n;i++) 
        for(j=0;j<r;j++) 
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++) 
        pp[i]=&p[i*r];
    return(pp);
}
inline int *** qx_alloci_3(int n,int r,int c,int padding=10)
{
	int *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(int*) malloc(sizeof(int)*(n*rc+padding));
	if(a==NULL) {printf("qx_allocd_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(int**) malloc(sizeof(int*)*n*r);
    pp=(int***) malloc(sizeof(int**)*n);
    for(i=0;i<n;i++) 
        for(j=0;j<r;j++) 
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++) 
        pp[i]=&p[i*r];
    return(pp);
}
inline unsigned char**** qx_allocu_4(int t,int n,int r,int c,int padding=10)
{
	unsigned char *a,**p,***pp,****ppp;
    int nrc=n*r*c,nr=n*r,rc=r*c;
    int i,j,k;
	a=(unsigned char*) malloc(sizeof(unsigned char)*(t*nrc+padding));
	if(a==NULL) {printf("qx_allocu_4() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(unsigned char**) malloc(sizeof(unsigned char*)*t*nr);
    pp=(unsigned char***) malloc(sizeof(unsigned char**)*t*n);
    ppp=(unsigned char****) malloc(sizeof(unsigned char***)*t);
    for(k=0;k<t;k++)
        for(i=0;i<n;i++)
            for(j=0;j<r;j++)
                p[k*nr+i*r+j]=&a[k*nrc+i*rc+j*c];
    for(k=0;k<t;k++)
        for(i=0;i<n;i++)
            pp[k*n+i]=&p[k*nr+i*r];
    for(k=0;k<t;k++)
        ppp[k]=&pp[k*n];
    return(ppp);
}

inline double**** qx_allocd_4(int t,int n,int r,int c,int padding=10)
{
	double *a,**p,***pp,****ppp;
    int nrc=n*r*c,nr=n*r,rc=r*c;
    int i,j,k;
	a=(double*) malloc(sizeof(double)*(t*nrc+padding));
	if(a==NULL) {printf("qx_allocd_4() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(double**) malloc(sizeof(double*)*t*nr);
    pp=(double***) malloc(sizeof(double**)*t*n);
    ppp=(double****) malloc(sizeof(double***)*t);
    for(k=0;k<t;k++)
        for(i=0;i<n;i++)
            for(j=0;j<r;j++)
                p[k*nr+i*r+j]=&a[k*nrc+i*rc+j*c];
    for(k=0;k<t;k++)
        for(i=0;i<n;i++)
            pp[k*n+i]=&p[k*nr+i*r];
    for(k=0;k<t;k++)
        ppp[k]=&pp[k*n];
    return(ppp);
}
inline unsigned char *** qx_allocu_3(int n,int r,int c,int padding=10)
{
	unsigned char *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(unsigned char*) malloc(sizeof(unsigned char )*(n*rc+padding));
	if(a==NULL) {printf("qx_allocu_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(unsigned char**) malloc(sizeof(unsigned char*)*n*r);
    pp=(unsigned char***) malloc(sizeof(unsigned char**)*n);
    for(i=0;i<n;i++) 
        for(j=0;j<r;j++) 
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++) 
        pp[i]=&p[i*r];
    return(pp);
}
inline void qx_freeu_3(unsigned char ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline float *** qx_allocf_3(int n,int r,int c,int padding=10)
{
	float *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(float*) malloc(sizeof(float)*(n*rc+padding));
	if(a==NULL) {printf("qx_allocf_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(float**) malloc(sizeof(float*)*n*r);
    pp=(float***) malloc(sizeof(float**)*n);
    for(i=0;i<n;i++) 
        for(j=0;j<r;j++) 
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++) 
        pp[i]=&p[i*r];
    return(pp);
}
inline void qx_freef_3(float ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}

void ctmf_img(unsigned char*** img,int hL,int wL)
{
	unsigned char*image_in;
	unsigned char*image_out;

	image_in=(unsigned char*) malloc(sizeof(unsigned char)*(hL*wL*3));
	image_out=(unsigned char*) malloc(sizeof(unsigned char)*(hL*wL*3));

	for(int h=0;h<hL;h++)
		for(int w=0;w<wL;w++)
		{
			int index=h*wL*3+w*3;
			image_in[index]=img[h][w][0];
			image_in[index+1]=img[h][w][1];
			image_in[index+2]=img[h][w][2];
		}

	ctmf(image_in,image_out,wL,hL,wL*3,wL*3,1,3,hL*wL*3);
	

	for(int h=0;h<hL;h++)
		for(int w=0;w<wL;w++)
		{
			int index=h*wL*3+w*3;
			img[h][w][0]=image_out[index];
			img[h][w][1]=image_out[index+1];
			img[h][w][2]=image_out[index+2];
		}

	free(image_in);
	image_in=NULL;
	free(image_out);
	image_out=NULL;

}

inline unsigned char rgb_2_gray(unsigned char*in){return(unsigned char(0.299*in[0]+0.587*in[1]+0.114*in[2]+0.5));}
inline void image_copy(unsigned char**out,unsigned char**in,int h,int w){memcpy(out[0],in[0],sizeof(unsigned char)*h*w);}
inline void qx_memcpy_u3(unsigned char a[3],unsigned char b[3]){*a++=*b++; *a++=*b++; *a++=*b++;}


void ToPoint(IplImage *imL,IplImage *imR,int hL,int wL,unsigned char***image_left,unsigned char***image_right)
{
	for(int x=0;x<hL;x++)
	{
		for(int y=0;y<wL;y++)
		{
			for(int c=0;c<3;c++)
			{
				CvScalar p;
				CvScalar q;
				p=cvGet2D(imL,x,y);
				image_left[x][y][0]=p.val[2];
				image_left[x][y][1]=p.val[1];
				image_left[x][y][2]=p.val[0];
				q=cvGet2D(imR,x,y);
				image_right[x][y][0]=q.val[2];
				image_right[x][y][1]=q.val[1];
				image_right[x][y][2]=q.val[0];
			}
		}
	}
}
inline unsigned char euro_dist_rgb_max(unsigned char *a,unsigned char *b) {unsigned char x,y,z; x=abs(a[0]-b[0]); y=abs(a[1]-b[1]); z=abs(a[2]-b[2]); return(max(max(x,y),z));}
double get_edge(int y1,int x1,int y2, int x2,int flag,double sigma_edge)
{
	
	double edge_weight=1;
	if(flag==0)
	{
		if(mage_l[y1][x1]!=mage_l[y2][x2])
		{
			double temp=mag_l[y1][x1]+mag_l[y2][x2];
			edge_weight=exp(-temp/(sigma_edge*mag_max_l));
		}
	}else
	{
		if(mage_r[y1][x1]!=mage_r[y2][x2])
		{
			double temp=mag_r[y1][x1]+mag_r[y2][x2];
			edge_weight=exp(-temp/(sigma_edge*mag_max_r));
		}	
	}
	return edge_weight;
}
void first_order_recursive_bilateral_filter(unsigned char**disparity,double***out,double***in,unsigned char***texture,double sigma_spatial,double sigma_range,int h,int w,int nr_channel,double***temp,double***temp_2w)
{
	double yp[100];
	double alpha=exp(-sqrt(2.0)/(sigma_spatial*w));//filter kernel size
	double inv_alpha=(1-alpha);
	double range_table[QX_DEF_CHAR_MAX+1];
	for(int i=0;i<=QX_DEF_CHAR_MAX;i++) range_table[i]=exp(-double(i)/(sigma_range*QX_DEF_CHAR_MAX));
	
	
	double***in_=in;/*horizontal filtering*/
	//double***out_=temp;
	double***out_=temp;
	for(int y=0;y<h;y++)
	{
		memcpy(out_[y][0],in_[y][0],sizeof(double)*nr_channel);
		memcpy(yp,out_[y][0],sizeof(double)*nr_channel);
		unsigned char*tc=texture[y][1],*tp=texture[y][0];
		for(int x=1;x<w;x++) 
		{
			double weight=range_table[euro_dist_rgb_max(texture[y][x],texture[y][x-1])];
			double alpha_=weight*alpha;
			for(int c=0;c<nr_channel;c++) 
			{
				double ycc;
				out_[y][x][c]=ycc=inv_alpha*in_[y][x][c]+alpha_*yp[c];
				yp[c]=ycc;
			}
		}
		int w1=w-1;
		for(int c=0;c<nr_channel;c++) out_[y][w1][c]=0.5*(out_[y][w1][c]+in_[y][w1][c]);
		memcpy(yp,out_[y][w1],sizeof(double)*nr_channel);
		for(int x=w-2;x>=0;x--) 
		{
			double weight=range_table[euro_dist_rgb_max(texture[y][x],texture[y][x+1])];
			double alpha_=weight*alpha;
			for(int c=0;c<nr_channel;c++) 
			{
				double ycc=inv_alpha*in_[y][x][c]+alpha_*yp[c];
				out_[y][x][c]=0.5*(out_[y][x][c]+ycc);
				yp[c]=ycc;
			}
		}
	}
	in_=temp;/*vertical filtering*/
	alpha=exp(-sqrt(2.0)/(sigma_spatial*h));//filter kernel size
	inv_alpha=(1-alpha);
	double**ycy,**ypy,**xcy,**xpy;
	unsigned char**tcy,**tpy;
	memcpy(out[0][0],in_[0][0],sizeof(double)*w*nr_channel);
	for(int y=1;y<h;y++)
	{
		tpy=texture[y-1];
		tcy=texture[y];
		xcy=in_[y];
		ypy=out[y-1];
		ycy=out[y];
		for(int x=0;x<w;x++)
		{
			double weight=range_table[euro_dist_rgb_max(tcy[x],tpy[x])];
			double alpha_=weight*alpha;
			for(int c=0;c<nr_channel;c++) ycy[x][c]=inv_alpha*xcy[x][c]+alpha_*ypy[x][c];
		}
	}
	int h1=h-1;
	ycy=temp_2w[0];
	ypy=temp_2w[1];
	memcpy(ypy[0],in_[h1][0],sizeof(double)*w*nr_channel);
	for(int x=0;x<w;x++) 
	{
		unsigned char disp=0; double min_cost=0.5*(out[h1][x][0]+ypy[x][0]);
		for(int c=1;c<nr_channel;c++)
		{
			double cost=0.5*(out[h1][x][c]+ypy[x][c]);
			if(cost<min_cost)
			{
				min_cost=cost;
				disp=c;
			}
		}
		disparity[h1][x]=disp;
	}
	for(int y=h1-1;y>=0;y--)
	{
		tpy=texture[y+1];
		tcy=texture[y];
		xcy=in_[y];
		for(int x=0;x<w;x++)
		{
			double weight=range_table[euro_dist_rgb_max(tcy[x],tpy[x])];
			double alpha_=weight*alpha;
			unsigned char disp=0; double min_cost=QX_DEF_DOUBLE_MAX;
			for(int c=0;c<nr_channel;c++) 
			{
				ycy[x][c]=inv_alpha*xcy[x][c]+alpha_*ypy[x][c];
				double cost=0.5*(out[y][x][c]+ycy[x][c]);
				if(cost<min_cost)
				{
					min_cost=cost;
					disp=c;
				}
			}
			disparity[y][x]=disp;
		}
		memcpy(ypy[0],ycy[0],sizeof(double)*w*nr_channel);
	}
}
void Recursive_tf(unsigned char**disparity,double***out,double***in,unsigned char***texture,double sigma_spatial,double sigma_range,int h,int w,int nr_channel,double***temp,double***out1,double***out2,int flag,double sigma_edge)
{
	///////////////////////////////////////////////////////////////////////////////这是按照RecursiveBF的算法，正向进行然后反向进行的结果
	double yp[100],yp1[100];
	double alpha=exp(-sqrt(2.0)/(sigma_spatial*w));//filter kernel size
	double inv_alpha=(1-alpha);
	double range_table[QX_DEF_CHAR_MAX+1];
	for(int i=0;i<=QX_DEF_CHAR_MAX;i++) range_table[i]=exp(-double(i)/(sigma_range*QX_DEF_CHAR_MAX));
	
	double***in_=in;/*horizontal filtering*/
	double***out_=temp;

	for(int y=0;y<h;y++)
	{
		memcpy(out_[y][0],in_[y][0], sizeof (double )*nr_channel);
        memcpy(out1[y][0],in_[y][0], sizeof (double )*nr_channel);
        
		for(int x=1;x<w;x++) 
		{
			double weight=range_table[euro_dist_rgb_max(texture[y][x],texture[y][x-1])];
			double edge=get_edge(y,x,y,x-1,flag,sigma_edge);
			double alpha_=weight*alpha;
			double alpha_2=sqrt(weight*alpha*edge);
			for(int c=0;c<nr_channel;c++) 
			{
				out_[y][x][c]=inv_alpha*in_[y][x][c]+alpha_*out_[y][x-1][c];
                out1[y][x][c]=inv_alpha*in_[y][x][c]+alpha_2*out1[y][x-1][c];
			}
		}
		int w1=w-1;
		for(int c=0;c<nr_channel;c++) 
		{
			out_[y][w1][c]=0.5*(out_[y][w1][c]+in_[y][w1][c]);
			out1[y][w1][c]=0.5*(out1[y][w1][c]+in_[y][w1][c]);
		}
		memcpy(yp,out_[y][w1],sizeof(double)*nr_channel);
		memcpy(yp1,out1[y][w1],sizeof(double)*nr_channel);
		for(int x=w-2;x>=0;x--) 
		{
			double weight=range_table[euro_dist_rgb_max(texture[y][x],texture[y][x+1])];
			double edge=get_edge(y,x,y,x+1,flag,sigma_edge);
            double alpha_=weight*alpha;
            double alpha_2=sqrt (weight*alpha*edge);

			for(int c=0;c<nr_channel;c++) 
			{
				yp[c]=inv_alpha*in_[y][x][c]+alpha_*yp[c];
				yp1[c]=inv_alpha*in_[y][x][c]+alpha_2*yp1[c];
				out_[y][x][c]=0.5*(out_[y][x][c]+yp[c]);
				out1[y][x][c]=0.5*(out1[y][x][c]+yp1[c]);
			}
		}
	}


	in_=out_;/*vertical filtering*/
	double***in_2=out1;
	alpha=exp(-sqrt(2.0)/(sigma_spatial*h));//filter kernel size
	inv_alpha=(1-alpha);

	for(int x=0;x<w;x++)
	{
		memcpy(out[0][x],in_[0][x],sizeof(double)*nr_channel);
		memcpy(out2[0][x],in_2[0][x],sizeof(double)*nr_channel);
		for(int y=1;y<h;y++)
		{
			double weight=range_table[euro_dist_rgb_max(texture[y][x],texture[y-1][x])];
			double edge=get_edge(y,x,y-1,x,flag,sigma_edge);
			double alpha_=weight*alpha;
			double alpha_2=sqrt (weight*alpha*edge);
			for(int c=0;c<nr_channel;c++) 
			{
				out[y][x][c]=inv_alpha*in_[y][x][c]+alpha_*out[y-1][x][c];
				out2[y][x][c]=inv_alpha*in_2[y][x][c]+alpha_2*out2[y-1][x][c];
			}
		}
	
		int h1=h-1;
		unsigned char disp=0; 
		double min_cost=QX_DEF_DOUBLE_MAX;
		for(int c=0;c<nr_channel;c++)
		{
			out[h1][x][c]=0.5*(out[h1][x][c]+in_[h1][x][c]);
			out2[h1][x][c]=0.5*(out2[h1][x][c]+in_2[h1][x][c]);
			if((out[h1][x][c]+out2[h1][x][c])<min_cost)
			{
				min_cost=out[h1][x][c]+out2[h1][x][c];
				disp=c;
			}
			//if((out[h1][x][c])<min_cost)
			//{
			//	min_cost=out[h1][x][c];
			//	disp=c;
			//}
		}
		disparity[h1][x]=disp;
	
		memcpy(yp,out[h1][x],sizeof(double)*nr_channel);
		memcpy(yp1,out2[h1][x],sizeof(double)*nr_channel);
		for(int y=h-2;y>=0;y--)
		{
			double weight=range_table[euro_dist_rgb_max(texture[y][x],texture[y+1][x])];
			double edge=get_edge(y,x,y+1,x,flag,sigma_edge);
			double alpha_=weight*alpha;
			double alpha_2=sqrt(weight*alpha*edge);
			unsigned char disp=0; double min_cost=QX_DEF_DOUBLE_MAX;
			for(int c=0;c<nr_channel;c++) 
			{
				yp[c]=inv_alpha*in_[y][x][c]+alpha_*yp[c];
				yp1[c]=inv_alpha*in_2[y][x][c]+alpha_2*yp1[c];
				out[y][x][c]=0.5*(out[y][x][c]+out2[y][x][c]+yp[c]+yp1[c]);
				//out[y][x][c]=0.5*(out[y][x][c]+yp[c]);
				if(out[y][x][c]<min_cost)
				{
					min_cost=out[y][x][c];
					disp=c;
				}
			}
			disparity[y][x]=disp;
		}
	}

}
//下面这个代码是正确的，保存
//void Recursive_tf(unsigned char**disparity,double***out,double***in,unsigned char***texture,double sigma_spatial,double sigma_range,int h,int w,int nr_channel,double***temp,double***out1,int flag,double sigma_edge)
//{
//	///////////////////////////////////////////////////////////////////////////////这是按照RecursiveBF的算法，正向进行然后反向进行的结果
//	double yp[100],yp1[100];
//	double alpha=exp(-sqrt(2.0)/(sigma_spatial*w));//filter kernel size
//	double inv_alpha=(1-alpha);
//	double range_table[256];
//	for(int i=0;i<256;i++) range_table[i]=exp(-double(i)/(sigma_range*255));
//	
//	double***in_=in;/*horizontal filtering*/
//	double***out_=temp;
//
//	for(int y=0;y<h;y++)
//	{
//		memcpy(out_[y][0],in_[y][0], sizeof (double )*nr_channel);
//        memcpy(out1[y][0],in_[y][0], sizeof (double )*nr_channel);
//        
//		for(int x=1;x<w;x++) 
//		{
//			double weight=range_table[euro_dist_rgb_max(texture[y][x],texture[y][x-1])];
//			double edge=get_edge(y,x,y,x-1,flag,sigma_edge);
//			double alpha_=weight*alpha;
//			double alpha_2=sqrt(weight*alpha*edge);
//			for(int c=0;c<nr_channel;c++) 
//			{
//				out_[y][x][c]=inv_alpha*in_[y][x][c]+alpha_*out_[y][x-1][c];
//                out1[y][x][c]=inv_alpha*in_[y][x][c]+alpha_2*out1[y][x-1][c];
//			}
//		}
//		int w1=w-1;
//		for(int c=0;c<nr_channel;c++) 
//		{
//			out_[y][w1][c]=0.5*(out_[y][w1][c]+in_[y][w1][c]);
//			out1[y][w1][c]=0.5*(out1[y][w1][c]+in_[y][w1][c]);
//		}
//		memcpy(yp,out_[y][w1],sizeof(double)*nr_channel);
//		memcpy(yp1,out1[y][w1],sizeof(double)*nr_channel);
//		for(int x=w-2;x>=0;x--) 
//		{
//			double weight=range_table[euro_dist_rgb_max(texture[y][x],texture[y][x+1])];
//			double edge=get_edge(y,x,y,x+1,flag,sigma_edge);
//            double alpha_=weight*alpha;
//            double alpha_2=sqrt (weight*alpha*edge);
//
//			for(int c=0;c<nr_channel;c++) 
//			{
//				yp[c]=inv_alpha*in_[y][x][c]+alpha_*yp[c];
//				yp1[c]=inv_alpha*in_[y][x][c]+alpha_2*yp1[c];
//				out_[y][x][c]=0.5*(out_[y][x][c]+out1[y][x][c]+yp[c]+yp1[c]);
//			}
//		}
//	}
//	in_=out_;/*vertical filtering*/
//	alpha=exp(-sqrt(2.0)/(sigma_spatial*h));//filter kernel size
//	inv_alpha=(1-alpha);
//
//	for(int x=0;x<w;x++)
//	{
//		memcpy(out[0][x],in_[0][x],sizeof(double)*nr_channel);
//		memcpy(out1[0][x],in_[0][x],sizeof(double)*nr_channel);
//		for(int y=1;y<h;y++)
//		{
//			double weight=range_table[euro_dist_rgb_max(texture[y][x],texture[y-1][x])];
//			double edge=get_edge(y,x,y-1,x,flag,sigma_edge);
//			double alpha_=weight*alpha;
//			double alpha_2=sqrt (weight*alpha*edge);
//			for(int c=0;c<nr_channel;c++) 
//			{
//				out[y][x][c]=inv_alpha*in_[y][x][c]+alpha_*out[y-1][x][c];
//				out1[y][x][c]=inv_alpha*in_[y][x][c]+alpha_2*out1[y-1][x][c];
//			}
//		}
//	
//		int h1=h-1;
//		unsigned char disp=0; 
//		double min_cost=QX_DEF_DOUBLE_MAX;
//		for(int c=0;c<nr_channel;c++)
//		{
//			out[h1][x][c]=0.5*(out[h1][x][c]+in_[h1][x][c]);
//			out1[h1][x][c]=0.5*(out1[h1][x][c]+in_[h1][x][c]);
//			if((out[h1][x][c]+out1[h1][x][c])<min_cost)
//			{
//				min_cost=out[h1][x][c]+out1[h1][x][c];
//				disp=c;
//			}
//		}
//		disparity[h1][x]=disp;
//	
//		memcpy(yp,out[h1][x],sizeof(double)*nr_channel);
//		memcpy(yp1,out1[h1][x],sizeof(double)*nr_channel);
//		for(int y=h-2;y>=0;y--)
//		{
//			double weight=range_table[euro_dist_rgb_max(texture[y][x],texture[y+1][x])];
//			double edge=get_edge(y,x,y+1,x,flag,sigma_edge);
//			double alpha_=weight*alpha;
//			double alpha_2=sqrt(weight*alpha*edge);
//			unsigned char disp=0; double min_cost=QX_DEF_DOUBLE_MAX;
//			for(int c=0;c<nr_channel;c++) 
//			{
//				yp[c]=inv_alpha*in_[y][x][c]+alpha_*yp[c];
//				yp1[c]=inv_alpha*in_[y][x][c]+alpha_2*yp1[c];
//				out[y][x][c]=0.5*(out[y][x][c]+out1[y][x][c]+yp[c]+yp1[c]);
//				if(out[y][x][c]<min_cost)
//				{
//					min_cost=out[y][x][c];
//					disp=c;
//				}
//			}
//			disparity[y][x]=disp;
//		}
//	}
//
//}
void compute_second_derivative(int m_h,int m_w,float**second_derivative,unsigned char***image)
{
	float gray,gray_minus,gray_plus;
	for(int y=0;y<m_h;y++)
	{
		gray_minus=rgb_2_gray(image[y][0]);
		gray=gray_plus=rgb_2_gray(image[y][1]);
		second_derivative[y][0]=gray_plus-gray_minus+127.5;
		for(int x=1;x<m_w-1;x++)
		{
			gray_plus=rgb_2_gray(image[y][x+1]);
			second_derivative[y][x]=0.5*(gray_plus-gray_minus)+127.5;
			gray_minus=gray;
			gray=gray_plus;
		}
		second_derivative[y][m_w-1]=gray_plus-gray_minus+127.5;
	}
}
inline void image_zero(double ***in,int h,int w,int d,double zero=0){memset(in[0][0],zero,sizeof(double)*h*w*d);}
inline float min(float a,float b){if(a<b) return(a); else return(b);}
void matching_cost_from_color_and_gradient(int m_h,int m_w,int m_nr_max_disp,double***cost_vol,unsigned char ***left,unsigned char ***right,int shift_sign)
{
	int max_color_difference=7;
	int max_second_derivative_difference=2;
	double weight_on_color=0.11;
	double weight_on_color_inv=1-weight_on_color;

	image_zero(cost_vol,m_h,m_w,m_nr_max_disp);
	compute_second_derivative(m_h,m_w,m_second_derivative_left,left);
	compute_second_derivative(m_h,m_w,m_second_derivative_right,right);

	for(int i=0;i<m_nr_max_disp;i++)
	{
		for(int y=0;y<m_h;y++) //shift the right image by i pixels
		{
			if(shift_sign>0)
			{
				image_copy((&m_image_shifted[y][i]),right[y],m_w-i,3);
				memcpy(&(m_second_derivative_shifted[y][i]),m_second_derivative_right[y],sizeof(float)*(m_w-i));
				for(int x=0;x<i;x++) 
				{
					qx_memcpy_u3(m_image_shifted[y][x],right[y][0]);
					m_second_derivative_shifted[y][x]=m_second_derivative_right[y][0];
				}
			}
			else
			{
				image_copy(&m_image_shifted[y][0],&right[y][i],m_w-i,3);
				memcpy(&(m_second_derivative_shifted[y][0]),&m_second_derivative_right[y][i],sizeof(float)*(m_w-i));
				for(int x=(m_w-i);x<m_w;x++) 
				{
					qx_memcpy_u3(m_image_shifted[y][x],right[y][(m_w-i)]);
					m_second_derivative_shifted[y][x]=m_second_derivative_right[y][(m_w-i)];
				}
			}
		}
		for(int y=0;y<m_h;y++) for(int x=0;x<(m_w);x++) 
		{
			float cost=0.f;
			for(int c=0;c<3;c++) cost+=abs(left[y][x][c]-m_image_shifted[y][x][c]);
			cost=min(cost/3,max_color_difference);
			float cost_2nd_derivative=min(abs(m_second_derivative_left[y][x]-m_second_derivative_shifted[y][x]),max_second_derivative_difference);
			cost_vol[y][x][i]=weight_on_color*cost+weight_on_color_inv*cost_2nd_derivative;
		}
	}
}
void qx_detect_occlusion_left_right(unsigned char**mask_left,unsigned char**depth_left,unsigned char**depth_right,int h,int w,int nr_max_disp)
{
	memset(mask_left[0],0,sizeof(char)*h*w);
	for(int y=0;y<h;y++) 
	{
		for(int x=0;x<w;x++) 
		{
			int d=depth_left[y][x];
			int xr=x-d;
			if(xr>=0)
			{
				if(d==0||abs(d-depth_right[y][xr])>=1)
				{
					mask_left[y][x]=255;
				}
			}
			else mask_left[y][x]=255;
		}
	}
}
void edge_strength_calculation(IplImage *im_input,int hL,int wL,double **mag,int **mage,double& mag_max)
{
	double Fe1[21][21]={-3.656e-005,-5.0751e-005,-6.3713e-005,-7.244e-005,-7.466e-005,-6.9798e-005,-5.9213e-005,-4.5597e-005,-3.1877e-005,-2.0235e-005,-1.1664e-005,-6.1062e-006,-2.9029e-006,-1.2534e-006,-4.915e-007,-1.7505e-007,-5.6622e-008,-1.6635e-008,-4.4386e-009,-1.0757e-009,-2.3676e-010,
-8.2098e-005,-0.00012849,-0.00018052,-0.00022859,-0.00026156,-0.00027092,-0.00025429,-0.00021645,-0.00016716,-0.00011717,-7.4556e-005,-4.3075e-005,-2.2598e-005,-1.0766e-005,-4.658e-006,-1.8303e-006,-6.5318e-007,-2.1171e-007,-6.232e-008,-1.6662e-008,-4.0459e-009,
-0.00012198,-0.00022524,-0.00036372,-0.00052151,-0.00066961,-0.00077377,-0.00080725,-0.00076186,-0.00065131,-0.0005048,-0.00035492,-0.00022645,-0.00013116,-6.8968e-005,-3.293e-005,-1.4278e-005,-5.622e-006,-2.0104e-006,-6.5293e-007,-1.9259e-007,-5.1594e-008,
-7.9714e-005,-0.00022707,-0.00047141,-0.00081084,-0.0012075,-0.0015883,-0.0018654,-0.0019682,-0.0018727,-0.0016108,-0.0012544,-0.00088537,-0.00056675,-0.00032919,-0.00017354,-8.3059e-005,-3.6094e-005,-1.4243e-005,-5.104e-006,-1.661e-006,-4.9094e-007,
0.00012564,3.6828e-005,-0.00020935,-0.00066932,-0.0013499,-0.0021782,-0.0030011,-0.0036281,-0.0039017,-0.0037612,-0.0032654,-0.0025604,-0.0018167,-0.0011678,-0.00068064,-0.0003599,-0.00017271,-7.5237e-005,-2.9757e-005,-1.0687e-005,-3.4854e-006,
0.00046066,0.00060446,0.0006197,0.00036196,-0.00029968,-0.001407,-0.0028489,-0.0043565,-0.0055784,-0.0062121,-0.0061249,-0.0053989,-0.0042786,-0.0030594,-0.0019781,-0.0011582,-0.00061474,-0.00029594,-0.00012928,-5.1261e-005,-1.8453e-005,
0.00070972,0.0011875,0.0017239,0.0021355,0.0021532,0.0015096,7.4856e-005,-0.0020261,-0.0043929,-0.0064673,-0.0077434,-0.0079665,-0.0072132,-0.0058191,-0.0042123,-0.0027477,-0.0016194,-0.00086382,-0.00041754,-0.00018301,-7.278e-005,
0.00061205,0.0012861,0.0022992,0.0035632,0.0048026,0.0055746,0.0054026,0.0039924,0.0014263,-0.0017885,-0.0048766,-0.0070933,-0.0080109,-0.0076442,-0.0063691,-0.0047082,-0.0031152,-0.0018546,-0.00099658,-0.0004844,-0.00021326,
0.00013184,0.00065064,0.0016784,0.0033372,0.0055589,0.0079841,0.0099746,0.010795,0.0099176,0.0073094,0.0035201,-0.00050205,-0.0037864,-0.0057077,-0.0061601,-0.0054811,-0.0042135,-0.0028583,-0.0017302,-0.00094058,-0.00046101,
-0.00044918,-0.00038684,8.2049e-005,0.0012562,0.0033727,0.0064274,0.010022,0.013359,0.015459,0.015548,0.013421,0.0095848,0.0050595,0.00096193,-0.0019181,-0.0033303,-0.0035072,-0.0029317,-0.0020866,-0.0013014,-0.00072145,
-0.00077263,-0.001167,-0.0014698,-0.0013791,-0.00048746,0.0015736,0.0049079,0.0091538,0.013454,0.016686,0.01789,0.016686,0.013454,0.0091538,0.0049079,0.0015736,-0.00048746,-0.0013791,-0.0014698,-0.001167,-0.00077263,
-0.00072145,-0.0013014,-0.0020866,-0.0029317,-0.0035072,-0.0033303,-0.0019181,0.00096193,0.0050595,0.0095848,0.013421,0.015548,0.015459,0.013359,0.010022,0.0064274,0.0033727,0.0012562,8.2049e-005,-0.00038684,-0.00044918,
-0.00046101,-0.00094058,-0.0017302,-0.0028583,-0.0042135,-0.0054811,-0.0061601,-0.0057077,-0.0037864,-0.00050205,0.0035201,0.0073094,0.0099176,0.010795,0.0099746,0.0079841,0.0055589,0.0033372,0.0016784,0.00065064,0.00013184,
-0.00021326,-0.0004844,-0.00099658,-0.0018546,-0.0031152,-0.0047082,-0.0063691,-0.0076442,-0.0080109,-0.0070933,-0.0048766,-0.0017885,0.0014263,0.0039924,0.0054026,0.0055746,0.0048026,0.0035632,0.0022992,0.0012861,0.00061205,
-7.278e-005,-0.00018301,-0.00041754,-0.00086382,-0.0016194,-0.0027477,-0.0042123,-0.0058191,-0.0072132,-0.0079665,-0.0077434,-0.0064673,-0.0043929,-0.0020261,7.4856e-005,0.0015096,0.0021532,0.0021355,0.0017239,0.0011875,0.00070972,
-1.8453e-005,-5.1261e-005,-0.00012928,-0.00029594,-0.00061474,-0.0011582,-0.0019781,-0.0030594,-0.0042786,-0.0053989,-0.0061249,-0.0062121,-0.0055784,-0.0043565,-0.0028489,-0.001407,-0.00029968,0.00036196,0.0006197,0.00060446,0.00046066,
-3.4854e-006,-1.0687e-005,-2.9757e-005,-7.5237e-005,-0.00017271,-0.0003599,-0.00068064,-0.0011678,-0.0018167,-0.0025604,-0.0032654,-0.0037612,-0.0039017,-0.0036281,-0.0030011,-0.0021782,-0.0013499,-0.00066932,-0.00020935,3.6828e-005,0.00012564,
-4.9094e-007,-1.661e-006,-5.104e-006,-1.4243e-005,-3.6094e-005,-8.3059e-005,-0.00017354,-0.00032919,-0.00056675,-0.00088537,-0.0012544,-0.0016108,-0.0018727,-0.0019682,-0.0018654,-0.0015883,-0.0012075,-0.00081084,-0.00047141,-0.00022707,-7.9714e-005,
-5.1594e-008,-1.9259e-007,-6.5293e-007,-2.0104e-006,-5.622e-006,-1.4278e-005,-3.293e-005,-6.8968e-005,-0.00013116,-0.00022645,-0.00035492,-0.0005048,-0.00065131,-0.00076186,-0.00080725,-0.00077377,-0.00066961,-0.00052151,-0.00036372,-0.00022524,-0.00012198,
-4.0459e-009,-1.6662e-008,-6.232e-008,-2.1171e-007,-6.5318e-007,-1.8303e-006,-4.658e-006,-1.0766e-005,-2.2598e-005,-4.3075e-005,-7.4556e-005,-0.00011717,-0.00016716,-0.00021645,-0.00025429,-0.00027092,-0.00026156,-0.00022859,-0.00018052,-0.00012849,-8.2098e-005,
-2.3676e-010,-1.0757e-009,-4.4386e-009,-1.6635e-008,-5.6622e-008,-1.7505e-007,-4.915e-007,-1.2534e-006,-2.9029e-006,-6.1062e-006,-1.1664e-005,-2.0235e-005,-3.1877e-005,-4.5597e-005,-5.9213e-005,-6.9798e-005,-7.466e-005,-7.244e-005,-6.3713e-005,-5.0751e-005,-3.656e-005
};
double Fe2[21][21]={-3.656e-005,-8.2098e-005,-0.00012198,-7.9714e-005,0.00012564,0.00046066,0.00070972,0.00061205,0.00013184,-0.00044918,-0.00077263,-0.00072145,-0.00046101,-0.00021326,-7.278e-005,-1.8453e-005,-3.4854e-006,-4.9094e-007,-5.1594e-008,-4.0459e-009,-2.3676e-010,
-5.0751e-005,-0.00012849,-0.00022524,-0.00022707,3.6828e-005,0.00060446,0.0011875,0.0012861,0.00065064,-0.00038684,-0.001167,-0.0013014,-0.00094058,-0.0004844,-0.00018301,-5.1261e-005,-1.0687e-005,-1.661e-006,-1.9259e-007,-1.6662e-008,-1.0757e-009,
-6.3713e-005,-0.00018052,-0.00036372,-0.00047141,-0.00020935,0.0006197,0.0017239,0.0022992,0.0016784,8.2049e-005,-0.0014698,-0.0020866,-0.0017302,-0.00099658,-0.00041754,-0.00012928,-2.9757e-005,-5.104e-006,-6.5293e-007,-6.232e-008,-4.4386e-009,
-7.244e-005,-0.00022859,-0.00052151,-0.00081084,-0.00066932,0.00036196,0.0021355,0.0035632,0.0033372,0.0012562,-0.0013791,-0.0029317,-0.0028583,-0.0018546,-0.00086382,-0.00029594,-7.5237e-005,-1.4243e-005,-2.0104e-006,-2.1171e-007,-1.6635e-008,
-7.466e-005,-0.00026156,-0.00066961,-0.0012075,-0.0013499,-0.00029968,0.0021532,0.0048026,0.0055589,0.0033727,-0.00048746,-0.0035072,-0.0042135,-0.0031152,-0.0016194,-0.00061474,-0.00017271,-3.6094e-005,-5.622e-006,-6.5318e-007,-5.6622e-008,
-6.9798e-005,-0.00027092,-0.00077377,-0.0015883,-0.0021782,-0.001407,0.0015096,0.0055746,0.0079841,0.0064274,0.0015736,-0.0033303,-0.0054811,-0.0047082,-0.0027477,-0.0011582,-0.0003599,-8.3059e-005,-1.4278e-005,-1.8303e-006,-1.7505e-007,
-5.9213e-005,-0.00025429,-0.00080725,-0.0018654,-0.0030011,-0.0028489,7.4856e-005,0.0054026,0.0099746,0.010022,0.0049079,-0.0019181,-0.0061601,-0.0063691,-0.0042123,-0.0019781,-0.00068064,-0.00017354,-3.293e-005,-4.658e-006,-4.915e-007,
-4.5597e-005,-0.00021645,-0.00076186,-0.0019682,-0.0036281,-0.0043565,-0.0020261,0.0039924,0.010795,0.013359,0.0091538,0.00096193,-0.0057077,-0.0076442,-0.0058191,-0.0030594,-0.0011678,-0.00032919,-6.8968e-005,-1.0766e-005,-1.2534e-006,
-3.1877e-005,-0.00016716,-0.00065131,-0.0018727,-0.0039017,-0.0055784,-0.0043929,0.0014263,0.0099176,0.015459,0.013454,0.0050595,-0.0037864,-0.0080109,-0.0072132,-0.0042786,-0.0018167,-0.00056675,-0.00013116,-2.2598e-005,-2.9029e-006,
-2.0235e-005,-0.00011717,-0.0005048,-0.0016108,-0.0037612,-0.0062121,-0.0064673,-0.0017885,0.0073094,0.015548,0.016686,0.0095848,-0.00050205,-0.0070933,-0.0079665,-0.0053989,-0.0025604,-0.00088537,-0.00022645,-4.3075e-005,-6.1062e-006,
-1.1664e-005,-7.4556e-005,-0.00035492,-0.0012544,-0.0032654,-0.0061249,-0.0077434,-0.0048766,0.0035201,0.013421,0.01789,0.013421,0.0035201,-0.0048766,-0.0077434,-0.0061249,-0.0032654,-0.0012544,-0.00035492,-7.4556e-005,-1.1664e-005,
-6.1062e-006,-4.3075e-005,-0.00022645,-0.00088537,-0.0025604,-0.0053989,-0.0079665,-0.0070933,-0.00050205,0.0095848,0.016686,0.015548,0.0073094,-0.0017885,-0.0064673,-0.0062121,-0.0037612,-0.0016108,-0.0005048,-0.00011717,-2.0235e-005,
-2.9029e-006,-2.2598e-005,-0.00013116,-0.00056675,-0.0018167,-0.0042786,-0.0072132,-0.0080109,-0.0037864,0.0050595,0.013454,0.015459,0.0099176,0.0014263,-0.0043929,-0.0055784,-0.0039017,-0.0018727,-0.00065131,-0.00016716,-3.1877e-005,
-1.2534e-006,-1.0766e-005,-6.8968e-005,-0.00032919,-0.0011678,-0.0030594,-0.0058191,-0.0076442,-0.0057077,0.00096193,0.0091538,0.013359,0.010795,0.0039924,-0.0020261,-0.0043565,-0.0036281,-0.0019682,-0.00076186,-0.00021645,-4.5597e-005,
-4.915e-007,-4.658e-006,-3.293e-005,-0.00017354,-0.00068064,-0.0019781,-0.0042123,-0.0063691,-0.0061601,-0.0019181,0.0049079,0.010022,0.0099746,0.0054026,7.4856e-005,-0.0028489,-0.0030011,-0.0018654,-0.00080725,-0.00025429,-5.9213e-005,
-1.7505e-007,-1.8303e-006,-1.4278e-005,-8.3059e-005,-0.0003599,-0.0011582,-0.0027477,-0.0047082,-0.0054811,-0.0033303,0.0015736,0.0064274,0.0079841,0.0055746,0.0015096,-0.001407,-0.0021782,-0.0015883,-0.00077377,-0.00027092,-6.9798e-005,
-5.6622e-008,-6.5318e-007,-5.622e-006,-3.6094e-005,-0.00017271,-0.00061474,-0.0016194,-0.0031152,-0.0042135,-0.0035072,-0.00048746,0.0033727,0.0055589,0.0048026,0.0021532,-0.00029968,-0.0013499,-0.0012075,-0.00066961,-0.00026156,-7.466e-005,
-1.6635e-008,-2.1171e-007,-2.0104e-006,-1.4243e-005,-7.5237e-005,-0.00029594,-0.00086382,-0.0018546,-0.0028583,-0.0029317,-0.0013791,0.0012562,0.0033372,0.0035632,0.0021355,0.00036196,-0.00066932,-0.00081084,-0.00052151,-0.00022859,-7.244e-005,
-4.4386e-009,-6.232e-008,-6.5293e-007,-5.104e-006,-2.9757e-005,-0.00012928,-0.00041754,-0.00099658,-0.0017302,-0.0020866,-0.0014698,8.2049e-005,0.0016784,0.0022992,0.0017239,0.0006197,-0.00020935,-0.00047141,-0.00036372,-0.00018052,-6.3713e-005,
-1.0757e-009,-1.6662e-008,-1.9259e-007,-1.661e-006,-1.0687e-005,-5.1261e-005,-0.00018301,-0.0004844,-0.00094058,-0.0013014,-0.001167,-0.00038684,0.00065064,0.0012861,0.0011875,0.00060446,3.6828e-005,-0.00022707,-0.00022524,-0.00012849,-5.0751e-005,
-2.3676e-010,-4.0459e-009,-5.1594e-008,-4.9094e-007,-3.4854e-006,-1.8453e-005,-7.278e-005,-0.00021326,-0.00046101,-0.00072145,-0.00077263,-0.00044918,0.00013184,0.00061205,0.00070972,0.00046066,0.00012564,-7.9714e-005,-0.00012198,-8.2098e-005,-3.656e-005
};
double Fe3[21][21]={-2.3676e-010,-4.0459e-009,-5.1594e-008,-4.9094e-007,-3.4854e-006,-1.8453e-005,-7.278e-005,-0.00021326,-0.00046101,-0.00072145,-0.00077263,-0.00044918,0.00013184,0.00061205,0.00070972,0.00046066,0.00012564,-7.9714e-005,-0.00012198,-8.2098e-005,-3.656e-005,
-1.0757e-009,-1.6662e-008,-1.9259e-007,-1.661e-006,-1.0687e-005,-5.1261e-005,-0.00018301,-0.0004844,-0.00094058,-0.0013014,-0.001167,-0.00038684,0.00065064,0.0012861,0.0011875,0.00060446,3.6828e-005,-0.00022707,-0.00022524,-0.00012849,-5.0751e-005,
-4.4386e-009,-6.232e-008,-6.5293e-007,-5.104e-006,-2.9757e-005,-0.00012928,-0.00041754,-0.00099658,-0.0017302,-0.0020866,-0.0014698,8.2049e-005,0.0016784,0.0022992,0.0017239,0.0006197,-0.00020935,-0.00047141,-0.00036372,-0.00018052,-6.3713e-005,
-1.6635e-008,-2.1171e-007,-2.0104e-006,-1.4243e-005,-7.5237e-005,-0.00029594,-0.00086382,-0.0018546,-0.0028583,-0.0029317,-0.0013791,0.0012562,0.0033372,0.0035632,0.0021355,0.00036196,-0.00066932,-0.00081084,-0.00052151,-0.00022859,-7.244e-005,
-5.6622e-008,-6.5318e-007,-5.622e-006,-3.6094e-005,-0.00017271,-0.00061474,-0.0016194,-0.0031152,-0.0042135,-0.0035072,-0.00048746,0.0033727,0.0055589,0.0048026,0.0021532,-0.00029968,-0.0013499,-0.0012075,-0.00066961,-0.00026156,-7.466e-005,
-1.7505e-007,-1.8303e-006,-1.4278e-005,-8.3059e-005,-0.0003599,-0.0011582,-0.0027477,-0.0047082,-0.0054811,-0.0033303,0.0015736,0.0064274,0.0079841,0.0055746,0.0015096,-0.001407,-0.0021782,-0.0015883,-0.00077377,-0.00027092,-6.9798e-005,
-4.915e-007,-4.658e-006,-3.293e-005,-0.00017354,-0.00068064,-0.0019781,-0.0042123,-0.0063691,-0.0061601,-0.0019181,0.0049079,0.010022,0.0099746,0.0054026,7.4856e-005,-0.0028489,-0.0030011,-0.0018654,-0.00080725,-0.00025429,-5.9213e-005,
-1.2534e-006,-1.0766e-005,-6.8968e-005,-0.00032919,-0.0011678,-0.0030594,-0.0058191,-0.0076442,-0.0057077,0.00096193,0.0091538,0.013359,0.010795,0.0039924,-0.0020261,-0.0043565,-0.0036281,-0.0019682,-0.00076186,-0.00021645,-4.5597e-005,
-2.9029e-006,-2.2598e-005,-0.00013116,-0.00056675,-0.0018167,-0.0042786,-0.0072132,-0.0080109,-0.0037864,0.0050595,0.013454,0.015459,0.0099176,0.0014263,-0.0043929,-0.0055784,-0.0039017,-0.0018727,-0.00065131,-0.00016716,-3.1877e-005,
-6.1062e-006,-4.3075e-005,-0.00022645,-0.00088537,-0.0025604,-0.0053989,-0.0079665,-0.0070933,-0.00050205,0.0095848,0.016686,0.015548,0.0073094,-0.0017885,-0.0064673,-0.0062121,-0.0037612,-0.0016108,-0.0005048,-0.00011717,-2.0235e-005,
-1.1664e-005,-7.4556e-005,-0.00035492,-0.0012544,-0.0032654,-0.0061249,-0.0077434,-0.0048766,0.0035201,0.013421,0.01789,0.013421,0.0035201,-0.0048766,-0.0077434,-0.0061249,-0.0032654,-0.0012544,-0.00035492,-7.4556e-005,-1.1664e-005,
-2.0235e-005,-0.00011717,-0.0005048,-0.0016108,-0.0037612,-0.0062121,-0.0064673,-0.0017885,0.0073094,0.015548,0.016686,0.0095848,-0.00050205,-0.0070933,-0.0079665,-0.0053989,-0.0025604,-0.00088537,-0.00022645,-4.3075e-005,-6.1062e-006,
-3.1877e-005,-0.00016716,-0.00065131,-0.0018727,-0.0039017,-0.0055784,-0.0043929,0.0014263,0.0099176,0.015459,0.013454,0.0050595,-0.0037864,-0.0080109,-0.0072132,-0.0042786,-0.0018167,-0.00056675,-0.00013116,-2.2598e-005,-2.9029e-006,
-4.5597e-005,-0.00021645,-0.00076186,-0.0019682,-0.0036281,-0.0043565,-0.0020261,0.0039924,0.010795,0.013359,0.0091538,0.00096193,-0.0057077,-0.0076442,-0.0058191,-0.0030594,-0.0011678,-0.00032919,-6.8968e-005,-1.0766e-005,-1.2534e-006,
-5.9213e-005,-0.00025429,-0.00080725,-0.0018654,-0.0030011,-0.0028489,7.4856e-005,0.0054026,0.0099746,0.010022,0.0049079,-0.0019181,-0.0061601,-0.0063691,-0.0042123,-0.0019781,-0.00068064,-0.00017354,-3.293e-005,-4.658e-006,-4.915e-007,
-6.9798e-005,-0.00027092,-0.00077377,-0.0015883,-0.0021782,-0.001407,0.0015096,0.0055746,0.0079841,0.0064274,0.0015736,-0.0033303,-0.0054811,-0.0047082,-0.0027477,-0.0011582,-0.0003599,-8.3059e-005,-1.4278e-005,-1.8303e-006,-1.7505e-007,
-7.466e-005,-0.00026156,-0.00066961,-0.0012075,-0.0013499,-0.00029968,0.0021532,0.0048026,0.0055589,0.0033727,-0.00048746,-0.0035072,-0.0042135,-0.0031152,-0.0016194,-0.00061474,-0.00017271,-3.6094e-005,-5.622e-006,-6.5318e-007,-5.6622e-008,
-7.244e-005,-0.00022859,-0.00052151,-0.00081084,-0.00066932,0.00036196,0.0021355,0.0035632,0.0033372,0.0012562,-0.0013791,-0.0029317,-0.0028583,-0.0018546,-0.00086382,-0.00029594,-7.5237e-005,-1.4243e-005,-2.0104e-006,-2.1171e-007,-1.6635e-008,
-6.3713e-005,-0.00018052,-0.00036372,-0.00047141,-0.00020935,0.0006197,0.0017239,0.0022992,0.0016784,8.2049e-005,-0.0014698,-0.0020866,-0.0017302,-0.00099658,-0.00041754,-0.00012928,-2.9757e-005,-5.104e-006,-6.5293e-007,-6.232e-008,-4.4386e-009,
-5.0751e-005,-0.00012849,-0.00022524,-0.00022707,3.6828e-005,0.00060446,0.0011875,0.0012861,0.00065064,-0.00038684,-0.001167,-0.0013014,-0.00094058,-0.0004844,-0.00018301,-5.1261e-005,-1.0687e-005,-1.661e-006,-1.9259e-007,-1.6662e-008,-1.0757e-009,
-3.656e-005,-8.2098e-005,-0.00012198,-7.9714e-005,0.00012564,0.00046066,0.00070972,0.00061205,0.00013184,-0.00044918,-0.00077263,-0.00072145,-0.00046101,-0.00021326,-7.278e-005,-1.8453e-005,-3.4854e-006,-4.9094e-007,-5.1594e-008,-4.0459e-009,-2.3676e-010
};
double Fe4[21][21]={-2.3676e-010,-1.0757e-009,-4.4386e-009,-1.6635e-008,-5.6622e-008,-1.7505e-007,-4.915e-007,-1.2534e-006,-2.9029e-006,-6.1062e-006,-1.1664e-005,-2.0235e-005,-3.1877e-005,-4.5597e-005,-5.9213e-005,-6.9798e-005,-7.466e-005,-7.244e-005,-6.3713e-005,-5.0751e-005,-3.656e-005,
-4.0459e-009,-1.6662e-008,-6.232e-008,-2.1171e-007,-6.5318e-007,-1.8303e-006,-4.658e-006,-1.0766e-005,-2.2598e-005,-4.3075e-005,-7.4556e-005,-0.00011717,-0.00016716,-0.00021645,-0.00025429,-0.00027092,-0.00026156,-0.00022859,-0.00018052,-0.00012849,-8.2098e-005,
-5.1594e-008,-1.9259e-007,-6.5293e-007,-2.0104e-006,-5.622e-006,-1.4278e-005,-3.293e-005,-6.8968e-005,-0.00013116,-0.00022645,-0.00035492,-0.0005048,-0.00065131,-0.00076186,-0.00080725,-0.00077377,-0.00066961,-0.00052151,-0.00036372,-0.00022524,-0.00012198,
-4.9094e-007,-1.661e-006,-5.104e-006,-1.4243e-005,-3.6094e-005,-8.3059e-005,-0.00017354,-0.00032919,-0.00056675,-0.00088537,-0.0012544,-0.0016108,-0.0018727,-0.0019682,-0.0018654,-0.0015883,-0.0012075,-0.00081084,-0.00047141,-0.00022707,-7.9714e-005,
-3.4854e-006,-1.0687e-005,-2.9757e-005,-7.5237e-005,-0.00017271,-0.0003599,-0.00068064,-0.0011678,-0.0018167,-0.0025604,-0.0032654,-0.0037612,-0.0039017,-0.0036281,-0.0030011,-0.0021782,-0.0013499,-0.00066932,-0.00020935,3.6828e-005,0.00012564,
-1.8453e-005,-5.1261e-005,-0.00012928,-0.00029594,-0.00061474,-0.0011582,-0.0019781,-0.0030594,-0.0042786,-0.0053989,-0.0061249,-0.0062121,-0.0055784,-0.0043565,-0.0028489,-0.001407,-0.00029968,0.00036196,0.0006197,0.00060446,0.00046066,
-7.278e-005,-0.00018301,-0.00041754,-0.00086382,-0.0016194,-0.0027477,-0.0042123,-0.0058191,-0.0072132,-0.0079665,-0.0077434,-0.0064673,-0.0043929,-0.0020261,7.4856e-005,0.0015096,0.0021532,0.0021355,0.0017239,0.0011875,0.00070972,
-0.00021326,-0.0004844,-0.00099658,-0.0018546,-0.0031152,-0.0047082,-0.0063691,-0.0076442,-0.0080109,-0.0070933,-0.0048766,-0.0017885,0.0014263,0.0039924,0.0054026,0.0055746,0.0048026,0.0035632,0.0022992,0.0012861,0.00061205,
-0.00046101,-0.00094058,-0.0017302,-0.0028583,-0.0042135,-0.0054811,-0.0061601,-0.0057077,-0.0037864,-0.00050205,0.0035201,0.0073094,0.0099176,0.010795,0.0099746,0.0079841,0.0055589,0.0033372,0.0016784,0.00065064,0.00013184,
-0.00072145,-0.0013014,-0.0020866,-0.0029317,-0.0035072,-0.0033303,-0.0019181,0.00096193,0.0050595,0.0095848,0.013421,0.015548,0.015459,0.013359,0.010022,0.0064274,0.0033727,0.0012562,8.2049e-005,-0.00038684,-0.00044918,
-0.00077263,-0.001167,-0.0014698,-0.0013791,-0.00048746,0.0015736,0.0049079,0.0091538,0.013454,0.016686,0.01789,0.016686,0.013454,0.0091538,0.0049079,0.0015736,-0.00048746,-0.0013791,-0.0014698,-0.001167,-0.00077263,
-0.00044918,-0.00038684,8.2049e-005,0.0012562,0.0033727,0.0064274,0.010022,0.013359,0.015459,0.015548,0.013421,0.0095848,0.0050595,0.00096193,-0.0019181,-0.0033303,-0.0035072,-0.0029317,-0.0020866,-0.0013014,-0.00072145,
0.00013184,0.00065064,0.0016784,0.0033372,0.0055589,0.0079841,0.0099746,0.010795,0.0099176,0.0073094,0.0035201,-0.00050205,-0.0037864,-0.0057077,-0.0061601,-0.0054811,-0.0042135,-0.0028583,-0.0017302,-0.00094058,-0.00046101,
0.00061205,0.0012861,0.0022992,0.0035632,0.0048026,0.0055746,0.0054026,0.0039924,0.0014263,-0.0017885,-0.0048766,-0.0070933,-0.0080109,-0.0076442,-0.0063691,-0.0047082,-0.0031152,-0.0018546,-0.00099658,-0.0004844,-0.00021326,
0.00070972,0.0011875,0.0017239,0.0021355,0.0021532,0.0015096,7.4856e-005,-0.0020261,-0.0043929,-0.0064673,-0.0077434,-0.0079665,-0.0072132,-0.0058191,-0.0042123,-0.0027477,-0.0016194,-0.00086382,-0.00041754,-0.00018301,-7.278e-005,
0.00046066,0.00060446,0.0006197,0.00036196,-0.00029968,-0.001407,-0.0028489,-0.0043565,-0.0055784,-0.0062121,-0.0061249,-0.0053989,-0.0042786,-0.0030594,-0.0019781,-0.0011582,-0.00061474,-0.00029594,-0.00012928,-5.1261e-005,-1.8453e-005,
0.00012564,3.6828e-005,-0.00020935,-0.00066932,-0.0013499,-0.0021782,-0.0030011,-0.0036281,-0.0039017,-0.0037612,-0.0032654,-0.0025604,-0.0018167,-0.0011678,-0.00068064,-0.0003599,-0.00017271,-7.5237e-005,-2.9757e-005,-1.0687e-005,-3.4854e-006,
-7.9714e-005,-0.00022707,-0.00047141,-0.00081084,-0.0012075,-0.0015883,-0.0018654,-0.0019682,-0.0018727,-0.0016108,-0.0012544,-0.00088537,-0.00056675,-0.00032919,-0.00017354,-8.3059e-005,-3.6094e-005,-1.4243e-005,-5.104e-006,-1.661e-006,-4.9094e-007,
-0.00012198,-0.00022524,-0.00036372,-0.00052151,-0.00066961,-0.00077377,-0.00080725,-0.00076186,-0.00065131,-0.0005048,-0.00035492,-0.00022645,-0.00013116,-6.8968e-005,-3.293e-005,-1.4278e-005,-5.622e-006,-2.0104e-006,-6.5293e-007,-1.9259e-007,-5.1594e-008,
-8.2098e-005,-0.00012849,-0.00018052,-0.00022859,-0.00026156,-0.00027092,-0.00025429,-0.00021645,-0.00016716,-0.00011717,-7.4556e-005,-4.3075e-005,-2.2598e-005,-1.0766e-005,-4.658e-006,-1.8303e-006,-6.5318e-007,-2.1171e-007,-6.232e-008,-1.6662e-008,-4.0459e-009,
-3.656e-005,-5.0751e-005,-6.3713e-005,-7.244e-005,-7.466e-005,-6.9798e-005,-5.9213e-005,-4.5597e-005,-3.1877e-005,-2.0235e-005,-1.1664e-005,-6.1062e-006,-2.9029e-006,-1.2534e-006,-4.915e-007,-1.7505e-007,-5.6622e-008,-1.6635e-008,-4.4386e-009,-1.0757e-009,-2.3676e-010
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////
double Fo1[21][21]={1.2077e-005,1.3684e-005,1.4075e-005,1.3146e-005,1.1149e-005,8.5864e-006,6.0056e-006,3.8149e-006,2.2009e-006,1.1532e-006,5.4877e-007,2.3719e-007,9.3108e-008,3.3196e-008,1.075e-008,3.1616e-009,8.4453e-010,2.049e-010,4.515e-011,9.0363e-012,1.6426e-012,
4.5634e-005,5.7165e-005,6.4966e-005,6.7004e-005,6.2732e-005,5.3325e-005,4.1159e-005,2.8849e-005,1.8364e-005,1.0616e-005,5.5736e-006,2.6577e-006,1.151e-006,4.5273e-007,1.6173e-007,5.2476e-008,1.5464e-008,4.1391e-009,1.0062e-009,2.2216e-010,4.455e-011,
0.00012733,0.00017695,0.00022269,0.00025403,0.00026283,0.00024677,0.00021029,0.0001627,0.00011429,7.2909e-005,4.2237e-005,2.2221e-005,1.0617e-005,4.6074e-006,1.8159e-006,6.5e-007,2.1132e-007,6.2398e-008,1.6734e-008,4.076e-009,9.0174e-010,
0.00025784,0.00040125,0.00056261,0.00071271,0.00081715,0.00084892,0.00079973,0.00068353,0.00053023,0.00037338,0.00023873,0.0001386,7.3071e-005,3.4986e-005,1.5213e-005,6.008e-006,2.1549e-006,7.0197e-007,2.0769e-007,5.5809e-008,1.3621e-008,
0.00036013,0.00064557,0.0010263,0.0014599,0.0018683,0.002158,0.0022545,0.0021334,0.00183,0.001424,0.0010055,0.0006445,0.00037505,0.00019817,9.5087e-005,4.1434e-005,1.6397e-005,5.8929e-006,1.9235e-006,5.7024e-007,1.5354e-007,
0.00028355,0.0006621,0.0012697,0.0021006,0.0030648,0.0039889,0.0046617,0.0049114,0.0046767,0.0040314,0.0031493,0.0022311,0.0014342,0.00083676,0.00044319,0.00021313,9.3073e-005,3.6909e-005,1.3293e-005,4.3477e-006,1.2915e-006,
-6.4024e-005,0.00020683,0.00080942,0.001851,0.0033384,0.0051159,0.0068639,0.0081825,0.0087385,0.0084012,0.0072931,0.0057281,0.0040755,0.0026291,0.0015385,0.00081709,0.00039391,0.00017241,6.8524e-005,2.4731e-005,8.1058e-006,
-0.00056148,-0.00064712,-0.00046354,0.00022693,0.0016327,0.0038054,0.0065302,0.0093147,0.01152,0.012596,0.012303,0.010799,0.0085494,0.0061194,0.0039661,0.0023298,0.0012413,0.00060011,0.00026334,0.00010491,3.7948e-005,
-0.00091712,-0.0014366,-0.0019472,-0.0021835,-0.001773,-0.00035926,0.0022066,0.0056955,0.0094701,0.012659,0.014478,0.014545,0.013005,0.010429,0.0075326,0.0049149,0.0029021,0.0015526,0.0007532,0.0003315,0.00013241,
-0.00092739,-0.0016852,-0.002742,-0.0039611,-0.0050025,-0.005351,-0.0044728,-0.00207,0.0016859,0.0060953,0.010129,0.012813,0.01361,0.01259,0.010319,0.0075649,0.0049876,0.0029675,0.0015967,0.00077808,0.00034371,
-0.00066047,-0.0013467,-0.0024814,-0.0041209,-0.0061428,-0.0081633,-0.0095564,-0.00963,-0.0079246,-0.0044951,0,0.0044951,0.0079246,0.00963,0.0095564,0.0081633,0.0061428,0.0041209,0.0024814,0.0013467,0.00066047,
-0.00034371,-0.00077808,-0.0015967,-0.0029675,-0.0049876,-0.0075649,-0.010319,-0.01259,-0.01361,-0.012813,-0.010129,-0.0060953,-0.0016859,0.00207,0.0044728,0.005351,0.0050025,0.0039611,0.002742,0.0016852,0.00092739,
-0.00013241,-0.0003315,-0.0007532,-0.0015526,-0.0029021,-0.0049149,-0.0075326,-0.010429,-0.013005,-0.014545,-0.014478,-0.012659,-0.0094701,-0.0056955,-0.0022066,0.00035926,0.001773,0.0021835,0.0019472,0.0014366,0.00091712,
-3.7948e-005,-0.00010491,-0.00026334,-0.00060011,-0.0012413,-0.0023298,-0.0039661,-0.0061194,-0.0085494,-0.010799,-0.012303,-0.012596,-0.01152,-0.0093147,-0.0065302,-0.0038054,-0.0016327,-0.00022693,0.00046354,0.00064712,0.00056148,
-8.1058e-006,-2.4731e-005,-6.8524e-005,-0.00017241,-0.00039391,-0.00081709,-0.0015385,-0.0026291,-0.0040755,-0.0057281,-0.0072931,-0.0084012,-0.0087385,-0.0081825,-0.0068639,-0.0051159,-0.0033384,-0.001851,-0.00080942,-0.00020683,6.4024e-005,
-1.2915e-006,-4.3477e-006,-1.3293e-005,-3.6909e-005,-9.3073e-005,-0.00021313,-0.00044319,-0.00083676,-0.0014342,-0.0022311,-0.0031493,-0.0040314,-0.0046767,-0.0049114,-0.0046617,-0.0039889,-0.0030648,-0.0021006,-0.0012697,-0.0006621,-0.00028355,
-1.5354e-007,-5.7024e-007,-1.9235e-006,-5.8929e-006,-1.6397e-005,-4.1434e-005,-9.5087e-005,-0.00019817,-0.00037505,-0.0006445,-0.0010055,-0.001424,-0.00183,-0.0021334,-0.0022545,-0.002158,-0.0018683,-0.0014599,-0.0010263,-0.00064557,-0.00036013,
-1.3621e-008,-5.5809e-008,-2.0769e-007,-7.0197e-007,-2.1549e-006,-6.008e-006,-1.5213e-005,-3.4986e-005,-7.3071e-005,-0.0001386,-0.00023873,-0.00037338,-0.00053023,-0.00068353,-0.00079973,-0.00084892,-0.00081715,-0.00071271,-0.00056261,-0.00040125,-0.00025784,
-9.0174e-010,-4.076e-009,-1.6734e-008,-6.2398e-008,-2.1132e-007,-6.5e-007,-1.8159e-006,-4.6074e-006,-1.0617e-005,-2.2221e-005,-4.2237e-005,-7.2909e-005,-0.00011429,-0.0001627,-0.00021029,-0.00024677,-0.00026283,-0.00025403,-0.00022269,-0.00017695,-0.00012733,
-4.455e-011,-2.2216e-010,-1.0062e-009,-4.1391e-009,-1.5464e-008,-5.2476e-008,-1.6173e-007,-4.5273e-007,-1.151e-006,-2.6577e-006,-5.5736e-006,-1.0616e-005,-1.8364e-005,-2.8849e-005,-4.1159e-005,-5.3325e-005,-6.2732e-005,-6.7004e-005,-6.4966e-005,-5.7165e-005,-4.5634e-005,
-1.6426e-012,-9.0363e-012,-4.515e-011,-2.049e-010,-8.4453e-010,-3.1616e-009,-1.075e-008,-3.3196e-008,-9.3108e-008,-2.3719e-007,-5.4877e-007,-1.1532e-006,-2.2009e-006,-3.8149e-006,-6.0056e-006,-8.5864e-006,-1.1149e-005,-1.3146e-005,-1.4075e-005,-1.3684e-005,-1.2077e-005
};
double Fo2[21][21]={-1.2077e-005,-4.5634e-005,-0.00012733,-0.00025784,-0.00036013,-0.00028355,6.4024e-005,0.00056148,0.00091712,0.00092739,0.00066047,0.00034371,0.00013241,3.7948e-005,8.1058e-006,1.2915e-006,1.5354e-007,1.3621e-008,9.0174e-010,4.455e-011,1.6426e-012,
-1.3684e-005,-5.7165e-005,-0.00017695,-0.00040125,-0.00064557,-0.0006621,-0.00020683,0.00064712,0.0014366,0.0016852,0.0013467,0.00077808,0.0003315,0.00010491,2.4731e-005,4.3477e-006,5.7024e-007,5.5809e-008,4.076e-009,2.2216e-010,9.0363e-012,
-1.4075e-005,-6.4966e-005,-0.00022269,-0.00056261,-0.0010263,-0.0012697,-0.00080942,0.00046354,0.0019472,0.002742,0.0024814,0.0015967,0.0007532,0.00026334,6.8524e-005,1.3293e-005,1.9235e-006,2.0769e-007,1.6734e-008,1.0062e-009,4.515e-011,
-1.3146e-005,-6.7004e-005,-0.00025403,-0.00071271,-0.0014599,-0.0021006,-0.001851,-0.00022693,0.0021835,0.0039611,0.0041209,0.0029675,0.0015526,0.00060011,0.00017241,3.6909e-005,5.8929e-006,7.0197e-007,6.2398e-008,4.1391e-009,2.049e-010,
-1.1149e-005,-6.2732e-005,-0.00026283,-0.00081715,-0.0018683,-0.0030648,-0.0033384,-0.0016327,0.001773,0.0050025,0.0061428,0.0049876,0.0029021,0.0012413,0.00039391,9.3073e-005,1.6397e-005,2.1549e-006,2.1132e-007,1.5464e-008,8.4453e-010,
-8.5864e-006,-5.3325e-005,-0.00024677,-0.00084892,-0.002158,-0.0039889,-0.0051159,-0.0038054,0.00035926,0.005351,0.0081633,0.0075649,0.0049149,0.0023298,0.00081709,0.00021313,4.1434e-005,6.008e-006,6.5e-007,5.2476e-008,3.1616e-009,
-6.0056e-006,-4.1159e-005,-0.00021029,-0.00079973,-0.0022545,-0.0046617,-0.0068639,-0.0065302,-0.0022066,0.0044728,0.0095564,0.010319,0.0075326,0.0039661,0.0015385,0.00044319,9.5087e-005,1.5213e-005,1.8159e-006,1.6173e-007,1.075e-008,
-3.8149e-006,-2.8849e-005,-0.0001627,-0.00068353,-0.0021334,-0.0049114,-0.0081825,-0.0093147,-0.0056955,0.00207,0.00963,0.01259,0.010429,0.0061194,0.0026291,0.00083676,0.00019817,3.4986e-005,4.6074e-006,4.5273e-007,3.3196e-008,
-2.2009e-006,-1.8364e-005,-0.00011429,-0.00053023,-0.00183,-0.0046767,-0.0087385,-0.01152,-0.0094701,-0.0016859,0.0079246,0.01361,0.013005,0.0085494,0.0040755,0.0014342,0.00037505,7.3071e-005,1.0617e-005,1.151e-006,9.3108e-008,
-1.1532e-006,-1.0616e-005,-7.2909e-005,-0.00037338,-0.001424,-0.0040314,-0.0084012,-0.012596,-0.012659,-0.0060953,0.0044951,0.012813,0.014545,0.010799,0.0057281,0.0022311,0.0006445,0.0001386,2.2221e-005,2.6577e-006,2.3719e-007,
-5.4877e-007,-5.5736e-006,-4.2237e-005,-0.00023873,-0.0010055,-0.0031493,-0.0072931,-0.012303,-0.014478,-0.010129,0,0.010129,0.014478,0.012303,0.0072931,0.0031493,0.0010055,0.00023873,4.2237e-005,5.5736e-006,5.4877e-007,
-2.3719e-007,-2.6577e-006,-2.2221e-005,-0.0001386,-0.0006445,-0.0022311,-0.0057281,-0.010799,-0.014545,-0.012813,-0.0044951,0.0060953,0.012659,0.012596,0.0084012,0.0040314,0.001424,0.00037338,7.2909e-005,1.0616e-005,1.1532e-006,
-9.3108e-008,-1.151e-006,-1.0617e-005,-7.3071e-005,-0.00037505,-0.0014342,-0.0040755,-0.0085494,-0.013005,-0.01361,-0.0079246,0.0016859,0.0094701,0.01152,0.0087385,0.0046767,0.00183,0.00053023,0.00011429,1.8364e-005,2.2009e-006,
-3.3196e-008,-4.5273e-007,-4.6074e-006,-3.4986e-005,-0.00019817,-0.00083676,-0.0026291,-0.0061194,-0.010429,-0.01259,-0.00963,-0.00207,0.0056955,0.0093147,0.0081825,0.0049114,0.0021334,0.00068353,0.0001627,2.8849e-005,3.8149e-006,
-1.075e-008,-1.6173e-007,-1.8159e-006,-1.5213e-005,-9.5087e-005,-0.00044319,-0.0015385,-0.0039661,-0.0075326,-0.010319,-0.0095564,-0.0044728,0.0022066,0.0065302,0.0068639,0.0046617,0.0022545,0.00079973,0.00021029,4.1159e-005,6.0056e-006,
-3.1616e-009,-5.2476e-008,-6.5e-007,-6.008e-006,-4.1434e-005,-0.00021313,-0.00081709,-0.0023298,-0.0049149,-0.0075649,-0.0081633,-0.005351,-0.00035926,0.0038054,0.0051159,0.0039889,0.002158,0.00084892,0.00024677,5.3325e-005,8.5864e-006,
-8.4453e-010,-1.5464e-008,-2.1132e-007,-2.1549e-006,-1.6397e-005,-9.3073e-005,-0.00039391,-0.0012413,-0.0029021,-0.0049876,-0.0061428,-0.0050025,-0.001773,0.0016327,0.0033384,0.0030648,0.0018683,0.00081715,0.00026283,6.2732e-005,1.1149e-005,
-2.049e-010,-4.1391e-009,-6.2398e-008,-7.0197e-007,-5.8929e-006,-3.6909e-005,-0.00017241,-0.00060011,-0.0015526,-0.0029675,-0.0041209,-0.0039611,-0.0021835,0.00022693,0.001851,0.0021006,0.0014599,0.00071271,0.00025403,6.7004e-005,1.3146e-005,
-4.515e-011,-1.0062e-009,-1.6734e-008,-2.0769e-007,-1.9235e-006,-1.3293e-005,-6.8524e-005,-0.00026334,-0.0007532,-0.0015967,-0.0024814,-0.002742,-0.0019472,-0.00046354,0.00080942,0.0012697,0.0010263,0.00056261,0.00022269,6.4966e-005,1.4075e-005,
-9.0363e-012,-2.2216e-010,-4.076e-009,-5.5809e-008,-5.7024e-007,-4.3477e-006,-2.4731e-005,-0.00010491,-0.0003315,-0.00077808,-0.0013467,-0.0016852,-0.0014366,-0.00064712,0.00020683,0.0006621,0.00064557,0.00040125,0.00017695,5.7165e-005,1.3684e-005,
-1.6426e-012,-4.455e-011,-9.0174e-010,-1.3621e-008,-1.5354e-007,-1.2915e-006,-8.1058e-006,-3.7948e-005,-0.00013241,-0.00034371,-0.00066047,-0.00092739,-0.00091712,-0.00056148,-6.4024e-005,0.00028355,0.00036013,0.00025784,0.00012733,4.5634e-005,1.2077e-005
};
double Fo3[21][21]={-1.6426e-012,-4.455e-011,-9.0174e-010,-1.3621e-008,-1.5354e-007,-1.2915e-006,-8.1058e-006,-3.7948e-005,-0.00013241,-0.00034371,-0.00066047,-0.00092739,-0.00091712,-0.00056148,-6.4024e-005,0.00028355,0.00036013,0.00025784,0.00012733,4.5634e-005,1.2077e-005,
-9.0363e-012,-2.2216e-010,-4.076e-009,-5.5809e-008,-5.7024e-007,-4.3477e-006,-2.4731e-005,-0.00010491,-0.0003315,-0.00077808,-0.0013467,-0.0016852,-0.0014366,-0.00064712,0.00020683,0.0006621,0.00064557,0.00040125,0.00017695,5.7165e-005,1.3684e-005,
-4.515e-011,-1.0062e-009,-1.6734e-008,-2.0769e-007,-1.9235e-006,-1.3293e-005,-6.8524e-005,-0.00026334,-0.0007532,-0.0015967,-0.0024814,-0.002742,-0.0019472,-0.00046354,0.00080942,0.0012697,0.0010263,0.00056261,0.00022269,6.4966e-005,1.4075e-005,
-2.049e-010,-4.1391e-009,-6.2398e-008,-7.0197e-007,-5.8929e-006,-3.6909e-005,-0.00017241,-0.00060011,-0.0015526,-0.0029675,-0.0041209,-0.0039611,-0.0021835,0.00022693,0.001851,0.0021006,0.0014599,0.00071271,0.00025403,6.7004e-005,1.3146e-005,
-8.4453e-010,-1.5464e-008,-2.1132e-007,-2.1549e-006,-1.6397e-005,-9.3073e-005,-0.00039391,-0.0012413,-0.0029021,-0.0049876,-0.0061428,-0.0050025,-0.001773,0.0016327,0.0033384,0.0030648,0.0018683,0.00081715,0.00026283,6.2732e-005,1.1149e-005,
-3.1616e-009,-5.2476e-008,-6.5e-007,-6.008e-006,-4.1434e-005,-0.00021313,-0.00081709,-0.0023298,-0.0049149,-0.0075649,-0.0081633,-0.005351,-0.00035926,0.0038054,0.0051159,0.0039889,0.002158,0.00084892,0.00024677,5.3325e-005,8.5864e-006,
-1.075e-008,-1.6173e-007,-1.8159e-006,-1.5213e-005,-9.5087e-005,-0.00044319,-0.0015385,-0.0039661,-0.0075326,-0.010319,-0.0095564,-0.0044728,0.0022066,0.0065302,0.0068639,0.0046617,0.0022545,0.00079973,0.00021029,4.1159e-005,6.0056e-006,
-3.3196e-008,-4.5273e-007,-4.6074e-006,-3.4986e-005,-0.00019817,-0.00083676,-0.0026291,-0.0061194,-0.010429,-0.01259,-0.00963,-0.00207,0.0056955,0.0093147,0.0081825,0.0049114,0.0021334,0.00068353,0.0001627,2.8849e-005,3.8149e-006,
-9.3108e-008,-1.151e-006,-1.0617e-005,-7.3071e-005,-0.00037505,-0.0014342,-0.0040755,-0.0085494,-0.013005,-0.01361,-0.0079246,0.0016859,0.0094701,0.01152,0.0087385,0.0046767,0.00183,0.00053023,0.00011429,1.8364e-005,2.2009e-006,
-2.3719e-007,-2.6577e-006,-2.2221e-005,-0.0001386,-0.0006445,-0.0022311,-0.0057281,-0.010799,-0.014545,-0.012813,-0.0044951,0.0060953,0.012659,0.012596,0.0084012,0.0040314,0.001424,0.00037338,7.2909e-005,1.0616e-005,1.1532e-006,
-5.4877e-007,-5.5736e-006,-4.2237e-005,-0.00023873,-0.0010055,-0.0031493,-0.0072931,-0.012303,-0.014478,-0.010129,0,0.010129,0.014478,0.012303,0.0072931,0.0031493,0.0010055,0.00023873,4.2237e-005,5.5736e-006,5.4877e-007,
-1.1532e-006,-1.0616e-005,-7.2909e-005,-0.00037338,-0.001424,-0.0040314,-0.0084012,-0.012596,-0.012659,-0.0060953,0.0044951,0.012813,0.014545,0.010799,0.0057281,0.0022311,0.0006445,0.0001386,2.2221e-005,2.6577e-006,2.3719e-007,
-2.2009e-006,-1.8364e-005,-0.00011429,-0.00053023,-0.00183,-0.0046767,-0.0087385,-0.01152,-0.0094701,-0.0016859,0.0079246,0.01361,0.013005,0.0085494,0.0040755,0.0014342,0.00037505,7.3071e-005,1.0617e-005,1.151e-006,9.3108e-008,
-3.8149e-006,-2.8849e-005,-0.0001627,-0.00068353,-0.0021334,-0.0049114,-0.0081825,-0.0093147,-0.0056955,0.00207,0.00963,0.01259,0.010429,0.0061194,0.0026291,0.00083676,0.00019817,3.4986e-005,4.6074e-006,4.5273e-007,3.3196e-008,
-6.0056e-006,-4.1159e-005,-0.00021029,-0.00079973,-0.0022545,-0.0046617,-0.0068639,-0.0065302,-0.0022066,0.0044728,0.0095564,0.010319,0.0075326,0.0039661,0.0015385,0.00044319,9.5087e-005,1.5213e-005,1.8159e-006,1.6173e-007,1.075e-008,
-8.5864e-006,-5.3325e-005,-0.00024677,-0.00084892,-0.002158,-0.0039889,-0.0051159,-0.0038054,0.00035926,0.005351,0.0081633,0.0075649,0.0049149,0.0023298,0.00081709,0.00021313,4.1434e-005,6.008e-006,6.5e-007,5.2476e-008,3.1616e-009,
-1.1149e-005,-6.2732e-005,-0.00026283,-0.00081715,-0.0018683,-0.0030648,-0.0033384,-0.0016327,0.001773,0.0050025,0.0061428,0.0049876,0.0029021,0.0012413,0.00039391,9.3073e-005,1.6397e-005,2.1549e-006,2.1132e-007,1.5464e-008,8.4453e-010,
-1.3146e-005,-6.7004e-005,-0.00025403,-0.00071271,-0.0014599,-0.0021006,-0.001851,-0.00022693,0.0021835,0.0039611,0.0041209,0.0029675,0.0015526,0.00060011,0.00017241,3.6909e-005,5.8929e-006,7.0197e-007,6.2398e-008,4.1391e-009,2.049e-010,
-1.4075e-005,-6.4966e-005,-0.00022269,-0.00056261,-0.0010263,-0.0012697,-0.00080942,0.00046354,0.0019472,0.002742,0.0024814,0.0015967,0.0007532,0.00026334,6.8524e-005,1.3293e-005,1.9235e-006,2.0769e-007,1.6734e-008,1.0062e-009,4.515e-011,
-1.3684e-005,-5.7165e-005,-0.00017695,-0.00040125,-0.00064557,-0.0006621,-0.00020683,0.00064712,0.0014366,0.0016852,0.0013467,0.00077808,0.0003315,0.00010491,2.4731e-005,4.3477e-006,5.7024e-007,5.5809e-008,4.076e-009,2.2216e-010,9.0363e-012,
-1.2077e-005,-4.5634e-005,-0.00012733,-0.00025784,-0.00036013,-0.00028355,6.4024e-005,0.00056148,0.00091712,0.00092739,0.00066047,0.00034371,0.00013241,3.7948e-005,8.1058e-006,1.2915e-006,1.5354e-007,1.3621e-008,9.0174e-010,4.455e-011,1.6426e-012
};
double Fo4[21][21]={-1.6426e-012,-9.0363e-012,-4.515e-011,-2.049e-010,-8.4453e-010,-3.1616e-009,-1.075e-008,-3.3196e-008,-9.3108e-008,-2.3719e-007,-5.4877e-007,-1.1532e-006,-2.2009e-006,-3.8149e-006,-6.0056e-006,-8.5864e-006,-1.1149e-005,-1.3146e-005,-1.4075e-005,-1.3684e-005,-1.2077e-005,
-4.455e-011,-2.2216e-010,-1.0062e-009,-4.1391e-009,-1.5464e-008,-5.2476e-008,-1.6173e-007,-4.5273e-007,-1.151e-006,-2.6577e-006,-5.5736e-006,-1.0616e-005,-1.8364e-005,-2.8849e-005,-4.1159e-005,-5.3325e-005,-6.2732e-005,-6.7004e-005,-6.4966e-005,-5.7165e-005,-4.5634e-005,
-9.0174e-010,-4.076e-009,-1.6734e-008,-6.2398e-008,-2.1132e-007,-6.5e-007,-1.8159e-006,-4.6074e-006,-1.0617e-005,-2.2221e-005,-4.2237e-005,-7.2909e-005,-0.00011429,-0.0001627,-0.00021029,-0.00024677,-0.00026283,-0.00025403,-0.00022269,-0.00017695,-0.00012733,
-1.3621e-008,-5.5809e-008,-2.0769e-007,-7.0197e-007,-2.1549e-006,-6.008e-006,-1.5213e-005,-3.4986e-005,-7.3071e-005,-0.0001386,-0.00023873,-0.00037338,-0.00053023,-0.00068353,-0.00079973,-0.00084892,-0.00081715,-0.00071271,-0.00056261,-0.00040125,-0.00025784,
-1.5354e-007,-5.7024e-007,-1.9235e-006,-5.8929e-006,-1.6397e-005,-4.1434e-005,-9.5087e-005,-0.00019817,-0.00037505,-0.0006445,-0.0010055,-0.001424,-0.00183,-0.0021334,-0.0022545,-0.002158,-0.0018683,-0.0014599,-0.0010263,-0.00064557,-0.00036013,
-1.2915e-006,-4.3477e-006,-1.3293e-005,-3.6909e-005,-9.3073e-005,-0.00021313,-0.00044319,-0.00083676,-0.0014342,-0.0022311,-0.0031493,-0.0040314,-0.0046767,-0.0049114,-0.0046617,-0.0039889,-0.0030648,-0.0021006,-0.0012697,-0.0006621,-0.00028355,
-8.1058e-006,-2.4731e-005,-6.8524e-005,-0.00017241,-0.00039391,-0.00081709,-0.0015385,-0.0026291,-0.0040755,-0.0057281,-0.0072931,-0.0084012,-0.0087385,-0.0081825,-0.0068639,-0.0051159,-0.0033384,-0.001851,-0.00080942,-0.00020683,6.4024e-005,
-3.7948e-005,-0.00010491,-0.00026334,-0.00060011,-0.0012413,-0.0023298,-0.0039661,-0.0061194,-0.0085494,-0.010799,-0.012303,-0.012596,-0.01152,-0.0093147,-0.0065302,-0.0038054,-0.0016327,-0.00022693,0.00046354,0.00064712,0.00056148,
-0.00013241,-0.0003315,-0.0007532,-0.0015526,-0.0029021,-0.0049149,-0.0075326,-0.010429,-0.013005,-0.014545,-0.014478,-0.012659,-0.0094701,-0.0056955,-0.0022066,0.00035926,0.001773,0.0021835,0.0019472,0.0014366,0.00091712,
-0.00034371,-0.00077808,-0.0015967,-0.0029675,-0.0049876,-0.0075649,-0.010319,-0.01259,-0.01361,-0.012813,-0.010129,-0.0060953,-0.0016859,0.00207,0.0044728,0.005351,0.0050025,0.0039611,0.002742,0.0016852,0.00092739,
-0.00066047,-0.0013467,-0.0024814,-0.0041209,-0.0061428,-0.0081633,-0.0095564,-0.00963,-0.0079246,-0.0044951,0,0.0044951,0.0079246,0.00963,0.0095564,0.0081633,0.0061428,0.0041209,0.0024814,0.0013467,0.00066047,
-0.00092739,-0.0016852,-0.002742,-0.0039611,-0.0050025,-0.005351,-0.0044728,-0.00207,0.0016859,0.0060953,0.010129,0.012813,0.01361,0.01259,0.010319,0.0075649,0.0049876,0.0029675,0.0015967,0.00077808,0.00034371,
-0.00091712,-0.0014366,-0.0019472,-0.0021835,-0.001773,-0.00035926,0.0022066,0.0056955,0.0094701,0.012659,0.014478,0.014545,0.013005,0.010429,0.0075326,0.0049149,0.0029021,0.0015526,0.0007532,0.0003315,0.00013241,
-0.00056148,-0.00064712,-0.00046354,0.00022693,0.0016327,0.0038054,0.0065302,0.0093147,0.01152,0.012596,0.012303,0.010799,0.0085494,0.0061194,0.0039661,0.0023298,0.0012413,0.00060011,0.00026334,0.00010491,3.7948e-005,
-6.4024e-005,0.00020683,0.00080942,0.001851,0.0033384,0.0051159,0.0068639,0.0081825,0.0087385,0.0084012,0.0072931,0.0057281,0.0040755,0.0026291,0.0015385,0.00081709,0.00039391,0.00017241,6.8524e-005,2.4731e-005,8.1058e-006,
0.00028355,0.0006621,0.0012697,0.0021006,0.0030648,0.0039889,0.0046617,0.0049114,0.0046767,0.0040314,0.0031493,0.0022311,0.0014342,0.00083676,0.00044319,0.00021313,9.3073e-005,3.6909e-005,1.3293e-005,4.3477e-006,1.2915e-006,
0.00036013,0.00064557,0.0010263,0.0014599,0.0018683,0.002158,0.0022545,0.0021334,0.00183,0.001424,0.0010055,0.0006445,0.00037505,0.00019817,9.5087e-005,4.1434e-005,1.6397e-005,5.8929e-006,1.9235e-006,5.7024e-007,1.5354e-007,
0.00025784,0.00040125,0.00056261,0.00071271,0.00081715,0.00084892,0.00079973,0.00068353,0.00053023,0.00037338,0.00023873,0.0001386,7.3071e-005,3.4986e-005,1.5213e-005,6.008e-006,2.1549e-006,7.0197e-007,2.0769e-007,5.5809e-008,1.3621e-008,
0.00012733,0.00017695,0.00022269,0.00025403,0.00026283,0.00024677,0.00021029,0.0001627,0.00011429,7.2909e-005,4.2237e-005,2.2221e-005,1.0617e-005,4.6074e-006,1.8159e-006,6.5e-007,2.1132e-007,6.2398e-008,1.6734e-008,4.076e-009,9.0174e-010,
4.5634e-005,5.7165e-005,6.4966e-005,6.7004e-005,6.2732e-005,5.3325e-005,4.1159e-005,2.8849e-005,1.8364e-005,1.0616e-005,5.5736e-006,2.6577e-006,1.151e-006,4.5273e-007,1.6173e-007,5.2476e-008,1.5464e-008,4.1391e-009,1.0062e-009,2.2216e-010,4.455e-011,
1.2077e-005,1.3684e-005,1.4075e-005,1.3146e-005,1.1149e-005,8.5864e-006,6.0056e-006,3.8149e-006,2.2009e-006,1.1532e-006,5.4877e-007,2.3719e-007,9.3108e-008,3.3196e-008,1.075e-008,3.1616e-009,8.4453e-010,2.049e-010,4.515e-011,9.0363e-012,1.6426e-012
};

	IplImage *im=cvCreateImage( cvGetSize(im_input), 8, 1);
	cvSetImageCOI(im_input,2);
	cvCopy(im_input,im); 
	cvResetImageROI(im_input);
	
	CvMat mat_Fe1=cvMat(21,21,CV_64FC1,Fe1);
	CvMat mat_Fe2=cvMat(21,21,CV_64FC1,Fe2);
	CvMat mat_Fe3=cvMat(21,21,CV_64FC1,Fe3);
	CvMat mat_Fe4=cvMat(21,21,CV_64FC1,Fe4);
	CvMat mat_Fo1=cvMat(21,21,CV_64FC1,Fo1);
	CvMat mat_Fo2=cvMat(21,21,CV_64FC1,Fo2);
	CvMat mat_Fo3=cvMat(21,21,CV_64FC1,Fo3);
	CvMat mat_Fo4=cvMat(21,21,CV_64FC1,Fo4);

	CvMat *FIe1 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *FIe2 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *FIe3 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *FIe4 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *FIo1 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *FIo2 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *FIo3 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *FIo4 = cvCreateMat(hL,wL,CV_64FC1);

	cvFilter2D(im,FIe1,&mat_Fe1, cvPoint( -1, -1 ));
	cvFilter2D(im,FIe2,&mat_Fe2, cvPoint( -1, -1 ));
	cvFilter2D(im,FIe3,&mat_Fe3, cvPoint( -1, -1 ));
	cvFilter2D(im,FIe4,&mat_Fe4, cvPoint( -1, -1 ));
	cvFilter2D(im,FIo1,&mat_Fo1, cvPoint( -1, -1 ));
	cvFilter2D(im,FIo2,&mat_Fo2, cvPoint( -1, -1 ));
	cvFilter2D(im,FIo3,&mat_Fo3, cvPoint( -1, -1 ));
	cvFilter2D(im,FIo4,&mat_Fo4, cvPoint( -1, -1 ));

	CvMat *sum_FBo = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *sum_FBe = cvCreateMat(hL,wL,CV_64FC1);
	
	CvMat *If_FBo1_pow2 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *If_FBo2_pow2 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *If_FBo3_pow2 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *If_FBo4_pow2 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *If_FBe1_pow2 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *If_FBe2_pow2 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *If_FBe3_pow2 = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *If_FBe4_pow2 = cvCreateMat(hL,wL,CV_64FC1);
	
	cvMul(FIo1, FIo1, If_FBo1_pow2); 
	cvMul(FIo2, FIo2, If_FBo2_pow2); 
	cvMul(FIo3, FIo3, If_FBo3_pow2); 
	cvMul(FIo4, FIo4, If_FBo4_pow2); 
	
	cvAdd(If_FBo1_pow2, If_FBo2_pow2, sum_FBo);
	cvAdd(sum_FBo, If_FBo3_pow2, sum_FBo);
	cvAdd(sum_FBo, If_FBo4_pow2, sum_FBo);

	cvMul(FIe1, FIe1, If_FBe1_pow2); 
	cvMul(FIe2, FIe2, If_FBe2_pow2); 
	cvMul(FIe3, FIe3, If_FBe3_pow2); 
	cvMul(FIe4, FIe4, If_FBe4_pow2);

	cvAdd(If_FBe1_pow2, If_FBe2_pow2, sum_FBe);
	cvAdd(sum_FBe, If_FBe3_pow2, sum_FBe);
	cvAdd(sum_FBe, If_FBe4_pow2, sum_FBe);

	CvMat *mag_temp = cvCreateMat(hL,wL,CV_64FC1);
	
	mag_max=-1;
	cvAdd(sum_FBe, sum_FBo, mag_temp);
	for(int h=0;h<hL;h++)
		for(int w=0;w<wL;w++)
		{
			double val=cvmGet(mag_temp,h,w);
			double valsqrt=sqrt(val);
			mag[h][w]=valsqrt;
			if(valsqrt>mag_max)
				mag_max=valsqrt;
		}
	
	CvMat *If_FBe1_sum = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *If_FBe2_sum = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *If_FBe3_sum = cvCreateMat(hL,wL,CV_64FC1);
	CvMat *If_FBe4_sum = cvCreateMat(hL,wL,CV_64FC1);
	
	cvAdd(If_FBo1_pow2, If_FBe1_pow2, If_FBe1_sum);
	cvAdd(If_FBo2_pow2, If_FBe2_pow2, If_FBe2_sum);
	cvAdd(If_FBo3_pow2, If_FBe3_pow2, If_FBe3_sum);
	cvAdd(If_FBo4_pow2, If_FBe4_pow2, If_FBe4_sum);

	//timer.start();
	
	for(int i=0;i<hL;i++)
		for(int j=0;j<wL;j++)
		{
			double c1=cvmGet(If_FBe1_sum,i,j);
			double c2=cvmGet(If_FBe2_sum,i,j);
			double c3=cvmGet(If_FBe3_sum,i,j);
			double c4=cvmGet(If_FBe4_sum,i,j);
			
			if(c1>=c2&&c1>=c3&&c1>=c4)
			{
				double d=cvmGet(FIe1,i,j);
				if(d>0)
					mage[i][j]=1;
				else
					mage[i][j]=-1;
			}	
			if(c2>=c1&&c2>=c3&&c2>=c4)
			{
				double d=cvmGet(FIe2,i,j);
				if(d>0)
					mage[i][j]=1;
				else
					mage[i][j]=-1;
			}	
			if(c3>=c1&&c3>=c2&&c3>=c4)
			{
				double d=cvmGet(FIe3,i,j);
				if(d>0)
					mage[i][j]=1;
				else
					mage[i][j]=-1;
			}	
			if(c4>=c1&&c4>=c2&&c4>=c3)
			{
				double d=cvmGet(FIe4,i,j);
				if(d>0)
					mage[i][j]=1;
				else
					mage[i][j]=-1;
			}	
		}
	

	cvReleaseMat(&FIe1);
	cvReleaseMat(&FIe2);
	cvReleaseMat(&FIe3);
	cvReleaseMat(&FIe4);
	cvReleaseMat(&FIo1);
	cvReleaseMat(&FIo2);
	cvReleaseMat(&FIo3);
	cvReleaseMat(&FIo4);
	cvReleaseMat(&sum_FBo);
	cvReleaseMat(&sum_FBe);

	cvReleaseMat(&If_FBo1_pow2);
	cvReleaseMat(&If_FBo2_pow2);
	cvReleaseMat(&If_FBo3_pow2);
	cvReleaseMat(&If_FBo4_pow2);
	cvReleaseMat(&If_FBe1_pow2);
	cvReleaseMat(&If_FBe2_pow2);
	cvReleaseMat(&If_FBe3_pow2);
	cvReleaseMat(&If_FBe4_pow2);

	cvReleaseMat(&mag_temp);
	cvReleaseMat(&If_FBe1_sum);
	cvReleaseMat(&If_FBe2_sum);
	cvReleaseMat(&If_FBe3_sum);
	cvReleaseMat(&If_FBe4_sum);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc,char*argv[])
{
	if(argc!=6)
	{
		printf("input wrong!");
			return (-1);
	}
	//////////////////////////////////////////////////////////////////////////////参数初始化
	char*filename_disparity_map=argv[1];//输出结果
	char*filename_left_image=argv[2];//左图
	char*filename_right_image=argv[3];//右图
	int max_disp=atoi(argv[4]);//disparity的取值范围
	int scale=atoi(argv[5]);
//	float sigma_edge=atof(argv[6])/100;//0.08;
//	float sigma_spatial=atof(argv[6])/100;//0.08;
//	float sigma_range=atof(argv[6])/100;//0.07;
	bool refinement=true;
	float sigma_edge=0.05;//0.07
	float sigma_spatial=0.03;//0.03
	float sigma_range=0.13;//0.11
//	float sigma_range2=0.04;
	////////////////////////////////////////////////////////////////////////////
	IplImage *imL = cvLoadImage(filename_left_image);
	IplImage *imR = cvLoadImage(filename_right_image);
	int hL=imL->height;
	int wL=imL->width;
	
	unsigned char***image_left=qx_allocu_3(hL,wL,3);//allocate memory
	unsigned char***image_right=qx_allocu_3(hL,wL,3);
	unsigned char**disparity=qx_allocu(hL,wL);
	
	ToPoint(imL,imR,hL,wL,image_left,image_right);
	int ctmf_radius=2;
	double ****m_buf_d3=qx_allocd_4(5,hL,wL,max_disp);
	double ***m_cost_vol=m_buf_d3[0];
	double ***m_cost_vol_A=m_buf_d3[1];
	double ***m_cost_vol_temp=m_buf_d3[2];
	double ***m_cost_vol_right=m_buf_d3[3];
	double ***m_cost_temp_2w=m_buf_d3[4];
	unsigned char ***m_buf_u2=qx_allocu_3(3,hL,wL);
	unsigned char **m_disparity=m_buf_u2[0];
	unsigned char **m_disparity_right=m_buf_u2[1];
	unsigned char **m_mask_occlusion=m_buf_u2[2];
	double****m_buf_temp_chen=qx_allocd_4(2,hL,wL,max_disp);
	double***m_out1=m_buf_temp_chen[0];
	double***m_out2=m_buf_temp_chen[1];
	float***m_buf_f2;
	m_buf_f2=qx_allocf_3(3,hL,wL);
	m_second_derivative_shifted=m_buf_f2[0];
	m_second_derivative_left=m_buf_f2[1];
	m_second_derivative_right=m_buf_f2[2];
	unsigned char****m_buf_u3;
	m_buf_u3=qx_allocu_4(2,hL,wL,3);
	m_image_shifted=m_buf_u3[0];
	m_image_temp=m_buf_u3[1];
	double ***m_buf_d2=qx_allocd_3(2,hL,wL);
	mag_l=m_buf_d2[0];
	mag_r=m_buf_d2[1];
	int ***m_buf_i2=qx_alloci_3(2,hL,wL);
	mage_l=m_buf_i2[0];
	mage_r=m_buf_i2[1];
	////////////////////////////////////////////////////////////////////////////////////////////
	edge_strength_calculation(imL,hL,wL,mag_l,mage_l,mag_max_l);
	edge_strength_calculation(imR,hL,wL,mag_r,mage_r,mag_max_r);
	////////////////////////////////////////////////////////////////////////////////////////////
	//printf("cost computation begin!");
	
	matching_cost_from_color_and_gradient(hL,wL,max_disp,m_cost_vol,image_left,image_right,1);
	matching_cost_from_color_and_gradient(hL,wL,max_disp,m_cost_vol_right,image_right,image_left,-1);
	
	ctmf_img(image_left,hL,wL);
	ctmf_img(image_right,hL,wL);
	start = clock();
	//printf("cost aggregation begin!");
	Recursive_tf(m_disparity,m_cost_vol_A,m_cost_vol,image_left,sigma_spatial,sigma_range,hL,wL,max_disp,m_cost_vol_temp,m_out1,m_out2,0,sigma_edge);
//	first_order_recursive_bilateral_filter(m_disparity,m_cost_vol_A,m_cost_vol,image_left,sigma_spatial,sigma_range,hL,wL,max_disp,m_cost_vol_temp,m_cost_temp_2w);
	
	ctmf(m_disparity[0],disparity[0],wL,hL,wL,wL,ctmf_radius,1,hL*wL);
	finish = clock();
//	printf("该程序运行时间为:%d ms", finish - start);
	if(refinement==true)
	{
		Recursive_tf(m_disparity,m_cost_vol_A,m_cost_vol_right,image_right,sigma_spatial,sigma_range,hL,wL,max_disp,m_cost_vol_temp,m_out1,m_out2,1,sigma_edge);
	//	first_order_recursive_bilateral_filter(m_disparity,m_cost_vol_A,m_cost_vol_right,image_right,sigma_spatial,sigma_range,hL,wL,max_disp,m_cost_vol_temp,m_cost_temp_2w);
		
		ctmf(m_disparity[0],m_disparity_right[0],wL,hL,wL,wL,ctmf_radius,1,hL*wL);

		qx_detect_occlusion_left_right(m_mask_occlusion,disparity,m_disparity_right,hL,wL,max_disp);
		image_zero(m_cost_vol,hL,wL,max_disp);

		for(int y=0;y<hL;y++) for(int x=0;x<wL;x++) if(!m_mask_occlusion[y][x])
		{
			for(int d=0;d<max_disp;d++)
			{
				m_cost_vol[y][x][d]=abs(disparity[y][x]-d);
			}
		}
		//for(int y=0;y<hL;y++) for(int x=0;x<wL;x++) if(m_mask_occlusion[y][x]==255)
		//{
		//	disparity[y][x]=0;
		//}
		Recursive_tf(disparity,m_cost_vol_A,m_cost_vol,image_left,sigma_spatial,sigma_range*0.3,hL,wL,max_disp,m_cost_vol_temp,m_out1,m_out2,0,sigma_edge);//sigma_range*0.5
	//	first_order_recursive_bilateral_filter(disparity,m_cost_vol_A,m_cost_vol,image_left,sigma_spatial,sigma_range*0.5,hL,wL,max_disp,m_cost_vol_temp,m_cost_temp_2w);
		
	//	ctmf(m_disparity[0],disparity[0],wL,hL,wL,wL,ctmf_radius,1,hL*wL);
	}
	//printf("export disparity!");
	

	Mat disparity_img(hL,wL,CV_8UC1);
	for(int h=0;h<hL;h++)
		for(int w=0;w<wL;w++)
		{
			disparity_img.at<uchar>(h,w)=disparity[h][w]*scale;//
		}
	imwrite(filename_disparity_map, disparity_img);
	
	qx_freeu_3(m_buf_u2); m_buf_u2=NULL;
	qx_freed_4(m_buf_d3); m_buf_d3=NULL;
	qx_freed_4(m_buf_temp_chen); m_buf_temp_chen=NULL;
	qx_freef_3(m_buf_f2); m_buf_f2=NULL;
	qx_freeu_4(m_buf_u3); m_buf_u3=NULL;
	qx_freed_3(m_buf_d2); m_buf_d2=NULL;
	qx_freei_3(m_buf_i2); m_buf_i2=NULL;
	cvReleaseImage(&imL);
	cvReleaseImage(&imR);
	return 0;
}