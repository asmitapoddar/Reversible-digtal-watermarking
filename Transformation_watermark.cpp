#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "vector"
#include"math.h"
#include"stdlib.h"
#include "iostream"
#include <stdio.h>
//#include <omp.h>
#include "omp.h"
 
using namespace cv;
using namespace std;

int LN, RN;
const double huge_val = 999;

struct histograms
{
	double error;
	int count;
};

struct Mat_overhead
{
	Mat watermarked_image;
	vector<int> overhead;
	int len_wm;
};

struct LM_RM
{
	int LM;
	int RM;
	Mat error_histogram;
};

struct errors
{
	Mat x;
	Mat e;
};
 
int rounding( double value )
{
    double intpart, fractpart;
    fractpart = modf(value, &intpart);
	
    //for +ve numbers, when fraction is 0.5, odd numbers are rounded up 
    //and even numbers are rounded down 
    //and vice versa for -ve numbers
	if (value > 0)
	{
		if(fractpart<0.5)
			return (int)intpart;
		else
			return (int)(intpart+1);
	}
	else
	{
		if(fractpart>-0.5)
			return (int)intpart;
		else
			return (int)(intpart-1);
	}

    //if ((fabs(fractpart) != 0.5) || ((((int)intpart) % 2) != 0))
    //    return (int)(value + (value >= 0 ? 0.5 : -0.5));    //(value >= 0 ? 0.5 : -0.5)
    //else
    //    return (int)intpart;
}
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


errors  Interpolated( Mat p)
{
	//Interpolate
	
	  int nr= p.size().height;
	  int nc= p.size().width;
	 
	  errors error_em;
	  
	   error_em.x = Mat_<double>(nr,nc);
	   error_em.x= -1 * Mat::ones(nr, nc, CV_64FC1);
	   
	   //Create Margin
	   int i, j;
	   
	    for (i = 0; i<nr; i++)
		{
			for (j = 0; j<nc; j++)
			{
				if( ( i == 0 ) || ( j == 0) || ( i == nr-1 ) || ( j == nc-1 ) )
				error_em.x.at<double>(i,j)=p.at<double>(i,j);
			}
		}
		
	   // Down Sample => Get the Low Resolution Pixels
		for (i = 0; i<nr;i++)
		{
			for (j = 0; j<nc; j++)
			{
				if( ( i%2 == 0 ) && (j%2 == 0 ) )
				error_em.x.at<double>(i,j)=p.at<double>(i,j);
			}
		}
		
		//Error Matrix
		
	    error_em.e = Mat_<double>(nr,nc);
		error_em.e = huge_val* Mat::ones( nr, nc, CV_64FC1);
				
		// STEP I (center high resolution pixels)
		
		double x_45, x_135, u;
		double sigma_e45, sigma_e135, w45, w135, S45_1, S45_2, S45_3, S135_1, S135_2, S135_3;

		for (i = 0; i<(nr/2 - 1); i++)
		 {
			 for (j = 0; j<(nc/2 -1); j++)
			 {  
			   x_45 = ( error_em.x.at<double>(2*i,2*j+2) + error_em.x.at<double>(2*i+2,2*j) ) / 2; 
			   x_135 = ( error_em.x.at<double>(2*i,2*j) + error_em.x.at<double>(2*i+2,2*j+2) ) / 2;
			   u =( x_45+x_135)/2;
			  // cout<<"index: "<<2*i+1<<" , "<<2*j+1<<endl;
			   S45_1 = error_em.x.at<double>(2*i,2*j+2);
			   S45_2=x_45;
			   S45_3= error_em.x.at<double>(2*i+2,2*j);
			   S135_1= error_em.x.at<double>(2*i,2*j);
			   S135_2= x_135;
			   S135_3= error_em.x.at<double>( 2*i+2, 2*j+2);

			   //cout <<S45_1<<" "<<S45_2<<" "<<S45_3<<endl;
			  sigma_e45 = ( (S45_1 - u)*(S45_1 - u) + (S45_2 - u)*(S45_2 - u) + (S45_3 - u)*(S45_3 - u) ) / 3;
			  sigma_e135 = ( (S135_1 - u)*(S135_1 - u) + (S135_2 - u)*(S135_2 - u) + (S135_3 - u)*(S135_3 - u) ) / 3;
			  w45 = sigma_e135 / ( sigma_e45 + sigma_e135 );
			  w135 = 1 - w45; 
			  if( (sigma_e45 == 0) && (sigma_e135 == 0) )            
			     error_em.x.at<double>((2*i+1),(2*j+1)) = u;
		      else
			      error_em.x.at<double>((2*i+1),(2*j+1)) =rounding( w45*x_45 + w135*x_135 );
			 
			 	  error_em.e.at<double>((2*i+1),(2*j+1)) = (error_em.x.at<double>((2*i+1),(2*j+1)) - p.at<double>((2*i+1),(2*j+1)));
			    }
		 }
		
		 // STEP II (residual high resolution pixels)

		 double x_0, x_90; double sigma_e0, sigma_e90, w0, w90, S0_1, S0_2, S0_3, S90_1, S90_2, S90_3;
		
		 for (i = 1; i<=(nr-2); i++)
		 {
			 for (j = 1; j<=(nc-2); j++)
			 {
				 if( (i+j)%2 == 1 )
				 {
					 //cout<<"Index"<<i<<" "<<j<<endl;
					x_0 = ( error_em.x.at<double>(i,j-1) + error_em.x.at<double>(i,j+1) ) / 2;
					x_90 = ( error_em.x.at<double>(i-1,j) + error_em.x.at<double>(i+1,j) ) / 2;
					 u = ( error_em.x.at<double>(i,j-1) + error_em.x.at<double>(i,j+1) + error_em.x.at<double>(i-1,j) + error_em.x.at<double>(i+1,j) ) / 4;

					 S0_1 = error_em.x.at<double>(i,j-1);
					 S0_2= x_0;
					 S0_3=error_em.x.at<double>(i,j+1);
					 S90_1 = error_em.x.at<double>(i-1,j);
					 S90_2= x_90;
					 S90_3 = error_em.x.at<double>(i+1,j);

					 //cout<<S90_1<<" "<<S90_2<<" "<<S90_3<<endl;

					sigma_e0 = ( (S0_1 - u)*(S0_1 - u) + (S0_2 - u)*(S0_2 - u) + (S0_3 - u)*(S0_3 - u) ) / 3;
				    sigma_e90 = ( (S90_1 - u)*(S90_1 - u) + (S90_2 - u)*(S90_2 - u) + (S90_3 - u)*(S90_3 - u) ) / 3;            
					w0 = sigma_e90 / ( sigma_e0 + sigma_e90 );
					w90 = 1 - w0;
					//cout<<sigma_e0<< " ";
					//cout<<w0<< " ";
					if( (sigma_e0 == 0) && (sigma_e90 == 0) )
						   error_em.x.at<double>(i,j) = u;
					else
						  error_em.x.at<double>(i,j) = rounding(( w0*x_0 + w90*x_90 ));
					error_em.e.at<double>(i,j) =(error_em.x.at<double>(i,j) - p.at<double>(i,j) );
				 }
			 }
		}
					  	   
	 return error_em;       //Started comment
		
}

std::vector<double> unique(const cv::Mat& input, bool sort = false)
{
    std::vector<double> out;
    for (int y = 0; y < input.rows; ++y)
    {
        const double* row_ptr = input.ptr<double>(y);
        for (int x = 0; x < input.cols; ++x)
        {
            double value = row_ptr[x];

            if ( std::find(out.begin(), out.end(), value) == out.end() )
                out.push_back(value);
        }
    }

    if (sort)
        std::sort(out.begin(), out.end());
    return out;
}


LM_RM createHistogram(Mat e_hist)
{
	int i, j, k;
	int nr= e_hist.size().height;
	int nc= e_hist.size().width;
	
	 double max_err = 0;
	 double min_err = 0; 
	
	LM_RM lm_rm;
	
	lm_rm.error_histogram= Mat_<double>(nr, nc);
	
	//Calculating unique element
	std::vector<double> unik = unique (e_hist, true);

	int histSize = unik.size();
	//for(i=0;i<unik.size();i++)
	//cout << unik[i]<<" ";   // Unique elements in the range of -90 to 78, 999
    
	Mat x_hist=Mat_<double>(nr,nc);
	for( i=0; i < nr; i++)
	{
		for( j = 0; j < nc; j++)
		x_hist.at<double>(i,j)=e_hist.at<double>(i,j);
	}
	//x_hist=e_hist;
	Mat new_hist = Mat_<double>(nr,nc);
	new_hist=e_hist;
	
	// Finding max and min error
	
	//string tyy11 =  type2str( e_hist.type() );   
	  //cout<<"e_hist IS"<<tyy11<<endl;
	for (i = 0; i< nr; i++)
	{
		for (j = 0; j<nc; j++)
		{
			if( e_hist.at<double>(i,j) == huge_val )    //Matrix type has to be double
             x_hist.at<double>(i,j) = 0;
			else if( e_hist.at<double>(i,j) > max_err )
            max_err = e_hist.at<double>(i,j);
            else if( e_hist.at<double>(i,j) < min_err )
             min_err = e_hist.at<double>(i,j);
		}
		
	}
	//cout<<"MAX ERROR: "<<max_err<<endl; cout<<"Min ERROR: "<<min_err<<endl;
	
    histograms *histogram= new histograms[histSize];
	 	 
	 for (i = 0; i<unik.size(); i++)
	 {
		 histogram[i].error = unik[i];
		 histogram[i].count = 0;
	 } 
	 
	 //Create histogram
	 for (i = 0; i<nr; i++)
	 {
		 for (j = 0; j<nc; j++)
		 {
			 if( e_hist.at<double>(i,j) == huge_val )
            ;
			 else
			 {
				 for (k = 0; k<unik.size(); k++)
				 {
					 if( e_hist.at<double>(i,j) == histogram[k].error )
                    histogram[k].count = histogram[k].count + 1;
				 }
			 }
		 }
	 }
	 for(i=0;i<nr; i++)
	 {
		 for(j=0;j<nc;j++)
			 lm_rm.error_histogram.at<double>(i,j)=e_hist.at<double>(i,j);
	 }
	

	//Display Histogram
	 //cout<<"Histogram:"<<endl;
	 //for(i=0;i<unik.size();i++)
	 //{
	//	cout<<histogram[i].error <<" "<< histogram[i].count<<endl;
	 //}   //OK
	 

	 //Compute LM RM

	int max_count = -1;
	for (i = 0; i<unik.size(); i++)
	{
		//X(i) = histogram(1,i).error; % For Plot
		// Y(i) = histogram(1,i).count; % For Plot
		 if( histogram[i].count > max_count )
		 {
			 max_count = histogram[i].count;
			 lm_rm.LM = histogram[i].error;
		 }
	}
	
	int second_max_count = -1;
	for (i = 0; i<unik.size(); i++)
	{
		if( histogram[i].count == max_count )
        ;
		else 
			{
				if( histogram[i].count > second_max_count )
				{
					second_max_count = histogram[i].count;
					lm_rm.RM = histogram[i].error;
				}
			}
	}
	
	//cout<<"LM: "<<lm_rm.LM<< "RM: "<<lm_rm.RM<<endl;  //LM=0, RM=-2

	//Calculate LE and RE
	histograms *LE= new histograms[unik.size()] ;
	histograms *RE= new histograms[unik.size()];
	int posLE = 0;
	int posRE = 0;
    for (i = 0; i<unik.size(); i++)
	{
		if( histogram[i].error <= lm_rm.LM )
		{
			posLE = posLE + 1;
			LE[posLE].error = histogram[i].error;
			LE[posLE].count = histogram[i].count;
		}
		if( histogram[i].error >= lm_rm.RM )
		{
			posRE = posRE + 1;
			RE[posRE].error = histogram[i].error;
			RE[posRE].count = histogram[i].count;
		}
	}
	//Compute LN RN
	int mincount_LE = LE[1].count;
	LN = LE[1].error;
	for (j = 0; j<posLE; j++)
	{
		if( LE[j].count < mincount_LE )
		{
			mincount_LE = LE[j].count;
	         LN = LE[j].error;
		}
	}
	int mincount_RE = RE[1].count;
	RN = RE[1].error;
    for (j = 0; j<posRE; j++)
	{
		if( RE[j].count < mincount_RE )
		{
			mincount_RE = RE[j].count;
			RN = RE[j].error;
		}
	}  */
	//cout<<"LN: "<<LN<<"RN: "<<RN<<endl;
	//for"(i=0;i<unik.size();i++)
	//{	cout<<"LE error: "<<LE[i].error<<" LE count: "<<LE[i].count<<endl;
		//cout<<"RE error: "<<RE[i].error<<" RE count: "<<RE[i].count<<endl;
	//}
	//cout<<e_hist; // getting 999
	 
	//cout<<lm_rm.error_histogram;
	return lm_rm;   //started comment
}

vector<int> EightBits(int I)
{
	vector<int> bitstring(8); int i=0;
	//[bitstring] = EightBits(I) converts integer I into its 8-bit binary
	//encoding, bitstring.
	//Example, EightBits(2) returns bitstring = 00000010
	//Input:   I: an integer
	//Output:  bitstring: a 1X8 array of 0s and 1s   
	
	
	//binary encoding of I consisting of 8 or lesser number of bits
	//bin = dec2bin(I8);
	if(I<0)
	I=-I;
	int bin[8];

	while(I>0)
	{
		bin[i]=I%2;
		I=I/2;
		i++;  //will signify no. of bits occupied
	}
	    
	int j;
	//If number of bits < 8, shift each bit right so that the rightmost bit occupies the 8th position.
	
	//Shift each bit of bin right by (8 - nc) positions
	 for (j = 0; j<i; j++)
		bitstring[ 7-j] = bin[j]; 
	 //Fill in the remaining (8 - nc) left bits by zeros.
    for (j = 1; j<(8 - i); j++)
        bitstring[j] = 0; 
		    
	 return bitstring;
}


Mat_overhead EmbedWatermarkInterpolation(Mat p, vector<int> &watermark_original)
{
	int i, j;
	int nr= p.size().height;
	int nc= p.size().width;
	LM_RM lm_rm;
	errors err;
	
	vector<int> watermark;   //to be embedded
	//int64 t= getTickCount();

	err = Interpolated(p);
	//cout<<"Time taken: "<< ( ((double)(getTickCount())-t)/ getTickFrequency() ); 
	long long int sume=0, sumx=0;
	
	lm_rm=createHistogram(err.e);
	
	 Mat e_new= Mat_<double>(nr,nc);
	 for( i =0 ; i < nr; i++)
	 {
		 for( j=0; j < nc; j++)
			 e_new.at<double>(i,j)=err.e.at<double>(i,j);
	 }
	//Create bitstream to be embedded-------------------------------------------
    //Location Map for overflow or underflow.
	
	 	int len_locmap = 0;  //Total no. of pixels with values 0, 255, 1, 254.
	
	for (i = 0; i<nr; i++)
	{
		for (j = 0; j<nc; j++)
		{  
			//Only 254 or 1 can change to 255 and 0 respectively
			 // So we take max length (for 0,255,1,254)
			 // actual loc map consists of those pixel positions which are either
			 // 0, 255 or have changed to 0, 255.
			 //  So actual length may be smaller than len_locmap.
			   
			if( (p.at<double>(i,j) == 0 ) || ( p.at<double>(i,j) == 255 ) || ( p.at<double>(i,j) == 1 ) || ( p.at<double>(i,j) == 254 ) )
			len_locmap = len_locmap + 1;    //NOT COMING CORRECTLY 
		}
	} 
	
	//cout<<"LENLOCMAP: "<<len_locmap<<endl;
	
	Mat_overhead ob1;
	ob1.watermarked_image= Mat_<double>(nr,nc);
	//watermarked_image = zeros(nr,nc);
	 for( i =0 ; i < nr; i++)
	 {
		 for( j=0; j < nc; j++)
			 ob1.watermarked_image.at<double>(i,j)=p.at<double>(i,j);
	 }
		
	int poswm = 0; // To keep track of bits to be embedded
	int cnt = 0; // To find Maximum Embedding capacity
	vector<int> locmap; // To prevent overflow/underflow

	//cout<< "LM ANND RM: "<<lm_rm.LM<<" "<<lm_rm.RM<<endl; 
	// Get Maximum Capacity----------
								
	for( i=0; i<nr; i++)
	{
		for (j = 0; j<nc; j++)
			{
				if( ( p.at<double>(i,j) == 0 ) || ( p.at<double>(i,j) == 255 ) ) ;     
				else        
				{
					if( err.e.at<double>(i,j) == huge_val ) ;
				    else if( err.e.at<double>(i,j) <= (double)lm_rm.LM )
						{
							if( err.e.at<double>(i,j) == (double)lm_rm.LM )
							cnt = cnt + 1;
						}
					 else if( err.e.at<double>(i,j) >= (double)lm_rm.RM )
						 {
							if( err.e.at<double>(i,j) == (double)lm_rm.RM )
							cnt = cnt + 1;
						 }
				}
			}
	}
	
	
	//cout<< " Wtaermark size: "<<watermark_original.size()<<endl;;
	//cout<<" Length of location map: "<<len_locmap<<endl;
	//cout<<"Count: "<<cnt<<endl;
	//for(i=0;i<watermark_original.size();i++)
	//cout<<watermark[i];
	if( watermark_original.size() > ( cnt - 18 - len_locmap ) )    //original watermrk
	{
		for( i=0; i < ( cnt - 18 - len_locmap ) ; i++)
			{
				watermark.push_back(watermark_original[i]);
				//cout<<watermark[i];
		}
	}
	
    ob1.len_wm = watermark.size();
	///cout<<"Watermark size: "<<ob1.len_wm;
	//cout<<"Maximum embedding capacity: "<<cnt - 18 - len_locmap <<endl<<endl;
	
	//Store just 18 bits for 2 signed integers along with 'len_locmap' bits for overflow/underflow.
	int k = 1; double v;
	for (i = 0; i<nr; i++)
	{
		for (j = 0; j<nc; j++)
		{
			if( (i == 0) || (j == 0) || (i == 7) || (j == 7) )
			{
				if( (i%2==0) && (j%2==1) )
				{
					if( k <= ( 18 + len_locmap ) )
					{
						if (p.at<double>(i,j)<0)
							v=-p.at<double>(i,j);
						else
							v=p.at<double>(i,j);

						watermark.push_back( ( (int) (v ) ) %2 );  //LSB= (int)(p.at<double>(i,j))%2
					    k = k + 1;
					} 
				}
				else if( (i%2==1) && (j%2==0) )
					 {
						 if( k <= ( 18 + len_locmap ) )
						 {
							 if (p.at<double>(i,j)<0)
							   v=-p.at<double>(i,j);
							 else
								v=p.at<double>(i,j);

							watermark.push_back( ( (int) (v ) )%2 );
							k = k + 1;
						 }
					}
			}
		}
	}
	//cout<<" k: "<<k<<endl;
	// Embed--------------------------------------------------------------------
	 //string somethings11 =  type2str( ob1.watermarked_image.type() );   
	 // cout<<"ob1.watermarked_imaged "<<somethings11<<endl;
	int ncwm = watermark.size(); int b, sign_e;
	
	for (i = 0;i<nr; i++)
	{
		for (j = 0; j<nc; j++)
		{
			if( ( p.at<double>(i,j) == 0 ) || ( p.at<double>(i,j) == 255 ) )
            locmap.push_back(0);
            else
			{
				if( lm_rm.error_histogram.at<double>(i,j) == huge_val )  // Do not Embed (Low resolution or marginal pixels)
                ob1.watermarked_image.at<double>(i,j) = p.at<double>(i,j);
			    else if( lm_rm.error_histogram.at<double>(i,j) <= lm_rm.LM )
				{
					sign_e = -1;   // sign(e) = -1 if e in LE i.e. <= LM
					if( lm_rm.error_histogram.at<double>(i,j) == lm_rm.LM ) // Embed next bit
						{
							//cnt = cnt + 1;
							if( poswm < ncwm )  // More bits to embed
								{
									poswm = poswm + 1;
									b = watermark[poswm-1];
								}
							 else
							    b = 0;  // No more bits => e will be same
						     e_new.at<double>(i,j) = lm_rm.error_histogram.at<double>(i,j) + sign_e * b;
						}
					else if( lm_rm.error_histogram.at<double>(i,j) < lm_rm.LM ) // Embed 1
						e_new.at<double>(i,j) = lm_rm.error_histogram.at<double>(i,j) + sign_e * 1;
				
					 ob1.watermarked_image.at<double>(i,j) = err.x.at<double>(i,j) - e_new.at<double>(i,j);
						// update locmap
					if( ( ob1.watermarked_image.at<double>(i,j) == 0 ) || ( ob1.watermarked_image.at<double>(i,j) == 255 ) )
                    locmap.push_back( 1 );
				}
				else if( lm_rm.error_histogram.at<double>(i,j) >= lm_rm.RM )
				{
					sign_e = 1;  //  sign(e) = 1 if e in RE i.e. <= RM
					 if( lm_rm.error_histogram.at<double>(i,j) == lm_rm.RM )  // Embed next bit
						{
							//cnt = cnt + 1;
							if( poswm < ncwm )  // More bits to embed
							{
								poswm = poswm + 1;
								b = watermark[poswm-1];
							}
							else
								b = 0;  // No more bits => e will be same

							e_new.at<double>(i,j) =lm_rm.error_histogram.at<double>(i,j) + sign_e * b;
						}
					 else if( lm_rm.error_histogram.at<double>(i,j) > lm_rm.RM )  // Embed 1
					   e_new.at<double>(i,j) = lm_rm.error_histogram.at<double>(i,j) + sign_e * 1;
				
                ob1.watermarked_image.at<double>(i,j) = err.x.at<double>(i,j) - e_new.at<double>(i,j);
                //update locmap
                if( ( ob1.watermarked_image.at<double>(i,j) == 0 ) || ( ob1.watermarked_image.at<double>(i,j) == 255 ) )
                    locmap.push_back(1);
				}
			}
		}
	}
	 //cout<<lm_rm.error_histogram;
	
	//cout<< "LOCMAP length: "<<locmap.size();
	//for(i=0;i<locmap.size();i++)
	//	cout<<locmap[i];
	//cout<<endl;
	//cout<<" LM"<<lm_rm.LM<<" RM "<<lm_rm.RM;  //LM=0, RM=1 
	
	// GET binary representations of LM and RM----------------
	vector<int> temp = EightBits(lm_rm.LM);
	vector<int> LM_bin(9);
	for(i=0 ; i<9; i++)
		LM_bin[i]=0;
	if( lm_rm.LM < 0 )
		LM_bin[0] = 1;
	else
		LM_bin[0] = 0;
	for (i = 0; i<8; i++)
	{
		if( temp[i] == 0 )
			LM_bin[i+1] = 0;
		else if( temp[i] == 1 )
			LM_bin[i+1] = 1;
	}
    
	temp = EightBits(lm_rm.RM);
	vector<int>RM_bin(9);
	for(i=0 ; i<9; i++)
		RM_bin[i]=0;
	if( lm_rm.RM < 0 )
		RM_bin[0] = 1;
	else
		RM_bin[0] = 0;
	for (i = 0; i<8; i++)
	{
		if( temp[i] == 0 )
			RM_bin[i+1] = 0;
		else if( temp[i] == 1 )
			RM_bin[i+1] = 1;
	}
	
	// cout<< "RM BIN: ";
	//for(i=0;i<9;i++)
	//	 cout<<RM_bin[i]<< lm_rm.RM; 

	// Concatenate binary reps of LM and RM and locmap
	vector<int> b_vect;
	
//	cout<< "LM PART: ";
	for(i=0; i<9;i++)
	{
		b_vect.push_back(LM_bin[i]);
	//	cout<<b_vect[i];
		}
	//cout<<endl;
//cout<< "RM PART: ";
	for(i=0; i<9;i++)
	{
		b_vect.push_back(RM_bin[i]);
		//cout<<b_vect[i];
	}
	//cout<<endl;
	//b_vect.push_back(LM_bin);

	for(i=0; i<locmap.size();i++)
	b_vect.push_back(locmap[i]);
	
	ob1.overhead = b_vect;
	
	//cout<<"Overhead:"<<endl;
	//for(i=0;i<ob1.overhead.size();i++)
	//	cout<<ob1.overhead[i];
	//cout<<endl; 
	//cout<<endl<<"Original overhead size: "<<ob1.overhead.size()<<endl;
	//long long int aaa=0;
	//for(i=0;i<nr;i++)
	//{
	//	for(j=0;j<nc;j++)
	//		aaa+=ob1.watermarked_image.at<double>(i,j);
	//}
	//cout<<endl<<" Embed watermarked imaage sum original: "<<aaa<<endl;
	return ob1;
}

int bi2de(vector<int> bin)
{
	int i=7, dec = 0, rem, num, base = 1;

    while (i>=0)
    {
       dec = dec + bin[i] * base;
       //cout<<dec<<endl;
        base = base * 2;
        i--;
    }
    return dec;
}

Mat YCoCg2RGB( Mat watermarked_image_ee)
{
	int nr= watermarked_image_ee.size().height;
	int nc= watermarked_image_ee.size().width;
	watermarked_image_ee.convertTo(watermarked_image_ee, CV_64FC3);

	// Split into Y, Co, Cg
	Mat Y = Mat_<double>(nr, nc); 
	Mat Co = Mat_<double>(nr, nc);
	Mat Cg = Mat_<double>(nr, nc); 
	Mat rgb_image;
	Mat YCoCg[3];

	split(watermarked_image_ee, YCoCg);

	 //for(int i=0; i< nr; i++)
	//	 {
	//		 for(int j=0; j<nc; j++)	
	//		{
	//			Y.at<double>(i,j)= watermarked_image_ee.at<Vec3d>(i,j)[0];
	//			 Co.at<double>(i,j) = watermarked_image_ee.at<Vec3d>(i,j)[1];
	//			 Cg.at<double>(i,j) = watermarked_image_ee.at<Vec3d>(i,j)[2];
	//		}
	//	 }
	Y=YCoCg[0];
	Co=YCoCg[1];
	Cg=YCoCg[2];
	 
  	  Mat R = Mat_<double>(nr, nc);
	  Mat G = Mat_<double>(nr, nc); 
	  Mat B = Mat_<double>(nr, nc);
	  Mat t = Mat_<double>(nr, nc);

	  int i, j;
	  
	  for(i=0;i<nr;i++)
	  {
		  for (j=0;j<nc; j++)
		  {	  t.at<double>(i,j) =  (double)(Y.at<double>(i,j)) - (double)(floor)(0.5*Cg.at<double>(i,j));
				G.at<double>(i,j) = (double)(Cg.at<double>(i,j)) + (double)(t.at<double>(i,j));
				B.at<double>(i,j) = (double)(t.at<double>(i,j)) - (double)(floor)(0.5*Co.at<double>(i,j));
				R.at<double>(i,j) = (double)(B.at<double>(i,j)) + (double)(Co.at<double>(i,j));
		  }
	  }
	
	 Mat YCoCg_new[3];
	 YCoCg_new[0] = B;
	 YCoCg_new[1] = G;
	 YCoCg_new[2] = R;
	 //cout<<YCoCg_new[0];
	  merge( YCoCg_new,3, rgb_image); 
	   //cout<<rgb_image;
	  
	   return rgb_image;
}

Mat RGB2YCoCg( Mat images)
{
	int nr= images.size().height;
	int nc= images.size().width;
	images.convertTo(images, CV_64FC3);
	//Split into R, G, B
	Mat b=Mat_<double>(nr, nc); 
	Mat g=Mat_<double>(nr, nc); 
	Mat r=Mat_<double>(nr, nc);
	Mat arr[3];
	split(images, arr);                  //GIVES Lesser time and better result

	//for(int i=0; i< nr; i++)
	//	 {
	//		 for(int j=0; j<nc; j++)	
	//		{
	//			b.at<double>(i,j)= images.at<Vec3d>(i,j)[0];
	//			 g.at<double>(i,j) = images.at<Vec3d>(i,j)[1];
	//			 r.at<double>(i,j) = images.at<Vec3d>(i,j)[2];
	//		}
	//	 }
	b=arr[0];
	g=arr[1];
	r=arr[2];
	  
	  Mat Co= Mat_<double>(nr, nc);
	  Mat t= Mat_<double>(nr, nc);
	  Mat Cg= Mat_<double>(nr, nc);
	  Mat Y= Mat_<double>(nr, nc);
	 	
	  int i,j;
	  for(i=0;i<nr;i++)
	  {
		  for (j=0;j<nc; j++)
		  {
			  Co.at<double>(i,j) = (double)(r.at<double>(i,j)) - (double)(b.at<double>(i,j));
			t.at<double>(i,j) = (double)(b.at<double>(i,j)) + (double)(floor)(0.5*Co.at<double>(i,j)); 
			Cg.at<double>(i,j) = (double)(g.at<double>(i,j)) - (double)(t.at<double>(i,j));
			Y.at<double>(i,j) = (double)(t.at<double>(i,j)) + (double)(floor)(0.5*Cg.at<double>(i,j));
		  }
	  }
	  
	   Mat arr_new[3];
	  arr_new[0]=Y;
	  arr_new[1]=Co;
	  arr_new[2]=Cg;
	  
	   Mat colors;
	   merge( arr_new,3, colors);
		
	    return colors;
}


Mat extractWatermarkInterpolation( Mat watermarked_image_e, int l, vector<int> overheads, vector<int> wmm)
{
	int nr= watermarked_image_e.size().height;
	int nc= watermarked_image_e.size().width;
	int i, j;
	errors errs;
	//cout<<watermarked_image;  //Some values around 1000
	   	
	//long long sums1=0;
	//for( i=0; i<nr; i++)
	//{
	//	for(j=0;j<nc;j++)
	//		sums1+=watermarked_image_e.at<double>(i,j);
	//}
	//cout<<endl<<"In extract function, embedded Watermarked image sum: "<<sums1<<endl;

	errs= Interpolated(watermarked_image_e);
	
	Mat e_new  = Mat_<double>(nr,nc);
	Mat xxx  = Mat_<double>(nr,nc);
	Mat eee = Mat_<double>(nr,nc);
	//cout<<e_new;
	for(i=0; i<nr; i++)
	{
		for(j=0; j<nc; j++)
		{
			e_new.at<double>(i,j) = errs.e.at<double>(i,j);
			eee.at<double>(i,j) = errs.e.at<double>(i,j);
			xxx.at<double>(i,j) = errs.x.at<double>(i,j);
		}
	}
	
	vector<int> watermark;
	Mat  p=Mat_<double>(nr, nc);
	//cout<<"Empty Watermark size:"<<watermark.size()<<endl;
	for(i=0;i<nr;i++)
	{
		for(j=0;j<nc;j++)
			p.at<double>(i,j) = watermarked_image_e.at<double>(i, j) ;
	}
	int LM_extract, RM_extract;
	
	// Compute LM, RM
	
	vector<int> LM_RM_e(overheads.size());
	for(i=0;i<overheads.size();i++)
		LM_RM_e[i]= overheads[i];
	   	

	//cout<<"LM NAD RM: "<<overhead.size()<<endl;
	//for(i=0; i<overheads.size(); i++)
	//cout<<overheads[i];

	std::vector<int>  temp(8);
	
	if( LM_RM_e[0] == 1 )
	{
		for(i=0;i<8;i++)
		temp[i]=overheads[i+1];
		LM_extract = (-1)*bi2de ( temp);
	}
	else if( LM_RM_e[0] == 0 )
	{
		for(i=0;i<8;i++)
		temp[i]=overheads[i+1]; 
		LM_extract = bi2de( temp );
	}

	if( LM_RM_e[9] == 1 )
	{
		for(i=10;i<=17;i++)
			temp[i-10]=overheads[i];
		RM_extract = (-1)*bi2de( temp );
	}
	else if( LM_RM_e[9] == 0 )
	{
		for(i=10;i<=17;i++)
			temp[i-10]=overheads[i];
		RM_extract = bi2de( temp);
	}
	//cout<<"EXTRACT"<<endl<<endl;
	//cout<<"LM_extract: "<<LM_extract<<"RM_extract: "<<RM_extract << endl;      //LM=0, RM=-2

	// Compute Locmap
	// locmap = LM_RM( 1, 19:( 19 + len_locmap - 1 ) );
	vector<int> locmap(overheads.size()-18);
	if( overheads.size() > 18 )
		{for( i=18; i<overheads.size(); i++)
		locmap[i-18]=overheads[i];
	    }

	//cout<<"LOCMAP ";
	//cout<<"Overhead size: "<<overheads.size()<<endl;
	//cout<<" Locmap size: "<<locmap.size()<<endl;                      

		// EXTRACT -----------------------------------------------------------------
   	
	int sign_e, b;
	Mat p1 = Mat_<double>(nr,nc);
	   p1= -1 * Mat::ones(nr, nc, CV_64FC1);
	 
	int pos_locmap = 0;
	
	
	for (i = 0; i<nr;i++)
	 {
		 for (j = 0; j<nc; j++)
	      {
			  if( e_new.at<double>(i,j) == huge_val )
				{
					//e(i,j) = 999;
					eee.at<double>(i,j) = 0;
					p1.at<double>(i,j) = p.at<double>(i,j);
				    if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
						pos_locmap = pos_locmap + 1;
				}
           else if( e_new.at<double>(i,j) == LM_extract )
		   {
			   if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
			   {
				   pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
						p1.at<double>(i,j) = p.at<double>(i,j);
			         else if( locmap[pos_locmap-1] == 1 )
					 {
						 sign_e = -1;
						 b = 0;
						watermark.push_back(b);
						eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					  }
			   }
				else
				{
					sign_e = -1;
					b = 0;
					watermark.push_back(b);
					eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
					p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				}
			}
          else if( e_new.at<double>(i,j) == RM_extract )
		  {
			  if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
                {
					pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
						p1.at<double>(i,j) = p.at<double>(i,j);
					else if( locmap[pos_locmap-1] == 1 )
                    {
						sign_e = 1;
						b = 0;
						watermark.push_back(b);
						eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					}
			  }
				else
				{
					sign_e = 1;
					b = 0;
					watermark.push_back(b);
					eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
					p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				}   
		   }
			else if( e_new.at<double>(i,j) == (LM_extract-1) )
			{
				if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
				{
					pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
						p1.at<double>(i,j) = p.at<double>(i,j);
				    else if( locmap[pos_locmap-1] == 1 )
					{
						sign_e = -1;
						b = 1;
						watermark.push_back(b);
						eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					}
				}
				else
				{
					sign_e = -1;
					b = 1;
					watermark.push_back(b);
					eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
				    p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				 }
			}
			else if( e_new.at<double>(i,j) == (RM_extract+1) )
            {
				if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
                {
					pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
					   p1.at<double>(i,j) = p.at<double>(i,j);
					else if( locmap[pos_locmap-1] == 1 )
					{ sign_e = 1;
						b = 1;
						watermark.push_back(b);
						eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					 }
				}
				else
				{
					sign_e = 1;
					b = 1;
					 watermark.push_back(b);
					eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
					p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				}
			}
            
			else if( e_new.at<double>(i,j) < (LM_extract-1) )
            {
				if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
                {
					pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
						p1.at<double>(i,j) = p.at<double>(i,j);
					else if( locmap[pos_locmap-1] == 1 )
                    {
						sign_e = -1;
						eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*1;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					}
				}
            else
			{
				sign_e = -1;
                eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*1;
                p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				}
			}
            
		 else if( e_new.at<double>(i,j) > (RM_extract+1) )
			{
				if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
			 {
					pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
						p1.at<double>(i,j) = p.at<double>(i,j);
					else if( locmap[pos_locmap-1] == 1 )
					{
						sign_e = 1;
						 eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*1;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					}
				}
				else
				{
					sign_e = 1;
					 eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*1;
					p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				}
			}
		}
	}

	vector<int> tempp(l);
	for(i=0; i<l;i++)
	tempp[i]=watermark[i];
	
	//long long sums=0;
	//for( i=0; i<nr; i++)
	//{
	//	for(j=0;j<nc;j++)
	//		sums+=p1.at<double>(i,j);
	//}
	//cout<<endl<<" Extracted sum: "<<sums<<endl;
	//int flag=0;
	//for(i=0;i<tempp.size();i++)
	//{
	//	//<<watermark[i];
	//	if(tempp[i]==wmm[i])
	//		flag++;
	//}
	//cout<<"Extracted Watermark size: "<<tempp.size();
	//cout<<" Flag_watermark: "<<flag<<endl;

	return p1;
} 

double MSE ( Mat matrix1, Mat matrix2)
{
	//   'nmse' gives the Mean Square Error of matrices 1 and 2
	int nr, nc, ns;  double nmse = 0, temp; int i, j, k;

	//Check dimensions of 2 matrices
	if ( matrix1.size() == matrix2.size() )
	{
		//Get dimensions of the matrices
		nr = matrix1.size().height;
		nc = matrix1.size().width;
		ns = matrix1.channels(); 
		for (i = 0; i < nr; i++)
		{
			for (j = 0; j < nc; j++)
				{
					for (k = 0; k < ns; k++)
						{
							if ( matrix1.at<Vec3d>(i,j)[k] == matrix2.at<Vec3d>(i,j)[k] )
							 ; 
							else
							{
								temp = matrix1.at<Vec3d>(i,j)[k] - matrix2.at<Vec3d>(i,j)[k];
								nmse = nmse + temp * temp;
							}
						}
				}
		}
	}
            
   else
		cout<< endl<< " Dimension mismatch in MSE calculation."<<endl; //Error if dimensions mismatch
	nmse = nmse / (nr*nc*ns);
	return nmse;
}

void PSNR(Mat image_original, Mat image_modified )
{
	//[ psnr ] = PSNR( image_original, image_modified )
	//   Computes PSNR value of an image.
	//   Input: image_original -> Original image filename.
	//          image_modified -> Modified image filename.
	//   Output: psnr -> PSNR of the modified image (image_modified) w.r.t the
	//                   original image (image_original).

	double mse = MSE( image_original, image_modified );
	double psnr = 10 * log10( 255*255 / mse );

	cout<< endl << "Calculated PSNR = " << psnr;
}



int main(int argc, char **argv ) 
{
     Mat image;
       image = imread("G:\\zelda.tiff", CV_LOAD_IMAGE_COLOR);  
 
       if(! image.data )                             
       {
              cout <<  "Could not open or find the image" << std::endl ;
              return -1;
       }
 	 
	 namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
       imshow( "Display window", image );             
	   

	  double t= getTickCount();
	   Mat color = RGB2YCoCg (image);
	 //  imshow("color", color);
	  	   
      int nor= color.size().height;
	  int noc= color.size().width;
     
	 long long int summ1=0, summ2=0, summ3=0;
	 
	  //image.convertTo(image, CV_64FC3);
	 //Create Channel
	   int i, j, k; 
	  
	  Mat p1 = Mat_<double>(nor,noc);
	  Mat p2 = Mat_<double>(nor,noc);
	  Mat p3 = Mat_<double>(nor,noc);
	  
	  Mat_overhead o1, o2, o3;
	  Mat watermarked_image_channels[3], watermarked_image_final;
	  Mat rgb_image_watermarked;
	      
	   for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{
	   			p1.at<double>(i,j)=color.at<Vec3d>(i,j)[0]; //Y
				p2.at<double>(i,j)=color.at<Vec3d>(i,j)[1];  //Co
				p3.at<double>(i,j)=color.at<Vec3d>(i,j)[2];  //Cg
			}
	   }
	   
	 /*   for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{ summ1+=p1.at<double>(i,j);
			summ2+=p2.at<double>(i,j);
			summ3+=p3.at<double>(i,j);
			}
		}
	   cout<<"SUM of 1st channel original: "<< summ1<<endl;
	   cout<<"SUM of 2nd channel original: "<< summ2<<endl;
	   cout<<"SUM of 3rd channel original: "<< summ3<<endl; */
	 
	/*   vector<int> watermarks(999999);     //Watermark in int
	  
	   for(i=0; i< 999999; i++)
		   {
			  // watermarks[i]=rand() % 2 ;
			   watermarks[i]=1;
		 }
			
	  o1 = EmbedWatermarkInterpolation(p1, watermarks);  //Y_wm
	  o2 = EmbedWatermarkInterpolation(p2, watermarks);  //Co_wm
	  o3 = EmbedWatermarkInterpolation(p3, watermarks);  //Cg_wm
	  //std::cout << "rgb_image:   " << rgb_image.size().width << " x " << rgb_image.size().height << " x " << rgb_image.channels() << std::endl;

	  watermarked_image_channels[0]= o1.watermarked_image;
	  watermarked_image_channels[1] = o2.watermarked_image;
	  watermarked_image_channels[2] = o3.watermarked_image;
	  merge( watermarked_image_channels, 3, watermarked_image_final);
	
	// imshow("Watermarked image in YCoCg", watermarked_image_final) ;
	rgb_image_watermarked= YCoCg2RGB(watermarked_image_final);
	//cout<<(rgb_image_watermarked-image);
	//rgb_image_watermarked.convertTo(rgb_image_watermarked, CV_8UC3);
	//imshow("Watermarked image in RGB", rgb_image_watermarked) ;
	
	
	//RECEIVER END
	
	Mat image_2b_extracted = RGB2YCoCg( rgb_image_watermarked);
	
	Mat z1 = Mat_<double>(nor,noc);
	Mat z2 = Mat_<double>(nor,noc);
	Mat z3 = Mat_<double>(nor,noc);
	
	for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{
	   			z1.at<double>(i,j)= image_2b_extracted.at<Vec3d>(i,j)[0];
				z2.at<double>(i,j)= image_2b_extracted.at<Vec3d>(i,j)[1];
				z3.at<double>(i,j)= image_2b_extracted.at<Vec3d>(i,j)[2];
			}
	   }
	 
	Mat extracted_image_channels[3], extracted_image;
	
	extracted_image_channels[0] = extractWatermarkInterpolation( z1, o1.len_wm, o1.overhead, watermarks); //WATERMARKED IN YCoCg
	extracted_image_channels[1] = extractWatermarkInterpolation( z2, o2.len_wm, o2.overhead, watermarks);
	extracted_image_channels[2] = extractWatermarkInterpolation( z3, o3.len_wm, o3.overhead, watermarks);
	//cout<<"LENGTH OF WATERMARK: "<<o1.len_wm + o2.len_wm + o3.len_wm;
	//cout<<(extracted_image_channels[1]);   // Breaking here

	merge( extracted_image_channels, 3, extracted_image);
	//imshow( "Extracted image in YCoCg", extracted_image);
	//cout<<"Extracted_image"<<extracted_image.at<Vec3d>(500,500)[0]<<endl;
	
	Mat final_image= YCoCg2RGB(extracted_image);
	cout<<"Time : "<< ( ((double)(getTickCount())-t)/ getTickFrequency() ); 
	//final_image.convertTo(final_image, CV_8UC3 );
	//imshow( "Extracted image final", final_image);
	//final_image.convertTo(final_image, CV_8UC3);
	image.convertTo(image, CV_64FC3);
	int test_count1=0, test_count2=0, test_count3=0;
	for(i=0;i<nor;i++)
	{
		for(j=0;j<noc; j++)
		{
				if ( extracted_image_channels[0].at<double>(i,j) != p1.at<double >(i,j) )
				test_count1 ++;
				if (extracted_image_channels[1].at<double>(i,j) != p2.at<double >(i,j) )
				test_count2 ++;
				if ( extracted_image_channels[2].at<double>(i,j) != p3.at<double >(i,j) )
				test_count3 ++;
		}
	}
	
	cout<<endl<<" Test count: "<<test_count1<< " " <<test_count2<< " "<< test_count3<<endl;
	int test_count11=0, test_count22=0, test_count33=0;
	for(i=0;i<nor;i++)
	{
		for(j=0;j<noc; j++)
		{
			
				if ( final_image.at<Vec3d>(i,j)[0] != image.at<Vec3d>(i,j)[0] )
				test_count11 ++;
				if (final_image.at<Vec3d>(i,j)[1] != image.at<Vec3d>(i,j)[1] )
				test_count22 ++;
				if ( final_image.at<Vec3d>(i,j)[2] != image.at<Vec3d>(i,j)[2] )
				test_count33 ++;
		}
	}
	
	cout<<endl<<" Test count: "<<test_count11<< " " <<test_count22<< " "<< test_count33<<endl;
	PSNR(image, rgb_image_watermarked);

	long long int z=0, zz=0, zzz=0;
	for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{ z+=extracted_image_channels[0].at<double>(i,j);
			zz+=extracted_image_channels[1].at<double>(i,j);
			zzz+=extracted_image_channels[2].at<double>(i,j);
			}

		}
	   cout<<"SUM of 1st channel final: "<< z<<endl;
	   cout<<"SUM of 2nd channel final: "<< zz<<endl;
	   cout<<"SUM of 3rd channel final: "<< zzz<<endl;
	   long long int summm1=0, summm2=0, summm3=0;
	     for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{ summm1+=image.at<Vec3d>(i,j)[0];
			summm2+=image.at<Vec3d>(i,j)[1];
			summm3+=image.at<Vec3d>(i,j)[2];
			}
		}
	   cout<<"SUM of 1st channel original: "<< summm1<<endl;
	   cout<<"SUM of 2nd channel original: "<< summm2<<endl;
	   cout<<"SUM of 3rd channel original: "<< summm3<<endl;
	   long long int summm11=0, summm22=0, summm33=0;
	     for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{ summm11+=final_image.at<Vec3d>(i,j)[0];
			summm22+=final_image.at<Vec3d>(i,j)[1];
			summm33+=final_image.at<Vec3d>(i,j)[2];
			}
		}
	   cout<<"SUM of 1st channel final: "<< summm11<<endl;
	   cout<<"SUM of 2nd channel final: "<< summm22<<endl;
	   cout<<"SUM of 3rd channel final: "<< summm33<<endl;
	
	
	cvWaitKey();
   return 0;
}
	
	/*vector<int> LM_RM(o1.overhead.size());
	for(i=0;i<o1.overhead.size();i++)
		LM_RM[i]= o1.overhead[i];
	*/	
	//namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
	
	/*Mat a=Mat_<double>(4,4);
	double k=-3.2;
	for(int i =0;i<4;i++)
		{for(int j=0;j<4;j++)
			a.at<double>(i,j)=k;
	k+=2.6;
	}
	for(int i =0;i<4;i++)
		{for(int j=0;j<4;j++)
		cout<<a.at<double>(i,j)<< " ";
	cout<<endl;
	}
	
	cout<<endl;
	for(int i =0;i<4;i++)
		{for(int j=0;j<4;j++)
		cout<<rounding(a.at<double>(i,j))<< " ";
	cout<<endl;
	}
	 cout<<endl<<floor(3/2); */

	/*int i=3, j;
	Mat y, z;
	for(int i =0;i<4;i++)
		{for(int j=0;j<4;j++)
		y.at<double>(i,j)=i;
	}
	for(int i =0;i<4;i++)
		{for(int j=0;j<4;j++)
		z.at<double>(i,j)=0;
	}
	cout<<endl;
	int64 t= getTickCount();
	//Test loop;
	parallel_for_ (cv::Range(0,100), Test(i,j));
		cout<<"Y: "<<y<<endl;
		cout<<endl<<"Time 1: "<< ( ((double)(getTickCount())-t)/ getTickFrequency() )<<endl;
		//double tt= getTickCount();

		for (i = 0; i < 100; ++i)
		{
			//cout<<" H";
			cout<< "Y[i]: "<<y[i];
		}
	
		//cout<<"Time 2: "<< ( ((double)(getTickCount())-tt)/ getTickFrequency() ); */
	//double t = getTickCount();
	//for( int i=0; i<11000; i++)
	//cout<<" H";
	//cout<<"Time 2: "<< ( ((double)(getTickCount())-t)/ getTickFrequency() );
	/*int64 t= getTickCount();
	long long int s=0; int i;
	 //opencv setNumThreads(2); int counter=2;
	//cout<<"No of threads:"<<o<<endl;;
	//omp_set_num_threads(2);
//	int o = omp_get_thread_num();
	omp_set_dynamic(0);
	omp_set_num_threads(4);
	
	#pragma omp parllel for
	
	for( i= 0; i<1000; i++)
	{
		int o = omp_get_num_threads();
		cout<<"No of threads:"<<o<<endl;
		//cout<<i<<" ";
				
	}
	
	
	cout<<"Sum: "<<s<<endl;
	cout<<"Time taken: "<< ( ((double)(getTickCount())-t)/ getTickFrequency() );
	int64 tt= getTickCount();
	long long int ss=0;
	omp_set_num_threads(2);
#pragma omp parallel 
{
    #pragma omp for
    for (int i = 0; i < 10; i++)
    {
         cout<<"Thread no.: "<<omp_get_thread_num()<<" ";
    }
}
    //#pragma omp parllel for
	//for(i=0; i<100000; i++)
	//	ss=ss+i;
	//cout<<"Sum: "<<ss<<endl;
	//cout<<"Time taken: "<< ( ((double)(getTickCount())-tt)/ getTickFrequency() );
    cvWaitKey();
   return 0;
}
*/
/*
class Means : public cv:: ParallelLoopBody 
{
	private: 
		Mat s;
		int i;
		int** x_45; 
	public:
		 
	Means (Mat ss, int ii, int** xx_45) : s(ss), i(ii), x_45(xx_45) {}
	//	void operator()( const cv::Range &r )  const {}
	 void operator()( const cv::Range &r )  const
	{
		int j;
		
		cout<<" HI ";
		for( j = r.start; j < r.end; ++j)
		{
		   x_45[i][j] = j; 
		  // s.at<int>(i,j)=j;
			 cout<<x_45[i][j]<<" ";  
	}

		//cout<<" Printing y: "<<yoyo<<endl;
	}
};

int main() {
	namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
	Mat a= Mat_<double>(4,4); int i, j;
	for(i =0;i<4;i++)
		{for( j=0;j<4;j++)
		a.at<double>(i,j)=i;
	}
	int** q = new int*[4];

	for(int i = 0; i < 4; ++i)
    q[i]= new int[512];

	for(i=0;i<4; i++)
		parallel_for_(cv::Range(0,4), Means(a, i, q));
		for(i=0;i<4; i++)
		{
			for(j=0;j<4;j++)
				cout<<q[i][j];
			cout<<endl;
		} 
cvWaitKey();
return 0;
} */
		#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "vector"
#include"math.h"
#include"stdlib.h"
#include "iostream"
 
using namespace cv;
using namespace std;

int LN, RN;
const double huge_val = 999;

struct histograms
{
	double error;
	int count;
};

struct Mat_overhead
{
	Mat watermarked_image;
	vector<int> overhead;
	int len_wm;
};

struct LM_RM
{
	int LM;
	int RM;
	Mat error_histogram;
};

struct errors
{
	Mat x;
	Mat e;
};
 
int rounding( double value )
{
    double intpart, fractpart;
    fractpart = modf(value, &intpart);
	
    //for +ve numbers, when fraction is 0.5, odd numbers are rounded up 
    //and even numbers are rounded down 
    //and vice versa for -ve numbers
	if (value > 0)
	{
		if(fractpart<0.5)
			return (int)intpart;
		else
			return (int)(intpart+1);
	}
	else
	{
		if(fractpart>-0.5)
			return (int)intpart;
		else
			return (int)(intpart-1);
	}

 }
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


errors  Interpolated( Mat p)
{
	//Interpolate
	
	  int nr= p.size().height;
	  int nc= p.size().width;
	 	  
	  errors error_em;
	  
	   error_em.x = Mat_<double>(nr,nc);
	   error_em.x= -1 * Mat::ones(nr, nc, CV_64FC1);
	   
	   //Create Margin
	   int i, j;
	   
	    for (i = 0; i<nr; i++)
		{
			for (j = 0; j<nc; j++)
			{
				if( ( i == 0 ) || ( j == 0) || ( i == nr-1 ) || ( j == nc-1 ) )
				error_em.x.at<double>(i,j)=p.at<double>(i,j);
			}
		}
		
	   // Down Sample => Get the Low Resolution Pixels
		for (i = 0; i<nr;i++)
		{
			for (j = 0; j<nc; j++)
			{
				if( ( i%2 == 0 ) && (j%2 == 0 ) )
				error_em.x.at<double>(i,j)=p.at<double>(i,j);
			}
		}
		
		//Error Matrix
		
	    error_em.e = Mat_<double>(nr,nc);
		error_em.e = huge_val* Mat::ones( nr, nc, CV_64FC1);
				
		// STEP I (center high resolution pixels)
		
		double x_45, x_135, u;
		double sigma_e45, sigma_e135, w45, w135, S45_1, S45_2, S45_3, S135_1, S135_2, S135_3;

		 for (i = 0; i<(nr/2 - 1); i++)
		 {
			 for (j = 0; j<(nc/2 -1); j++)
			 {  
			   x_45 = ( error_em.x.at<double>(2*i,2*j+2) + error_em.x.at<double>(2*i+2,2*j) ) / 2; 
			   x_135 = ( error_em.x.at<double>(2*i,2*j) + error_em.x.at<double>(2*i+2,2*j+2) ) / 2;
			   u =( x_45+x_135)/2;
			   S45_1 = error_em.x.at<double>(2*i,2*j+2);
			   S45_2=x_45;
			   S45_3= error_em.x.at<double>(2*i+2,2*j);
			   S135_1= error_em.x.at<double>(2*i,2*j);
			   S135_2= x_135;
			   S135_3= error_em.x.at<double>( 2*i+2, 2*j+2);
			  sigma_e45 = ( (S45_1 - u)*(S45_1 - u) + (S45_2 - u)*(S45_2 - u) + (S45_3 - u)*(S45_3 - u) ) / 3;
			  sigma_e135 = ( (S135_1 - u)*(S135_1 - u) + (S135_2 - u)*(S135_2 - u) + (S135_3 - u)*(S135_3 - u) ) / 3;
			  w45 = sigma_e135 / ( sigma_e45 + sigma_e135 );
			  w135 = 1 - w45; 
			  if( (sigma_e45 == 0) && (sigma_e135 == 0) )            
			     error_em.x.at<double>((2*i+1),(2*j+1)) = u;
		      else
			      error_em.x.at<double>((2*i+1),(2*j+1)) =rounding( w45*x_45 + w135*x_135 );
			 
			 	  error_em.e.at<double>((2*i+1),(2*j+1)) = (error_em.x.at<double>((2*i+1),(2*j+1)) - p.at<double>((2*i+1),(2*j+1)));
			    }
		 }
		 
		 // STEP II (residual high resolution pixels)

		 double x_0, x_90; double sigma_e0, sigma_e90, w0, w90, S0_1, S0_2, S0_3, S90_1, S90_2, S90_3;
		 for (i = 1; i<=(nr-2); i++)
		 {
			 for (j = 1; j<=(nc-2); j++)
			 {
				 if( (i+j)%2 == 1 )
				 {
					 //cout<<"Index"<<i<<" "<<j<<endl;
					x_0 = ( error_em.x.at<double>(i,j-1) + error_em.x.at<double>(i,j+1) ) / 2;
					x_90 = ( error_em.x.at<double>(i-1,j) + error_em.x.at<double>(i+1,j) ) / 2;
					 u = ( error_em.x.at<double>(i,j-1) + error_em.x.at<double>(i,j+1) + error_em.x.at<double>(i-1,j) + error_em.x.at<double>(i+1,j) ) / 4;

					 S0_1 = error_em.x.at<double>(i,j-1);
					 S0_2= x_0;
					 S0_3=error_em.x.at<double>(i,j+1);
					 S90_1 = error_em.x.at<double>(i-1,j);
					 S90_2= x_90;
					 S90_3 = error_em.x.at<double>(i+1,j);

					sigma_e0 = ( (S0_1 - u)*(S0_1 - u) + (S0_2 - u)*(S0_2 - u) + (S0_3 - u)*(S0_3 - u) ) / 3;
				        sigma_e90 = ( (S90_1 - u)*(S90_1 - u) + (S90_2 - u)*(S90_2 - u) + (S90_3 - u)*(S90_3 - u) ) / 3;            
					w0 = sigma_e90 / ( sigma_e0 + sigma_e90 );
					w90 = 1 - w0;
					//cout<<sigma_e0<< " ";
					//cout<<w0<< " ";
					if( (sigma_e0 == 0) && (sigma_e90 == 0) )
						   error_em.x.at<double>(i,j) = u;
					else
						  error_em.x.at<double>(i,j) = rounding(( w0*x_0 + w90*x_90 ));
					error_em.e.at<double>(i,j) =(error_em.x.at<double>(i,j) - p.at<double>(i,j) );
				 }
			 }
		 } 
		
	 return error_em;       //Started comment
		
}

std::vector<double> unique(const cv::Mat& input, bool sort = false)
{
    std::vector<double> out;
    for (int y = 0; y < input.rows; ++y)
    {
        const double* row_ptr = input.ptr<double>(y);
        for (int x = 0; x < input.cols; ++x)
        {
            double value = row_ptr[x];

            if ( std::find(out.begin(), out.end(), value) == out.end() )
                out.push_back(value);
        }
    }

    if (sort)
        std::sort(out.begin(), out.end());
    return out;
}


LM_RM createHistogram(Mat e_hist)
{
	int i, j, k;
	int nr= e_hist.size().height;
	int nc= e_hist.size().width;
	
	 double max_err = 0;
	 double min_err = 0; 
	
	LM_RM lm_rm;
	
	lm_rm.error_histogram= Mat_<double>(nr, nc);
	
	//Calculating unique element
	std::vector<double> unik = unique (e_hist, true);

	int histSize = unik.size();
	
	Mat x_hist=Mat_<double>(nr,nc);
	for( i=0; i < nr; i++)
	{
		for( j = 0; j < nc; j++)
		x_hist.at<double>(i,j)=e_hist.at<double>(i,j);
	}
	//x_hist=e_hist;
	Mat new_hist = Mat_<double>(nr,nc);
	new_hist=e_hist;
	
	// Finding max and min error
	
	for (i = 0; i< nr; i++)
	{
		for (j = 0; j<nc; j++)
		{
			if( e_hist.at<double>(i,j) == huge_val )    //Matrix type has to be double
             x_hist.at<double>(i,j) = 0;
			else if( e_hist.at<double>(i,j) > max_err )
            max_err = e_hist.at<double>(i,j);
            else if( e_hist.at<double>(i,j) < min_err )
             min_err = e_hist.at<double>(i,j);
		}
		
	}
		
	 histograms *histogram= new histograms[histSize];
	 	 
	 for (i = 0; i<unik.size(); i++)
	 {
		 histogram[i].error = unik[i];
		 histogram[i].count = 0;
	 } 
	 
	 //Create histogram
	 for (i = 0; i<nr; i++)
	 {
		 for (j = 0; j<nc; j++)
		 {
			 if( e_hist.at<double>(i,j) == huge_val )
            ;
			 else
			 {
				 for (k = 0; k<unik.size(); k++)
				 {
					 if( e_hist.at<double>(i,j) == histogram[k].error )
                    histogram[k].count = histogram[k].count + 1;
				 }
			 }
		 }
	 }
	 for(i=0;i<nr; i++)
	 {
		 for(j=0;j<nc;j++)
			 lm_rm.error_histogram.at<double>(i,j)=e_hist.at<double>(i,j);
	 }
	

	//Display Histogram
	 
	 //Compute LM RM

	int max_count = -1;
	for (i = 0; i<unik.size(); i++)
	{
		//X(i) = histogram(1,i).error; % For Plot
		// Y(i) = histogram(1,i).count; % For Plot
		 if( histogram[i].count > max_count )
		 {
			 max_count = histogram[i].count;
			 lm_rm.LM = histogram[i].error;
		 }
	}
	
	int second_max_count = -1;
	for (i = 0; i<unik.size(); i++)
	{
		if( histogram[i].count == max_count )
        ;
		else 
			{
				if( histogram[i].count > second_max_count )
				{
					second_max_count = histogram[i].count;
					lm_rm.RM = histogram[i].error;
				}
			}
	}
	
	/*
	//Calculate LE and RE
	histograms *LE= new histograms[unik.size()] ;
	histograms *RE= new histograms[unik.size()];
	int posLE = 0;
	int posRE = 0;
    for (i = 0; i<unik.size(); i++)
	{
		if( histogram[i].error <= lm_rm.LM )
		{
			posLE = posLE + 1;
			LE[posLE].error = histogram[i].error;
			LE[posLE].count = histogram[i].count;
		}
		if( histogram[i].error >= lm_rm.RM )
		{
			posRE = posRE + 1;
			RE[posRE].error = histogram[i].error;
			RE[posRE].count = histogram[i].count;
		}
	}
	//Compute LN RN
	int mincount_LE = LE[1].count;
	LN = LE[1].error;
	for (j = 0; j<posLE; j++)
	{
		if( LE[j].count < mincount_LE )
		{
			mincount_LE = LE[j].count;
	         LN = LE[j].error;
		}
	}
	int mincount_RE = RE[1].count;
	RN = RE[1].error;
    for (j = 0; j<posRE; j++)
	{
		if( RE[j].count < mincount_RE )
		{
			mincount_RE = RE[j].count;
			RN = RE[j].error;
		}
	}  */
	
		return lm_rm;   
}

vector<int> EightBits(int I)
{
	vector<int> bitstring(8); int i=0;
	//[bitstring] = EightBits(I) converts integer I into its 8-bit binary
	//encoding, bitstring.
	//Example, EightBits(2) returns bitstring = 00000010
	//Input:   I: an integer
	//Output:  bitstring: a 1X8 array of 0s and 1s   
	
	
	//binary encoding of I consisting of 8 or lesser number of bits
	//bin = dec2bin(I8);
	if(I<0)
	I=-I;
	int bin[8];

	while(I>0)
	{
		bin[i]=I%2;
		I=I/2;
		i++;  //will signify no. of bits occupied
	}
	    
	int j;
	//If number of bits < 8, shift each bit right so that the rightmost bit occupies the 8th position.
	
	//Shift each bit of bin right by (8 - nc) positions
	 for (j = 0; j<i; j++)
		bitstring[ 7-j] = bin[j]; 
	 //Fill in the remaining (8 - nc) left bits by zeros.
    for (j = 1; j<(8 - i); j++)
        bitstring[j] = 0; 
		    
	 return bitstring;
}


Mat_overhead EmbedWatermarkInterpolation(Mat p, vector<int> &watermark_original)
{
	int i, j;
	int nr= p.size().height;
	int nc= p.size().width;
	LM_RM lm_rm;
	errors err;
	
	vector<int> watermark;   //to be embedded
	err = Interpolated(p);
	
	lm_rm=createHistogram(err.e);
	
	 Mat e_new= Mat_<double>(nr,nc);
	 for( i =0 ; i < nr; i++)
	 {
		 for( j=0; j < nc; j++)
			 e_new.at<double>(i,j)=err.e.at<double>(i,j);
	 }
	 
	//Create bitstream to be embedded-------------------------------------------
    //Location Map for overflow or underflow.
	
	 	int len_locmap = 0;  //Total no. of pixels with values 0, 255, 1, 254.
	
	for (i = 0; i<nr; i++)
	{
		for (j = 0; j<nc; j++)
		{  
			//Only 254 or 1 can change to 255 and 0 respectively
			 // So we take max length (for 0,255,1,254)
			 // actual loc map consists of those pixel positions which are either
			 // 0, 255 or have changed to 0, 255.
			 //  So actual length may be smaller than len_locmap.
			   
			if( (p.at<double>(i,j) == 0 ) || ( p.at<double>(i,j) == 255 ) || ( p.at<double>(i,j) == 1 ) || ( p.at<double>(i,j) == 254 ) )
			len_locmap = len_locmap + 1;    
		}
	} 
	
	Mat_overhead ob1;
	ob1.watermarked_image= Mat_<double>(nr,nc);
	//watermarked_image = zeros(nr,nc);
	 for( i =0 ; i < nr; i++)
	 {
		 for( j=0; j < nc; j++)
			 ob1.watermarked_image.at<double>(i,j)=p.at<double>(i,j);
	 }
		
	int poswm = 0; // To keep track of bits to be embedded
	int cnt = 0; // To find Maximum Embedding capacity
	vector<int> locmap; // To prevent overflow/underflow

	// Get Maximum Capacity----------
								cout<<"RM: "<<lm_rm.RM;

	for( i=0; i<nr; i++)
	{
		for (j = 0; j<nc; j++)
			{
				if( ( p.at<double>(i,j) == 0 ) || ( p.at<double>(i,j) == 255 ) ) ;     
				else        
				{
					if( err.e.at<double>(i,j) == huge_val ) ;
				    else if( err.e.at<double>(i,j) <= (double)lm_rm.LM )
						{
							if( err.e.at<double>(i,j) == (double)lm_rm.LM )
							cnt = cnt + 1;
						}
					 else if( err.e.at<double>(i,j) >= (double)lm_rm.RM )
						 {
							if( err.e.at<double>(i,j) == (double)lm_rm.RM )
							cnt = cnt + 1;
						 }
				}
			}
	}
	
	
	if( watermark_original.size() > ( cnt - 18 - len_locmap ) )    //original watermrk
	{
		for( i=0; i < ( cnt - 18 - len_locmap ) ; i++)
			{
				watermark.push_back(watermark_original[i]);
			}
	}
	
    ob1.len_wm = watermark.size();
	cout<<"Watermark size: "<<ob1.len_wm;
	cout<<"Maximum embedding capacity: "<<cnt - 18 - len_locmap <<endl<<endl;
	
	//Store just 18 bits for 2 signed integers along with 'len_locmap' bits for overflow/underflow.
	int k = 1; double v;
	for (i = 0; i<nr; i++)
	{
		for (j = 0; j<nc; j++)
		{
			if( (i == 0) || (j == 0) || (i == 7) || (j == 7) )
			{
				if( (i%2==0) && (j%2==1) )
				{
					if( k <= ( 18 + len_locmap ) )
					{
						if (p.at<double>(i,j)<0)
							v=-p.at<double>(i,j);
						else
							v=p.at<double>(i,j);

						watermark.push_back( ( (int) (v ) ) %2 );  //LSB= (int)(p.at<double>(i,j))%2
					    k = k + 1;
					} 
				}
				else if( (i%2==1) && (j%2==0) )
					 {
						 if( k <= ( 18 + len_locmap ) )
						 {
							 if (p.at<double>(i,j)<0)
							   v=-p.at<double>(i,j);
							 else
								v=p.at<double>(i,j);

							watermark.push_back( ( (int) (v ) )%2 );
							k = k + 1;
						 }
					}
			}
		}
	}
	
	// Embed--------------------------------------------------------------------
	 
	int ncwm = watermark.size(); int b, sign_e;
	
	for (i = 0;i<nr; i++)
	{
		for (j = 0; j<nc; j++)
		{
			if( ( p.at<double>(i,j) == 0 ) || ( p.at<double>(i,j) == 255 ) )
            locmap.push_back(0);
            else
			{
				if( lm_rm.error_histogram.at<double>(i,j) == huge_val )  // Do not Embed (Low resolution or marginal pixels)
                ob1.watermarked_image.at<double>(i,j) = p.at<double>(i,j);
			    else if( lm_rm.error_histogram.at<double>(i,j) <= lm_rm.LM )
				{
					sign_e = -1;   // sign(e) = -1 if e in LE i.e. <= LM
					if( lm_rm.error_histogram.at<double>(i,j) == lm_rm.LM ) // Embed next bit
						{
							//cnt = cnt + 1;
							if( poswm < ncwm )  // More bits to embed
								{
									poswm = poswm + 1;
									b = watermark[poswm-1];
								}
							 else
							    b = 0;  // No more bits => e will be same
						     e_new.at<double>(i,j) = lm_rm.error_histogram.at<double>(i,j) + sign_e * b;
						}
					else if( lm_rm.error_histogram.at<double>(i,j) < lm_rm.LM ) // Embed 1
						e_new.at<double>(i,j) = lm_rm.error_histogram.at<double>(i,j) + sign_e * 1;
				
					 ob1.watermarked_image.at<double>(i,j) = err.x.at<double>(i,j) - e_new.at<double>(i,j);
						// update locmap
					if( ( ob1.watermarked_image.at<double>(i,j) == 0 ) || ( ob1.watermarked_image.at<double>(i,j) == 255 ) )
                    locmap.push_back( 1 );
				}
				else if( lm_rm.error_histogram.at<double>(i,j) >= lm_rm.RM )
				{
					sign_e = 1;  //  sign(e) = 1 if e in RE i.e. <= RM
					 if( lm_rm.error_histogram.at<double>(i,j) == lm_rm.RM )  // Embed next bit
						{
							//cnt = cnt + 1;
							if( poswm < ncwm )  // More bits to embed
							{
								poswm = poswm + 1;
								b = watermark[poswm-1];
							}
							else
								b = 0;  // No more bits => e will be same

							e_new.at<double>(i,j) =lm_rm.error_histogram.at<double>(i,j) + sign_e * b;
						}
					 else if( lm_rm.error_histogram.at<double>(i,j) > lm_rm.RM )  // Embed 1
					   e_new.at<double>(i,j) = lm_rm.error_histogram.at<double>(i,j) + sign_e * 1;
				
                ob1.watermarked_image.at<double>(i,j) = err.x.at<double>(i,j) - e_new.at<double>(i,j);
                //update locmap
                if( ( ob1.watermarked_image.at<double>(i,j) == 0 ) || ( ob1.watermarked_image.at<double>(i,j) == 255 ) )
                    locmap.push_back(1);
				}
			}
		}
	}
		
	// GET binary representations of LM and RM----------------
	vector<int> temp = EightBits(lm_rm.LM);
	vector<int> LM_bin(9);
	for(i=0 ; i<9; i++)
		LM_bin[i]=0;
	if( lm_rm.LM < 0 )
		LM_bin[0] = 1;
	else
		LM_bin[0] = 0;
	for (i = 0; i<8; i++)
	{
		if( temp[i] == 0 )
			LM_bin[i+1] = 0;
		else if( temp[i] == 1 )
			LM_bin[i+1] = 1;
	}
    
	temp = EightBits(lm_rm.RM);
	vector<int>RM_bin(9);
	for(i=0 ; i<9; i++)
		RM_bin[i]=0;
	if( lm_rm.RM < 0 )
		RM_bin[0] = 1;
	else
		RM_bin[0] = 0;
	for (i = 0; i<8; i++)
	{
		if( temp[i] == 0 )
			RM_bin[i+1] = 0;
		else if( temp[i] == 1 )
			RM_bin[i+1] = 1;
	}
	
	
	// Concatenate binary reps of LM and RM and locmap
	vector<int> b_vect;

	for(i=0; i<9;i++)
	b_vect.push_back(LM_bin[i]);
	
	
	for(i=0; i<9;i++)
	b_vect.push_back(RM_bin[i]);
	
	for(i=0; i<locmap.size();i++)
	b_vect.push_back(locmap[i]);
	
	ob1.overhead = b_vect;
	
	long long int aaa=0;
	
	return ob1;
}

int bi2de(vector<int> bin)
{
	int i=7, dec = 0, rem, num, base = 1;

    while (i>=0)
    {
       dec = dec + bin[i] * base;
       //cout<<dec<<endl;
        base = base * 2;
        i--;
    }
    return dec;
}

Mat YCoCg2RGB( Mat watermarked_image_ee)
{
	int nr= watermarked_image_ee.size().height;
	int nc= watermarked_image_ee.size().width;
	watermarked_image_ee.convertTo(watermarked_image_ee, CV_64FC3);

	// Split into Y, Co, Cg
	Mat Y = Mat_<double>(nr, nc); 
	Mat Co = Mat_<double>(nr, nc);
	Mat Cg = Mat_<double>(nr, nc); 
	Mat rgb_image;
	//Mat YCoCg[3];

	//split(watermarked_image, YCoCg);

	 for(int i=0; i< nr; i++)
		 {
			 for(int j=0; j<nc; j++)	
			{
				Y.at<double>(i,j)= watermarked_image_ee.at<Vec3d>(i,j)[0];
				 Co.at<double>(i,j) = watermarked_image_ee.at<Vec3d>(i,j)[1];
				 Cg.at<double>(i,j) = watermarked_image_ee.at<Vec3d>(i,j)[2];
			}
		 }
	 
  	  Mat R = Mat_<double>(nr, nc);
	  Mat G = Mat_<double>(nr, nc); 
	  Mat B = Mat_<double>(nr, nc);
	  Mat t = Mat_<double>(nr, nc);

	  int i, j;
	  
	  for(i=0;i<nr;i++)
	  {
		  for (j=0;j<nc; j++)
		  {	  t.at<double>(i,j) =  (double)(Y.at<double>(i,j)) - (double)(floor)(0.5*Cg.at<double>(i,j));
				G.at<double>(i,j) = (double)(Cg.at<double>(i,j)) + (double)(t.at<double>(i,j));
				B.at<double>(i,j) = (double)(t.at<double>(i,j)) - (double)(floor)(0.5*Co.at<double>(i,j));
				R.at<double>(i,j) = (double)(B.at<double>(i,j)) + (double)(Co.at<double>(i,j));
		  }
	  }
	
	 Mat YCoCg_new[3];
	 YCoCg_new[0] = B;
	 YCoCg_new[1] = G;
	 YCoCg_new[2] = R;
	
	  merge( YCoCg_new,3, rgb_image); 
	   	  
	   return rgb_image;
}

Mat RGB2YCoCg( Mat images)
{
	int nr= images.size().height;
	int nc= images.size().width;
	images.convertTo(images, CV_64FC3);
	//Split into R, G, B
	Mat b=Mat_<double>(nr, nc); 
	Mat g=Mat_<double>(nr, nc); 
	Mat r=Mat_<double>(nr, nc);
	//Mat arr[3];
	//split(images, arr);                  //GIVES Lesser time and better result

	for(int i=0; i< nr; i++)
		 {
			 for(int j=0; j<nc; j++)	
			{
				b.at<double>(i,j)= images.at<Vec3d>(i,j)[0];
				 g.at<double>(i,j) = images.at<Vec3d>(i,j)[1];
				 r.at<double>(i,j) = images.at<Vec3d>(i,j)[2];
			}
		 }
	  
	  Mat Co= Mat_<double>(nr, nc);
	  Mat t= Mat_<double>(nr, nc);
	  Mat Cg= Mat_<double>(nr, nc);
	  Mat Y= Mat_<double>(nr, nc);
	 	
	  int i,j;
	  for(i=0;i<nr;i++)
	  {
		  for (j=0;j<nc; j++)
		  {
			  Co.at<double>(i,j) = (double)(r.at<double>(i,j)) - (double)(b.at<double>(i,j));
			t.at<double>(i,j) = (double)(b.at<double>(i,j)) + (double)(floor)(0.5*Co.at<double>(i,j)); 
			Cg.at<double>(i,j) = (double)(g.at<double>(i,j)) - (double)(t.at<double>(i,j));
			Y.at<double>(i,j) = (double)(t.at<double>(i,j)) + (double)(floor)(0.5*Cg.at<double>(i,j));
		  }
	  }
	  
	   Mat arr_new[3];
	  arr_new[0]=Y;
	  arr_new[1]=Co;
	  arr_new[2]=Cg;
	  
	   Mat colors;
	  
	    merge( arr_new,3, colors);
		
	    return colors;
}


Mat extractWatermarkInterpolation( Mat watermarked_image_e, int l, vector<int> overheads, vector<int> wmm)
{
	int nr= watermarked_image_e.size().height;
	int nc= watermarked_image_e.size().width;
	int i, j;
	errors errs;
		   	
	long long sums1=0;
	for( i=0; i<nr; i++)
	{
		for(j=0;j<nc;j++)
			sums1+=watermarked_image_e.at<double>(i,j);
	}
	cout<<endl<<"In extract function, embedded Watermarked image sum: "<<sums1<<endl;

	errs= Interpolated(watermarked_image_e);
	
	Mat e_new  = Mat_<double>(nr,nc);
	Mat xxx  = Mat_<double>(nr,nc);
	Mat eee = Mat_<double>(nr,nc);
	
	for(i=0; i<nr; i++)
	{
		for(j=0; j<nc; j++)
		{
			e_new.at<double>(i,j) = errs.e.at<double>(i,j);
			eee.at<double>(i,j) = errs.e.at<double>(i,j);
			xxx.at<double>(i,j) = errs.x.at<double>(i,j);
		}
	}
	
	vector<int> watermark;
	Mat  p=Mat_<double>(nr, nc);
	
	for(i=0;i<nr;i++)
	{
		for(j=0;j<nc;j++)
			p.at<double>(i,j) = watermarked_image_e.at<double>(i, j) ;
	}
	int LM_extract, RM_extract;
	
	// Compute LM, RM
	
	vector<int> LM_RM_e(overheads.size());
	for(i=0;i<overheads.size();i++)
		LM_RM_e[i]= overheads[i];
	
	std::vector<int>  temp(8);
	
	if( LM_RM_e[0] == 1 )
	{
		for(i=0;i<8;i++)
		temp[i]=overheads[i+1];
		LM_extract = (-1)*bi2de ( temp);
	}
	else if( LM_RM_e[0] == 0 )
	{
		for(i=0;i<8;i++)
		temp[i]=overheads[i+1]; 
		LM_extract = bi2de( temp );
	}

	if( LM_RM_e[9] == 1 )
	{
		for(i=10;i<=17;i++)
			temp[i-10]=overheads[i];
		RM_extract = (-1)*bi2de( temp );
	}
	else if( LM_RM_e[9] == 0 )
	{
		for(i=10;i<=17;i++)
			temp[i-10]=overheads[i];
		RM_extract = bi2de( temp);
	}
	
	// Compute Locmap
	// locmap = LM_RM( 1, 19:( 19 + len_locmap - 1 ) );
	vector<int> locmap(overheads.size()-18);
	if( overheads.size() > 18 )
		{for( i=18; i<overheads.size(); i++)
		locmap[i-18]=overheads[i];
	    }

	// EXTRACT -----------------------------------------------------------------
   	cout<<" HI";
	int sign_e, b;
	Mat p1 = Mat_<double>(nr,nc);
	   p1= -1 * Mat::ones(nr, nc, CV_64FC1);
	 
	int pos_locmap = 0;
	
	
	for (i = 0; i<nr;i++)
	 {
		 for (j = 0; j<nc; j++)
	      {
			  if( e_new.at<double>(i,j) == huge_val )
				{
					//e(i,j) = 999;
					eee.at<double>(i,j) = 0;
					p1.at<double>(i,j) = p.at<double>(i,j);
				    if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
						pos_locmap = pos_locmap + 1;
				}
           else if( e_new.at<double>(i,j) == LM_extract )
		   {
			   if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
			   {
				   pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
						p1.at<double>(i,j) = p.at<double>(i,j);
			         else if( locmap[pos_locmap-1] == 1 )
					 {
						 sign_e = -1;
						 b = 0;
						watermark.push_back(b);
						eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					  }
			   }
				else
				{
					sign_e = -1;
					b = 0;
					watermark.push_back(b);
					eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
					p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				}
			}
          else if( e_new.at<double>(i,j) == RM_extract )
		  {
			  if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
                {
					pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
						p1.at<double>(i,j) = p.at<double>(i,j);
					else if( locmap[pos_locmap-1] == 1 )
                    {
						sign_e = 1;
						b = 0;
						watermark.push_back(b);
						eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					}
			  }
				else
				{
					sign_e = 1;
					b = 0;
					watermark.push_back(b);
					eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
					p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				}   
		   }
			else if( e_new.at<double>(i,j) == (LM_extract-1) )
			{
				if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
				{
					pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
						p1.at<double>(i,j) = p.at<double>(i,j);
				    else if( locmap[pos_locmap-1] == 1 )
					{
						sign_e = -1;
						b = 1;
						watermark.push_back(b);
						eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					}
				}
				else
				{
					sign_e = -1;
					b = 1;
					watermark.push_back(b);
					eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
				    p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				 }
			}
			else if( e_new.at<double>(i,j) == (RM_extract+1) )
            {
				if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
                {
					pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
					   p1.at<double>(i,j) = p.at<double>(i,j);
					else if( locmap[pos_locmap-1] == 1 )
					{ sign_e = 1;
						b = 1;
						watermark.push_back(b);
						eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					 }
				}
				else
				{
					sign_e = 1;
					b = 1;
					 watermark.push_back(b);
					eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*b;
					p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				}
			}
            
			else if( e_new.at<double>(i,j) < (LM_extract-1) )
            {
				if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
                {
					pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
						p1.at<double>(i,j) = p.at<double>(i,j);
					else if( locmap[pos_locmap-1] == 1 )
                    {
						sign_e = -1;
						eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*1;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					}
				}
            else
			{
				sign_e = -1;
                eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*1;
                p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				}
			}
            
		 else if( e_new.at<double>(i,j) > (RM_extract+1) )
			{
				if( (p.at<double>(i,j)==0) || (p.at<double>(i,j)==255) )
			 {
					pos_locmap = pos_locmap + 1;
					if( locmap[pos_locmap-1] == 0 )
						p1.at<double>(i,j) = p.at<double>(i,j);
					else if( locmap[pos_locmap-1] == 1 )
					{
						sign_e = 1;
						 eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*1;
						p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
					}
				}
				else
				{
					sign_e = 1;
					 eee.at<double>(i,j) = e_new.at<double>(i,j) - sign_e*1;
					p1.at<double>(i,j) = xxx.at<double>(i,j) - eee.at<double>(i,j);
				}
			}
		}
	}

	vector<int> tempp(l);
	for(i=0; i<l;i++)
	tempp[i]=watermark[i];
	
	long long sums=0;
	for( i=0; i<nr; i++)
	{
		for(j=0;j<nc;j++)
			sums+=p1.at<double>(i,j);
	}
	cout<<endl<<" Extracted sum: "<<sums<<endl;
	int flag=0;
	for(i=0;i<tempp.size();i++)
	{
		//<<watermark[i];
		if(tempp[i]==wmm[i])
			flag++;
	}
	cout<<"Extracted Watermark size: "<<tempp.size();
	cout<<" Flag_watermark: "<<flag<<endl;

	return p1;
} 

double MSE ( Mat matrix1, Mat matrix2)
{
	//   'nmse' gives the Mean Square Error of matrices 1 and 2
	int nr, nc, ns;  double nmse = 0, temp; int i, j, k;

	//Check dimensions of 2 matrices
	if ( matrix1.size() == matrix2.size() )
	{
		//Get dimensions of the matrices
		nr = matrix1.size().height;
		nc = matrix1.size().width;
		ns = matrix1.channels(); 
		for (i = 0; i < nr; i++)
		{
			for (j = 0; j < nc; j++)
				{
					for (k = 0; k < ns; k++)
						{
							if ( matrix1.at<Vec3d>(i,j)[k] == matrix2.at<Vec3d>(i,j)[k] )
							 ; 
							else
							{
								temp = matrix1.at<Vec3d>(i,j)[k] - matrix2.at<Vec3d>(i,j)[k];
								nmse = nmse + temp * temp;
							}
						}
				}
		}
	}
            
   else
		cout<< endl<< " Dimension mismatch in MSE calculation."<<endl; //Error if dimensions mismatch
	nmse = nmse / (nr*nc*ns);
	return nmse;
}

void PSNR(Mat image_original, Mat image_modified )
{
	//[ psnr ] = PSNR( image_original, image_modified )
	//   Computes PSNR value of an image.
	//   Input: image_original -> Original image filename.
	//          image_modified -> Modified image filename.
	//   Output: psnr -> PSNR of the modified image (image_modified) w.r.t the
	//                   original image (image_original).

	double mse = MSE( image_original, image_modified );
	double psnr = 10 * log10( 255*255 / mse );

	cout<< endl << "Calculated PSNR = " << psnr;
}



int main(int argc, char **argv ) 
{
     Mat image;
       image = imread("G:\\lena.tiff", CV_LOAD_IMAGE_COLOR);  
 
       if(! image.data )                             
       {
              cout <<  "Could not open or find the image" << std::endl ;
              return -1;
       }
 	 
	 namedWindow( "Display window", CV_WINDOW_AUTOSIZE );  
       imshow( "Display window", image );             
	   

	  double t= getTickCount();
	   Mat color = RGB2YCoCg (image);
	 //  imshow("color", color);
	  	   
      int nor= color.size().height;
	  int noc= color.size().width;
     
	 long long int summ1=0, summ2=0, summ3=0;
	 
	  //image.convertTo(image, CV_64FC3);
	 //Create Channel
	   int i, j, k; 
	   



	  Mat p1 = Mat_<double>(nor,noc);
	  Mat p2 = Mat_<double>(nor,noc);
	  Mat p3 = Mat_<double>(nor,noc);
	  
	  Mat_overhead o1, o2, o3;
	  Mat watermarked_image_channels[3], watermarked_image_final;
	  Mat rgb_image_watermarked;
	      
	   for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{
	   			p1.at<double>(i,j)=color.at<Vec3d>(i,j)[0]; //Y
				p2.at<double>(i,j)=color.at<Vec3d>(i,j)[1];  //Co
				p3.at<double>(i,j)=color.at<Vec3d>(i,j)[2];  //Cg
			}
	   }
	   
	  /*  for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{ summ1+=p1.at<double>(i,j);
			summ2+=p2.at<double>(i,j);
			summ3+=p3.at<double>(i,j);
			}
		}
	   cout<<"SUM of 1st channel original: "<< summ1<<endl;
	   cout<<"SUM of 2nd channel original: "<< summ2<<endl;
	   cout<<"SUM of 3rd channel original: "<< summ3<<endl;
*/
	 
	   vector<int> watermarks(999999);     //Watermark in int
	  
	   for(i=0; i< 999999; i++)
		   {
			  // watermarks[i]=rand() % 2 ;
			   watermarks[i]=1;
		 }
			
	  o1 = EmbedWatermarkInterpolation(p1, watermarks);  //Y_wm
	  o2 = EmbedWatermarkInterpolation(p2, watermarks);  //Co_wm
	  o3 = EmbedWatermarkInterpolation(p3, watermarks);  //Cg_wm
	  
	  watermarked_image_channels[0]= o1.watermarked_image;
	  watermarked_image_channels[1] = o2.watermarked_image;
	  watermarked_image_channels[2] = o3.watermarked_image;
	  merge( watermarked_image_channels, 3, watermarked_image_final);
		
	rgb_image_watermarked= YCoCg2RGB(watermarked_image_final);
		
	//RECEIVER END
	
	Mat image_2b_extracted = RGB2YCoCg( rgb_image_watermarked);
	
	Mat z1 = Mat_<double>(nor,noc);
	Mat z2 = Mat_<double>(nor,noc);
	Mat z3 = Mat_<double>(nor,noc);
	
	for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{
	   			z1.at<double>(i,j)= image_2b_extracted.at<Vec3d>(i,j)[0];
				z2.at<double>(i,j)= image_2b_extracted.at<Vec3d>(i,j)[1];
				z3.at<double>(i,j)= image_2b_extracted.at<Vec3d>(i,j)[2];
			}
	   }
	 
	Mat extracted_image_channels[3], extracted_image;
	
	extracted_image_channels[0] = extractWatermarkInterpolation( z1, o1.len_wm, o1.overhead, watermarks); //WATERMARKED IN YCoCg
	extracted_image_channels[1] = extractWatermarkInterpolation( z2, o2.len_wm, o2.overhead, watermarks);
	extracted_image_channels[2] = extractWatermarkInterpolation( z3, o3.len_wm, o3.overhead, watermarks);
	//cout<<"LENGTH OF WATERMARK: "<<o1.len_wm + o2.len_wm + o3.len_wm;
	
	merge( extracted_image_channels, 3, extracted_image);
	
	Mat final_image= YCoCg2RGB(extracted_image);

	//final_image.convertTo(final_image, CV_8UC3 );
	//imshow( "Extracted image final", final_image);
	//final_image.convertTo(final_image, CV_8UC3);
	image.convertTo(image, CV_64FC3);
	
	/*int test_count1=0, test_count2=0, test_count3=0;
	for(i=0;i<nor;i++)
	{
		for(j=0;j<noc; j++)
		{
				if ( extracted_image_channels[0].at<double>(i,j) != p1.at<double >(i,j) )
				test_count1 ++;
				if (extracted_image_channels[1].at<double>(i,j) != p2.at<double >(i,j) )
				test_count2 ++;
				if ( extracted_image_channels[2].at<double>(i,j) != p3.at<double >(i,j) )
				test_count3 ++;
		}
	}
	
	cout<<endl<<" Test count: "<<test_count1<< " " <<test_count2<< " "<< test_count3<<endl;
	int test_count11=0, test_count22=0, test_count33=0;
	for(i=0;i<nor;i++)
	{
		for(j=0;j<noc; j++)
		{
			
				if ( final_image.at<Vec3d>(i,j)[0] != image.at<Vec3d>(i,j)[0] )
				test_count11 ++;
				if (final_image.at<Vec3d>(i,j)[1] != image.at<Vec3d>(i,j)[1] )
				test_count22 ++;
				if ( final_image.at<Vec3d>(i,j)[2] != image.at<Vec3d>(i,j)[2] )
				test_count33 ++;
		}
	}
	
	cout<<endl<<" Test count: "<<test_count11<< " " <<test_count22<< " "<< test_count33<<endl;
	
	long long int z=0, zz=0, zzz=0;
	for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{ z+=extracted_image_channels[0].at<double>(i,j);
			zz+=extracted_image_channels[1].at<double>(i,j);
			zzz+=extracted_image_channels[2].at<double>(i,j);
			}

		}
	   cout<<"SUM of 1st channel final: "<< z<<endl;
	   cout<<"SUM of 2nd channel final: "<< zz<<endl;
	   cout<<"SUM of 3rd channel final: "<< zzz<<endl;
	   long long int summm1=0, summm2=0, summm3=0;
	     for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{ summm1+=image.at<Vec3d>(i,j)[0];
			summm2+=image.at<Vec3d>(i,j)[1];
			summm3+=image.at<Vec3d>(i,j)[2];
			}
		}
	   cout<<"SUM of 1st channel original: "<< summm1<<endl;
	   cout<<"SUM of 2nd channel original: "<< summm2<<endl;
	   cout<<"SUM of 3rd channel original: "<< summm3<<endl;
	   long long int summm11=0, summm22=0, summm33=0;
	     for (i = 0; i<nor; i++)
		{
			for (j = 0; j<noc; j++)
			{ summm11+=final_image.at<Vec3d>(i,j)[0];
			summm22+=final_image.at<Vec3d>(i,j)[1];
			summm33+=final_image.at<Vec3d>(i,j)[2];
			}
		}
	   cout<<"SUM of 1st channel final: "<< summm11<<endl;
	   cout<<"SUM of 2nd channel final: "<< summm22<<endl;
	   cout<<"SUM of 3rd channel final: "<< summm33<<endl;

	*/
	PSNR(image, rgb_image_watermarked);

	cout<<"Time taken: "<< ( ((double)(getTickCount())-t)/ getTickFrequency() ); 
	
	 cvWaitKey();
   return 0;
}

