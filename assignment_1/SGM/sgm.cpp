#include <algorithm>
#include <vector>
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>

#include "sgm.h"

#define NUM_DIRS 3
#define PATHS_PER_SCAN 8

using namespace std;
using namespace cv;
static char hamLut[256][256];
static int directions[NUM_DIRS] = {0, -1, 1};

//compute values for hamming lookup table
void compute_hamming_lut()
{
    for (uchar i = 0; i < 255; i++)
    {
        for (uchar j = 0; j < 255; j++)
        {
            uchar census_xor = i^j;
            uchar dist=0;
            while(census_xor)
            {
                ++dist;
                census_xor &= census_xor-1;
            }

            hamLut[i][j] = dist;
        }
    }
}

namespace sgm {
    SGM::SGM(unsigned int disparity_range, unsigned int p1, unsigned int p2, unsigned int window_height, unsigned window_width):
            disparity_range_(disparity_range), p1_(p1), p2_(p2), window_height_(window_height), window_width_(window_width)
    {
        compute_hamming_lut();
    }

    // set images and initialize all the desired values
    void SGM::set(const  cv::Mat &left_img, const  cv::Mat &right_img)
    {
        views_[0] = left_img;
        views_[1] = right_img;


        height_ = left_img.rows;
        width_ = right_img.cols;
        pw_.north = window_height_/2;
        pw_.south = height_ - window_height_/2;
        pw_.west = window_width_/2;
        pw_.east = width_ - window_height_/2;
        init_paths();
        cost_.resize(height_, ul_array2D(width_, ul_array(disparity_range_)));
        aggr_cost_.resize(height_, ul_array2D(width_, ul_array(disparity_range_)));
        path_cost_.resize(PATHS_PER_SCAN, ul_array3D(height_, ul_array2D(width_, ul_array(disparity_range_)))
        );
    }

    //initialize path directions
    void SGM::init_paths()
    {
        for(int i = 0; i < NUM_DIRS; ++i)
        {
            for(int j = 0; j < NUM_DIRS; ++j)
            {
                // skip degenerate path
                if (i==0 && j==0)
                    continue;
                paths_.push_back({directions[i], directions[j]});
            }
        }
    }

    //compute costs and fill volume cost cost_
    void SGM::calculate_cost_hamming()
    {
        uchar census_left, census_right, shift_count;
        cv::Mat_<uchar> census_img[2];

        cout << "\nApplying Census Transform" <<endl;

        for( int view = 0; view < 2; view++)
        {
            census_img[view] = cv::Mat_<uchar>::zeros(height_,width_);
            for (int r = 1; r < height_ - 1; r++)
            {
                uchar *p_center = views_[view].ptr<uchar>(r),
                        *p_census = census_img[view].ptr<uchar>(r);
                p_center += 1;
                p_census += 1;
                for(int c = 1; c < width_ - 1; c++, p_center++, p_census++)
                {
                    uchar census_val = 0, shift_count = 0;
                    for (int wr = r - 1; wr <= r + 1; wr++)
                    {
                        for (int wc = c - 1; wc <= c + 1; wc++)
                        {

                            if( shift_count != 4 )//skip the center pixel
                            {
                                census_val <<= 1;
                                if(views_[view].at<uchar>(wr,wc) < *p_center ) //compare pixel values in the neighborhood
                                    census_val = census_val | 0x1;
                            }
                            shift_count ++;
                        }
                    }
                    *p_census = census_val;
                }
            }
        }

        cout <<"\nFinding Hamming Distance" <<endl;

        for(int r = window_height_/2 + 1; r < height_ - window_height_/2 - 1; r++)
        {
            for(int c = window_width_/2 + 1; c < width_ - window_width_/2 - 1; c++)
            {
                for(int d=0; d<disparity_range_; d++)
                {
                    long cost = 0;
                    for(int wr = r - window_height_/2; wr <= r + window_height_/2; wr++)
                    {
                        uchar *p_left = census_img[0].ptr<uchar>(wr),
                                *p_right = census_img[1].ptr<uchar>(wr);
                        int wc = c - window_width_/2;
                        p_left += wc;
                        p_right += wc + d;
                        const uchar out_val = census_img[1].at<uchar>(wr, width_ - window_width_/2 - 1);

                        for(; wc <= c + window_width_/2; wc++, p_left++, p_right++)
                        {
                            uchar census_left, census_right;
                            census_left = *p_left;

                            if (c+d < width_ - window_width_/2)
                                census_right= *p_right;
                            else
                                census_right= out_val;

                            uchar census_xor = census_left^census_right; //Hamming Distance
                            uchar dist=0;
                            cost += hamLut[census_left][census_right];
                        }
                    }
                    cost_[r][c][d]=cost;
                }
            }
        }
    }

    //compute final costs per path
    void SGM::compute_path_cost(int direction_y, int direction_x, int cur_y, int cur_x, int cur_path)
    {
        //use this variables if needed
        unsigned long prev_cost;
        unsigned long best_prev_cost;
        unsigned long no_penalty_cost;
        unsigned long penalty_cost;
        unsigned long small_penalty_cost;
        unsigned long big_penalty_cost;

        // if the processed pixel is the first:
        if(cur_y == pw_.north || cur_y == pw_.south || cur_x == pw_.east || cur_x == pw_.west)
        {
            for(int d=0; d < disparity_range_; d++)
            {
                path_cost_[cur_path][cur_y][cur_x][d] = cost_[cur_y][cur_x][d];
            }
        }

        else
        {
            // Compute Big penalty factor and the value that allow to restrict the range of resulting values only one time
            best_prev_cost = ULONG_MAX;
            for(int p_disparity = 0; p_disparity < disparity_range_; p_disparity++)
            {
                best_prev_cost = std::min(path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][p_disparity], best_prev_cost);
            }
            big_penalty_cost = best_prev_cost + p2_;

            for(int d=0; d < disparity_range_; d++)
            {
                no_penalty_cost = path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][d];

                // Compute Small penalty factor
                if(d==0)
                {
                    small_penalty_cost = path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][d+1] + p1_;
                }
                else if(d == disparity_range_-1)
                {
                    small_penalty_cost = path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][d-1] + p1_;
                }
                else
                {
                    small_penalty_cost = std::min(path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][d-1] + p1_,
                                                  path_cost_[cur_path][cur_y - direction_y][cur_x - direction_x][d+1] + p1_);
                }

                // Works fine even without restricting the range of resulting values
                //path_cost_[cur_path][cur_y][cur_x][d] = cost_[cur_y][cur_x][d] + std::min({no_penalty_cost, small_penalty_cost, big_penalty_cost});
                path_cost_[cur_path][cur_y][cur_x][d] = cost_[cur_y][cur_x][d] + std::min({no_penalty_cost, small_penalty_cost, big_penalty_cost}) - best_prev_cost;
            }
        }
    }

    //aggregate the costs
    void SGM::aggregation()
    {
        //for all defined paths
        for(int cur_path = 0; cur_path < PATHS_PER_SCAN; ++cur_path)
        {

            int dir_x = paths_[cur_path].direction_x;
            int dir_y = paths_[cur_path].direction_y;

            int start_x, start_y, end_x, end_y, step_x, step_y;

            //TO DO: initialize the variables start_x, start_y, end_x, end_y, step_x, step_y with the right values
            //after that uncomment the code below

            // init_paths sequence [x,y]
            // {[0, -1], [0, 1], [-1, 0], [-1, -1] [-1, 1], [1, 0], [1, -1], [1, 1]}

            std::cout << std::endl << "Computing path: " << cur_path << " -> x: " << dir_x << ", y: " << dir_y << std::endl;
            if (dir_x == 0 || dir_x == 1)
            {
                start_x = pw_.west;
                end_x = pw_.east;
                if (dir_x == 0)
                {
                    step_x = 1;
                }
                else
                {
                    step_x = dir_x;
                }
            }
            // dir_x = -1
            else
            {
                start_x = pw_.east;
                end_x = pw_.west;
                step_x = dir_x;
            }

            if (dir_y == 0 || dir_y == 1)
            {
                start_y = pw_.north;
                end_y = pw_.south;
                if (dir_y == 0)
                {
                    step_y = 1;
                }
                else
                {
                    step_y = dir_y;
                }
            }
            // dir_y = -1
            else
            {
                start_y = pw_.south;
                end_y = pw_.north;
                step_y = dir_y;
            }

            for(int y = start_y; y != end_y ; y+=step_y)
            {
                for(int x = start_x; x != end_x ; x+=step_x)
                {
                    compute_path_cost(dir_y, dir_x, y, x, cur_path);
                }
            }
        }

        //TO DO: aggregate the costs for all direction into the aggr_cost_ tensor
        std::cout << std::endl << "Aggregating the costs for all direction" << std::endl << std::endl;
        for (int row = 0; row < height_; ++row)
        {
            for (int col = 0; col < width_; ++col)
            {
                for(int d = 0; d<disparity_range_; d++)
                {
                    for(int path = 0; path < PATHS_PER_SCAN; path ++)
                    {
                        aggr_cost_[row][col][d] += path_cost_[path][row][col][d]; //Aggregation
                    }
                }
            }
        }
    }

    void SGM::compute_disparity()
    {
        calculate_cost_hamming();
        aggregation();
        disp_ = Mat(Size(width_, height_), CV_8UC1, Scalar::all(0));

        for (int row = 0; row < height_; ++row)
        {
            for (int col = 0; col < width_; ++col)
            {
                unsigned long smallest_cost =aggr_cost_[row][col][0];
                int smallest_disparity = 0;
                for(int d=disparity_range_-1; d>=0; --d)
                {

                    if(aggr_cost_[row][col][d]<smallest_cost)
                    {
                        smallest_cost = aggr_cost_[row][col][d];
                        smallest_disparity = d;

                    }
                }

                disp_.at<uchar>(row, col) = smallest_disparity*255.0/disparity_range_;

            }
        }

    }

    float SGM::compute_mse(const cv::Mat &gt)
    {
        cv::Mat1f container[2];
        cv::normalize(gt, container[0], 0, 85, cv::NORM_MINMAX);
        cv::normalize(disp_, container[1], 0, disparity_range_, cv::NORM_MINMAX);

        cv::Mat1f  mask = min(gt, 1);
        cv::multiply(container[1], mask, container[1], 1);
        float error = 0;
        for (int y=0; y<height_; ++y)
        {
            for (int x=0; x<width_; ++x)
            {
                float diff = container[0](y,x) - container[1](y,x);
                error+=(diff*diff);
            }
        }
        error = error/(width_*height_);
        return error;
    }

    void SGM::save_disparity(char* out_file_name)
    {
        imwrite(out_file_name, disp_);
        return;
    }

}

