//
//  write.cpp
//  multiResolution
//
//  Created by kokofan on 3/9/20.
//  Copyright © 2020 koko Fan. All rights reserved.
//

#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <unistd.h>
#include <mpi.h>

static int global_box_size[3];
static int local_box_size[3];
static char file_name[512];
static char write_file_name[512];
static int image_format;
static int wavelet_level = -1;
static int max_wavelet_level;
static int idx_box_size[3];
static int local_box_offset[3];
static int sub_div[3];
static int rank = 0;
static int process_count = 1;

static void parse_args(int argc, char * argv[]);
static void check_args(int argc, char * argv[]);
static void calculate_per_process_offsets();
static void wavelet_transform(unsigned char * buf);
static void reorganisation(unsigned char * buf1,  unsigned char * buf2, int flag);
static void sub_filling(unsigned char * idx_buf, unsigned char * idx_level_buf);


int main(int argc, char * argv[]) {
    
    // Parse arguments
    parse_args(argc, argv);
    
    // The max number of wavelet levels
    auto min = *std::min_element(local_box_size, local_box_size + 3);
    max_wavelet_level = log2(min);
    
    // Check correctness of arguments if users don't require help
    std::string arg = argv[1];
    if (arg.compare("-h") != 0)
        check_args(argc, argv);
    
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n";
    if (MPI_Comm_size(MPI_COMM_WORLD, &process_count) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Comm_size error\n";
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Comm_rank error\n";
    
    // Calculate offsets per process
    calculate_per_process_offsets();
    
    // File handle
    MPI_File fh[2];
    MPI_Status status[2];
    int count[2];
    unsigned char * buf;
    int local_size = local_box_size[0] * local_box_size[1] * local_box_size[2];
    int offset =  local_size * sizeof(unsigned char);
    buf = (unsigned char *)malloc(offset);
    
    // Create subarray for data partition
    MPI_Datatype subarray;
    MPI_Type_create_subarray(3, global_box_size, local_box_size, local_box_offset, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &subarray);
    MPI_Type_commit(&subarray);
    
    // Read file parallel
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh[0]);
    MPI_File_set_view(fh[0], offset, MPI_UNSIGNED_CHAR, subarray, "native", MPI_INFO_NULL);
    MPI_File_read(fh[0], buf, offset, MPI_UNSIGNED_CHAR, &status[0]);
    MPI_Get_count(&status[0], MPI_UNSIGNED_CHAR, &count[0]);
    if(count[0] != offset)
    {
        std::cerr << "ERROR: Only Read " << count << " bits (Required " << offset << ")\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    MPI_File_close(&fh[0]);
    
    wavelet_transform(buf);

    // Calculate the size of DC component for IDX component
    for (int i = 0; i < 3; i++)
    {
        idx_box_size[i] = local_box_size[i]/pow(2, wavelet_level);
    }
    
    // Reorganize the DC component
    int idx_size = idx_box_size[0] * idx_box_size[1] * idx_box_size[2];
    unsigned char * idx_buf;
    idx_buf = (unsigned char *)malloc(idx_size * sizeof(unsigned char));
    reorganisation(idx_buf, buf, 0);
    
    // IDX coding
    unsigned char * idx_level_buf;
    idx_level_buf = (unsigned char *)malloc(idx_size * sizeof(unsigned char));
    sub_filling(idx_buf, idx_level_buf);
    free(idx_buf);
    
    reorganisation(buf, idx_level_buf, 1);
    free(idx_level_buf);
    
    char * name = strtok(file_name, ".");
    sprintf(write_file_name, "%s_output_%d", name, rank);
    
    MPI_File_open(MPI_COMM_SELF, write_file_name, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh[1]);
    MPI_File_write(fh[1], buf, offset, MPI_UNSIGNED_CHAR, &status[1]);
    MPI_Get_count(&status[0], MPI_UNSIGNED_CHAR, &count[1]);
    if(count[1] != offset)
    {
        std::cerr << "ERROR: Only Write " << count << " bits (Required " << offset << ")\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    MPI_File_close(&fh[1]);
    
    
    free(buf);
    MPI_Finalize();
    return 0;
}


static void parse_args(int argc, char * argv[])
{
    char options[] = "hg:l:f:i:w:";
    int one_opt = 0;
    
    while((one_opt = getopt(argc, argv, options)) != EOF)
    {
        switch (one_opt)
        {
            case('h'): // show help
                std::cout << "Help:\n" << "-g Specify the global box size (e.g., axbxc)\n" << "-l Specify the local box size (e.g., axbxc)\n" << "-f Specify the input file path (e.g., ./example.txt)\n" << "-i Specify the image format (e.g., unit8)\n" << "-w Specify the number of wavelet levels (e.g., 2)" << "-d Specify the number of sub-filling levels (e.g., 2)\n";
                break;
                
            case('g'): //global dimention, e.g., 256x256x256
                if((sscanf(optarg, "%dx%dx%d", &global_box_size[0], &global_box_size[1], &global_box_size[2]) == EOF))
                    exit(-1);
                break;
                
            case('l'): //local dimention, e.g., 32x32x32
                if((sscanf(optarg, "%dx%dx%d", &local_box_size[0], &local_box_size[1], &local_box_size[2]) == EOF))
                    exit(-1);
                break;
                
            case('f'): // read file name
                if (sprintf(file_name, "%s", optarg) < 0)
                    exit(-1);
                break;
                
            case('i'): // image formate, e.g., uint8
                if((sscanf(optarg, "uint%d", &image_format) == EOF) || image_format > 64)
                    exit(-1);
                break;
                
            case('w'): // the number of wavelet levels
                if((sscanf(optarg, "%d", &wavelet_level) == EOF))
                    exit(-1);
                break;
        }
    }
}


static void check_args(int argc, char * argv[])
{
    if (global_box_size[0] == 0 || global_box_size[1] == 0 || global_box_size[2] == 0 || local_box_size[0] == 0 || local_box_size[1] == 0 || local_box_size[2] == 0 || file_name[0] == ' ' || wavelet_level == -1)
    {
        std::cout << "Plese using -h to get help\n";
        exit(-1);
    }
    if (global_box_size[0] < local_box_size[0] || global_box_size[1] < local_box_size[1] || global_box_size[2] < local_box_size[2])
    {
        std::cerr << "ERROR: Global box is smaller than local box in one of the dimensions\n";
        exit(-1);
    }
    if (wavelet_level > max_wavelet_level)
    {
        std::cerr << "ERROR: Wavelet levels should be smaller than " << max_wavelet_level << "\n";
        exit(-1);
    }
}


static void calculate_per_process_offsets()
{
    sub_div[0] = (global_box_size[0] / local_box_size[0]);
    sub_div[1] = (global_box_size[1] / local_box_size[1]);
    sub_div[2] = (global_box_size[2] / local_box_size[2]);
    local_box_offset[2] = (rank / (sub_div[0] * sub_div[1])) * local_box_size[2];
    int slice = rank % (sub_div[0] * sub_div[1]);
    local_box_offset[1] = (slice / sub_div[0]) * local_box_size[1];
    local_box_offset[0] = (slice % sub_div[0]) * local_box_size[0];
}


static void wavelet_helper(unsigned char * buf, int step, int ng_step, int flag)
{
    int si = ng_step; int sj = ng_step; int sk = ng_step;
    
    if (flag == 0)
        sj = step;
    if (flag == 1)
        si = step;
    if (flag == 2)
        sk = step;
    
    int neighbor = 1;
    
    for (int k = 0; k < local_box_size[2]; k+=sk)
    {
        for (int i = 0; i < local_box_size[0]; i+=si)
        {
            for (int j = 0; j < local_box_size[1]; j+=sj)
            {
                int position = k*local_box_size[0]*local_box_size[1] + i*local_box_size[1] + j;
                
                if (flag == 0)
                    neighbor = position + ng_step;
                if (flag == 1)
                    neighbor = position + ng_step*local_box_size[1];
                if (flag == 2)
                    neighbor = position + ng_step*local_box_size[0]*local_box_size[1];

                buf[position] = (buf[position] + buf[neighbor])/2;
                buf[neighbor] = buf[position] - buf[neighbor];
            }
        }
    }
}


static void wavelet_transform(unsigned char * buf)
{
    for (int level = 1; level <= wavelet_level; level++)
    {
        int step = pow(2, level);
        int ng_step = step/2;
        
        // Calculate x-dir
        wavelet_helper(buf, step, ng_step, 0);
        
        // Calculate y-dir
        wavelet_helper(buf, step, ng_step, 1);

        // Calculate z-dir
        wavelet_helper(buf, step, ng_step, 2);
    }
}

static void reorganisation(unsigned char * buf1,  unsigned char * buf2, int flag)
{
    int step = pow(2, wavelet_level);
    int index = 0;
    
    for (int k = 0; k < local_box_size[2]; k+=step)
    {
        for (int i = 0; i < local_box_size[0]; i+=step)
        {
            for (int j = 0; j < local_box_size[1]; j+=step)
            {
                int position = k*local_box_size[0]*local_box_size[1] + i*local_box_size[1] + j;
                if(flag == 0)
                    buf1[index] = buf2[position];
                if(flag == 1)
                    buf1[position] = buf2[index];
                index++;
            }
        }
    }
}


static void idx_helper(unsigned char * buf1, unsigned char * buf2, int si, int sj, int sk, int ti, int tj, int tk, int* index)
{
    for (int k = sk; k < idx_box_size[2]; k+=tk)
    {
        for (int i = si; i < idx_box_size[0]; i+=ti)
        {
            for (int j = sj; j < idx_box_size[1]; j+=tj)
            {
                int position = k*idx_box_size[0]*idx_box_size[1] + i*idx_box_size[1] + j;
                buf2[*index] = buf1[position];
                *index += 1;
            }
        }
    }
}


static void sub_filling(unsigned char * buf1, unsigned char * buf2)
{
    buf2[0] = buf1[0];
    int index = 1;
    
    int ti = idx_box_size[0];
    int tj = idx_box_size[1];
    int tk = idx_box_size[2];
    
    int si, sj, sk;
    
    while (ti > 1 && tj > 1 && tk > 1)
    {
        // divide i
        sj = tj/2; si = 0; sk = 0;
        idx_helper(buf1, buf2, si, sj, sk, ti, tj, tk, &index);
        tj = tj/2;
        
        // divide j
        si = ti/2; sj = 0; sk = 0;
        idx_helper(buf1, buf2, si, sj, sk, ti, tj, tk, &index);
        ti = ti/2;
        
        // divide k
        sk = tk/2; si = 0; sj = 0;
        idx_helper(buf1, buf2, si, sj, sk, ti, tj, tk, &index);
        tk = tk/2;
    }
}
