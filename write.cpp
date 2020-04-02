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
#include "zfp/include/zfp.h"

static int global_box_size[3];
static int local_box_size[3];
static char file_name[512];
static char write_file_name[512];
static int zfp_comp_flag = -1;
static float zfp_err_ratio = -1.0;
static int wavelet_level = -1;
static int max_wavelet_level;
static int idx_box_size[3];
static int local_box_offset[3];
static int sub_div[3];
static int rank = 0;
static int process_count = 1;
static int out_size = 0;

std::vector<std::string> metadata;
unsigned char *out_buf;

static void parse_args(int argc, char * argv[]);
static void check_args(int argc, char * argv[]);
static void MPI_Initial(int argc, char * argv[]);
static void calculate_per_process_offsets();
static void read_file_parallel(float *buf, int size);
static void wavelet_transform(float *buf);
static void read_DC(float *buf1,  float *buf2);
static void idx_encoding(float *buf1, float *buf2);
static void read_levels(float *buf1,  float *buf2, int level);
static void calculate_level_dimension(int* size, int level);
static void compressed_subbands(float *buf);
static void array_to_string(int* array);
static void write_file_parallel();

std::vector<unsigned char>
compress_3D_float(float* buf, int dim_x, int dim_y, int dim_z, float param, int flag);


int main(int argc, char * argv[]) {
    
    // MPI environment initialization
    MPI_Initial(argc, argv);
    
    // Parse arguments
    parse_args(argc, argv);
    
    // Check correctness of arguments if users don't require help
    std::string arg = argv[1];
    if (arg.compare("-h") == 0)
        MPI_Abort(MPI_COMM_WORLD, -1);
    
    // The max number of wavelet levels
    auto min = *std::min_element(local_box_size, local_box_size + 3);
    max_wavelet_level = log2(min);
    
    // Check arguments
    check_args(argc, argv);
    
    array_to_string(global_box_size);
    array_to_string(local_box_size);
    metadata.push_back(std::to_string(wavelet_level));
    metadata.push_back(std::to_string(zfp_comp_flag));
    metadata.push_back(std::to_string(zfp_err_ratio));

    // Calculate offsets per process
    calculate_per_process_offsets();

    // Mallocate buffer per process, and its size is the local dimension
    float * buf;
    int local_size = local_box_size[0] * local_box_size[1] * local_box_size[2];
    int offset =  local_size * sizeof(float);
    buf = (float *)malloc(offset);
    
    out_buf = (unsigned char *)malloc(offset);

    // Read file in parallel (chunks)
    read_file_parallel(buf, local_size);

    // Wavelet transform (inplace method)
    wavelet_transform(buf);

    // Calculate the dimensions of DC component
    calculate_level_dimension(idx_box_size, wavelet_level);
    int idx_size = idx_box_size[0] * idx_box_size[1] * idx_box_size[2];

    // Read DC compoment from buffer
    float *dc_buf = (float *)malloc(idx_size * sizeof(float));
    read_DC(buf, dc_buf);

    // IDX encoding
    float *idx_buf = (float *)malloc(idx_size * sizeof(float));
    idx_encoding(dc_buf, idx_buf);
    free(dc_buf);
    
    // zfp compression of idx encoding (DC component)
    std::vector<unsigned char> output = compress_3D_float(idx_buf, idx_box_size[0], idx_box_size[1], idx_box_size[2], zfp_err_ratio, zfp_comp_flag);
    free(idx_buf);
    
    // Combine buffer
    memcpy(&out_buf[out_size], output.data(), output.size());
    out_size += output.size();
    metadata.push_back(std::to_string(output.size()));

    // zfp compression of seven subbands per level
    compressed_subbands(buf);
    free(buf);
    
    // write file in parallel
    write_file_parallel();
    
    free(out_buf);
    MPI_Finalize();
    return 0;
}

// Parse arguments
static void parse_args(int argc, char * argv[])
{
    char options[] = "hg:l:f:w:z:e:";
    int one_opt = 0;
    
    while((one_opt = getopt(argc, argv, options)) != EOF)
    {
        switch (one_opt)
        {
            case('h'): // show help
                if (rank == 0)
                std::cout << "Help:\n" << "-g Specify the global box size (e.g., axbxc)\n" << "-l Specify the local box size (e.g., axbxc)\n" << "-f Specify the input file path (e.g., ./example.txt)\n" << "-w Specify the number of wavelet levels (e.g., 2)\n" << "-z Specify the zfp compress flag (0 means accuracy, 1 means precision)\n" << "-e Specify the error tolerant rate of zfp compression (range: [0,1])\n\n";
                break;
                
            case('g'): //global dimention, e.g., 256x256x256
                if((sscanf(optarg, "%dx%dx%d", &global_box_size[0], &global_box_size[1], &global_box_size[2]) == EOF))
                    MPI_Abort(MPI_COMM_WORLD, -1);
                break;
                
            case('l'): //local dimention, e.g., 32x32x32
                if((sscanf(optarg, "%dx%dx%d", &local_box_size[0], &local_box_size[1], &local_box_size[2]) == EOF))
                    MPI_Abort(MPI_COMM_WORLD, -1);
                break;
                
            case('f'): // read file name
                if (sprintf(file_name, "%s", optarg) < 0)
                    MPI_Abort(MPI_COMM_WORLD, -1);
                break;
                
            case('w'): // the number of wavelet levels
                if((sscanf(optarg, "%d", &wavelet_level) == EOF))
                    MPI_Abort(MPI_COMM_WORLD, -1);
                break;
            
            case('z'): // 0 means accuracy, 1 means precision
                if((sscanf(optarg, "%d", &zfp_comp_flag) == EOF) || zfp_comp_flag > 1)
                    MPI_Abort(MPI_COMM_WORLD, -1);
                break;
                
            case('e'): // the error tolerant rate of zfp compression, range: [0,1]
                if((sscanf(optarg, "%f", &zfp_err_ratio) == EOF))
                    MPI_Abort(MPI_COMM_WORLD, -1);
                break;
        }
    }
}

// Check arguments
static void check_args(int argc, char * argv[])
{
    if (global_box_size[0] == 0 || global_box_size[1] == 0 || global_box_size[2] == 0 || local_box_size[0] == 0 || local_box_size[1] == 0 || local_box_size[2] == 0 || file_name[0] == ' ' || wavelet_level == -1 || zfp_comp_flag == -1 || zfp_err_ratio == -1.0)
    {
        std::cout << "Plese using -h to get help\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (global_box_size[0] < local_box_size[0] || global_box_size[1] < local_box_size[1] || global_box_size[2] < local_box_size[2])
    {
        std::cerr << "ERROR: Global box is smaller than local box in one of the dimensions\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (wavelet_level > max_wavelet_level)
    {
        std::cerr << "ERROR: Wavelet levels should be smaller than " << max_wavelet_level << "\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}


static void MPI_Initial(int argc, char * argv[])
{
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n";
    if (MPI_Comm_size(MPI_COMM_WORLD, &process_count) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Comm_size error\n";
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Comm_rank error\n";
}


// Calculate the start position of data partition for each process
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

// Create a subarray for read non contiguous data
MPI_Datatype create_subarray()
{
    MPI_Datatype subarray;
    MPI_Type_create_subarray(3, global_box_size, local_box_size, local_box_offset, MPI_ORDER_FORTRAN, MPI_FLOAT, &subarray);
    MPI_Type_commit(&subarray);
    return subarray;
}

// Read file in parallel
static void read_file_parallel(float * buf, int size)
{
    MPI_Datatype subarray = create_subarray();

    MPI_File fh;
    MPI_Status status;

    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_FLOAT, subarray, "native", MPI_INFO_NULL);
    MPI_File_read(fh, buf, size, MPI_FLOAT, &status);
    MPI_File_close(&fh);
}

// A wavelet helper
static void wavelet_helper(float * buf, int step, int ng_step, int flag)
{
    int si = ng_step; int sj = ng_step; int sk = ng_step;
    
    // Define the start position based on orientations (x, y, z)
    if (flag == 0)
        sj = step;
    if (flag == 1)
        si = step;
    if (flag == 2)
        sk = step;
    
    int neighbor = 1;
    
    for (int k = 0; k < local_box_size[2]; k+=sk)
    {
        for (int i = 0; i < local_box_size[1]; i+=si)
        {
            for (int j = 0; j < local_box_size[0]; j+=sj)
            {
                int position = k*local_box_size[0]*local_box_size[1] + i*local_box_size[0] + j;
                
                // Define the neighbor position based on orientations (x, y, z)
                if (flag == 0)
                    neighbor = position + ng_step;
                if (flag == 1)
                    neighbor = position + ng_step*local_box_size[0];
                if (flag == 2)
                    neighbor = position + ng_step*local_box_size[0]*local_box_size[1];
                
                buf[position] = (buf[position] + buf[neighbor])/2.0;
                buf[neighbor] = buf[position] - buf[neighbor];
            }
        }
    }
}

// Wavelet transform (inplace)
static void wavelet_transform(float * buf)
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

// A reorganisation hepler
static void reorg_helper(float * buf1,  float * buf2, int step, int *index, int sk, int si, int sj)
{
    for (int k = sk; k < local_box_size[2]; k+=step)
    {
        for (int i = si; i < local_box_size[1]; i+=step)
        {
            for (int j = sj; j < local_box_size[0]; j+=step)
            {
                int position = k*local_box_size[0]*local_box_size[1] + i*local_box_size[0] + j;
                buf2[*index] = buf1[position];
                *index += 1;
            }
        }
    }
}

// Copy DC component
static void read_DC(float * buf1,  float * buf2)
{
    int step = pow(2, wavelet_level);
    int index = 0;
    reorg_helper(buf1, buf2, step, &index, 0, 0, 0);
}

// A helper for Idx Encoding
static void idx_helper(float *buf1, float *buf2, int si, int sj, int sk, int ti, int tj, int tk, int *index)
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

// IDX encoding
static void idx_encoding(float *buf1, float *buf2)
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

// Calulate the dimension of each level
static void calculate_level_dimension(int* size, int level)
{
    for (int i = 0; i < 3; i++)
    {
        size[i] = local_box_size[i]/pow(2, level);
    }
}

// ZFP compression
std::vector<unsigned char>
compress_3D_float(float* buf, int dim_x, int dim_y, int dim_z, float param, int flag)
{
    zfp_type type = zfp_type_float;
    zfp_field* field = zfp_field_3d(buf, type, dim_x, dim_y, dim_z);
    zfp_stream* zfp = zfp_stream_open(nullptr);
    if (flag == 0)
        zfp_stream_set_accuracy(zfp, param);
    else if (flag == 1)
        zfp_stream_set_precision(zfp, param);
    else
    {
        std::cout << "-z should be followed by (0 or 1)\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    size_t max_compressed_bytes = zfp_stream_maximum_size(zfp, field);
    std::vector<unsigned char> output(max_compressed_bytes);
    bitstream* stream = stream_open(&output[0], max_compressed_bytes);
    zfp_stream_set_bit_stream(zfp, stream);
    size_t compressed_bytes = zfp_compress(zfp, field);
    if (compressed_bytes == 0)
        puts("ERROR: Something wrong happened during compression\n");
    output.resize(compressed_bytes);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    return output;
}

// ZFP compression of each subband per level
static void compressed_subbands(float * buf)
{
    int level_size[3];
    
    for (int level = wavelet_level; level > 0; level--)
    {
        // Calculate dimention per level
        calculate_level_dimension(level_size, level);
        int size = level_size[0] * level_size[1] * level_size[2];
        float* level_buf = (float *)malloc(size * sizeof(float));
        
        int step = pow(2, level);
        int n_step[2] = {0, step/2};
    
        // Define the start points for HHL, HLL, HLH ...
        for(int k = 0; k < 2; k++)
        {
            for(int i = 0; i < 2; i++)
            {
                for(int j = 0; j < 2; j++)
                {
                    int index = 0;
                    if (j == 0 && i == 0 && k == 0)
                        continue;
                    else
                    {
                        // Read subbands per level
                        reorg_helper(buf, level_buf, step, &index, n_step[k], n_step[i], n_step[j]);
                        // ZFP compression per subbands of each level
                        std::vector<unsigned char> output = compress_3D_float(level_buf, level_size[0], level_size[1], level_size[2], zfp_err_ratio, zfp_comp_flag);
                        // Combine buffer
                        memcpy(&out_buf[out_size], output.data(), output.size());
                        out_size += output.size();
                        // Push the compressed size of each subband to metadata
                        metadata.push_back(std::to_string(output.size()));
                    }
                }
            }
        }
        free(level_buf);
    }
}

// Convert dimension array to string and push it into metadata
static void array_to_string(int* array)
{
    std::string dimension = "";
    int size = 3;
    for(int i = 0; i < size; i++)
    {
        if(i < size - 1)
            dimension += std::to_string(array[i]) + "x";
        else
            dimension += std::to_string(array[i]);
    }
    metadata.push_back(dimension);
}

// Write file in parallel (a file per process)
static void write_file_parallel()
{
    std::string name = "./output";
    sprintf(write_file_name, "%s_%d", name.data(), rank);
    metadata.push_back(write_file_name);
    metadata.push_back(std::to_string(process_count));
    
    MPI_File fh;
    MPI_Status status;

    MPI_File_open(MPI_COMM_SELF, write_file_name, MPI_MODE_WRONLY|MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    MPI_File_write(fh, out_buf, out_size, MPI_UNSIGNED_CHAR, &status);
    MPI_File_close(&fh);
    
    // Process 0 writes metadata file
    if (rank == 0)
    {
        std::ofstream outfile("./metadata");
        for (const auto &e: metadata) outfile << e << "\n";
    }
}


