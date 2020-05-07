//
//  write.cpp
//  multiResolution
//
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

static int rank = 0;
static int process_count = 1;

static int global_box_size[3];
static int local_box_size[3];
static char file_name[512];
static char write_file_name[512];
static int zfp_comp_flag = -1;
static float zfp_err_ratio = -1.0;
static int encoding_mode = -1;
static int wavelet_level = -1;
static int idx_level = -1;
static int local_box_offset[3];
static int sub_div[3];
static int out_size = 0;

std::vector<float> metadata;
unsigned char *out_buf;

static void parse_args(int argc, char * argv[]);
static void check_args(int argc, char * argv[]);
static void MPI_Initial(int argc, char * argv[]);
static void calculate_per_process_offsets();
static void read_file_parallel(float *buf, int size);
static void push_array(int* array);
static void wavelet_transform(float *buf);
static void compressed_wavelet(float * buf);
static void idx_encoding(float *buf1, float *buf2);
static void compress_idx(float * buf, int idx_size);
static void write_file_parallel();


std::vector<unsigned char>
compress_3D_float(float* buf, int dim_x, int dim_y, int dim_z, float param, int flag);

int main(int argc, char * argv[])
{
    // MPI environment initialization
    MPI_Initial(argc, argv);
    
    // Parse arguments
    parse_args(argc, argv);
    
    // Check correctness of arguments if users don't require help
    std::string arg = argv[1];
    if (arg.compare("-h") == 0)
        MPI_Abort(MPI_COMM_WORLD, -1);
    
    // Check arguments
    check_args(argc, argv);
    
    // Calculate offsets per process
    calculate_per_process_offsets();
    
    // Check the number of processes which should be equal to the (global_size/local_size)
    int processes = sub_div[0] * sub_div[1] * sub_div[2];
    if(process_count != processes)
    {
        std::cout << "Error: The number of processes should equal to " << processes << ". (global_size/local_size)\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    // Push basic parameters into metadata
    push_array(global_box_size);
    push_array(local_box_size);
    metadata.push_back(encoding_mode);
    metadata.push_back(zfp_comp_flag);
    metadata.push_back(zfp_err_ratio);
    metadata.push_back(process_count);
    
    // Allocate memory for a buf per process
    float * buf;
    int local_size = local_box_size[0] * local_box_size[1] * local_box_size[2];
    int offset =  local_size * sizeof(float);
    buf = (float *)malloc(offset);
    
    // Output buffer
    out_buf = (unsigned char *)malloc(offset);
    
    // Returns an elapsed time on the calling processor
    double io_read_start = MPI_Wtime();
    // Read file in parallel (chunks)
    read_file_parallel(buf, local_size);
    double io_read_end = MPI_Wtime();
    
    double encoding_start; double encoding_end;
    double zfp_start; double zfp_end;
    // Calculate the wavelet levels or idx levels based on the encoding mode
    if(encoding_mode == 0)
    {
        // The max number of wavelet levels
        auto min = *std::min_element(local_box_size, local_box_size + 3);
        wavelet_level = log2(min);
        // Perform wavelet transform
        encoding_start = MPI_Wtime();
        wavelet_transform(buf);
        encoding_end = MPI_Wtime();
        // ZFP Compress
        zfp_start = MPI_Wtime();
        compressed_wavelet(buf);
        zfp_end = MPI_Wtime();
        free(buf);
    }
    else
    {
        // Calculate the levels of idx
        idx_level = log2(local_size) + 1;
        float* idx_buf = (float *)malloc(offset);
        // IDX Encoding
        encoding_start = MPI_Wtime();
        idx_encoding(buf, idx_buf);
        encoding_end = MPI_Wtime();
        free(buf);
        // ZFP Compress
        zfp_start = MPI_Wtime();
        compress_idx(idx_buf, local_size);
        zfp_end = MPI_Wtime();
        free(idx_buf);
    }
    
    // Write file in parallel
    double io_write_start = MPI_Wtime();
    write_file_parallel();
    double io_write_end = MPI_Wtime();
    
    std::string mode = (encoding_mode == 0)? "Wavelet": "IDX";
    
    // Calculating the total time and other decomposistion time
    double total_time = io_write_end - io_read_start;
    double max_time;
    MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (max_time == total_time)
    {
        printf("Time take for %s is %f \nTime decomposistion (IO + ENCOD + COMPRESS + IO): %f + %f + %f + %f \n", mode.c_str(), max_time, (io_read_end - io_read_start), (encoding_end - encoding_start), (zfp_end - zfp_start), (io_write_end - io_write_start));
    }
    
    free(out_buf);
    MPI_Finalize();
    return 0;
}

// MPI Environment Initialization
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

// Parse arguments
static void parse_args(int argc, char * argv[])
{
    char options[] = "hg:l:f:m:z:e:";
    int one_opt = 0;
    
    while((one_opt = getopt(argc, argv, options)) != EOF)
    {
        switch (one_opt)
        {
            case('h'): // show help
                if (rank == 0)
                    std::cout << "Help:\n" << "-g Specify the global box size (e.g., axbxc)\n" << "-l Specify the local box size (e.g., axbxc)\n" << "-f Specify the input file path (e.g., ./example.txt)\n" << "-m Specify encoding mode (0 means wavelet, 1 means idx) \n" << "-z Specify the zfp compress flag (0 means accuracy, 1 means precision)\n" << "-e Specify the error tolerant rate of zfp compression (range: [0,1])\n\n";
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
                
            case('m'): // the encoding mode (0 means wavelet, 1 means idx)
                if((sscanf(optarg, "%d", &encoding_mode) == EOF) || encoding_mode > 1)
                {
                    std::cout << "ERROR: Encoding mode should be 0(wavelet) or 1(idx)\n";
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
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
    if (global_box_size[0] == 0 || global_box_size[1] == 0 || global_box_size[2] == 0 || local_box_size[0] == 0 || local_box_size[1] == 0 || local_box_size[2] == 0 || file_name[0] == ' ' || encoding_mode == -1 || zfp_comp_flag == -1 || zfp_err_ratio == -1.0)
    {
        std::cout << "Plese using -h to get help\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (global_box_size[0] < local_box_size[0] || global_box_size[1] < local_box_size[1] || global_box_size[2] < local_box_size[2])
    {
        std::cerr << "ERROR: Global box is smaller than local box in one of the dimensions\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
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

// Convert dimension array to string and push it into metadata
static void push_array(int* array)
{
    int size = 3;
    for(int i = 0; i < size; i++)
    {
        metadata.push_back(array[i]);
    }
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

// Calulate the dimension of each level
static void calculate_level_dimension(int* size, int level)
{
    for (int i = 0; i < 3; i++)
    {
        size[i] = local_box_size[i]/pow(2, level);
    }
}

// Compressed top two levels after wavelet transfrom
static void compressed_top_two_level(float * buf)
{
    float* level_buf = (float *)malloc(64 * sizeof(float));
    
    // Read DC component
    int step = pow(2, wavelet_level);
    int index = 0;
    reorg_helper(buf, level_buf, step, &index, 0, 0, 0);
    
    // Read each subbands for top two level
    for (int level = wavelet_level; level > wavelet_level-2; level--){
        step = pow(2, level);
        int n_step[2] = {0, step/2};
        
        for(int k = 0; k < 2; k++){
            for(int i = 0; i < 2; i++){
                for(int j = 0; j < 2; j++){
                    if (j == 0 && i == 0 && k == 0)
                        continue;
                    reorg_helper(buf, level_buf, step, &index, n_step[k], n_step[i], n_step[j]);
                }
            }
        }
    }
    // ZFP compress
    std::vector<unsigned char> output = compress_3D_float(level_buf, 4, 4, 4, zfp_err_ratio, zfp_comp_flag);
    free(level_buf);
    // Combine buffer
    memcpy(&out_buf[out_size], output.data(), output.size());
    out_size += output.size();
    metadata.push_back(output.size());
}

// ZFP compression of each subband per level
static void compressed_subbands(float * buf)
{
    int level_size[3];
    
    for (int level = wavelet_level-2; level > 0; level--)
    {
        // Calculate dimention per level
        calculate_level_dimension(level_size, level);
        int size = level_size[0] * level_size[1] * level_size[2];
        float* level_buf = (float *)malloc(size * sizeof(float));
        
        int step = pow(2, level);
        int n_step[2] = {0, step/2};
        
        // Define the start points for HHL, HLL, HLH ...
        for(int k = 0; k < 2; k++){
            for(int i = 0; i < 2; i++){
                for(int j = 0; j < 2; j++){
                    if (j == 0 && i == 0 && k == 0)
                        continue;
                    else{
                        int index = 0;
                        // Read subbands per level
                        reorg_helper(buf, level_buf, step, &index, n_step[k], n_step[i], n_step[j]);
                        // ZFP compression per subbands of each level
                        std::vector<unsigned char> output = compress_3D_float(level_buf, level_size[0], level_size[1], level_size[2], zfp_err_ratio, zfp_comp_flag);
                        
                        // Combine buffer
                        memcpy(&out_buf[out_size], output.data(), output.size());
                        out_size += output.size();
                        metadata.push_back(output.size());
                    }
                }
            }
        }
        free(level_buf);
    }
}

// compressed data after wavelet transform
static void compressed_wavelet(float * buf)
{
    compressed_top_two_level(buf);
    compressed_subbands(buf);
}

// A helper for Idx Encoding
static void idx_helper(float *buf1, float *buf2, int si, int sj, int sk, int ti, int tj, int tk, int *index)
{
    for (int k = sk; k < local_box_size[2]; k+=tk)
    {
        for (int i = si; i < local_box_size[1]; i+=ti)
        {
            for (int j = sj; j < local_box_size[0]; j+=tj)
            {
                int position = k*local_box_size[0]*local_box_size[1] + i*local_box_size[1] + j;
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
    
    int ti = local_box_size[0];
    int tj = local_box_size[1];
    int tk = local_box_size[2];
    
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

// compressed each level after idx encoding
static void compress_idx(float * buf, int idx_size)
{
    int dim_x = 4; int dim_y = 4; int dim_z = 4;
    int size = dim_x * dim_y * dim_z;
    float* init_buf = (float *)malloc(size*sizeof(float));
    memcpy(init_buf, buf, size*sizeof(float));
    std::vector<unsigned char> output = compress_3D_float(init_buf, dim_x, dim_y, dim_z, zfp_err_ratio, zfp_comp_flag);
    free(init_buf);
    
    memcpy(&out_buf[out_size], output.data(), output.size());
    out_size += output.size();
    metadata.push_back(output.size());
    
    int count = size;
    int i = 0;
    
    while (size < idx_size-size+1) {
        float* level_buf = (float *)malloc(size*sizeof(float));
        memcpy(level_buf, &buf[count], size*sizeof(float));
        output = compress_3D_float(level_buf, dim_x, dim_y, dim_z, zfp_err_ratio, zfp_comp_flag);
        
        memcpy(&out_buf[out_size], output.data(), output.size());
        out_size += output.size();
        metadata.push_back(output.size());
        
        count += size;
        free(level_buf);
        
        if (i%3 == 0)
            dim_x *= 2;
        else if (i%3 == 1)
            dim_y *= 2;
        else
            dim_z *= 2;
        size *= 2;
        i++;
    }
}

// Write file in parallel (a file per process)
static void write_file_parallel()
{
    std::string name = "./output";
    sprintf(write_file_name, "%s_%d", name.data(), rank);
    
    MPI_File fh;
    MPI_Status status;
    
    MPI_File_open(MPI_COMM_SELF, write_file_name, MPI_MODE_WRONLY|MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    MPI_File_write(fh, out_buf, out_size, MPI_UNSIGNED_CHAR, &status);
    MPI_File_close(&fh);
    
    // Process 0 writes metadata file
    std::string meta_name = "./metadata";
    char metadata_file[512];
    sprintf(metadata_file, "%s_%d", meta_name.data(), rank);
    std::ofstream outfile(metadata_file);
    for (const auto &e: metadata) outfile << e << "\n";
    outfile.close();
}
