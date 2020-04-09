//
//  read.cpp
//  multiResolution
//
//  Created by kokofan on 4/2/20.
//  Copyright © 2020 koko Fan. All rights reserved.
//

#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include "zfp/include/zfp.h"

static int global_box_size[3];
static int local_box_size[3];
static int wavelet_level = -1;
static int zfp_comp_flag = -1;
static float zfp_err_ratio = -1.0;
static int write_processes_num = -1;
static int rank = 0;
static int process_count = 1;
static int idx_level = -1;
static int require_level = -1;
static int total_level = -1;
static int dc_size = 1;

std::vector<float> metadata;
std::vector<int> compress_size;
std::vector<std::string> file_names;

unsigned char *out_buf;

static void MPI_Initial(int argc, char * argv[]);
static void read_metadata();
static void parse_metadata();
static void parse_args(int argc, char * argv[]);
static void check_args(int argc, char * argv[]);
static int calculate_buffer_offset();
static void read_file_decompress(float* buf);

bool decompress_3D_float(const char* input, size_t bytes, int dim_x, int dim_y, int dim_z, float param, char** output, int flag);

int main(int argc, char * argv[])
{
    read_metadata();
    parse_metadata();
    
    // MPI environment initialization
    MPI_Initial(argc, argv);
    double starttime = MPI_Wtime();
    
    for(int i = 0; i < 3; i++)
    {
        dc_size = dc_size * (local_box_size[i]/pow(2, wavelet_level));
    }
    
    // First three levels of idx can be treated as first level
    idx_level = log2(dc_size) + 1 - 2;
    total_level = wavelet_level + idx_level;
    
    // Prase arguments
    parse_args(argc, argv);
    
    std::string arg = argv[1];
    if (arg.compare("-h") == 0)
        MPI_Abort(MPI_COMM_WORLD, -1);
    
    check_args(argc, argv);
    
    int offset = calculate_buffer_offset();

    int divis = write_processes_num/process_count;
    int rem = write_processes_num%process_count;
    int buf_num = (rank < rem)?(divis+1):divis;
    
    float * buf = (float *)malloc(offset*buf_num);
    read_file_decompress(buf);
    
    MPI_Finalize();
    return 0;
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

// Read metadata file
static void read_metadata()
{
    std::ifstream infile("./metadata");
    float a;
    while (infile >> a)
    {
        metadata.push_back(a);
    }
    infile.close();
}

// Parse meradata
static void parse_metadata()
{
    if (metadata.size() > 0)
    {
        for(int i = 0; i < 3; i++)
            global_box_size[i] = (int)metadata[i];
        for(int i = 3; i < 6; i++)
            local_box_size[i-3] = (int)metadata[i];
        wavelet_level = metadata[6];
        zfp_comp_flag = metadata[7];
        zfp_err_ratio = metadata[8];
        int num_blocks = wavelet_level * 7 + 1;
        for(int i = 9; i < num_blocks+9; i++)
            compress_size.push_back(int(metadata[i]));
        write_processes_num = metadata[num_blocks+9];
    }
    else
    {
        std::cout << "Error: Read Metadata Fail!!!\n";
        exit(-1);
    }
}


static void parse_args(int argc, char * argv[])
{
    char options[] = "hl:";
    int one_opt = 0;
    
    while((one_opt = getopt(argc, argv, options)) != EOF)
    {
        switch (one_opt)
        {
            case('h'): // show help
                if (rank == 0)
                    std::cout << "\nHelp: There are " << total_level << " levels, please use -l to specify which level you want to read.\n" << "Range[0, " << total_level << "] (level 0 means the whole data).\n\n";
                break;
            case('l'): // specify which level to read
                if((sscanf(optarg, "%d", &require_level) == EOF) || require_level < 0 || require_level > total_level)
                {
                    std::cout << "Required level should be in range [0, " << total_level << "]\n";
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
                 break;
        }
    }
}

// Check arguments
static void check_args(int argc, char * argv[])
{
    if (require_level == -1)
    {
        std::cout << "Plese using -h to get help\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

static int calculate_buffer_offset()
{
    int offset = sizeof(float);
    if (require_level < wavelet_level)
    {
        for(int i = 0; i < 3; i++)
        {
            offset = offset * (local_box_size[i]/pow(2, require_level));
        }
    }
    else
    {
        offset = offset * dc_size;
    }
    return offset;
}


static void calculate_level_dimension(int* size, int level)
{
    for (int i = 0; i < 3; i++)
    {
        size[i] = local_box_size[i]/pow(2, level);
    }
}


static void read_file_decompress(float* buf)
{
    int num = rank;
    int ind = (wavelet_level - require_level > 0)? (wavelet_level - require_level):0;
    int start_point = 0;
    
    int size = 0;
    for(int i = 0; i < ind*7+1; i++)
    {
        size += compress_size[i];
    }
    
    std::string name = "./output";
    while (num < write_processes_num)
    {
        // Get file name
        char file_name[512];
        sprintf(file_name, "%s_%d", name.data(), num);
        
        // Read file
        unsigned char* tmp_buf = (unsigned char*)malloc(size);
        MPI_File fh;
        MPI_Status status;
        MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        MPI_File_read(fh, tmp_buf, size, MPI_UNSIGNED_CHAR, &status);
        MPI_File_close(&fh);
        
        // Initial parameters
        int out_size = 0;
        int level = wavelet_level;
        int level_size[3];
        calculate_level_dimension(level_size, level);
        int size = level_size[0] * level_size[1] * level_size[2];

        // ZFP decompress each sub-bands
        for(int i = 0; i < ind*7+1; i++)
        {
            // Calculate dimension per level
            if (i > 1 && (i-1)%7 == 0)
            {
                level--;
                calculate_level_dimension(level_size, level);
                size = level_size[0] * level_size[1] * level_size[2];
            }
            
            // ZFP decompression
            float* decompress_buf = (float *)malloc(size * sizeof(float));
            decompress_3D_float((const char*)&tmp_buf[out_size], compress_size[i], level_size[0], level_size[1], level_size[2], zfp_err_ratio, (char**)&decompress_buf, zfp_comp_flag);
            
            // Combine each decompression buf
            memcpy(&buf[start_point], decompress_buf, size*sizeof(float));

            start_point += size;
            free(decompress_buf);
            out_size += compress_size[i];
        }
        
        free(tmp_buf);
        num += process_count;
    }
}


bool decompress_3D_float(const char* input, size_t bytes, int dim_x, int dim_y, int dim_z, float param, char** output, int flag)
{
    assert(input);
    assert(output);
    
    zfp_field* field = zfp_field_3d(*output, zfp_type_float, dim_x, dim_y, dim_z);
    zfp_stream* zfp = zfp_stream_open(nullptr);
    //zfp_stream_set_accuracy(zfp, accuracy, zfp_type_double);
    //zfp_stream_set_rate(zfp, 2, zfp_type_double, 2, 0);
    if (flag == 0)
        zfp_stream_set_accuracy(zfp, param);
    else if (flag == 1)
        zfp_stream_set_precision(zfp, param);
    else
    {
        std::cout << "-z should be followed by (0 or 1)\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    bitstream* stream = stream_open((void*)input, bytes);
    zfp_stream_set_bit_stream(zfp, stream);
    if (!zfp_decompress(zfp, field)) {
        std::cout << "Something wrong happened during decompression\n";
        return false;
    }
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    return true;
}
