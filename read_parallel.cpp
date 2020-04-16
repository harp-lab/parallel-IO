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
#include <tuple>
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
static int dc_box_size[3];
static int sub_div[3];
static int local_box_offset[3];
static int rank = 0;
static int process_count = 1;
static int idx_level = -1;
static int require_level = -1;
static int total_level = -1;
static int dc_size = 1;
static int buf_num;
static char file_name[512];

std::vector<float> metadata;
std::vector<std::string> file_names;

unsigned char *out_buf;

static void MPI_Initial(int argc, char * argv[]);
static void read_metadata(int num);
static void parse_metadata();
static void parse_args(int argc, char * argv[]);
static void check_args(int argc, char * argv[]);
static int calculate_buffer_offset();
static void read_file_decompress(float* buf, int num);
static void idx_decoding(float *buf1, float *buf2, int level);
static void wavelet_recover(float * buf);
static void calculate_per_process_offsets();

bool decompress_3D_float(const char* input, size_t bytes, int dim_x, int dim_y, int dim_z, float param, char** output, int flag);


int main(int argc, char * argv[])
{
    read_metadata(0);
    parse_metadata();
    
    // MPI environment initialization
    MPI_Initial(argc, argv);
    double starttime = MPI_Wtime();
    
    
    for(int i = 0; i < 3; i++)
    {
        dc_box_size[i] = local_box_size[i]/pow(2, wavelet_level);
    }
    dc_size = dc_box_size[0] * dc_box_size[1] * dc_box_size[2];
    
    int idx_level_size = 4*4*4;
    idx_level = 1;
    while (idx_level_size < dc_size) {
        idx_level_size *= 2;
        idx_level++;
    }
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
    buf_num = (rank < rem)?(divis+1):divis;
    
    int num = rank;
    int diff = total_level - require_level;
    while (num < write_processes_num)
    {
        float *buf = (float *)malloc(offset);
        read_file_decompress(buf, num);
        
        float* idx_buf = (float *)calloc(dc_size, sizeof(float));
        idx_decoding(buf, idx_buf, diff);
        
        if(diff > idx_level)
        {
            memcpy(buf, idx_buf, dc_size*sizeof(float));
            wavelet_recover(buf);
        }
        
        //        calculate_per_process_offsets();
        
        free(idx_buf);
        num += process_count;
        free(buf);
    }
    
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
static void read_metadata(int num)
{
    std::string meta_name = "./metadata";
    char metadata_file[512];
    sprintf(metadata_file, "%s_%d", meta_name.data(), num);
    
    std::ifstream infile(metadata_file);
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
        write_processes_num = metadata[9];
    }
    else
    {
        std::cout << "Error: Read Metadata Fail!!!\n";
        exit(-1);
    }
}


static void parse_args(int argc, char * argv[])
{
    char options[] = "hl:f:";
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
            case('f'): // read file name
                if (sprintf(file_name, "%s", optarg) < 0)
                    MPI_Abort(MPI_COMM_WORLD, -1);
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
    if (require_level <= wavelet_level)
    {
        for(int i = 0; i < 3; i++)
        {
            offset *= (local_box_size[i]/pow(2, require_level));
        }
    }
    else if (require_level == total_level)
    {
        offset *= 64;
    }
    else
    {
        int dif = require_level-wavelet_level;
        offset *= 2*dc_size/(pow(2, dif));
    }
    return offset;
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


static void calculate_level_dimension(int* size, int level)
{
    for (int i = 0; i < 3; i++)
    {
        size[i] = local_box_size[i]/pow(2, level);
    }
}

static void calculate_idx_level_dimension(int* size, int diff)
{
    size[0] = 4; size[1] = 4; size[2] = 4;
    
    for(int i = 1; i < diff; i++)
    {
        if (i%3 == 1)
            size[0] *= 2;
        else if (i%3 == 2)
            size[1] *= 2;
        else
            size[2] *= 2;
    }
}


std::vector<int> parse_compress_size()
{
    std::vector<int> compress_size;
    int num_blocks = wavelet_level*7 + idx_level;
    for(int i = 10; i < num_blocks+10; i++)
        compress_size.push_back(int(metadata[i]));
    return compress_size;
}


std::tuple<int, int> idx_decompress(float* buf,  unsigned char* tmp_buf, std::vector<int> compress_size, int max_i)
{
    int level_size = 0;
    int level_dimen[3];
    int start_point = 0;
    int out_size = 0;
    
    for(int i = 0; i < max_i; i++)
    {
        calculate_idx_level_dimension(level_dimen, i);
        level_size = level_dimen[0] * level_dimen[1] * level_dimen[2];
        
        float* decompress_buf = (float *)malloc(level_size * sizeof(float));
        decompress_3D_float((const char*)&tmp_buf[out_size], compress_size[i], level_dimen[0], level_dimen[1], level_dimen[2], zfp_err_ratio, (char**)&decompress_buf, zfp_comp_flag);
        
        memcpy(&buf[start_point], decompress_buf, level_size*sizeof(float));
        start_point += level_size;
        out_size += compress_size[i];
        free(decompress_buf);
    }
    
    return std::make_tuple(start_point, out_size);
}


static void wavelet_decompress(float* buf, unsigned char* tmp_buf, std::vector<int> compress_size, int start_point, int out_size)
{
    int ind = (wavelet_level - require_level > 0)? (wavelet_level - require_level):0;
    int level = wavelet_level;
    int level_dimen[3];
    
    for(int i = idx_level; i < ind*7+idx_level; i++)
    {
        calculate_level_dimension(level_dimen, level);
        int level_size = level_dimen[0] * level_dimen[1] * level_dimen[2];
        
        if (i > idx_level && (i-idx_level)%7 == 0)
        {
            level--;
            calculate_level_dimension(level_dimen, level);
            level_size = level_dimen[0] * level_dimen[1] * level_dimen[2];
        }
        
        float* decompress_buf = (float *)malloc(level_size * sizeof(float));
        decompress_3D_float((const char*)&tmp_buf[out_size], compress_size[i], level_dimen[0], level_dimen[1], level_dimen[2], zfp_err_ratio, (char**)&decompress_buf, zfp_comp_flag);
        
        memcpy(&buf[start_point], decompress_buf, level_size*sizeof(float));
        start_point += level_size;
        out_size += compress_size[i];
        free(decompress_buf);
    }
}


static void read_file_decompress(float* buf, int num)
{
    int ind = (wavelet_level - require_level > 0)? (wavelet_level - require_level):0;
    int level_dimen[3];
    int diff = 0;
    if(require_level > wavelet_level)
    {
        diff = total_level - require_level;
        calculate_idx_level_dimension(level_dimen, diff);
    }
    
    std::string name = "./output";
    read_metadata(num);
    std::vector<int> compress_size = parse_compress_size();
    
    // Get file name
    char file_name[512];
    sprintf(file_name, "%s_%d", name.data(), num);
    
    int size = 0;
    if (require_level > wavelet_level)
    {
        for(int i = 0; i < diff+1; i++)
            size += compress_size[i];
    }
    else
    {
        for(int i = 0; i < ind*7+idx_level; i++)
            size += compress_size[i];
    }
    
    unsigned char* tmp_buf = (unsigned char*)malloc(size);
    
    MPI_File fh;
    MPI_Status status;
    MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read(fh, tmp_buf, size, MPI_UNSIGNED_CHAR, &status);
    MPI_File_close(&fh);
    
    if (require_level > wavelet_level)
    {
        idx_decompress(buf, tmp_buf, compress_size, diff+1);
    }
    else
    {
        int start_point; int out_size;
        std::tie(start_point, out_size)  = idx_decompress(buf, tmp_buf, compress_size, idx_level);
        wavelet_decompress(buf, tmp_buf, compress_size, start_point, out_size);
        
    }
    free(tmp_buf);
}


static void idx_helper(float *buf1, float *buf2, int si, int sj, int sk, int ti, int tj, int tk, int *index)
{
    for (int k = sk; k < dc_box_size[2]; k+=tk)
    {
        for (int i = si; i < dc_box_size[1]; i+=ti)
        {
            for (int j = sj; j < dc_box_size[0]; j+=tj)
            {
                int position = k*dc_box_size[0]*dc_box_size[1] + i*dc_box_size[0] + j;
                buf2[position] = buf1[*index];
                *index += 1;
            }
        }
    }
}


static void idx_decoding(float *buf1, float *buf2, int level)
{
    level = (level < idx_level)? level: (idx_level-1);
    
    buf2[0] = buf1[0];
    int index = 1;
    
    int si, sj, sk;
    int level_dimen[3];
    
    calculate_idx_level_dimension(level_dimen, level);
    
    int ti = dc_box_size[0];
    int tj = dc_box_size[1];
    int tk = dc_box_size[2];
    
    int level_size = 64;
    for(int l = 0; l < level; l++)
    {
        level_size *= 2;
    }
    int count = log2(level_size);
    
    int i = 0;
    while ( i < count)
    {
        sj = tj/2; si = 0; sk = 0;
        idx_helper(buf1, buf2, si, sj, sk, ti, tj, tk, &index);
        tj = tj/2;
        i++;
        if( i < count)
        {
            si = ti/2; sj = 0; sk = 0;
            idx_helper(buf1, buf2, si, sj, sk, ti, tj, tk, &index);
            ti = ti/2;
            i++;
        }
        if( i < count)
        {
            sk = tk/2; si = 0; sj = 0;
            idx_helper(buf1, buf2, si, sj, sk, ti, tj, tk, &index);
            tk = tk/2;
            i++;
        }
    }
}


static void wavelet_recover(float* buf)
{
    int level_dimen[3];
    for(int level = wavelet_level; level > require_level; level--)
    {
        calculate_level_dimension(level_dimen, level);
        int level_size = level_dimen[0] * level_dimen[1] * level_dimen[2];
        
        // z
        float* dc_buf = (float *)malloc(4*level_size*sizeof(float));
        float* sub_buf = (float *)malloc(4*level_size*sizeof(float));
        memcpy(dc_buf, buf, 4*level_size*sizeof(float));
        memcpy(sub_buf, &buf[4*level_size], 4*level_size*sizeof(float));
        
        float* tmp_buf = (float *)malloc(2*level_size*sizeof(float));
        for(int s = 0; s < 4; s++)
        {
            for(int k = 0; k < level_dimen[2]; k++)
            {
                int page = level_dimen[0] * level_dimen[1];
                for(int i = 0; i < page; i++)
                {
                    int position = s*level_size + k*page + i;
                    int index = 2*k*page + i;
                    int neighbor = index + page;
                    tmp_buf[index] = dc_buf[position] + sub_buf[position];
                    tmp_buf[neighbor] = dc_buf[position] - sub_buf[position];
                }
            }
            memcpy(&buf[s*level_size], &tmp_buf[0], level_size*sizeof(float));
            memcpy(&buf[(s+4)*level_size], &tmp_buf[level_size], level_size*sizeof(float));
        }
        
        // y
        int start = 0;
        for(int i = 0; i < 8; i+=4)
        {
            memcpy(&dc_buf[start], &buf[i*level_size], 2*level_size*sizeof(float));
            memcpy(&sub_buf[start], &buf[(i+2)*level_size], 2*level_size*sizeof(float));
            start += 2*level_size;
        }
        
        
        for(int s = 0; s < 4; s++)
        {
            for(int k = 0; k < level_dimen[2]; k++)
            {
                for(int i = 0; i < level_dimen[1]; i++)
                {
                    for(int j = 0; j < level_dimen[0]; j++)
                    {
                        int position = s*level_size + k*level_dimen[1]*level_dimen[0] + i*level_dimen[0] + j;
                        int index = 2*k*level_dimen[1]*level_dimen[0] + 2*i*level_dimen[0] + j;
                        int neighbor = index + level_dimen[0];
                        tmp_buf[index] = dc_buf[position] + sub_buf[position];
                        tmp_buf[neighbor] = dc_buf[position] - sub_buf[position];
                    }
                }
            }
            if(s < 2)
            {
                memcpy(&buf[s*level_size], &tmp_buf[0], level_size*sizeof(float));
                memcpy(&buf[(s+2)*level_size], &tmp_buf[level_size], level_size*sizeof(float));
            }
            else
            {
                memcpy(&buf[(s+2)*level_size], &tmp_buf[0], level_size*sizeof(float));
                memcpy(&buf[(s+4)*level_size], &tmp_buf[level_size], level_size*sizeof(float));
            }
        }
        
        // x
        start = 0;
        for(int i = 0; i < 8; i+=2)
        {
            memcpy(&dc_buf[start], &buf[i*level_size], level_size*sizeof(float));
            memcpy(&sub_buf[start], &buf[(i+1)*level_size], level_size*sizeof(float));
            start += level_size;
        }
        
        for(int s = 0; s < 4; s++)
        {
            for(int k = 0; k < level_dimen[2]; k++)
            {
                for(int i = 0; i < level_dimen[1]; i++)
                {
                    for(int j = 0; j < level_dimen[0]; j++)
                    {
                        int position = s*level_size + k*level_dimen[1]*level_dimen[0] + i*level_dimen[0] + j;
                        int index = 2*k*level_dimen[1]*level_dimen[0] + 2*i*level_dimen[0] + 2*j;
                        int neighbor = index + 1;
                        tmp_buf[index] = dc_buf[position] + sub_buf[position];
                        tmp_buf[neighbor] = dc_buf[position] - sub_buf[position];
                    }
                }
            }
            memcpy(&buf[2*s*level_size], &tmp_buf[0], level_size*sizeof(float));
            memcpy(&buf[(2*s+1)*level_size], &tmp_buf[level_size], level_size*sizeof(float));
        }
        free(tmp_buf);
        free(dc_buf);
        free(sub_buf);
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

//static void write_one_file(float * buf, int size)
//{
//    MPI_Datatype subarray = create_subarray();
//    MPI_File fh;
//    MPI_Status status;
//
//    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
//    MPI_File_set_view(fh, 0, MPI_FLOAT, subarray, "native", MPI_INFO_NULL);
//    MPI_File_read(fh, buf, size, MPI_FLOAT, &status);
//    MPI_File_close(&fh);
//}



