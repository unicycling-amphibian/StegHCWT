//wavelet steg
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//h.264 vid
#include <stdio.h>
#include <stdlib.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

int decode_video(const char *input_filename, uint8_t ***pixels) {
    AVFormatContext *format_context = NULL;
    AVCodecContext *codec_context = NULL;
    AVCodec *codec = NULL;
    AVFrame *frame = NULL;
    AVPacket packet;
    int video_stream_index;
    int ret, got_frame;

    // Initialize the FFmpeg library
    av_register_all();

    // Open the input file
    if ((ret = avformat_open_input(&format_context, input_filename, NULL, NULL)) < 0) {
        fprintf(stderr, "Error opening input file: %s\n", av_err2str(ret));
        return ret;
    }

    // Retrieve stream information
    if ((ret = avformat_find_stream_info(format_context, NULL)) < 0) {
        fprintf(stderr, "Error finding stream information: %s\n", av_err2str(ret));
        return ret;
    }

    // Find the video stream index
    video_stream_index = -1;
    for (int i = 0; i < format_context->nb_streams; i++) {
        if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }
    if (video_stream_index == -1) {
        fprintf(stderr, "Error finding video stream\n");
        return -1;
    }

    // Find the decoder for the video stream
    codec = avcodec_find_decoder(format_context->streams[video_stream_index]->codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "Error finding codec\n");
        return -1;
    }

    // Allocate a codec context for the decoder
    codec_context = avcodec_alloc_context3(codec);
    if (!codec_context) {
        fprintf(stderr, "Error allocating codec context\n");
        return -1;
    }

    // Copy the codec parameters from the video stream to the codec context
    if ((ret = avcodec_parameters_to_context(codec_context, format_context->streams[video_stream_index]->codecpar)) < 0) {
        fprintf(stderr, "Error copying codec parameters: %s\n", av_err2str(ret));
        return ret;
    }

    // Open the codec
    if ((ret = avcodec_open2(codec_context,codec, NULL)) < 0) {
        fprintf(stderr, "Error opening codec: %s\n", av_err2str(ret));
        return ret;
    }
    // Allocate a frame to store the decoded video
    frame = av_frame_alloc();
    if (!frame) {
    fprintf(stderr, "Error allocating frame\n");
    return -1;
    }

    // Read the video packets and decode them into the frame
    while (av_read_frame(format_context, &packet) >= 0) {
        if (packet.stream_index == video_stream_index) {
            ret = avcodec_decode_video2(codec_context, frame, &got_frame, &packet);
            if (ret < 0) {
                fprintf(stderr, "Error decoding video: %s\n", av_err2str(ret));
                break;
            }
            if (got_frame) {
                // Allocate memory to store the pixels
                int width = codec_context->width;
                int height = codec_context->height;
                *pixels = (uint8_t **)malloc(height * sizeof(uint8_t *));
                for (int i = 0; i < height; i++) {
                    (*pixels)[i] = (uint8_t *)malloc(width * 3 * sizeof(uint8_t));
                }

                // Convert the decoded frame into a 2D array of pixels
                struct SwsContext *sws_context = sws_getContext(width, height, codec_context->pix_fmt, width, height, AV_PIX_FMT_BGR24, SWS_BILINEAR, NULL, NULL, NULL);
                uint8_t *dst_data[4];
                int dst_linesize[4];
                av_image_alloc(dst_data, dst_linesize, width, height, AV_PIX_FMT_BGR24, 1);
                sws_scale(sws_context, (const uint8_t *const *)frame->data, frame->linesize, 0, height, dst_data, dst_linesize);
                for (int i = 0; i < height; i++) {
                    memcpy((*pixels)[i], dst_data[0] + i * dst_linesize[0], width * 3 * sizeof(uint8_t));
                }
                av_freep(&dst_data[0]);
                sws_freeContext(sws_context);

                // Break the loop after converting the first frame
                break;
            }
        }
        av_packet_unref(&packet);
    }

    // Free the resources
    avcodec_free_context(&codec_context);
    av_frame_free(&frame);
    avformat_close_input(&format_context);

    return ret;
}

int encode_video(const char *output_filename, uint8_t **pixels) {
AVFormatContext *format_context = NULL;
AVCodecContext *codec_context= NULL;
AVCodec *codec = NULL;
AVFrame *frame = NULL;
AVPacket packet;
int ret, got_frame;
// Initialize the FFmpeg library
av_register_all();

// Allocate an AVFormatContext to store the output format information
ret = avformat_alloc_output_context2(&format_context, NULL, NULL, output_filename);
if (ret < 0) {
    fprintf(stderr, "Error allocating output context: %s\n", av_err2str(ret));
    return ret;
}

// Find the H.264 video codec
codec = avcodec_find_encoder(AV_CODEC_ID_H264);
if (!codec) {
    fprintf(stderr, "Error finding codec\n");
    return -1;
}

// Allocate a codec context for the H.264 codec
codec_context = avcodec_alloc_context3(codec);
if (!codec_context) {
    fprintf(stderr, "Error allocating codec context\n");
    return -1;
}

// Set the codec parameters
codec_context->bit_rate = 400000;
codec_context->width = 640;
codec_context->height = 480;
codec_context->time_base = (AVRational){1, 25};
codec_context->framerate = (AVRational){25, 1};
codec_context->gop_size = 10;
codec_context->max_b_frames = 1;
codec_context->pix_fmt = AV_PIX_FMT_YUV420P;

// Open the codec
if ((ret = avcodec_open2(codec_context, codec, NULL)) < 0) {
    fprintf(stderr, "Error opening codec: %s\n", av_err2str(ret));
    return ret;
}

// Allocate a frame to store the encoded video
frame = av_frame_alloc();
if (!frame) {
    fprintf(stderr, "Error allocating frame\n");
    return -1;
}
frame->format = codec_context->pix_fmt;
frame->width = codec_context->width;
frame->height = codec_context->height;

// Allocate the data buffers for the frame
ret = av_frame_get_buffer(frame, 32);
if (ret < 0) {
    fprintf(stderr, "Error allocating frame data: %s\n", av_err2str(ret));
    return ret;
}

// Open the output file
if (!(format_context->flags & AVFMT_NOFILE)) {
    ret = avio_open(&format_context->pb, output_filename, AVIO_FLAG_WRITE);
    if (ret < 0) {
        fprintf(stderr, "Error opening output file: %s\n", av_err2str(ret));
        return ret;
    }
}

// Write the output format header
ret = avformat_write_header(format_context,NULL);
if (ret < 0) {
fprintf(stderr, "Error writing output format header: %s\n", av_err2str(ret));
return ret;
}
// Initialize the AVPacket
av_init_packet(&packet);
packet.data = NULL;
packet.size = 0;

// Re-encode the 2D array of pixels into the H.264 video format
for (int i = 0; i < num_frames; i++) {
    // Fill the frame with the pixel data
    for (int y = 0; y < codec_context->height; y++) {
        for (int x = 0; x < codec_context->width; x++) {
            frame->data[0][y * frame->linesize[0] + x] = pixels[i][y][x];
        }
    }

    // Encode the video frame
    ret = avcodec_send_frame(codec_context, frame);
    if (ret < 0) {
        fprintf(stderr, "Error encoding video frame: %s\n", av_err2str(ret));
        return ret;
    }

    // Receive the encoded packets
    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_context, &packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            fprintf(stderr, "Error receiving encoded packet: %s\n", av_err2str(ret));
            return ret;
        }

        // Write the encoded packet to the output file
        packet.pts = packet.dts = i;
        ret = av_interleaved_write_frame(format_context, &packet);
        av_packet_unref(&packet);
        if (ret < 0) {
            fprintf(stderr, "Error writing encoded packet: %s\n", av_err2str(ret));
            return ret;
        }
    }
}

// Write the output format trailer
ret = av_write_trailer(format_context);
if (ret < 0) {
    fprintf(stderr, "Error writing output format trailer: %s\n", av_err2str(ret));
    return ret;
}

// Clean up
avcodec_free_context(&codec_context);
av_frame_free(&frame);
avio_closep(&format_context->pb);
avformat_free_context(format_context);

return 0;
}
/*
DONE:
-Reduce the number of computations: The lifting scheme based Haar wavelet transform uses two operations for each coefficient in the input: the prediction and the update. To reduce the number of computations, you can use a lifting scheme implementation that requires fewer operations, such as the 9/7 lifting scheme.
-Parallelize the computations: The lifting scheme based Haar wavelet transform can be easily parallelized, as each coefficient can be processed independently. You can use multi-threading or GPU computing to achieve this.

MAYBE:
-Reduce the size of the input: The lifting scheme based Haar wavelet transform has a time complexity of O(n), where n is the size of the input. To reduce the computation time, you can reduce the size of the input by using a video codec that compresses the video into a smaller representation or by only processing a portion of the video at a time.
-Use a more efficient data structure: The current implementation uses an array to represent the video frame. To improve the performance, you can use a more efficient data structure, such as a matrix or a linked list, to store the video frame.
*/




#define MAX_FRAME_SIZE 4096
#define MAX_SECRET_MESSAGE_LENGTH 128

double frame_data[MAX_FRAME_SIZE];
char secret_message[MAX_SECRET_MESSAGE_LENGTH];

void lifting_scheme_997_based_wavelet_transform(double *sig, int len)
{
    int i, j;
    double *tmp = (double *) malloc(len * sizeof(double));

    // Predict 1
    #pragma omp parallel for
    for (i = 1; i < len - 1; i += 2)
        sig[i] += (sig[i - 1] + sig[i + 1]) / 4.0;

    // Update 1
    #pragma omp parallel for
    for (i = 2; i < len; i += 2)
        sig[i] -= (sig[i - 1] + sig[i + 1]) / 2.0;

    // Predict 2
    #pragma omp parallel for
    for (i = 1; i < len - 1; i += 2)
        sig[i] -= (sig[i - 1] + sig[i + 1]) / 4.0;

    // Update 2
    #pragma omp parallel for
    for (i = 2; i < len; i += 2)
        sig[i] += (sig[i - 1] + sig[i + 1]) / 2.0;

    // Scaling
    for (i = 0; i < len; i++)
        tmp[i] = sig[i] / 2.0;

    // Interleave the coefficients
    for (i = 0, j = 0; i < len; i += 2, j++)
        sig[j] = tmp[i];
    for (i = 1, j = len / 2; i < len; i += 2, j++)
        sig[j] = tmp[i];

    free(tmp);
}

inverse_lifting_scheme_997_based_wavelet_transform(double *sig, int len)
{
    int i, j;
    double *tmp = (double *) malloc(len * sizeof(double));

    // De-interleave the coefficients
    for (i = 0, j = 0; i < len / 2; i++, j += 2)
        tmp[j] = sig[i];
    for (i = len / 2, j = 1; i < len; i++, j += 2)
        tmp[j] = sig[i];

    // Scaling
    for (i = 0; i < len; i++)
        tmp[i] *= 2.0;

    // Predict 2
    #pragma omp parallel for
    for (i = 1; i < len - 1; i += 2)
        tmp[i] += (tmp[i - 1] + tmp[i + 1]) / 4.0;

    // Update 2
    #pragma omp parallel for
    for (i = 2; i < len; i += 2)
        tmp[i] -= (tmp[i - 1] + tmp[i + 1]) / 2.0;

    // Predict 1
    #pragma omp parallel for
    for (i = 1; i < len - 1; i += 2)
        tmp[i] -= (tmp[i - 1] + tmp[i + 1]) / 4.0;

    // Update 1
    #pragma omp parallel for
    for (i = 2; i < len; i += 2)
        tmp[i] += (tmp[i - 1] + tmp[i + 1]) / 2.0;


    // Copy the result back to the input array
    for (i = 0; i < len; i++)
        sig[i] = tmp[i];

    // Deallocate memory
    free(tmp);
}
//DEPRECATED FUNCTIONALITY
//Reduce the number of computations: The lifting scheme based Haar wavelet transform uses two operations for each coefficient in the input: the prediction and the update. To reduce the number of computations, you can use a lifting scheme implementation that requires fewer operations, such as the 9/7 lifting scheme.
/*
void lifting_scheme_based_haar_wavelet_transform(double *data, int size) {
    int i;
    int h = size / 2;

    // Perform the prediction and update steps for the lifting scheme
    for (i = 0; i < h; i++) {
        double prediction = (data[i * 2 + 1] + data[i * 2]) / 2;
        double update = data[i * 2 + 1] - prediction;
        data[i * 2] = prediction;
        data[i * 2 + 1] = update;
    }

    // Perform the scaling step for the lifting scheme
    for (i = 0; i < h; i++) {
        data[i] *= sqrt(2);
        data[i + h] /= sqrt(2);
    }
}

void inverse_lifting_scheme_based_haar_wavelet_transform(double *data, int size) {
    int i;
    int h = size / 2;

    // Perform the inverse scaling step for the lifting scheme
    for (i = 0; i < h; i++) {
        data[i] /= sqrt(2);
        data[i + h] *= sqrt(2);
    }

    // Perform the inverse prediction and update steps for the lifting scheme
    for (i = h - 1; i >= 0; i--) {
        double update = data[i * 2 + 1];
        double prediction = data[i * 2];
        data[i * 2 + 1] = prediction + update;
        data[i * 2] = prediction - update;
    }
}
*/

void encode_secret_message(double *data, char *message, int size, int message_length) {
    int i;
    int h = size / 2;
    int message_index = 0;

    for (i = 0; i < h && message_index < message_length; i++) {
        // Encode one bit of the secret message in the least significant bit of the update coefficient
        int bit = (secret_message[message_index] >> (7 - (i % 8))) & 1;
        double update = data[i * 2 + 1];
        data[i * 2 + 1] = update + bit;

        // Move to the next byte of the secret message if all 8 bits of the current byte have been encoded
        if ((i + 1) % 8 == 0) {
            message_index++;
        }
    }
}

void decode_secret_message(double *data, char *message, int message_length) {
    int i;
    int h = message_length / 2;
    int message_index = 0;

    for (i = 0; i < h && message_index < message_length; i++) {
        // Decode one bit of the secret message from the least significant bit of the update coefficient
        int bit = (int)round(data[i * 2 + 1]) & 1;
        secret_message[message_index] |= bit << (7 - (i % 8));

        // Move to the next byte of the secret message if all 8 bits of the current byte have been decoded
        if ((i + 1) % 8 == 0) {
            message_index++;
        }
    }
}


int run_video_steganography(int argc, char *argv[]) {
FILE *cover_video, *secret_payload, *stego_video;
char *cover_video_filename, *secret_payload_filename, *stego_video_filename;
int secret_message_length = 0;
if (argc == 4 && strcmp(argv[1], "-e") == 0) {
    // Encode mode
    cover_video_filename = argv[2];
    secret_payload_filename = argv[3];
    stego_video_filename = argv[4];

    // Open the cover video file
    cover_video = fopen(cover_video_filename, "rb");
    if (cover_video == NULL) {
        printf("Error: Unable to open cover video file\n");
        return 1;
    }

    // Read a frame of the cover video
    fread(frame_data, sizeof(double), MAX_FRAME_SIZE, cover_video);

    // Close the cover video file
    fclose(cover_video);

    // Open the secret payload file
    secret_payload = fopen(secret_payload_filename, "rb");
    if (secret_payload == NULL) {
        printf("Error: Unable to open secret payload file\n");
        return 1;
    }

    // Read the secret message from the secret payload file
    fread(secret_message, sizeof(char), MAX_SECRET_MESSAGE_LENGTH, secret_payload);
    secret_message_length = strlen(secret_message);

    // Close the secret payload file
    fclose(secret_payload);

    // Perform the Haar wavelet transform on the frame data
    lifting_scheme_997_based_wavelet_transform(frame_data, MAX_FRAME_SIZE);

    // Encode the secret message into the frame data
    encode_secret_message(frame_data, secret_message, MAX_FRAME_SIZE, secret_message_length);
     // Perform the inverse Haar wavelet transform on the frame data
    inverse_lifting_scheme_based_haar_wavelet_transform(frame_data, MAX_FRAME_SIZE);

    // Open the stego video file
    stego_video = fopen(stego_video_filename, "wb");
    if (stego_video == NULL) {
        printf("Error: Unable to open stego video file\n");
        return 1;
    }

    // Write the stego video to the stego video file
    fwrite(frame_data, sizeof(double), MAX_FRAME_SIZE, stego_video);

    // Close the stego video file
    fclose(stego_video);
} else if (argc == 3 && strcmp(argv[1], "-d") == 0) {
    // Decode mode
    stego_video_filename = argv[2];

    // Open the stego video file
    stego_video = fopen(stego_video_filename, "rb");
    if (stego_video == NULL) {
        printf("Error: Unable to open stego video file\n");
        return 1;
    }

    // Read a frame of the stego video
    fread(frame_data, sizeof(double), MAX_FRAME_SIZE, stego_video);

    // Close the stego video file
    fclose(stego_video);

    // Perform the Haar wavelet transform on the frame data
    lifting_scheme_997_based_wavelet_transform(frame_data, MAX_FRAME_SIZE);

    // Decode the secret message from the frame data
    decode_secret_message(frame_data, secret_message, secret_message_length);

    // Perform the inverse Haar wavelet transform on the frame data
    inverse_lifting_scheme_based_haar_wavelet_transform(frame_data, MAX_FRAME_SIZE);

    // Output the secret message
    printf("Secret message: %s\n", secret_message);
} else {
    // Invalid usage
    printf("Usage: video_steganography [-e cover_video secret_payload stego_video] [-d stego_video]\n");
    return 1;
}

return 0;
}

int main(int argc, char *argv[]) {

    int i, j, best_num_threads = 1;
    double best_time = -1;
    int len = 100000; // length of the signal
    double *sig = (double *) malloc(len * sizeof(double));
    // Initialize the signal
    for (i = 0; i < len; i++)
        sig[i] = (double) i;

    // Try different values of omp_set_num_threads
    for (i = 1; i <= omp_get_max_threads(); i++) {
        omp_set_num_threads(i);
        clock_t start = clock();
        #pragma omp parallel for
        for (j = 0; j < 100; j++)
            lifting_scheme_based_haar_wavelet_transform(sig, len);
        double time = ((double) (clock() - start)) / CLOCKS_PER_SEC;
        if (best_time < 0 || time < best_time) {
            best_time = time;
            best_num_threads = i;
        }
    }
omp_set_num_threads(best_num_threads);  
return run_video_steganography(argc, argv);
}