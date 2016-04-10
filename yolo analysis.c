/* How YOLO procedes:
 * After calling ./darknet yolo train cfg/yolo-tiny.cfg
 * darknet detects the invocation of yolo.
 * In the run_yolo procedure, the subroutine first reads in a bunch of 
 * labelling pictures for visulization then call train_yolo with modified arguments.
 * In the train_yolo procedure, the model name is first read in by calling basecfg(),
 * parameters initialized and then the network is parsed.
 * In the parse_network_cfg, read_cfg is first called. This function builds a list
 * containing each section's name and its set parameters(as a sublist in the section list).
 * Then a network is made by make_network. Note that make_network generates a network construct with
 * #sections - 1(excluding the [net]) layers. The int *seen is currently confusing.
 * After the parse_network_cfg is called, the network is properly initialized and parameters
 * specified in the .cfg file are read.
 *
 * load_weights reverses the procedure of save_weights, basically just saving the
 * weights in the layer into the file.
 *
 * Now the training data is loaded. It is loaded by a dedicated thread. The thread reads by first
 * reading the file that stores the paths of images. Details are commented in the source file.
 *
 * Finally, the training takes place, iterating the number of batches times and computing loss
 * while saving weights at specified batches.
 * In the train_network function, each subdivision of a batch is trained and the average
 * minibatch error is returned.
 *
 * As for the train_network_datum function that takes an subdivision of a minibatch, it 
 * feeds the data forward to the net work and propagate the error back, updating the weights
 * at the end of each subdivision, similarly to a sub-minibatch training scheme.
 * 
 */

typedef struct{
    int index;
    int class;
    float **probs;
} sortable_bbox;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA
} data_type;

typedef struct load_args{
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    float jitter;
    data *d;
    image *im;
    image *resized;
    data_type type;
} load_args;

 typedef struct {
    int h;
    int w;
    int c;
    float *data;
} image;

 typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
} data;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;

 typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
} size_params;

 typedef struct{
    char *key;
    char *val;
    int used;
} kvp;


typedef struct{
    char *type;
    list *options;
}section;

// list is the ADT of Bidirectional List that stores nodes of void *

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;


typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG
} learning_rate_policy;

typedef struct network{
    int n;
    int batch;
    int *seen;
    float epoch;
    int subdivisions;
    float momentum;
    float decay;
    layer *layers;
    int outputs;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;

    int inputs;
    int h, w, c;
    int max_crop;

    #ifdef GPU
    float **input_gpu;
    float **truth_gpu;
    #endif
} network;

typedef struct network_state {
    float *truth;
    float *input;
    float *delta;
    int train;
    int index;
    network net;
} network_state;

struct layer;
typedef struct layer layer;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    CRNN
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, SMOOTH
} COST_TYPE;

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY
}ACTIVATION;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int groups;
    int size;
    int side;
    int stride;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int steps;
    int hidden;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float class_scale;

    int dontload;
    int dontloadscales;

    float temperature;
    float probability;
    float scale;

    int *indexes;
    float *rand;
    float *cost;
    float *filters;
    char  *cfilters;
    float *filter_updates;
    float *state;

    float *binary_filters;

    float *biases;
    float *bias_updates;

    float *scales;
    float *scale_updates;

    float *weights;
    float *weight_updates;

    float *col_image;
    int   * input_layers;
    int   * input_sizes;
    float * delta;
    float * output;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    #ifdef GPU
    int *indexes_gpu;
    float * state_gpu;
    float * filters_gpu;
    float * filter_updates_gpu;

    float *binary_filters_gpu;
    float *mean_filters_gpu;

    float * spatial_mean_gpu;
    float * spatial_variance_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * spatial_mean_delta_gpu;
    float * spatial_variance_delta_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * col_image_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;

    float * output_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;
    #endif
};