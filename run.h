#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include <stdint.h>

// Transformer model configuration structure
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

// Transformer model weights structure
typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
} TransformerWeights;

// Run state structure
typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float* key_cache;
    float* value_cache;
} RunState;

// Main transformer structure
typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

// Tokenizer structure
typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

// Sampler structure
typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

// Function declarations
void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);
void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights);
void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights, int* fd, float** data, ssize_t* file_size);
void build_transformer(Transformer *t, char* checkpoint_path);
void free_transformer(Transformer* t);

void rmsnorm(float* o, float* x, float* weight, int size);
void softmax(float* x, int size);
void matmul(float* xout, float* x, float* w, int n, int d);

float* forward(Transformer* transformer, int token, int pos);
int compare_tokens(const void *a, const void *b);


void build_tokenizer(char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
char* decode(Tokenizer* t, int prev_token, int token);
void safe_printf(char *piece);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);

int sample_argmax(float* probabilities, int n);
int sample_mult(float* probabilities, int n, float coin);
int compare(const void* a, const void* b);
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin);


void build_sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);

unsigned int random_u32(unsigned long long *state);
float random_f32(unsigned long long *state);
int sample(Sampler* sampler, float* logits);
long time_in_ms();


void generate(Transformer *transformer, char *prompt, int steps);
void read_stdin(const char* guide, char* buffer, size_t bufsize);

void chat(Transformer *transformer, char *cli_user_prompt, char *cli_system_prompt, int steps);

void error_usage();
void goLLM();

#endif // TRANSFORMER_MODEL_H