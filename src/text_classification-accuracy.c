#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <locale.h>
#include <limits.h>

#include "text_classification-accuracy.h"

char data_file[MAX_STRING];
char word_file[MAX_STRING];
char subword_file[MAX_STRING];
char output_weight_file[MAX_STRING];

int hidden_size;
int n_of_label;
int n_of_vocab;

struct WORD* vocab;
int* word_hash;
int size_of_word_hash = MAX_VOCAB_SIZE;
int size_of_subword_hash = 2000000; // 2M

// Model parameters
float* word_vec;
float* subword_vec;
float* output_layer_vec;

int minn=3;
int maxn=6;
char BOW = '<';
char EOW = '>';

int utf8_charlen(unsigned char c){
    if ((c & 0x80) == 0x00) return 1;       // 0xxxxxxx
    else if ((c & 0xE0) == 0xC0) return 2;  // 110xxxxx
    else if ((c & 0xF0) == 0xE0) return 3;  // 1110xxxx
    else if ((c & 0xF8) == 0xF0) return 4;  // 11110xxx
    return 1; // invalid fallback  
}

int utf8_strlen(const char *s){
    int len = 0;
    int i = 0;
    int nbytes = strlen(s);

    while (i < nbytes){
        int char_len = utf8_charlen((unsigned char)s[i]);
        i += char_len;
        len++;
    }
    return len;
}

int max_index(const float* arr, int size){
    int max_idx = 0;
    float max_val = arr[0];
    for(int i=1; i<size; i++){
        if(arr[i] > max_val){
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main(int argc, char** argv){

    if(argc<7){
        printf("Usage: ./text-classification minn maxn data_file word_file subword_file output_weight_file\n");
        return -1;
    }
    else{
        minn = atoi(argv[1]);
        maxn = atoi(argv[2]);
        strcpy(data_file, argv[3]);
        strcpy(word_file, argv[4]);
        strcpy(subword_file, argv[5]);
        strcpy(output_weight_file, argv[6]);
    }

    // Read word file
    FILE* word_fp = fopen(word_file, "rb");
    char* buff = (char*)calloc(MAX_STRING, sizeof(char));

    fgets(buff, MAX_STRING-1, word_fp);
    if(sscanf(buff, "%d %d %d", &n_of_vocab, &hidden_size, &n_of_label) != 3){
        printf("Error: failed to read header from word file\n");
        fclose(word_fp);
        free(buff);
        return -1;
    }

    vocab = (struct WORD*)malloc(sizeof(struct WORD)*n_of_vocab);
    word_hash = (int*)malloc(sizeof(int)*size_of_word_hash);
    for(int i=0; i<size_of_word_hash; i++) word_hash[i] = -1;

    word_vec = (float*)malloc(sizeof(float)*n_of_vocab*hidden_size);
    char* curr_word = (char*)calloc(MAX_STRING-1, sizeof(char));
    int word_id=0;
    while(fgets(buff, MAX_STRING-1, word_fp)){
        char* ptr = buff;
        int n=0;
        sscanf(ptr, "%s%n", curr_word, &n); // read word
        ptr += n;

        unsigned int hash_key = getHash(curr_word, size_of_word_hash);
        while(word_hash[hash_key]!=-1){
            hash_key = (hash_key+1)%size_of_word_hash;
        }
        word_hash[hash_key] = word_id;
        strcpy(vocab[word_id].word, curr_word);

        for(int j=0; j<hidden_size; j++){
            float val;
            sscanf(ptr, "%f%n", &val, &n);
            word_vec[word_id*hidden_size+j] = val;
            ptr += n;
        }
        word_id++;
    }
    fclose(word_fp);

    // Read subword file
    FILE* subword_fp = fopen(subword_file, "rb");

    fgets(buff, MAX_STRING-1, subword_fp);
    if(sscanf(buff, "%d %d", &size_of_subword_hash, &hidden_size) != 2){
        printf("Error: failed to read header from subword file\n");
        fclose(subword_fp);
        free(buff);
        return -1;
    }

    subword_vec = (float*)malloc(sizeof(float)*size_of_subword_hash*hidden_size);
    int subword_idx=0;
    while(fgets(buff, MAX_STRING-1, word_fp)){
        char* ptr = buff;
        int n=0;

        for(int j=0; j<hidden_size; j++){
            float val;
            sscanf(ptr, "%f%n", &val, &n);
            subword_vec[subword_idx*hidden_size+j] = val;
            ptr += n;
        }
        subword_idx++;
    }
    fclose(subword_fp);

    // Read output layer weight file
    FILE* output_layer_fp = fopen(output_weight_file, "rb");
    output_layer_vec = (float*)malloc(sizeof(float)*hidden_size*n_of_label);
    for(int l=0; l<n_of_label; l++){
        fgets(buff, MAX_STRING-1, output_layer_fp);
        char* ptr = buff;
        int n=0;
        for(int h=0; h<hidden_size; h++){
            float val;
            sscanf(ptr, "%f%n", &val, &n);
            output_layer_vec[l*hidden_size+h] = val;
            ptr += n;
        }
    }
    fclose(output_layer_fp);
    free(buff);

    // Read test data file
    FILE* data_fp = fopen(data_file, "rb");
    int total_samples=0;
    int correct_samples=0;
    buff = (char*)calloc(MAX_SENTENCE_LENGTH, sizeof(char));
    
    // Predict
    while(fgets(buff, MAX_SENTENCE_LENGTH-1, data_fp)){
        char* ptr = buff;
        int n=0;
        
        // Read label of this sample
        sscanf(ptr, "%s%n", curr_word, &n);
        int curr_label = atoi(curr_word+9)-1;

        // Calculate senntence vector of sample sentence
        float sentence_vector[hidden_size];
        while(sscanf(ptr, "%s%n", curr_word, &n) == 2){
            int oov = 0;

            unsigned int hash_key = getHash(curr_word, size_of_word_hash);
            while(strcmp(vocab[word_hash[hash_key]].word, curr_word)!=0){
                hash_key = (hash_key + 1) % size_of_word_hash;
                if(word_hash[hash_key]==-1) {oov=1; break;}
            }

            if(oov==0){ // word in vocab
                for(int h=0; h<hidden_size; h++){
                    sentence_vector[h] += word_vec[word_hash[hash_key]*hidden_size+h];
                }
            }
            else if (oov==1){ // word not in vocab
                float oov_word[hidden_size];
                getWordVectorFromString(curr_word, oov_word);
                for(int h=0; h<hidden_size; h++){
                    sentence_vector[h] += oov_word[h];
                }
            }    

            ptr += n;
        }

        // Output layer
        float prediction[n_of_label];
        for(int l=0; l<n_of_label; l++){
            for(int h=0; h<hidden_size; h++){
                prediction[l] += sentence_vector[h] * output_layer_vec[l*hidden_size+h];
            }
        }

        // Final prediction
        int answer = max_index(prediction, n_of_label);

        if(answer == curr_label) correct_samples++;
        total_samples++;
    }

    fclose(data_fp);

    printf("Accuracy: %f (%d / %d)\n", (float)correct_samples/(float)total_samples, correct_samples, total_samples);

    free(curr_word);
    free(vocab);
    free(word_hash);
    free(word_vec);
    free(subword_vec);
    free(output_layer_vec);
    return 0;
}


unsigned int getHash(char* word, int max_hash_size){
    max_hash_size = (unsigned int)max_hash_size;

    unsigned int hash_key = 2166136261; // basis
    for (int i=0; i<strlen(word); i++){
        hash_key = hash_key ^ (unsigned int)((signed char)word[i]);
        hash_key *= 16777619;
    }
    hash_key = hash_key%max_hash_size;
    return hash_key;
}

void getWordVectorFromString(char* word, float* result_vec){

    char* tmp = (char*)calloc(strlen(word)+3, sizeof(char));
    tmp[0] = BOW;
    strncpy(tmp+1, word, strlen(word));
    tmp[strlen(word)+1] = EOW;

    int pos;
    int len = utf8_strlen(tmp);
    int word_bytes;
    int initial_char_len, char_len;

    unsigned int hash_key;
    char* cur_subword = (char*)malloc(sizeof(char)*MAX_STRING);

    if(len > maxn){
        hash_key = getHash(tmp, size_of_subword_hash);
        for(int h=0; h<hidden_size; h++){
            result_vec[h] += subword_vec[hash_key*hidden_size+h];
        }
    }

    for(int n=minn; n<maxn; n++){
        pos = 0;
        for(int cnt=0; cnt<=len-n; cnt++){
            int p = pos;
            word_bytes = 0;
            initial_char_len = utf8_charlen((unsigned char)word[pos]);
            for(int c=0; c<n; c++){
                char_len = utf8_charlen((unsigned char)word[p]);
                word_bytes += char_len;
                p += char_len;
            }
            strncpy(cur_subword, word+pos, word_bytes);
            hash_key = getHash(cur_subword, size_of_subword_hash);

            for(int h=0; h<hidden_size; h++){
                result_vec[h] += subword_vec[hash_key*hidden_size+h];
            }

            pos += initial_char_len;
        }
    }

    return;
}