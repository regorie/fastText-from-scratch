#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <locale.h>
#include <limits.h>

#include "fasttext_text_classification.h" // change later!

// strings for IO
char input_file[MAX_STRING];
char output_file[MAX_STRING-9];
char output_file_word[MAX_STRING]="word_";
char output_file_subword[MAX_STRING]="subword_";
char output_file_output_layer[MAX_STRING]="outlayer_";

// file saving mode
int binary;

// dataset info
int min_count=5;

// word structures
char BOW = '<';
char EOW = '>';
int minn = 3, maxn = 6;

struct WORD* vocab;
int size_of_vocab = 2048;

int* word_hash;
int size_of_subword_hash = 2000000; // 2M
int size_of_word_hash = MAX_VOCAB_SIZE;

// training info
long long file_size = 0;
int n_of_thread;

int n_of_vocab = 0;
int n_of_samples = 0;
int n_of_trained_samples = 0;

struct LABEL* label;
int size_of_label = 1;
int n_of_label = 1;

int n_of_features;

// training hyperparameters
//int window_size;
int hidden_size;
int epoch;
float starting_lr;
float lr;

// for efficiency
float* expTable;

// model parameters
float* word_vec;
float* subword_vec;
float* output_layer;

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


void* training_thread(void* id_ptr){
    long long id = (long long)id_ptr;

    int* sentence = (int*)malloc(sizeof(int)*MAX_SENTENCE_WORD);
    int sentence_len;

    float sentence_vector[hidden_size];
    int word_features[MAX_SENTENCE_WORD];
    int subword_features[MAX_SENTENCE_WORD]; // Will this be small...?
    int word_feature_idx;
    int subword_feature_idx;
    int cur_label;

    int n_of_local_trained_sample;
    int n_of_local_last_trained_sample;
    int sample_per_thread = n_of_samples/n_of_thread+1;

    float input_layer_grad[hidden_size];
    float output_layer_grad[hidden_size*n_of_label];

    float logit_grad[n_of_label];
    float logits[n_of_label];
    float probs[n_of_label];

    FILE* infp = fopen(input_file, "r");

    for(int ep=0; ep<epoch; ep++){
        clock_t start = time(NULL);

        fseek(infp, (file_size / (long long)n_of_thread) * (long long)id, SEEK_SET);
        n_of_local_trained_sample = 0;
        n_of_local_last_trained_sample = 0;

        if (id==0) { printf("\nRunning Epoch %d\n", ep+1); }
        while(1){

            // Calculate learning rate
            n_of_trained_samples += n_of_local_trained_sample - n_of_local_last_trained_sample;
            n_of_local_last_trained_sample = n_of_local_trained_sample;
            lr = starting_lr*(1-n_of_trained_samples/(float)(epoch*n_of_samples+1));

            if(lr < starting_lr*0.0001) lr = starting_lr*0.0001;
            if(id==0){
                printf("\rLearning rate: %f, Progress: %.4f ( %d / %d ), time: %ld", lr, (float)(n_of_local_trained_sample)/(float)(sample_per_thread), n_of_local_trained_sample, sample_per_thread, time(NULL)-start);
                fflush(stdout);
            }

            for(int h=0; h<hidden_size; h++){
                input_layer_grad[h] = 0.0;
            }
            for(int h=0; h<hidden_size*n_of_label; h++){
                output_layer_grad[h] = 0.0;
            }
            for(int l=0; l<n_of_label; l++){
                logit_grad[l] = 0.0;
            }

            // forward pass
            char** unknown_words = (char**)malloc(sizeof(char*)*MAX_SENTENCE_WORD);
            for(int i=0; i<MAX_SENTENCE_WORD; i++) {
                unknown_words[i] = NULL;
            }
            sentence_len = getSentenceSample(infp, &cur_label, sentence, unknown_words);
            if (sentence_len <= 0) break;

            getSentenceVector(sentence, sentence_len, unknown_words, sentence_vector, 
                word_features, &word_feature_idx, subword_features, &subword_feature_idx);

            for(int l=0; l<n_of_label; l++){
                logits[l] = 0.0;
                for(int h=0; h<hidden_size; h++){
                    logits[l] += sentence_vector[h] * output_layer[l*hidden_size+h];
                }
            }

            // build label one hot vector
            float label_one_hot[n_of_label];
            for(int l=0; l<n_of_label; l++){
                label_one_hot[l]=0.0;
            }
            label_one_hot[cur_label] = 1.0;

            // softmax
            float max_logit = logits[0];
            for (int l = 1; l < n_of_label; l++) if (logits[l] > max_logit) max_logit = logits[l];
            float sum_exp = 0.0;
            for (int l = 0; l < n_of_label; l++) {
                logits[l] = exp(logits[l] - max_logit);
                sum_exp += logits[l];
            }
            for (int l = 0; l < n_of_label; l++) {
                probs[l] = logits[l] / sum_exp;
            }

            // calculate gradient
            for(int l=0; l<n_of_label; l++){
                logit_grad[l] = probs[l] - label_one_hot[l];
            }

            // calculate output_layer gradient
            for(int l=0; l<n_of_label; l++){
                for(int h=0; h<hidden_size; h++){
                    output_layer_grad[l*hidden_size+h] += logit_grad[l] * sentence_vector[h];
                }
            }
            // calculate input_layer gradient
            for(int l=0; l<n_of_label; l++){
                for(int h=0; h<hidden_size; h++){
                    input_layer_grad[h] += logit_grad[l] * output_layer[l*hidden_size+h];
                }
            }
            for(int h=0; h<hidden_size; h++){
                input_layer_grad[h] /= n_of_features;
            }
            
            //update in_layers
            for(int i=0; i<word_feature_idx; i++){
                for(int h=0; h<hidden_size; h++){
                    word_vec[word_features[i]*hidden_size+h] -= lr*input_layer_grad[h];
                }
            }
            for(int i=0; i<subword_feature_idx; i++){
                for(int h=0; h<hidden_size; h++){
                    subword_vec[subword_features[i]*hidden_size+h] -= lr*input_layer_grad[h];
                }
            }
            //update output layer
            for(int l=0; l<n_of_label; l++){
                for(int h=0; h<hidden_size; h++){
                    output_layer[l*hidden_size+h] -= lr*output_layer_grad[l*hidden_size+h];
                }
            }
            n_of_local_trained_sample++;

            for(int i=0; i<MAX_SENTENCE_WORD; i++) {
                if(unknown_words[i] != NULL) free(unknown_words[i]);
            }
            free(unknown_words);
        }
    }
    fclose(infp);
    free(sentence);
}

int main(int argc, char** argv){
    if(argc < 6){
        printf("Usage example: ./fasttext-text hidden_size thread_number epoch data_file output_file\n");
        return -1;
    }
    else{
        hidden_size = atoi(argv[1]);
        n_of_thread = atoi(argv[2]);
        epoch = atoi(argv[3]);
        strcpy(input_file, argv[4]);
        strcpy(output_file, argv[5]);
    }
    starting_lr = 0.05;
    lr=starting_lr;
    printf("Starting learning rate : %f\n", starting_lr);

    // 1. Preperation
    word_hash = (int*)calloc(size_of_word_hash, sizeof(int));
    vocab = (struct WORD*)calloc(size_of_vocab, sizeof(struct WORD));
    label = (struct LABEL*)calloc(size_of_label, sizeof(struct LABEL));

    readWordsFromFile(input_file);
    reduceWords();
    calculateSubwordIDs();

    expTable = (float*)malloc((EXP_TABLE_SIZE+1)*sizeof(float));
    for(int i=0; i<EXP_TABLE_SIZE; i++){
        expTable[i] = exp((i/(float)EXP_TABLE_SIZE*2-1)*MAX_EXP);
        expTable[i] = expTable[i] / (expTable[i] + 1);
    }
    
    // Initialize model
    subword_vec = (float*)malloc(sizeof(float)*(hidden_size*size_of_subword_hash));
    long long random_number = time(NULL);
    for(int i=0; i<size_of_subword_hash; i++){
        for(int h=0; h<hidden_size; h++){
            random_number = random_number * (unsigned long long)25214903917 + 11;
            subword_vec[i*hidden_size + h] = (((random_number & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
        }
    }
    word_vec = (float*)malloc(sizeof(float)*hidden_size*n_of_vocab);
    for(int i=0; i<n_of_vocab; i++){
        for(int h=0; h<hidden_size; h++){
            random_number = random_number * (unsigned long long)25214903917 + 11;
            word_vec[i*hidden_size + h] = (((random_number & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
        }
    }
    output_layer = (float*)malloc(sizeof(float)*hidden_size*n_of_label);
    for(int i=0; i<n_of_label; i++){
        for(int h=0; h<hidden_size; h++){
            random_number = random_number * (unsigned long long)25214903917 + 11;
            output_layer[i*hidden_size+h] = (((random_number & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
        }
    }

    // Train
    printf("Training... ");
    time_t start_time = time(NULL);
    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t)*n_of_thread);

    int* id = (int*)malloc(sizeof(int)*n_of_thread);
    for(int a=0; a<n_of_thread; a++){
        id[a] = a;
        pthread_create(&threads[a], NULL, training_thread, (void*)(long)a);
    }
    printf("all threads created\n");
    for(int a=0; a<n_of_thread; a++){
        pthread_join(threads[a], NULL);
    }
    time_t end_time = time(NULL);
    printf("\n Training done... took %ld, last learning rate: %f\n", end_time-start_time, lr);
    free(threads);

    // 3. Save vectors
    strcat(output_file_subword, output_file);
    printf("output file: %s\n", output_file_subword);
    FILE* outfp = fopen(output_file_subword, "wb");
    if(outfp == NULL) {printf("subword file open error\n"); exit(1); }
    else{
        fprintf(outfp, "%d %d\n", size_of_subword_hash, hidden_size);
        for(int i=0; i<size_of_subword_hash; i++){
            if(binary){
                for(int h=0; h<hidden_size; h++){
                    fwrite(&subword_vec[i*hidden_size+h], sizeof(float), 1, outfp);
                }
            }
            else{
                for(int h=0; h<hidden_size; h++){
                    fprintf(outfp, "%lf ", subword_vec[i*hidden_size+h]);
                }
            }
            fprintf(outfp, "\n");
        }
    }
    fclose(outfp);

    int* tmp=(int*)malloc(sizeof(int)*1000);
    int tmp2 = 0;

    strcat(output_file_word, output_file);
    printf("output file: %s\n", output_file_word);
    outfp = fopen(output_file_word, "wb");
    if(outfp == NULL) {printf("word file open error\n"); exit(1);}
    else{
        fprintf(outfp, "%d %d %d\n", n_of_vocab, hidden_size, n_of_label);

        float target_vector[hidden_size];
        for(int i=0; i<n_of_vocab; i++){
            tmp2=0;
            getWordVector(i, target_vector, tmp, &tmp2);

            fprintf(outfp, "%s ", vocab[i].word);
            if(binary) {
                for(int h=0; h<hidden_size; h++){
                    fwrite(&target_vector[h], sizeof(float), 1, outfp);
                }
            }
            else{
                for(int h=0; h<hidden_size; h++){
                    fprintf(outfp, "%lf ", target_vector[h]);
                }
            }
            fprintf(outfp, "\n");
        }
    }
    fclose(outfp);

    strcat(output_file_output_layer, output_file);
    printf("output file: %s\n", output_file_output_layer);
    outfp = fopen(output_file_output_layer, "wb");
    if(outfp == NULL) {printf("output layer file open error\n"); exit(1);}
    else{
        for(int i=0; i<n_of_label; i++){
            if(binary){
                for(int h=0; h<hidden_size; h++){
                    fwrite(&output_layer[i*hidden_size+h], sizeof(float), 1, outfp);
                }
            }
            else{
                for(int h=0; h<hidden_size; h++){
                    fprintf(outfp, "%lf ", output_layer[i*hidden_size+h]);
                }
            }
            fprintf(outfp, "\n");
        }
    }
    fclose(outfp);

    // 4. Free everything
    free(id);
    free(word_vec);
    free(subword_vec);
    free(expTable);
    for(int i=0; i<n_of_vocab; i++){
        free(vocab[i].subword_ids);
    }
    free(vocab);
    free(word_hash);
    free(tmp);
    free(label);

    printf("Done\n");
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


void readWordsFromFile(char* file_name){
    printf("Reading words from file... ");
    resetHashTable();

    FILE* fp = fopen(file_name, "r");
    if(fp==NULL) {printf("Data file not found\n"); exit(1);}
    printf("File name: %s\n", file_name);

    char ch;
    char* cur_word = (char*)calloc(MAX_STRING, sizeof(char));
    
    int word_length = 0;
    unsigned int hash_key;

    while((ch = fgetc(fp)) != EOF){
        if(ch==13) continue;
        if(ch==' ' || ch=='\n' || ch=='\t' || ch=='\v' || ch=='\f' || ch=='\r' || ch=='\0'){
            if(word_length==0) continue;
            if(strncmp(cur_word, "__label__", 9) == 0) {
                // This is label!
                cur_word[word_length] = 0;
                word_length = 0;
                n_of_samples++;

                char* tmp = cur_word+9;
                int label_num = atoi(tmp);

                if(label_num>n_of_label){ // new label added
                    
                    size_of_label = label_num;
                    label = realloc(label, size_of_label*sizeof(struct LABEL));
                    if(label==NULL){ printf("Label reallocation failed\n"); exit(1);}
                
                    for(int k=n_of_label; k<label_num; k++){
                        label[k].count = 0;
                    }
                    label[label_num-1].count=1;

                    n_of_label=label_num;
                }
                else {
                    label[label_num-1].count++;
                }

                continue;
            }

            //total_word_count++;
            cur_word[word_length] = 0;
            word_length = 0;
            hash_key = getHash(cur_word, size_of_word_hash);

            // reduce the size of vocab if it reaches the limit
            if(n_of_vocab >= size_of_word_hash){
                for(int idx=0; idx<n_of_vocab; idx++){
                    if(vocab[idx].count < min_count){
                        vocab[idx].count = 0;
                        memset(&vocab[idx].word, 0, sizeof(vocab[idx].word));
                        word_hash[idx] = -1;
                        n_of_vocab--;
                    }
                }
            }

            while(1){
                if(word_hash[hash_key]==-1){
                    if(n_of_vocab >= size_of_vocab){
                        // allocate more space if necessary
                        size_of_vocab += 2048;
                        vocab = realloc(vocab, size_of_vocab*sizeof(struct WORD));
                        if(vocab==NULL){ printf("Vocab reallocation failed\n"); exit(1); }
                    }
                    word_hash[hash_key] = n_of_vocab;
                    vocab[n_of_vocab].count = 1;
                    strcpy(vocab[n_of_vocab].word, cur_word);
                    n_of_vocab++;
                    break;
                }
                if(strcmp(vocab[word_hash[hash_key]].word, cur_word)==0){
                    vocab[word_hash[hash_key]].count++;
                    break;
                }
                hash_key = (hash_key+1)%size_of_word_hash;
            }
        }
        else {
            cur_word[word_length++] = ch;
            if(word_length>=MAX_STRING-3) word_length--;
        }
    }

    free(cur_word);
    file_size = ftell(fp);
    fclose(fp);
    printf("Done\n");
    printf("Samples: %d\n", n_of_samples);
}

int _comp(const void* a, const void* b){
    return ((struct WORD*)b)->count - ((struct WORD*)a)->count;
}

void reduceWords(){
    printf("Reducing words... ");

    // 1. Sort vocab by count
    qsort(vocab, n_of_vocab, sizeof(struct WORD), _comp);

    // 2. Allocate spaces for words, discard those that appear too less
    resetHashTable();
    unsigned int hash_key;
    for(int i=0; i<n_of_vocab; i++){
        if(vocab[i].count < min_count || i>= MAX_VOCAB_SIZE){
            n_of_vocab = i;
            break;
        }

        if(utf8_strlen(vocab[i].word)+2 > maxn) vocab[i].n_of_subwords=1;
        else vocab[i].n_of_subwords = 0;
        for(int n=minn; n<=maxn; n++){
            if(utf8_strlen(vocab[i].word)+2 < n) break;
            vocab[i].n_of_subwords += utf8_strlen(vocab[i].word)+3-n;
        }
        vocab[i].subword_ids = (unsigned int*)malloc(sizeof(unsigned int)*vocab[i].n_of_subwords);
        vocab[i].subwords = (char**)malloc(sizeof(char*) * vocab[i].n_of_subwords);
        for(int k=0; k<vocab[i].n_of_subwords; k++){
            vocab[i].subwords[k] = (char*)calloc((utf8_strlen(vocab[i].word)+2)*4+1, sizeof(char));
        }

        hash_key = getHash(vocab[i].word, size_of_word_hash);
        while(word_hash[hash_key]!=-1){
            hash_key = (hash_key+1)%size_of_word_hash;
        }
        word_hash[hash_key] = i;
    }

    printf("Done\n");
    printf("number of vocab: %d\n", n_of_vocab);
}

void calculateSubwordIDs(){
    printf("Assigning subword IDs to each vocab... ");
    
    unsigned int hash_key;
    int current_id = 0;

    char* cur_word = (char*)calloc(MAX_STRING, sizeof(char));
    cur_word[0] = BOW;
    for(int i=0; i<n_of_vocab; i++){
        memcpy(cur_word+1, vocab[i].word, strlen(vocab[i].word));
        cur_word[strlen(vocab[i].word)+1] = EOW;
        cur_word[strlen(vocab[i].word)+2] = '\0';

        calculateSubwords(cur_word, i);

        for(int j=0; j<vocab[i].n_of_subwords; j++){
            hash_key = getHash(vocab[i].subwords[j], size_of_subword_hash);
            vocab[i].subword_ids[j] = hash_key;
            free(vocab[i].subwords[j]);
        }
        free(vocab[i].subwords);
    }

    free(cur_word);
    printf("Done\n");
}

void calculateSubwords(char* word, int vocab_id){
    int idx = 0;
    int pos;
    int len = utf8_strlen(word);
    char current_subword[MAX_STRING];

    int char_len;
    int initial_char_len;
    int word_bytes;

    if(len > maxn){
        strncpy(vocab[vocab_id].subwords[idx], word, len); // seg fault happens here
        idx++;
    }

    for(int n=minn; n<=maxn; n++){
        pos = 0;
        for(int cnt=0; cnt<=len-n; cnt++){
            int p=pos; // p is the currently estimated 'character' position
            word_bytes=0;
            initial_char_len=utf8_charlen((unsigned char)word[pos]);
            for(int c=0; c<n; c++){
                char_len = utf8_charlen((unsigned char)word[p]);
                word_bytes += char_len;
                p += char_len;
            }
            strncpy(vocab[vocab_id].subwords[idx], word+pos, word_bytes);
            vocab[vocab_id].subwords[idx][word_bytes] = '\0';
            idx++;

            pos += initial_char_len; // move to next character
        }
    }
    return;
}

void calculateSubwordsToBuff(char* word, char** subwords){
    // subwords needs to be allocated prior
    int idx = 0;
    int pos;
    int len = utf8_strlen(word);
    char current_subword[MAX_STRING];

    int char_len;
    int initial_char_len;
    int word_bytes;

    if(len > maxn){
        strncpy(subwords[idx], word, len);
        idx++;
    }

    for(int n=minn; n<=maxn; n++){
        pos = 0;
        for(int cnt=0; cnt<=len-n; cnt++){
            int p=pos; // p is the currently estimated 'character' position
            word_bytes=0;
            initial_char_len=utf8_charlen((unsigned char)word[pos]);
            for(int c=0; c<n; c++){
                char_len = utf8_charlen((unsigned char)word[p]);
                word_bytes += char_len;
                p += char_len;
            }
            strncpy(subwords[idx], word+pos, word_bytes);
            subwords[idx][word_bytes] = '\0';
            idx++;

            pos += initial_char_len; // move to next character
        }
    }
}


int getWordVector(int id, float* result_vec, int* subword_features, int* subword_idx){
    if(id >= n_of_vocab){
        printf("ID out of bound\n");
        exit(1);
    }

    for(int h=0; h<hidden_size; h++){
        result_vec[h] = 0.0;
    }

    for(int i=0; i< vocab[id].n_of_subwords; i++){
        subword_features[*subword_idx] = vocab[id].subword_ids[i];
        *subword_idx += 1;
        for(int h=0; h<hidden_size; h++){
            result_vec[h] += subword_vec[hidden_size*vocab[id].subword_ids[i] + h];
        }
    }
    for(int h=0; h<hidden_size; h++){
        result_vec[h] += word_vec[id*hidden_size + h];
    }

    return vocab[id].n_of_subwords+1;
}

int getWordVectorFromString(char* word, float* result_vec, int* subword_features, int* subword_idx){
    int _n_of_subwords = 0;
    if(utf8_strlen(word)+2 > maxn) _n_of_subwords=1;
    for (int n=minn; n<=maxn; n++){
        if(utf8_strlen(word)+2 < n) break;
        _n_of_subwords += utf8_strlen(word)+3-n;
    }

    unsigned int* subwords_id = (unsigned int*)calloc(_n_of_subwords, sizeof(unsigned int));
    char** subwords = (char**)malloc(sizeof(char*)*_n_of_subwords);
    char* tmp = (char*)calloc((utf8_strlen(word)+2)*4+1, sizeof(char));
    tmp[0] = BOW;
    strncpy(tmp+1, word, strlen(word));
    tmp[strlen(word)+1] = EOW;

    for(int i=0; i<_n_of_subwords; i++){
        subwords[i] = (char*)calloc((utf8_strlen(word)+2)*4+1, sizeof(char));
    }

    calculateSubwordsToBuff(tmp, subwords);

    for(int i=0; i<_n_of_subwords; i++){
        subwords_id[i] = getHash(subwords[i], size_of_subword_hash);
    }

    for(int h=0; h<hidden_size; h++){
        result_vec[h] = 0.0;
    }
    for(int i=0; i<_n_of_subwords; i++){
        subword_features[*subword_idx] = subwords_id[i];
        *subword_idx += 1;
        for(int h=0; h<hidden_size; h++){
            result_vec[h] += subword_vec[hidden_size*subwords_id[i] + h];
        }
    }

    free(subwords_id);
    for(int i=0; i<_n_of_subwords; i++){
        free(subwords[i]);
    }
    free(subwords);
    free(tmp);

    return _n_of_subwords;
}

void getSentenceVector(int* sentence, int sentence_len, char** unknown_words, float* sent_vec, int* word_features, int* word_idx, int* subword_features, int* subword_idx){
    float buf_vec[hidden_size];
    *subword_idx=0;
    *word_idx=0;
    n_of_features = 0;

    // reset sentence vector first
    for(int h=0; h<hidden_size; h++){
        sent_vec[h] = 0.0;
    }

    for(int i=0; i<sentence_len; i++){
        // if the word is not in vocab
        if(sentence[i] == -1){
            n_of_features += getWordVectorFromString(unknown_words[i], buf_vec, subword_features, subword_idx);
            for(int h=0; h<hidden_size; h++){
                sent_vec[h] += buf_vec[h];
            }
        }
        else{ // word is in vocab
            word_features[*word_idx] = sentence[i];
            *word_idx += 1;
            n_of_features += getWordVector(sentence[i], buf_vec, subword_features, subword_idx);
            for(int h=0; h<hidden_size; h++){
                sent_vec[h] += buf_vec[h];
            }
        }
    }

    for(int h=0; h<hidden_size; h++){
        sent_vec[h] *= (1/(float)n_of_features);
    }

    for(int i=0; i<sentence_len; i++){
        if(unknown_words[i] != NULL){
            free(unknown_words[i]);
            unknown_words[i] = NULL;
        }
    }
    return;
}

int wordToID(char* word){
    unsigned int hash_key = getHash(word, size_of_word_hash);

    if(word_hash[hash_key]==-1) return -1;
    while(strcmp(vocab[word_hash[hash_key]].word, word)!=0){
        hash_key = (hash_key+1)%(unsigned int)size_of_word_hash;
        if(word_hash[hash_key]==-1) return -1;
    }
    return word_hash[hash_key];
}

void resetHashTable(){
    for(int i=0; i<size_of_word_hash; i++){
        word_hash[i] = -1;
    }
    return;
}

// Reads one line of training data
int getSentenceSample(FILE* fp, int* _label, int* sentence, char** unknown_words){
    char* buff = (char*)malloc(sizeof(char)*MAX_SENTENCE_LENGTH);
    char ch;
    char cur_word[MAX_STRING];
    int word_length=0;
    int sentence_length=0;
    int id_found;
    
    char* result = fgets(buff, MAX_SENTENCE_LENGTH-1, fp);
    if(result==NULL) return -1;

    if(strlen(buff)<=1) return -1;

    for(int pos=0; pos<strlen(buff); pos++){
        ch = buff[pos];
        if(ch==' ' || ch=='\n' || ch=='\t' || ch=='\v' || ch=='\f' || ch=='\r' || ch=='\0'){
            if(word_length==0) continue;
            cur_word[word_length] = 0;
            word_length = 0;

            if(strncmp(cur_word, "__label__", 9)==0){
                // label
                char* tmp = cur_word+9;
                *_label = atoi(tmp)-1;

                continue;
            }

            id_found = wordToID(cur_word);
            if(id_found==-1){
                if(unknown_words[sentence_length] != NULL){
                    free(unknown_words[sentence_length]);
                    unknown_words[sentence_length] = NULL;
                }
                unknown_words[sentence_length] = (char*)calloc(MAX_STRING, sizeof(char));
                strcpy((unknown_words[sentence_length]), cur_word);
            }
            sentence[sentence_length++] = id_found;
            if(sentence_length >= MAX_SENTENCE_WORD-1) {
                int c;
                while((c=fgetc(fp))!= '\n' && c != EOF){continue;}
                free(buff);
                return sentence_length;
            }
            if(ch=='\n') {
                free(buff);
                return sentence_length;
            }
        }
        else {
            if(word_length >= MAX_STRING-3) word_length--;
            cur_word[word_length++] = ch;
        }
    }
    free(buff);
    return sentence_length;
}
