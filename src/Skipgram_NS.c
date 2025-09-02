#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <limits.h>

#include <ctype.h>

#include "word2vec_NS.h"

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

char input_file[MAX_STRING];
char output_file[MAX_STRING];
int binary = 1;
long long file_size = 0;

int n_of_thread;

// structures, vars for training
int* hash;
float* nodes;
int size_of_hash = MAX_VOCAB_SIZE;
struct WORD* vocab;
int size_of_vocab = 2048;
int n_of_words = 0;
int n_of_words_limit;
long long total_words = 0;
long long trained_word = 0;
int window_size = 5;
int min_count = 5;
int epoch;

float* expTable;

// negative sampling
int ns_sample=5;
int* unigram_table;
int unigram_table_size=1e8;

float starting_lr;
float lr;

// sub-sampling
float sample = 1e-4;
long long *skip_cnt;
long long total_skip_cnt = 0;

// model var
int hidden_size;
float* in_layer;
float* out_layer;

void* training_thread(void* id_ptr){
    long long id = (long long)id_ptr;
    
    FILE* infp = fopen(input_file, "r");
    long long* sentence = (long long*)malloc(sizeof(long long)*MAX_SENTENCE_LENGTH);
    long long target, target_pos;
    long long context, context_pos;
    long long sentence_length;

    long long random_window;
    unsigned long long next_random = (long long)id;

    long long local_trained_word = 0;
    long long local_last_trained_word = 0;
    
    float* layer_grad = (float*)calloc(hidden_size, sizeof(float));
    
    long long word_per_thread = total_words / n_of_thread;
    long long local_skipped_total = 0;

    lr = starting_lr;
    for(int ep=0; ep<epoch; ep++){
        clock_t start = time(NULL);

        fseek(infp, (file_size / (long long)n_of_thread) * (long long)id, SEEK_SET);
        local_trained_word = 0;
        local_last_trained_word = 0;

        if(id==0) printf("\nRunning epoch %d\n", ep);
        while(1){
            
            sentence_length = readSentenceFromFile(infp, sentence, id, ep+1);
            if(sentence_length < 0) break;
            local_trained_word += skip_cnt[id];
            local_skipped_total += skip_cnt[id];

            for(target_pos=0; target_pos<sentence_length; target_pos++){
                // traverse current sentence -> target
                // 0. calculate current learning rate

                if(local_trained_word - local_last_trained_word > 100){
                    trained_word += local_trained_word - local_last_trained_word;
                    local_last_trained_word = local_trained_word;
                    lr = starting_lr*(1-(float)trained_word / (float)(epoch*total_words+1));
                    if(lr<starting_lr*0.0001) lr = starting_lr*0.0001;
                    if(id==0){
                        printf("\rLearning rate: %f, Progress: %.4f, current skipped words: %lld, time: %ld", lr, (float)(local_trained_word)/(float)(total_words/n_of_thread), local_skipped_total, time(NULL)-start);
                        fflush(stdout);
                    }
                }
                // 1. Set target
                target = sentence[target_pos];
                if(target==-1) continue;

                // 2. forward pass
                // reset gradient
                for (int b=0; b<hidden_size; b++){
                    layer_grad[b] = 0.0;
                }
                next_random = next_random * (unsigned long long)25214903917 + 11;
                random_window = next_random % window_size;
                for(context_pos=target_pos-random_window; context_pos<=target_pos+random_window; context_pos++){
                    if(context_pos<0) continue;
                    if(context_pos>=sentence_length) break;
                    
                    if(context_pos != target_pos){
                        float g, f;
                        int current_sample, label;

                        context = sentence[context_pos];
                        if(context==-1) continue;

                        for(int d=0; d<ns_sample+1; d++){
                            // pick sample
                            if(d==0){
                                current_sample = context;
                                label=1;
                            }
                            else{
                                next_random = next_random*(unsigned long long)25214903917 + 11;
                                current_sample = unigram_table[(next_random >> 16) % unigram_table_size];
                                if(current_sample==context){
                                    continue;
                                }
                                label=0;
                            }

                            f = 0.0;
                            for(int c=0; c<hidden_size; c++){
                                f += in_layer[target*hidden_size + c] * out_layer[current_sample*hidden_size + c];
                            }
                            // sigmoid
                            if (f > MAX_EXP) g = (label-1)*lr;
                            else if (f < -MAX_EXP) g = (label-0)*lr;
                            else g = (label - expTable[(int)((f + MAX_EXP)*(EXP_TABLE_SIZE/MAX_EXP/2))]) * lr;
                        
                            for(int c=0; c<hidden_size; c++) layer_grad[c] += g*out_layer[current_sample*hidden_size + c];
                            for(int c=0; c<hidden_size; c++) out_layer[current_sample*hidden_size + c] += g*in_layer[target*hidden_size + c];
                        }
                    }
                }
                for (int b=0; b<hidden_size; b++){ // updating in_layer
                    in_layer[target*hidden_size+b] += layer_grad[b];
                }

                local_trained_word++;
            }

            if(local_trained_word > word_per_thread){
                trained_word += local_trained_word - local_last_trained_word;
                lr = starting_lr*(1-trained_word/(float)(epoch*total_words+1));
                if(lr < starting_lr*0.0001) lr = starting_lr*0.0001;
                if(id==0){
                    printf("\rLearning rate: %f, Progress: %.4f, current skipped words: %lld, time: %ld", lr, (float)(local_trained_word)/(float)(total_words/n_of_thread), local_skipped_total, time(NULL)-start);
                    fflush(stdout);
                }
                break;
            }
        }
    }

    free(layer_grad);
    free(sentence);
    fclose(infp);

    return NULL;
}

int main(int argc, char** argv){
    if(argc < 11){
        printf("Usage example: ./skipgramns hidden_size window_size n_of_words_limit ns_sample sampling_param min_count thread_number epoch data_file output_file\n");
        return -1;
    }
    else{
        hidden_size = atoi(argv[1]);
        window_size = atoi(argv[2]);
        n_of_words_limit = atoi(argv[3]);
        ns_sample = atof(argv[4]);
        sample = atof(argv[5]);
        min_count = atoi(argv[6]);
        n_of_thread = atoi(argv[7]);
        epoch = atoi(argv[8]);
        strcpy(input_file, argv[9]);
        strcpy(output_file, argv[10]);
    }
    starting_lr = 0.025;
    printf("Starting learning rate : %f\n", starting_lr);
    printf("Sampling param: %f\n", sample);
    printf("Negative sampling number: %d\n", ns_sample);

    // prepare for training
    hash = (int*)calloc(size_of_hash, sizeof(int));
    vocab = (struct WORD*)calloc(size_of_vocab, sizeof(struct WORD));

    buildHash(input_file);
    reduceHash();

    initUnigramTable();

    expTable = (float*)malloc((EXP_TABLE_SIZE + 1)*sizeof(float));
    for(int i=0; i<EXP_TABLE_SIZE; i++){
        expTable[i] = exp((i/(float)EXP_TABLE_SIZE*2 - 1)*MAX_EXP);
        expTable[i] = expTable[i] / (expTable[i] + 1);
    }

    // initialize model
    in_layer = (float*)malloc(sizeof(float)*hidden_size*n_of_words);
    out_layer = (float*)malloc(sizeof(float)*hidden_size*n_of_words);
    long long random_number = time(NULL);
    long long random_number2 = time(NULL)*11;
    for(int a=0; a<n_of_words; a++){
        for(int b=0; b<hidden_size; b++){
            random_number = random_number * (unsigned long long)25214903917 + 11;
            random_number2 = random_number2 * (unsigned long long)25214903917 + 11;
            in_layer[a*hidden_size + b] = (((random_number & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
            out_layer[a*hidden_size + b] = (((random_number2 & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
            //out_layer[a*hidden_size + b] = 0.0;
        }
    } 

    // train
    printf("Training...");
    time_t start_time = time(NULL);
    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t)*n_of_thread);
    
    int* id = (int*)malloc(sizeof(int)*n_of_thread);
    skip_cnt = (long long*)malloc(sizeof(long long)*n_of_thread);
    for(int a=0; a<n_of_thread; a++){
        id[a] = a;
        pthread_create(&threads[a], NULL, training_thread, (void*)(long)a);
    }
    printf("all threads created\n");
    for(int a=0; a<n_of_thread; a++){
        pthread_join(threads[a], NULL);
    }
    time_t end_time = time(NULL);
    printf("\nTraining done... took %ld, last learning rate: %f trained_words: %lld skipped_words: %lld\n", end_time-start_time, lr, trained_word, total_skip_cnt);
    
    // save word vectors
    FILE* outfp = fopen(output_file, "wb");
    long long nonalphabet_cnt=0;
    fprintf(outfp, "%lld %lld\n", (long long)n_of_words, (long long)hidden_size);
    for(int a=0; a<n_of_words; a++){
        fprintf(outfp, "%s ", vocab[a].word);
        
        if(binary) {
            for(int k=0;k<strlen(vocab[a].word);k++){
                if(!isalpha(vocab[a].word[k])){ nonalphabet_cnt++;}
            }
            
            for(int b=0; b<hidden_size; b++){
                fwrite(&in_layer[a*hidden_size + b], sizeof(float), 1, outfp);
            }
        }
        else{
            for(int b=0; b<hidden_size; b++){
                fprintf(outfp, "%lf ", in_layer[a*hidden_size+b]);
            }
        }
        fprintf(outfp, "\n");
    }
    fclose(outfp);
    // free everything
    
    free(id);
    free(hash);
    free(vocab);
    free(skip_cnt);
    free(threads);

    free(in_layer);
    free(out_layer);

    printf("non-alphabet characters: %lld\n", nonalphabet_cnt);
    fflush(stdout);
    return 0;
}

void resetHashTable(){
    for(int i=0; i<size_of_hash; i++){
        hash[i] = -1;
    }
    return;
}

int getWordHash(char* word){
    int hash_key = 0;
    for(int i=0; i<strlen(word); i++){
        hash_key += (hash_key << 5)*i + word[i];
    }
    hash_key = hash_key % MAX_VOCAB_SIZE;
    return abs(hash_key);
}

void buildHash(char* file_name){
    printf("building hash table...\n");
    resetHashTable();

    FILE* infp = fopen(file_name, "r");
    printf("file_name %s\n", file_name);

    if(infp==NULL){ printf("file not found\n"); exit(1);}

    char ch;
    char* cur_word = (char*)calloc(MAX_STRING, sizeof(char));
    int word_length = 0;
    int hash_key;

    while((ch = fgetc(infp)) != EOF){
        if(ch==13) continue;
        if(ch == ' ' || ch == '\n' || ch == '\t' || ch == '\0'){
            if (word_length==0) continue;

            total_words++;

            cur_word[word_length] = 0;
            word_length = 0;
            hash_key = getWordHash(cur_word);

            while(1){
                if(hash[hash_key]==-1){
                    if(n_of_words >= size_of_vocab){
                        size_of_vocab += 2048;
                        vocab = realloc(vocab,size_of_vocab*sizeof(struct WORD));
                        if(vocab==NULL){
                            printf("Reallocation failed\n");
                            exit(1);
                        }
                    }
                    hash[hash_key] = n_of_words;
                    vocab[n_of_words].count = 1;
                    strcpy(vocab[n_of_words].word, cur_word);
                    n_of_words++;
                    break;
                }
                if(strcmp(vocab[hash[hash_key]].word, cur_word)==0){
                    vocab[hash[hash_key]].count++;
                    break;
                }
                hash_key = (hash_key + 1) % MAX_VOCAB_SIZE;
            }
        }
        else{
            cur_word[word_length++] = ch;
            if(word_length >= MAX_STRING) word_length--;
        }
    }

    free(cur_word);
    file_size = ftell(infp);
    fclose(infp);
    printf("done... n_of_words = %d total_words = %lld\n", n_of_words, total_words);
    return;
}

int _comp(const void* a, const void* b){
    return ((struct WORD*)b)->count - ((struct WORD*)a)->count;
}

void reduceHash(){
    printf("Reducing hash table...\n");
    // 1. Sort vocab by count
    qsort(vocab, n_of_words, sizeof(struct WORD), _comp); // descending order

    // 2. Discard too less frequently appeared words
    // 3. Allocate space for codes
    // 4. recompute hash
    resetHashTable();
    total_words = 0;
    int hash_key;
    for(int i=0; i<n_of_words; i++){
        if (vocab[i].count < min_count || i >= MAX_VOCAB_SIZE) {
            n_of_words = i;
            break;
        }
        vocab[i].code = (char*)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[i].point = (int*)calloc(MAX_CODE_LENGTH, sizeof(int));

        hash_key = getWordHash(vocab[i].word);
        while(hash[hash_key]!=-1){
            hash_key = (hash_key + 1) % MAX_VOCAB_SIZE;
        }
        hash[hash_key] = i;

        total_words += vocab[i].count;
    }
    
    printf("n_of_words after excluding rare words %d\n", n_of_words);
    printf("total words after excluding rare words %lld\n", total_words);
}

void initModel(){

}

int readSentenceFromFile(FILE* fp, long long* sentence, long long thread_id, int iter){
    char ch;
    char cur_word[MAX_STRING] = {0};
    int word_length = 0;
    int sentence_length = 0;
    int id_found;
    unsigned long long next_random = thread_id;
    next_random += (unsigned long long)iter*17;

    skip_cnt[thread_id] = 0;
    while(!feof(fp)){
        ch = fgetc(fp);
        if(ch==' ' || ch=='\t' || ch=='\n'){
            
            if(word_length==0) continue;
            cur_word[word_length] = 0;
            word_length = 0;

            id_found = searchVocabID(cur_word);
            if(id_found != -1){
                if (sample > 0){
                    float ran = (sqrt(vocab[id_found].count / (sample * total_words)) + 1) * (sample * total_words) / vocab[id_found].count;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if(ran < (next_random & 0xFFFF) / (float)65536) {
                        skip_cnt[thread_id]++;
                        total_skip_cnt++;
                        continue;
                    }
                }
                sentence[sentence_length++] = id_found;
            }

            if(ch=='\n') { return sentence_length;}
            if(sentence_length >= MAX_SENTENCE_LENGTH){
                return sentence_length;
            }
        }
        else if(ch=='\r') continue;
        else{
            if(word_length >= MAX_STRING - 1) word_length--;
            cur_word[word_length++] = ch;
        }
    }

    if(word_length > 0){
        // add the last word
        cur_word[word_length] = 0;
        word_length = 0;

        id_found = searchVocabID(cur_word);
        if(id_found != -1){
            if (sample > 0){
                float ran = (sqrt(vocab[id_found].count / (sample * total_words)) + 1) * (sample * total_words) / vocab[id_found].count;
                next_random = next_random * (unsigned long long)25214903917 + 11;
                if(ran < (next_random & 0xFFFF) / (float)65536) {
                    skip_cnt[thread_id]++;
                    total_skip_cnt++;
                    return sentence_length;
                }
            }
            sentence[sentence_length++] = id_found;
        }
    }
    if(sentence_length==0) return -1;
    return sentence_length;
}

int searchVocabID(char* word){
    int hash_key = getWordHash(word);
    while(1){
        if(hash[hash_key]==-1) return -1;
        if(strcmp(vocab[hash[hash_key]].word, word)==0) return hash[hash_key];
        hash_key = (hash_key+1) % MAX_VOCAB_SIZE;
    }
}

void initUnigramTable(){
    int a, i;
    double train_words_pow = 0;
    double d1, power = 0.75;
    unigram_table = (int*)malloc(unigram_table_size * sizeof(int));
    for (a = 0; a < n_of_words; a++) {
        train_words_pow += pow(vocab[a].count, power);
    }
    i=0;
    d1 = pow(vocab[a].count, power) / train_words_pow;
    for(a=0; a<unigram_table_size; a++){
        unigram_table[a] = i;
        if(a / (double)unigram_table_size > d1){
            i++;
            d1 += pow(vocab[i].count, power) / train_words_pow;                
        }
        if(i >= n_of_words) i = n_of_words - 1;
    }
}

char* IDtoWord(int id){
    return vocab[id].word;
}
