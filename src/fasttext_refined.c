#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <locale.h>

#include "fasttext.h"

// strings for IO
char input_file[MAX_STRING];
char output_file[MAX_STRING-9];
char output_file_word[MAX_STRING]="word_";
char output_file_subword[MAX_STRING]="subword_";

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

//int* subword_hash;
int* word_hash;
int size_of_subword_hash = 2000000; // 2M
int size_of_word_hash = MAX_VOCAB_SIZE;

// training info
long long file_size = 0;
int n_of_thread;

int n_of_vocab = 0;
long long total_word_count = 0;
long long trained_word_count = 0;

// training hyperparameters
int window_size;
int hidden_size;
int epoch;
float starting_lr;
float lr;

// subsampling
float sample = 1e-4;
long long *skip_cnt;
long long total_skip_cnt=0;

// negative sampling
int ns_sample;
int* unigram_table;
int size_of_unigram_table=1e8;

// for efficiency
float* expTable;

// model parameters
float* word_vec;
float* subword_vec;
float* out_layer_vec;


void* training_thread(void* id_ptr){
    // Set thread info
    long long id = (long long)id_ptr;

    // Prepare training
    unsigned long long next_random = id;
    long long random_window;

    int* sentence = (int*)malloc(sizeof(int)*(MAX_SENTENCE_LENGTH+1));
    int sentence_len;
    int* unknown_sub_ids; // used to handle unknown words
    char* unknown_words[MAX_SENTENCE_LENGTH];
    int n_of_unknown_sub;
    int flag;

    int target, target_pos;
    float target_vector[hidden_size];
    int context, context_pos;

    long long local_trained_word;
    long long local_last_trained_word;
    long long word_per_thread = total_word_count / n_of_thread;

    float layer_grad[hidden_size];

    FILE* fp = fopen(input_file, "r");

    // Start training
    for (int ep=0; ep<epoch; ep++){
        clock_t start = time(NULL);

        fseek(fp, (file_size / (long long)n_of_thread) * (long long)id, SEEK_SET);
        local_trained_word = 0;
        local_last_trained_word = 0;

        if (id==0) { printf("\nRunning Epoch %d\n", ep+1); }
        while(1){
            sentence_len = readSentenceFromFile(fp, sentence, (long long)id, ep, unknown_words);
            local_trained_word += skip_cnt[id];
            if (sentence_len < 0) break;

            for (target_pos=0; target_pos<sentence_len; target_pos++){
                // 0. Calculate current learning rate
                if (local_trained_word - local_last_trained_word > 10000 || local_trained_word==0){
                    trained_word_count += local_trained_word - local_last_trained_word;
                    local_last_trained_word = local_trained_word;

                    lr = starting_lr*(1-(float)trained_word_count/(float)(epoch*total_word_count+1));
                    if(lr<starting_lr*0.0001) lr = starting_lr*0.0001;
                    if(id==0){
                        printf("\rLearning rate: %f, Progress: %.4f, time: %ld", lr, (float)(local_trained_word)/(float)(total_word_count/n_of_thread), time(NULL)-start);
                        fflush(stdout);
                    }
                }

                // 1. Set target
                target = sentence[target_pos];
                
                if(target==-1){ // unknown word in sentence
                    n_of_unknown_sub=1;
                    for (int n=minn; n<=maxn; n++){
                        if(n>strlen(unknown_words[target_pos])+2) break;
                        n_of_unknown_sub += strlen(unknown_words[target_pos])-n+3;
                    }
                    unknown_sub_ids = (int*)malloc(sizeof(int)*n_of_unknown_sub);
                    getWordVectorFromString(unknown_words[target_pos], target_vector, unknown_sub_ids, n_of_unknown_sub);
                    free(unknown_words[target_pos]);
                    flag=1;
                } else { // target word in vocab
                    getWordVector(target, target_vector);
                    flag = 0;
                }

                // 2. forward pass
                for (int h=0; h<hidden_size; h++) { // reset gradient
                    layer_grad[h] = 0.0;
                }

                next_random = next_random * (unsigned long long)25214903917 + 11;
                random_window = next_random % window_size + 1;

                for (context_pos=target_pos-random_window; context_pos<=target_pos+random_window; context_pos++){
                    if(context_pos < 0) continue;
                    if(context_pos >= sentence_len) break;
                    if(context_pos == target_pos) continue;

                    float g, f;
                    int current_sample, label;

                    // set context
                    context = sentence[context_pos];
                    if(context==-1) continue;

                    for(int d=0; d<ns_sample+1; d++){
                        // pick sample
                        if(d==0){
                            current_sample = context;
                            label=1;
                        } else {
                            next_random = next_random*(unsigned long long)25214903917 + 11;
                            current_sample = unigram_table[(next_random >> 16)%size_of_unigram_table];
                            if(current_sample==context) continue;
                            label=0;
                        }

                        // dot product
                        f = 0.0;
                        for (int h=0; h<hidden_size; h++){
                            f += target_vector[h] * out_layer_vec[current_sample*hidden_size+h];
                        }

                        // sigmoid
                        if ( f>MAX_EXP) g = (label-1)*lr;
                        else if (f<-MAX_EXP) g = label*lr;
                        else g = (label - expTable[(int)((f+MAX_EXP)*(EXP_TABLE_SIZE/MAX_EXP/2))]) * lr;

                        // backward pass
                        for (int h=0; h<hidden_size; h++){
                            layer_grad[h] += g*out_layer_vec[current_sample*hidden_size+h];
                        }
                        for (int h=0; h<hidden_size; h++){
                            out_layer_vec[current_sample*hidden_size+h] += g* target_vector[h];
                        }
                    }
                }

                // updating vectors
                if(flag==0){ // target word is in vocab
                    for(int i=0; i<vocab[target].n_of_subwords; i++){
                        for(int h=0; h<hidden_size; h++){
                            subword_vec[vocab[target].subword_ids[i]*hidden_size + h] += layer_grad[h];
                        }
                    }
                    for(int h=0; h<hidden_size; h++){
                        word_vec[target*hidden_size+h] += layer_grad[h];
                    }
                } else { // target word not in vocab
                    for(int i=0; i<n_of_unknown_sub; i++){
                        for(int h=0; h<hidden_size; h++){
                            subword_vec[unknown_sub_ids[i]*hidden_size + h] += layer_grad[h];
                        }
                    }
                    free(unknown_sub_ids);
                }
                local_trained_word++;
            }

            if (local_trained_word > word_per_thread) {
                trained_word_count += local_trained_word - local_last_trained_word;
                local_last_trained_word = local_trained_word;

                lr = starting_lr*(1-(float)trained_word_count/(float)(epoch*total_word_count+1));
                if(lr<starting_lr*0.0001) lr = starting_lr*0.0001;
                if(id==0){
                    printf("\rLearning rate: %f, Progress: %.4f, time: %ld", lr, (float)(local_trained_word)/(float)(total_word_count/n_of_thread), time(NULL)-start);
                    fflush(stdout);
                }
                break;
            }
        } 
    }

    free(sentence);
    fclose(fp);

    printf("\nThread %lld returning\n", id);
    fflush(stdout);

    return NULL;
}


int main(int argc, char** argv){
    setlocale(LC_ALL, ".UTF8");

    if(argc!=12){
        printf("Usage: ./fasttext hidden_size window_size minn maxn min_count ns_sample n_of_thread_number epoch binary data_file output_file\n");
        return -1;
    }
    else{
        hidden_size = atoi(argv[1]);
        window_size = atoi(argv[2]);
        minn = atoi(argv[3]);
        maxn = atoi(argv[4]);
        min_count = atoi(argv[5]);
        ns_sample = atoi(argv[6]);
        n_of_thread = atoi(argv[7]);
        epoch = atoi(argv[8]);
        binary = atoi(argv[9]);
        strcpy(input_file, argv[10]);
        strcpy(output_file, argv[11]);
    }
    starting_lr = 0.05;
    lr = starting_lr;
    printf("Starting learning rate : %f\n", starting_lr);

    // 1. Preperation
    word_hash = (int*)calloc(size_of_word_hash, sizeof(int));
    vocab = (struct WORD*)calloc(size_of_vocab, sizeof(struct WORD));
    //subword_hash = (int*)calloc(size_of_subword_hash, sizeof(int));

    readWordsFromFile(input_file);
    reduceWords();
    initUnigramTable();
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
    out_layer_vec = (float*)malloc(sizeof(float)*hidden_size*n_of_vocab);
    for(int i=0; i<n_of_vocab; i++){
        for(int h=0; h<hidden_size; h++){
            random_number = random_number * (unsigned long long)25214903917 + 11;
            out_layer_vec[i*hidden_size + h] = (((random_number & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
        }
    }
    word_vec = (float*)malloc(sizeof(float)*hidden_size*n_of_vocab);
    for(int i=0; i<n_of_vocab; i++){
        for(int h=0; h<hidden_size; h++){
            random_number = random_number * (unsigned long long)25214903917 + 11;
            word_vec[i*hidden_size + h] = (((random_number & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
        }
    }

    // 2. Train
    printf("Training... ");
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
    printf("\n Training done... took %ld, last learning rate: %f trained_words: %lld\n", end_time-start_time, lr, trained_word_count);


    // 3. Save vectors
    strcat(output_file_subword, output_file);
    printf("output file: %s\n", output_file_subword);
    FILE* outfp = fopen(output_file_subword, "wb");
    if(outfp == NULL) {printf("subword file open error\n");}
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
        fclose(outfp);
    }

    strcat(output_file_word, output_file);
    printf("output file: %s\n", output_file_word);
    outfp = fopen(output_file_word, "wb");
    if(outfp == NULL) printf("word file open error\n");
    else{
        fprintf(outfp, "%d %d\n", n_of_vocab, hidden_size);

        float target_vector[hidden_size];
        for(int i=0; i<n_of_vocab; i++){
            getWordVector(i, target_vector);

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
        fclose(outfp);
    }

    // 4. Free everything
    free(skip_cnt);
    free(id);
    free(word_vec);
    free(out_layer_vec);
    free(subword_vec);
    free(expTable);
    for(int i=0; i<n_of_vocab; i++){
        free(vocab[i].code);
        free(vocab[i].point);
        free(vocab[i].subword_ids);
    }
    free(vocab);
    free(word_hash);

    printf("Done\n");
    return 0;
}


unsigned int getHash(char* word, int max_hash){
    max_hash = (unsigned int)max_hash;

    unsigned int hash_key = 2166136261; // basis
    for (int i=0; i<strlen(word); i++){
        hash_key = hash_key ^ (unsigned int)((signed char)word[i]);
        hash_key *= 16777619;
    }
    hash_key = hash_key%max_hash;
    return hash_key;
}

void readWordsFromFile(char* file_name){
    printf("Reading words from file... ");
    resetHashTable();

    FILE* fp = fopen(file_name, "r");
    if(fp==NULL) {printf("Data file not found\n"); exit(1);}
    printf("File name: %s \n", file_name);

    char ch;
    char* cur_word = (char*)calloc(MAX_STRING, sizeof(char));
    int word_length = 0;
    unsigned int hash_key;

    while((ch = fgetc(fp)) != EOF){
        if(ch==13) continue;
        if(ch == ' ' || ch == '\n' || ch == '\t' || ch == '\0'){
            if (word_length == 0) continue;

            total_word_count++;
            cur_word[word_length] = 0;
            word_length = 0;
            hash_key = getHash(cur_word, size_of_word_hash);

            while(1){
                if(word_hash[hash_key]==-1){
                    if(n_of_vocab >= size_of_vocab){
                        // allocate more space if neccessary
                        size_of_vocab += 2048;
                        vocab = realloc(vocab, size_of_vocab*sizeof(struct WORD));
                        if(vocab==NULL){
                            printf("Reallocation failed\n"); exit(1);
                        }
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
        else{
            cur_word[word_length++] = ch;
            if(word_length >= MAX_STRING - 3) word_length--;
        }
    }

    free(cur_word);
    fclose(fp);
    printf("Done\n");
}

int _comp(const void* a, const void* b){
    return ((struct WORD*)b)->count - ((struct WORD*)a)->count;
}

void reduceWords(){
    printf("Reducing words.... ");

    // 1. Sort vocab by count
    qsort(vocab, n_of_vocab, sizeof(struct WORD), _comp);

    // 2. Allocate spaces for words, discard those that appear too less
    resetHashTable(0);
    total_word_count = 0;
    unsigned int hash_key;
    for(int i=0; i<n_of_vocab; i++){
        if(vocab[i].count < min_count || i>= MAX_VOCAB_SIZE){
            n_of_vocab = i;
            break;
        }
        vocab[i].code = (char*)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[i].point = (int*)calloc(MAX_CODE_LENGTH, sizeof(int));

        vocab[i].n_of_subwords = 1;
        for(int n=minn; n<=maxn; n++){
            vocab[i].n_of_subwords += strlen(vocab[i].word)+3-n;
        }
        vocab[i].subword_ids = (unsigned int*)malloc(sizeof(unsigned int)*vocab[i].n_of_subwords);
        vocab[i].subwords = (char**)malloc(sizeof(char*) * vocab[i].n_of_subwords);
        for(int k=0; k<vocab[i].n_of_subwords; k++){
            vocab[i].subwords[k] = (char*)calloc((strlen(vocab[i].word)+2)*4+1, sizeof(char));
        }

        hash_key = getHash(vocab[i].word, size_of_word_hash);
        while(word_hash[hash_key]!=-1){
            hash_key = (hash_key+1)%size_of_word_hash;
        }
        word_hash[hash_key] = i;
        //total_word_count += vocab[i].count; // get rid of this, fastText will train on all words
    }
    printf("Done\n");
    printf("number of vocab: %d\n", n_of_vocab);
    printf("total words: %lld\n", total_word_count);
}

void calculateSubwordIDs(){
    printf("assigning subword IDs to each vocab... ");
    
    unsigned int hash_key;
    int current_id=0;

    char* cur_word = (char*)calloc(MAX_STRING, sizeof(char));
    cur_word[0] = BOW;
    for(int i=0; i<n_of_vocab; i++){
        memcpy(cur_word+1, vocab[i].word, strlen(vocab[i].word));
        cur_word[strlen(vocab[i].word)+1] = EOW;

        calculateSubwords(cur_word, vocab[i].subwords);

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

// Finds all subwords and add to passed "subwords"
void calculateSubwords(char* word, char** subwords){
    int idx = 0;
    int pos;
    int len = utf8_strlen(word);
    char current_subword[MAX_STRING];

    int char_len;
    int initial_char_len;
    int word_bytes;

    strncpy(subwords[idx], word, len);
    idx++;

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

// reset word hash
void resetHashTable(){
    for(int i=0; i<size_of_word_hash; i++){
        word_hash[i] = -1;
    }
    return;
}


// builds list of word ids, returns the length of it
int readSentenceFromFile(FILE* fp, int* sentence, long long thread_id, int iter, char** unknown_words){
    char ch;
    char cur_word[MAX_STRING] = {0};
    int word_length=0;
    int sentence_length=0;
    int id_found;
    unsigned long long next_random = thread_id;
    next_random += (unsigned long long)iter * 17;

    skip_cnt[thread_id] = 0;
    while(!feof(fp)){
        ch = fgetc(fp);
        if(ch==' ' || ch=='\t' || ch=='\n'){

            if(word_length==0) continue;
            cur_word[word_length] = 0;
            word_length = 0;

            id_found = searchVocabID(cur_word);

            if(sample > 0){
                float ran;
                if (id_found==-1){ ran = (sqrt(1 / (sample * total_word_count)) + 1) * (sample * total_word_count) / 1; }
                else{ ran = (sqrt(vocab[id_found].count / (sample * total_word_count)) + 1) * (sample * total_word_count) / vocab[id_found].count; }

                next_random = next_random * (unsigned long long)25214903917 + 11;
                if(ran < (next_random & 0xFFFF) / (float)65536) {
                    skip_cnt[thread_id]++;
                    total_skip_cnt++;
                    continue;
                }
            }
            if (id_found==-1){
                unknown_words[sentence_length] = (char*)calloc(MAX_STRING, sizeof(char));
                strcpy((unknown_words[sentence_length]), cur_word);
            }
            sentence[sentence_length++] = id_found;

            if(ch=='\n') { return sentence_length;}
            if(sentence_length >= MAX_SENTENCE_LENGTH){ return sentence_length;}
        }
        else {
            if(word_length >= MAX_STRING-3) word_length--;
            cur_word[word_length++] = ch;
        }
    }

    if(word_length > 0){
        // add the last word
        cur_word[word_length] = 0;
        word_length = 0;

        id_found = searchVocabID(cur_word);
        if (sample > 0){
            float ran;
            if (id_found==-1){ ran = (sqrt(1 / (sample * total_word_count)) + 1) * (sample * total_word_count) / 1; }
            else{ ran = (sqrt(vocab[id_found].count / (sample * total_word_count)) + 1) * (sample * total_word_count) / vocab[id_found].count; }

            next_random = next_random * (unsigned long long)25214903917 + 11;
            if(ran < (next_random & 0xFFFF) / (float)65536) {
                skip_cnt[thread_id]++;
                total_skip_cnt++;
                return sentence_length;
            }
        }

        if (id_found==-1){
            unknown_words[sentence_length] = (char*)calloc(MAX_STRING, sizeof(char));
            strcpy((unknown_words[sentence_length]), cur_word);
        }
        sentence[sentence_length++] = id_found;
    }
    if(sentence_length==0) return -1;
    return sentence_length;
}

int searchVocabID(char* word){
    unsigned int hash_key = getHash(word, size_of_word_hash);

    if(word_hash[hash_key]==-1) return -1;
    while(strcmp(vocab[word_hash[hash_key]].word, word)!=0){
        hash_key = (hash_key+1)%(unsigned int)size_of_word_hash;
        if(word_hash[hash_key]==-1) return -1;
    }
    return word_hash[hash_key];
}

char* IDtoWord(int id){
    return vocab[id].word;
}


// Calculates vectors for OOV words
void getWordVectorFromString(char* word, float* target_vec, int* subwords_id, int n_of_subwords){

    if(n_of_subwords==0){
        n_of_subwords=1;
        for (int n=minn; n<=maxn; n++){
            if(n>strlen(word)+2) break;
            n_of_subwords += strlen(word)-n+3;
        }
    }

    char** subwords = (char**)malloc(sizeof(char*)*n_of_subwords);
    char* tmp = (char*)calloc(strlen(word)+3, sizeof(char));
    tmp[0] = BOW;
    strncpy(tmp[1], word, strlen(word));
    tmp[strlen(word)+1] = EOW;

    for(int i=0; i<n_of_subwords; i++){
        subwords[i] = (char*)calloc((strlen(word)+2)*4+1, sizeof(char));
    }

    calculateSubwords(tmp, subwords);

    unsigned int hash_key;
    for(int i=0; i<n_of_subwords; i++){
        subwords_id[i] = getHash(subwords[i], size_of_subword_hash);
    }

    for(int h=0; h<hidden_size; h++){
        target_vec[h] = 0.0;
    }

    for(int i=0; i<n_of_subwords; i++){
        for(int h=0; h<hidden_size; h++){
            target_vec[h] += subword_vec[hidden_size*subwords_id[i] + h];
        }
    }
    for(int h=0; h<hidden_size; h++){
        target_vec[h] *= (1/(float)n_of_subwords);
    }

    for(int i=0; i<n_of_subwords; i++){
        free(subwords[i]);
    }
    free(subwords);
    free(tmp);
}

void getWordVector(int id, float* target_vec){

    if(id >= n_of_vocab){
        printf("ID out of bound\n");
        exit(1);
    }

    for(int h=0; h<hidden_size; h++){
        target_vec[h] = 0.0;
    }

    for(int i=0; i< vocab[id].n_of_subwords; i++){
        for(int h=0; h<hidden_size; h++){
            target_vec[h] += subword_vec[hidden_size*vocab[id].subword_ids[i] + h];
        }
    }
    for(int h=0; h<hidden_size; h++){
        target_vec[h] += word_vec[id*hidden_size + h];
    }
    for(int i=0; i<hidden_size; i++){
        target_vec[i] *= (1/(float)(vocab[id].n_of_subwords+1));
    }
}

void initUnigramTable(){
    int a, i;
    double train_words_pow = 0;
    double d1, power = 0.75;
    
    unigram_table = (int*)malloc(sizeof(int) * size_of_unigram_table);
    for(a=0; a<n_of_vocab; a++){
        train_words_pow += pow(vocab[a].count, power);
    }
    i=0;
    d1 = pow(vocab[a].count, power) / train_words_pow;
    for(a=0; a<size_of_unigram_table; a++){
        unigram_table[a] = i;
        if(a / (double)size_of_unigram_table > d1){
            i++;
            d1 += pow(vocab[i].count, power) / train_words_pow;
        }
        if(i>=n_of_vocab) i = n_of_vocab - 1;
    }
}
