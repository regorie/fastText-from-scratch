#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <locale.h>

#include "fasttext.h"

// cars for IO
char input_file[MAX_STRING];
char output_file[MAX_STRING];
char output_file_word[MAX_STRING]="word_";
char output_file_subword[MAX_STRING]="subword_";
int binary = 1;
long long file_size = 0;

// training params
int n_of_thread;
int window_size = 5;
int hidden_size;
int min_count = 5;
int epoch;
float starting_lr;
float lr;

char BOW = '<', EOW = '>';
int minn = 3, maxn = 6;

// subsampling
float sample = 1e-4;
long long *skip_cnt;
long long total_skip_cnt = 0;

// Structures, vars used for training
int* subword_hash;
int* word_hash;
int size_of_subword_hash = 2000000; // 2M
int size_of_word_hash = MAX_VOCAB_SIZE;
struct WORD* vocab;
int size_of_vocab = 2048;

int n_of_words = 0;
long long int total_words = 0;
long long int trained_word = 0;
float* expTable;

// negative sampling
int ns_sample=5;
int* unigram_table;
int unigram_table_size=1e8;

float* subwords_vec;
float* out_words_vec;
float* words_vec;

void* training_thread(void* id_ptr){
    long long id = (long long)id_ptr;

    FILE* infp = fopen(input_file, "r");
    long long* sentence = (long long*)malloc(sizeof(long long)*MAX_SENTENCE_LENGTH);
    long long target, target_pos;
    float target_vector[hidden_size];
    long long context, context_pos;
    long long sentence_length;
    int unkown_sub_ids[MAX_SENTENCE_LENGTH];
    char* unkown_words[MAX_SENTENCE_LENGTH];
    int flag = 0;

    long long random_window;
    unsigned long long next_random = (long long)id;

    long long local_trained_word = 0;
    long long local_last_trained_word = 0;
    long long local_skipped_total = 0;

    float* layer_grad = (float*)calloc(hidden_size, sizeof(float));

    long long word_per_thread = total_words / n_of_thread;
    
    lr = starting_lr;
    for(int ep=0; ep<epoch; ep++){
        clock_t start = time(NULL);

        fseek(infp, (file_size / (long long)n_of_thread) * (long long)id, SEEK_SET);
        local_trained_word = 0;
        local_last_trained_word = 0;

        if(id==0) printf("\nRunning epoch %d\n", ep);
        while(1){
            sentence_length = readSentenceFromFile(infp, sentence, id, ep+1, unkown_words);
            //printf(" sentence read\n");
            if(sentence_length < 0 ) break;
            local_trained_word += skip_cnt[id];
            local_skipped_total += skip_cnt[id];
            
            for(target_pos=0; target_pos<sentence_length; target_pos++){
                // Traverse current sentence -> target

                // 0. Calculate current learning rate
                if(local_trained_word - local_last_trained_word > 10000){
                    trained_word += local_trained_word - local_last_trained_word;
                    local_last_trained_word = local_trained_word;
                    // TODO check for learning rate decay
                    lr = starting_lr*(1-(float)trained_word / (float)(epoch*total_words+1));
                    if(lr<starting_lr*0.0001) lr = starting_lr*0.0001;
                    if(id==0){
                        printf("\r Learning rate: %f, Progress: %.4f, time: %ld", lr, (float)(local_trained_word)/(float)(total_words/n_of_thread), time(NULL)-start);
                        fflush(stdout);
                    }
                }

                // 1. Set target
                target = sentence[target_pos];
                int n_of_subwords_ = 0;
                if(target==-1) {
                    //printf(" target not in vocab %d\n", target);
                    for(int n=minn; n<=maxn; n++){
                        if(n>strlen(unkown_words[target_pos])+2) break;
                        n_of_subwords_ += strlen(unkown_words[target_pos])+2-n+1;
                    }
                    //printf(" finding target vector\n");
                    getWordVectorFromString(unkown_words[target_pos], target_vector, unkown_sub_ids, n_of_subwords_);
                    free(unkown_words[target_pos]);
                    flag=1;
                    //printf(" target set\n");
                }
                else{
                    //printf(" target in vocab %d\n", target);
                    getWordVector(target, target_vector);
                    for(int b=0; b<hidden_size; b++){
                        target_vector[b] += words_vec[target*hidden_size + b]*(1/(float)(vocab[target].n_of_subwords+1));
                    }
                    flag=0;
                }
                //printf(" target set %d\n", target);
                // 2. Forward pass
                // reset gradient
                for (int b=0; b<hidden_size; b++){
                    layer_grad[b] = 0.0;
                }
                next_random = next_random * (unsigned long long)25214903917 + 11;
                random_window = next_random % window_size;
                //printf(" window size set\n");
                for(context_pos = target_pos-random_window; context_pos<=target_pos+random_window; context_pos++){
                    if(context_pos<0) continue;
                    if(context_pos>=sentence_length) break;

                    if(context_pos != target_pos){
                        float g, f;
                        int current_sample, label;

                        context = sentence[context_pos];
                        if(context==-1) {
                            continue;
                        }

                        for(int d=0; d<ns_sample+1; d++){
                            // pick sample
                            if(d==0){
                                current_sample = context;
                                label=1;
                            }
                            else{
                                next_random = next_random*(unsigned long long)25214903917 + 11;
                                current_sample = unigram_table[(next_random >> 16)%unigram_table_size];
                                if(current_sample==context) continue;
                                label=0;
                            }

                            // dot product
                            f = 0.0;
                            for(int c=0; c<hidden_size; c++){
                                f += target_vector[c] * out_words_vec[current_sample*hidden_size + c];
                            }

                            // sigmoid
                            if ( f > MAX_EXP) g = (label-1)*lr;
                            else if (f < -MAX_EXP) g = (label-0)*lr;
                            else g = (label - expTable[(int)((f+MAX_EXP)*(EXP_TABLE_SIZE/MAX_EXP/2))]) * lr;
                            
                            // 2. backward pass
                            for (int c=0 ;c< hidden_size; c++){
                                layer_grad[c] += g * out_words_vec[current_sample*hidden_size + c];
                            }
                            for (int c=0; c<hidden_size; c++){
                                out_words_vec[current_sample*hidden_size + c] += g*target_vector[c];
                            }
                        }
                    }
                }
                // updating subwords/words vector
                if(flag==0){ // target word in vocab
                    //for (int b=0; b<hidden_size; b++){
                    //    words_vec[target*hidden_size + b] += layer_grad[b];
                    //}
                    for(int b=0; b<hidden_size; b++){ // hmm... is this unnecessary
                        layer_grad[b] *= (1/(float)(vocab[target].n_of_subwords+1));
                    }
                    for (int sub=0; sub<vocab[target].n_of_subwords; sub++){
                        for(int b=0; b<hidden_size; b++){
                            subwords_vec[vocab[target].subword_ids[sub]*hidden_size + b] += layer_grad[b];
                        }
                    }
                    for (int b=0; b<hidden_size; b++){
                        words_vec[target*hidden_size + b] += layer_grad[b];
                    }
                }
                else if(flag==1){ // target word not in vocab
                    for(int b=0; b<hidden_size; b++){ // hmm... is this unnecessary
                        layer_grad[b] *= (1/(float)n_of_subwords_);
                    }
                    for (int sub=0; sub<n_of_subwords_; sub++){
                        for(int b=0; b<hidden_size; b++){
                            subwords_vec[unkown_sub_ids[sub]*hidden_size + b] += layer_grad[b];
                        }
                    }
                }
                local_trained_word++;
            }

            if(local_trained_word > word_per_thread){
                trained_word += local_trained_word - local_last_trained_word;
                lr = starting_lr*(1-trained_word/(float)(epoch*total_words+1));
                if(lr < starting_lr*0.0001) lr = starting_lr*0.0001;
                if(id==0){
                    printf("\rLearning rate: %f, Progress: %.4f, time: %ld", lr, (float)(local_trained_word)/(float)(total_words/n_of_thread), time(NULL)-start);
                    fflush(stdout);
                }
                break;
            }
        }
    }

    free(layer_grad);
    free(sentence);
    fclose(infp);

    printf("Thread %lld returning\n", id);
    fflush(stdout);

    return NULL;
}

int main(int argc, char** argv){
    setlocale(LC_ALL, ".UTF8");

    if(argc!=12){
        printf("Usage example: ./fasttext hidden_size window_size minn maxn min_count ns_sample n_of_thread_number epoch binary data_file output_file\n");
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
    printf("Starting learning rate : %f\n", starting_lr);
    
    // 1. Prepare for training
    word_hash = (int*)calloc(size_of_word_hash, sizeof(int));
    vocab = (struct WORD*)calloc(size_of_vocab, sizeof(struct WORD));
    subword_hash = (int*)calloc(size_of_subword_hash, sizeof(int));

    readWordsFromFile(input_file);
    reduceWords();

    initUnigramTable();
    buildSubwordHash();

    expTable = (float*)malloc((EXP_TABLE_SIZE+1)*sizeof(float));
    for(int i=0; i<EXP_TABLE_SIZE; i++){
        expTable[i] = exp((i/(float)EXP_TABLE_SIZE*2-1)*MAX_EXP);
        expTable[i] = expTable[i] / (expTable[i] + 1);
    }

    // Initialize model
    subwords_vec = (float*)malloc(sizeof(float)*(hidden_size*size_of_subword_hash));
    long long random_number = time(NULL);
    for(int a=0; a<size_of_subword_hash; a++){
        for(int b=0; b<hidden_size; b++){
            random_number = random_number * (unsigned long long)25214903917 + 11;
            subwords_vec[a*hidden_size + b] = (((random_number & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
        }
    }
    out_words_vec = (float*)malloc(sizeof(float)*hidden_size*n_of_words);
    for(int a=0; a<n_of_words; a++){
        for(int b=0; b<hidden_size; b++){
            random_number = random_number * (unsigned long long)25214903917 + 11;
            out_words_vec[a*hidden_size + b] = (((random_number & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
        }
    }
    words_vec = (float*)malloc(sizeof(float)*hidden_size*n_of_words);
    for(int a=0; a<n_of_words; a++){
        for(int b=0; b<hidden_size; b++){
            random_number = random_number * (unsigned long long)25214903917 + 11;
            words_vec[a*hidden_size + b] = (((random_number & 0xFFFF) / (float)65536) - 0.5) / hidden_size;
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
    printf("\n Training done... took %ld, last learning rate: %f trained_words: %lld\n", end_time-start_time, lr, trained_word);

    // 3. Save subword vectors
    strcat(output_file_subword, output_file);
    printf("output file: %s\n", output_file_subword);
    FILE* outfp = fopen(output_file_subword, "wb");
    if(outfp == NULL) printf("subword file open error\n");

    fprintf(outfp, "%d %d\n", size_of_subword_hash, hidden_size);

    for(int a=0; a<size_of_subword_hash; a++){
        //fprintf(outfp, "%s ", subword_list+(maxn*4+1)*a);

        if(binary) {
            for(int b=0; b<hidden_size; b++){
                fwrite(&subwords_vec[a*hidden_size + b], sizeof(float), 1, outfp);
            }
        }
        else{
            for(int b=0; b<hidden_size; b++){
                fprintf(outfp, "%lf ", subwords_vec[a*hidden_size+b]);
            }
        }
        fprintf(outfp, "\n");
    }
    fclose(outfp);

    //4. save word vectors
    strcat(output_file_word, output_file);
    printf("output file: %s\n", output_file_word);
    outfp = fopen(output_file_word, "wb");
    if(outfp == NULL) printf("word file open error\n");

    fprintf(outfp, "%lld %lld\n", (long long)n_of_words, (long long)hidden_size);
    for(int a=0; a<n_of_words; a++){

        fprintf(outfp, "%s ", vocab[a].word);

        if(binary) {
            for(int b=0; b<hidden_size; b++){
                fwrite(&words_vec[a*hidden_size + b], sizeof(float), 1, outfp);
            }
        }
        else{
            for(int b=0; b<hidden_size; b++){
                fprintf(outfp, "%lf ", words_vec[a*hidden_size + b]);
            }
        }
        fprintf(outfp, "\n");
    }
    fclose(outfp);

    printf("Saving done\n");

    free(id);
    free(word_hash);
    free(subword_hash);
    free(words_vec);
    free(subwords_vec);
    free(expTable);
    free(skip_cnt);

    for(int i=0; i<n_of_words; i++){
        free(vocab[i].code);
        free(vocab[i].point);
        free(vocab[i].subword_ids);
    }
    free(vocab);

    printf("Done\n");

    return 0;
}

unsigned int getHash(char* word, long long int max_hash){
    unsigned int hash_key = 2166136261;
    for(int i=0; i<strlen(word); i++){
        hash_key = hash_key^(unsigned int)((signed char)word[i]); // might need to change order
        hash_key = hash_key* 16777619;
    }
    hash_key = hash_key%max_hash;
    return hash_key;
}

void readWordsFromFile(char* file_name){
    printf("Reading words from file... ");
    resetHashTable(0);

    FILE* infp = fopen(file_name, "r");
    if(infp == NULL) { printf("data file not found\n"); exit(1); }
    printf("File name: %s ...", file_name);

    char ch;
    char* cur_word = (char*)calloc(MAX_STRING, sizeof(char));
    int word_length = 0;
    int hash_key;

    while((ch = fgetc(infp)) != EOF){
        if(ch==13) continue;
        if(ch == ' ' || ch == '\n' || ch == '\t' || ch == '\0'){
            if (word_length == 0) continue;

            total_words++;
            cur_word[word_length] = 0;
            word_length = 0;
            hash_key = getHash(cur_word, size_of_word_hash);

            while(1){
                if(word_hash[hash_key]==-1){
                    if(n_of_words >= size_of_vocab){
                        // allocate more space if neccessary
                        size_of_vocab += 2048;
                        vocab = realloc(vocab, size_of_vocab*sizeof(struct WORD));
                        if(vocab==NULL){
                            printf("Reallocation failed\n"); exit(1);
                        }
                    }
                    word_hash[hash_key] = n_of_words;
                    vocab[n_of_words].count = 1;
                    strcpy(vocab[n_of_words].word, cur_word);
                    n_of_words++;
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
            if(word_length >= MAX_STRING - 2) word_length--;
        }
    }

    free(cur_word);
    fclose(infp);
    printf("Done\n");
}

int _comp(const void* a, const void* b){
    return ((struct WORD*)b)->count - ((struct WORD*)a)->count;
}

void reduceWords(){
    // 1. Sort vocab by count
    // 2. Discard too less frequently appearing words
    // 3. allocate space for codes
    // 4. recompute hash
    printf("Reducing words... ");
    qsort(vocab, n_of_words, sizeof(struct WORD), _comp);

    resetHashTable(0);
    total_words = 0;
    int hash_key;
    for(int i=0; i<n_of_words; i++){
        if(vocab[i].count < min_count || i >= MAX_VOCAB_SIZE) {
            n_of_words = i;
            break;
        }
        vocab[i].code = (char*)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[i].point = (int*)calloc(MAX_CODE_LENGTH, sizeof(int));
        vocab[i].n_of_subwords = 0;
        for(int n=minn; n<=maxn; n++){
            if(n>strlen(vocab[i].word)+2) break;
            vocab[i].n_of_subwords += strlen(vocab[i].word)+2-n+1;
        }

        vocab[i].subword_ids = (int*)malloc(sizeof(int) * vocab[i].n_of_subwords);
        vocab[i].subwords = (char**)malloc(sizeof(char*) * vocab[i].n_of_subwords);
        for(int subs=0; subs<vocab[i].n_of_subwords; subs++){
            vocab[i].subwords[subs] = (char*)calloc(maxn*4+1, sizeof(char));
        }

        hash_key = getHash(vocab[i].word, size_of_word_hash);
        while(word_hash[hash_key]!=-1){
            hash_key = (hash_key+1)%size_of_word_hash;
        }
        word_hash[hash_key] = i;
        total_words += vocab[i].count;
    }
    printf("Done\n");
    printf("n_of_words after reducing vocab %d\n", n_of_words);
    printf("total words after reducing vocab %lld\n", total_words);
}


void buildSubwordHash(){
    printf("Building subword hash table... ");
    resetHashTable(1);
    
    unsigned int hash_key;
    int current_id = 0;
    
    char* cur_word = (char*)malloc(sizeof(char) * MAX_STRING);
    cur_word[0] = BOW;
    for(int i=0; i<n_of_words; i++){

        memcpy(cur_word+1, vocab[i].word, strlen(vocab[i].word));
        cur_word[strlen(vocab[i].word)+1] = EOW;
        cur_word[strlen(vocab[i].word)+2] = '\0';

        calculateSubwords(cur_word, vocab[i].subwords);

        for(int j=0; j< vocab[i].n_of_subwords; j++){
            hash_key = getHash(vocab[i].subwords[j], size_of_subword_hash);
            
            if(subword_hash[hash_key]==-1){
                subword_hash[hash_key] = current_id;
                current_id++;
            }
            vocab[i].subword_ids[j] = subword_hash[hash_key]; // store as id
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

int utf8_strlen(const char *s) {
    int len = 0;
    int i = 0;
    int nbytes = strlen(s);

    while (i < nbytes) {
        int char_len = utf8_charlen((unsigned char)s[i]);
        i += char_len;
        len++;
    }
    return len;
}

void calculateSubwords(char* word, char** subwords){
    // find all subwords and add to "subwords"
    int idx = 0;
    int pos;
    int n, len = utf8_strlen(word);
    char current_subword[MAX_STRING];

    int char_len;
    int initial_char_len;
    int word_bytes = 0;

    //strncpy(subwords[idx], word, len);
    //idx++;

    for (n = minn; n <= maxn; n++) {
        pos = 0;
        for(int cnt=0; cnt<=len-n; cnt++){ // pos is the starting position of current subword

            int p=pos;  // p is the currently estimated character position
            word_bytes=0;
            initial_char_len=utf8_charlen((unsigned char)word[pos]);
            for(int c=0; c<n; c++){ // calculate the byte length of n characters
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

void resetHashTable(int mode){
    if(mode==0){
        // reset word hash
        for(int i=0; i<size_of_word_hash; i++){
            word_hash[i] = -1;
        }
        return;
    }
    else if(mode==1){
        // reset subword hash
        for(int i=0; i<size_of_subword_hash; i++){
            subword_hash[i] = -1;
        }
        return;
    }
}

// returns list of word ids
int readSentenceFromFile(FILE* fp, long long* sentence, long long thread_id, int iter, char** unkown_words){
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

            if (sample > 0){
                float ran;
                if (id_found==-1){ ran = (sqrt(1 / (sample * total_words)) + 1) * (sample * total_words) / 1; }
                else{ ran = (sqrt(vocab[id_found].count / (sample * total_words)) + 1) * (sample * total_words) / vocab[id_found].count; }

                next_random = next_random * (unsigned long long)25214903917 + 11;
                if(ran < (next_random & 0xFFFF) / (float)65536) {
                    skip_cnt[thread_id]++;
                    total_skip_cnt++;
                    continue;
                }
            }
            if (id_found==-1){
                unkown_words[sentence_length] = (char*)calloc(MAX_STRING, sizeof(char));
                unkown_words[sentence_length][0] = BOW;
                strcpy((unkown_words[sentence_length])+1, cur_word);
                unkown_words[sentence_length][strlen(cur_word)+1] = EOW;
                unkown_words[sentence_length][strlen(cur_word)+2] = '\0';
            }
            sentence[sentence_length++] = id_found;

            if(ch=='\n') { return sentence_length;}
            if(sentence_length >= MAX_SENTENCE_LENGTH){
                return sentence_length;
            }
        }
        else if(ch=='\r') continue;
        else{
            if(word_length >= MAX_STRING - 3) word_length--;
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
            if (id_found==-1){ ran = (sqrt(1 / (sample * total_words)) + 1) * (sample * total_words) / 1; }
            else{ ran = (sqrt(vocab[id_found].count / (sample * total_words)) + 1) * (sample * total_words) / vocab[id_found].count; }

            next_random = next_random * (unsigned long long)25214903917 + 11;
            if(ran < (next_random & 0xFFFF) / (float)65536) {
                skip_cnt[thread_id]++;
                total_skip_cnt++;
                return sentence_length;
            }
        }

        if (id_found==-1){
            unkown_words[sentence_length] = (char*)calloc(MAX_STRING, sizeof(char));
            unkown_words[sentence_length][0] = BOW;
            strcpy((unkown_words[sentence_length])+1, cur_word);
            unkown_words[sentence_length][strlen(cur_word)+1] = EOW;
            unkown_words[sentence_length][strlen(cur_word)+2] = '\0';
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

// this function is used to calculate vectors that is not in vocab
void getWordVectorFromString(char* word, float* word_vec, int* subwords_id, int n_of_subwords){

    char** subwords = (char**)malloc(sizeof(char*)*n_of_subwords);
    for(int subs=0; subs<n_of_subwords; subs++){
            subwords[subs] = (char*)calloc(maxn*4+1, sizeof(char));
    }
    calculateSubwords(word, subwords);

    unsigned int hash_key;
    for(int i=0; i<n_of_subwords; i++){
        subwords_id[i] = getHash(subwords[i], size_of_subword_hash);
    }

    for(int h=0; h<hidden_size; h++){
        word_vec[h] = 0.0;
    }
    for (int i=0; i<n_of_subwords; i++){
        for (int h=0; h<hidden_size; h++){
            word_vec[h] += subwords_vec[hidden_size*subwords_id[i] + h];
        }
    }
    for(int i=0; i<hidden_size; i++){
        word_vec[i] *= (1/(float)n_of_subwords);
    }
}

void getWordVector(int id, float* word_vec){

    if(id >= n_of_words){
        printf("ID out of bound\n");
        exit(1);
    }

    for(int h=0; h<hidden_size; h++){
        word_vec[h] = 0.0;
    }

    for (int i=0; i<vocab[id].n_of_subwords; i++){
        for (int h=0; h<hidden_size; h++){
            word_vec[h] += subwords_vec[hidden_size*vocab[id].subword_ids[i] + h];
        }
    }
    for( int i=0; i<hidden_size; i++){
        word_vec[i] *= (1/(float)(vocab[id].n_of_subwords+1));
    }
}

void initUnigramTable(){
    int a, i;
    double train_words_pow = 0;
    double d1, power = 0.75;
    
    unigram_table = (int*)malloc(sizeof(int) * unigram_table_size);
    for(a=0; a<n_of_words; a++){
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
        if(i>=n_of_words) i = n_of_words - 1;
    }
}
