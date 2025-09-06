#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_VOCAB_SIZE 10000000 // maximum vocabulary size 10M
#define MAX_SENTENCE_LENGTH 1024*1024 // maximum number of characters in one sentence
#define MAX_SENTENCE_WORD 160000 // maximum number of words in one sentence
#define MAX_STRING 100

#define MAX_CODE_LENGTH 40

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

struct WORD { // idx is id(hash table)
    int count;
    char word[MAX_STRING];
    //char* code;
    //int codelen;
    //int* point;

    char** subwords;
    unsigned int* subword_ids;
    int n_of_subwords;
};

struct LABEL {
    int count;
    char label[16];

    char* code;
    int codelen;
    int* point;
};

unsigned int getHash(char* word, int max_hash_size); // calculate hash value

// preparation
void readWordsFromFile(char* file_name);
void reduceWords();
void calculateSubwordIDs();
void buildBinaryTree();

void calculateSubwords(char* word, int vocab_id);
void calculateSubwordsToBuff(char* word, char** subwords);

int getWordVector(int id, float* result_vec, int* subword_features, int* subword_idx);
int getWordVectorFromString(char* word, float* result_vec, int* subword_features, int* subword_idx);
void getSentenceVector(int* sentence, int sentence_len, char** unknown_words, float* sent_vec, int* word_features, int* word_feature_idx, int* subword_features, int* subword_feature_idx);

int getSentenceSample(FILE* fp, int* _label, int* sentence, char** unknown_words);

// some utils
int wordToID(char* word);
//char* IDToWord(int id);

void resetHashTable();
