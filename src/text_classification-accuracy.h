#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_VOCAB_SIZE 10000000 // maximum vocabulary size 10M
#define MAX_SENTENCE_LENGTH 1024*1024 // maximum number of characters in one sentence
#define MAX_SENTENCE_WORD 160000 // maximum number of words in one sentence
#define MAX_STRING 100

struct WORD { // idx is id(hash table)
    char word[MAX_STRING];
};


unsigned int getHash(char* word, int max_hash_size); // calculate hash value

void getWordVectorFromString(char* word, float* result_vec);

