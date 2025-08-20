// same with ver1, but to avoid confusion
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_VOCAB_SIZE 3000000 // maximum vocabulary size 3M
#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

struct WORD { // idx is id(hash table)
    int count;
    char word[MAX_STRING];
    char* code;
    int codelen;
    int* point;

    char** subwords;
    unsigned long long int* subword_ids;
    int n_of_subwords;
};

unsigned int getHash(char* word, long long int max_hash);
void readWordsFromFile(char* file_name);
void reduceWords();
void buildSubwordHash();

void calculateSubwords(char* word, char** subwords, int n_of_subwords);

void getWordVectorFromString(char* word, float* word_vec);
void getWordVector(int id, float* word_vec);
int searchVocabID(char* word);
char* IDtoWord(int id);

void resetHashTable(int mode);
void initUnigramTable();

int readSentenceFromFile(FILE* fp, long long* sentence, long long id, int iter);

