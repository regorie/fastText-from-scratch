// same with ver1, but to avoid confusion
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_VOCAB_SIZE 10000000 // maximum vocabulary size 10M
#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

struct WORD { // idx is id(hash table)
    int count;
    char word[MAX_STRING];
    char* code;
    int codelen;
    int* point;

    char** subwords;
    unsigned int* subword_ids;
    int n_of_subwords;
};

unsigned int getHash(char* word, int max_hash);
void readWordsFromFile(char* file_name);
void reduceWords();
//void buildSubwordHash();
void calculateSubwordIDs();

void calculateSubwords(char* word, char** subwords);

void getWordVectorFromString(char* word, float* word_vec, int* subwords_id, int n_of_subwords);
void getWordVector(int id, float* word_vec);
int searchVocabID(char* word);
char* IDtoWord(int id);

void resetHashTable();
void initUnigramTable();

int readSentenceFromFile(FILE* fp, long long* sentence, long long id, int iter, char** unkown_words);

